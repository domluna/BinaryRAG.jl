mutable struct HNSW
    const connectivity::Int
    const connectivity0::Int
    const mL::Float64

    # graphs[level] is a Matrix{Int} of size (mx, max_elements)
    const graphs::Vector{Matrix{Int}}
    # counts[level] is a Vector{Int} of length max_elements (no longer atomic!)
    const counts::Vector{Vector{Int}}
    # Striped lock pool to protect node updates without allocating a lock per node
    const locks::Vector{Threads.SpinLock}
    const entry_lock::ReentrantLock

    const enter_point::Threads.Atomic{Int}
    const enter_point_level::Threads.Atomic{Int}
    const data::Vector{SVector{8,UInt64}}
    const current_size::Threads.Atomic{Int}

    function HNSW(
        connectivity::Int,
        max_elements::Int;
        connectivity0::Int = connectivity * 2,
        mL::Float64 = 1 / log(connectivity),
        max_levels::Int = 10,
        lock_pool_size::Int = 2048,
    )
        graphs = [zeros(Int, level == 1 ? connectivity0 : connectivity, max_elements) for level in 1:max_levels]
        counts = [zeros(Int, max_elements) for _ in 1:max_levels]
        locks = [Threads.SpinLock() for _ in 1:lock_pool_size]
        new(
            connectivity,
            connectivity0,
            mL,
            graphs,
            counts,
            locks,
            ReentrantLock(),
            Threads.Atomic{Int}(1),
            Threads.Atomic{Int}(1),
            Vector{SVector{8,UInt64}}(undef, max_elements),
            Threads.Atomic{Int}(0),
        )
    end
end

# Outer constructor for backward compatibility
function HNSW(
    connectivity::Int;
    connectivity0::Int = connectivity * 2,
    mL::Float64 = 1 / log(connectivity),
)
    return HNSW(connectivity, 100_000; connectivity0 = connectivity0, mL = mL)
end

function rand_level(hnsw::HNSW)::Int
    floor(Int, (-log(rand()) * hnsw.mL) + 1)
end

# Helper to get the lock index for a node in the striped lock pool
@inline function get_lock(hnsw::HNSW, n::Int)
    @inbounds return hnsw.locks[mod1(n, length(hnsw.locks))]
end

# SearchContext encapsulating heaps and visited buffer for 100% allocation-free queries
mutable struct SearchContext
    const candidates::MinHeap
    const W::MaxHeap
    visited::Vector{Int}
    epoch::Int
    const neighbors_buf::Vector{Int}
    const selected_neighbors::Vector{Int}
    const prune_candidates::Vector{Pair{Int,Int}}

    function SearchContext(hnsw::HNSW, expansion_search::Int = 30, maximum_candidates::Int = 1000)
        new(
            MinHeap(maximum_candidates),
            MaxHeap(expansion_search),
            zeros(Int, length(hnsw.data)),
            1,
            Vector{Int}(undef, hnsw.connectivity0),
            Vector{Int}(undef, hnsw.connectivity0),
            Vector{Pair{Int,Int}}(undef, hnsw.connectivity0 + 1)
        )
    end
end

# In-place insertion sort for small vectors (0 allocations, extremely fast for size <= 100)
@inline function insertion_sort!(data::Vector{Pair{Int,Int}}, len::Int)
    for i = 2:len
        @inbounds val = data[i]
        j = i - 1
        @inbounds while j >= 1 && data[j].first > val.first
            data[j+1] = data[j]
            j -= 1
        end
        @inbounds data[j+1] = val
    end
end

# Heuristic 2 Neighbor Selection (0 heap allocations)
# Selects neighbors that are close to q, but also diverse (far from each other)
function select_neighbors_heuristic!(
    selected::Vector{Int},
    hnsw::HNSW,
    candidates::Vector{Pair{Int,Int}}, # Sorted by distance to q in ascending order
    len_W::Int,
    M::Int
)::Int
    n_sel = 0
    for i in 1:len_W
        if n_sel >= M
            break
        end
        @inbounds pair = candidates[i]
        e = pair.second
        d_eq = pair.first
        
        # Check if e is closer to q than to any node already selected
        keep = true
        for j in 1:n_sel
            @inbounds r = selected[j]
            d_er = hamming_distance(hnsw.data[e], hnsw.data[r])
            if d_er < d_eq
                keep = false
                break
            end
        end
        
        if keep
            n_sel += 1
            @inbounds selected[n_sel] = e
        end
    end
    
    # Keep pruned connections: if not enough neighbors selected, fill with the remaining closest ones
    if n_sel < M
        for i in 1:len_W
            if n_sel >= M
                break
            end
            @inbounds pair = candidates[i]
            e = pair.second
            
            already_selected = false
            for j in 1:n_sel
                @inbounds if selected[j] == e
                    already_selected = true
                    break
                end
            end
            
            if !already_selected
                n_sel += 1
                @inbounds selected[n_sel] = e
            end
        end
    end
    return n_sel
end

# Fast greedy search for upper layers of the HNSW graph (0 allocations, thread-safe)
function _greedy_search(hnsw::HNSW, ctx::SearchContext, query::SVector{8,UInt64}, ep::Int, level::Int)::Int
    curr = ep
    curr_d = hamming_distance(query, hnsw.data[curr])
    changed = true
    neighbors_buf = ctx.neighbors_buf
    
    while changed
        changed = false
        
        lk = get_lock(hnsw, curr)
        lock(lk)
        c_neighbors_count = hnsw.counts[level][curr]
        for i in 1:c_neighbors_count
            @inbounds neighbors_buf[i] = hnsw.graphs[level][i, curr]
        end
        unlock(lk)
        
        for i in 1:c_neighbors_count
            @inbounds e = neighbors_buf[i]
            d_e = hamming_distance(query, hnsw.data[e])
            if d_e < curr_d
                curr = e
                curr_d = d_e
                changed = true
            end
        end
    end
    return curr
end

# In-place search layer using SearchContext (0 allocations, thread-safe)
function _search_layer!(
    hnsw::HNSW,
    ctx::SearchContext,
    query::SVector{8,UInt64},
    ep::Int,
    level::Int,
)
    candidates = ctx.candidates
    W = ctx.W
    visited_buf = ctx.visited
    epoch = ctx.epoch
    
    visited_buf[ep] = epoch
    d = hamming_distance(query, hnsw.data[ep])
    insert!(candidates, d => ep)
    insert!(W, d => ep)
    
    neighbors_buf = ctx.neighbors_buf

    while length(candidates) > 0
        d_c, c = pop!(candidates)

        if length(W) >= W.k && d_c > W.data[1].first
            break
        end

        lk = get_lock(hnsw, c)
        lock(lk)
        c_neighbors_count = hnsw.counts[level][c]
        for i in 1:c_neighbors_count
            @inbounds neighbors_buf[i] = hnsw.graphs[level][i, c]
        end
        unlock(lk)
        
        for i in 1:c_neighbors_count
            @inbounds e = neighbors_buf[i]
            if visited_buf[e] == epoch
                continue
            end
            visited_buf[e] = epoch

            d_e = hamming_distance(query, hnsw.data[e])
            if length(W) < W.k || d_e < W.data[1].first
                insert!(W, d_e => e)
                insert!(candidates, d_e => e)
            end
        end
    end
end

function Base.insert!(
    hnsw::HNSW,
    q::SVector{8,UInt64};
    expansion_factor::Int = 100,
    maximum_candidates::Int = 1000,
)
    ctx = SearchContext(hnsw, expansion_factor, maximum_candidates)
    insert!(hnsw, ctx, q)
end

function Base.insert!(hnsw::HNSW, ctx::SearchContext, q::SVector{8,UInt64})
    # Increment current_size atomically to get a unique index for this node
    ind = Threads.atomic_add!(hnsw.current_size, 1) + 1
    
    # Write the data
    hnsw.data[ind] = q
    
    l = rand_level(hnsw)
    # Limit l to max_levels
    l = min(l, length(hnsw.graphs))
    
    if ind == 1
        # Set enter point atomically
        hnsw.enter_point[] = 1
        hnsw.enter_point_level[] = l
        return
    end

    L = length(hnsw.graphs)
    ep = hnsw.enter_point[]
    
    # 1. Greedy search on upper layers down to l+1
    for level = L:-1:l+1
        ep = _greedy_search(hnsw, ctx, q, ep, level)
    end

    # Reset heaps and epoch before starting local search
    reset!(ctx.candidates)
    reset!(ctx.W)
    if ctx.epoch == typemax(Int)
        fill!(ctx.visited, 0)
        ctx.epoch = 1
    else
        ctx.epoch += 1
    end

    # Ensure visited buffer is large enough (in case graph grew)
    if length(ctx.visited) < ind
        resize!(ctx.visited, max(ind, length(ctx.visited) * 2))
        fill!(ctx.visited, 0)
        ctx.epoch = 1
    end

    # 2. Local search on layers l down to 1
    for level = l:-1:1
        mx = level == 1 ? hnsw.connectivity0 : hnsw.connectivity
        
        _search_layer!(hnsw, ctx, q, ep, level)
        
        len_W = length(ctx.W)
        insertion_sort!(ctx.W.data, len_W)
        
        # Select neighbors using Heuristic 2
        n_neighbors = select_neighbors_heuristic!(
            ctx.selected_neighbors,
            hnsw,
            ctx.W.data,
            len_W,
            mx
        )
        
        # Copy selected neighbors
        neighbors = [ctx.selected_neighbors[i] for i in 1:n_neighbors]

        # Write neighbors of ind under its own lock
        lk_ind = get_lock(hnsw, ind)
        lock(lk_ind)
        try
            for i in 1:n_neighbors
                @inbounds hnsw.graphs[level][i, ind] = neighbors[i]
            end
            Threads.atomic_fence()
            hnsw.counts[level][ind] = n_neighbors
        finally
            unlock(lk_ind)
        end
        
        # Add back-edges from neighbors to ind (each under their respective locks)
        for i in 1:n_neighbors
            @inbounds n = neighbors[i]
            
            lk_n = get_lock(hnsw, n)
            lock(lk_n)
            try
                n_count = hnsw.counts[level][n]
                if n_count < mx
                    @inbounds hnsw.graphs[level][n_count + 1, n] = ind
                    Threads.atomic_fence()
                    hnsw.counts[level][n] = n_count + 1
                else
                    # Prune n's neighbor list using Heuristic 2
                    # Candidates are n's existing neighbors plus the new node ind
                    prune_len = n_count + 1
                    for j in 1:n_count
                        @inbounds x = hnsw.graphs[level][j, n]
                        @inbounds ctx.prune_candidates[j] = hamming_distance(hnsw.data[x], hnsw.data[n]) => x
                    end
                    @inbounds ctx.prune_candidates[prune_len] = hamming_distance(hnsw.data[ind], hnsw.data[n]) => ind
                    
                    insertion_sort!(ctx.prune_candidates, prune_len)
                    
                    # Select mx diverse neighbors
                    n_pruned = select_neighbors_heuristic!(
                        ctx.selected_neighbors,
                        hnsw,
                        ctx.prune_candidates,
                        prune_len,
                        mx
                    )
                    
                    # Overwrite n's neighbor list in-place
                    for j in 1:n_pruned
                        @inbounds hnsw.graphs[level][j, n] = ctx.selected_neighbors[j]
                    end
                    Threads.atomic_fence()
                    hnsw.counts[level][n] = n_pruned
                end
            finally
                unlock(lk_n)
            end
        end

        if n_neighbors > 0
            @inbounds ep = ctx.W.data[1].second
        end

        reset!(ctx.candidates)
        reset!(ctx.W)
        ctx.epoch += 1
    end

    # Update entry point atomically if our level is higher
    if l > hnsw.enter_point_level[]
        lock(hnsw.entry_lock)
        try
            if l > hnsw.enter_point_level[]
                hnsw.enter_point[] = ind
                hnsw.enter_point_level[] = l
            end
        finally
            unlock(hnsw.entry_lock)
        end
    end
    return
end

# Multi-threaded HNSW index construction
function construct(
    n::Int;
    connectivity::Int = 16,
    expansion_factor = 100,
    maximum_candidates::Int = 1000,
)::HNSW
    hnsw = HNSW(connectivity, n; connectivity0 = connectivity * 2)
    
    # Pre-generate random vectors
    vectors = [SVector{8,UInt64}(reinterpret(UInt64, rand(Int8, 64))) for _ in 1:n]
    
    # 1. Insert the first node sequentially to establish the entry point
    ctx1 = SearchContext(hnsw, expansion_factor, maximum_candidates)
    insert!(hnsw, ctx1, vectors[1])
    
    # 2. Insert the remaining n-1 nodes in parallel using a task-pool with atomic work-stealing
    next_idx = Threads.Atomic{Int}(2)
    T = Threads.nthreads()
    tasks = map(1:T) do _
        Threads.@spawn begin
            ctx = SearchContext(hnsw, expansion_factor, maximum_candidates)
            while true
                i = Threads.atomic_add!(next_idx, 1)
                if i > n
                    break
                end
                @inbounds insert!(hnsw, ctx, vectors[i])
            end
        end
    end
    wait.(tasks)
    
    for lc = 1:length(hnsw.graphs)
        cnt = count(i -> i == hnsw.enter_point[] || hnsw.counts[lc][i] > 0, 1:hnsw.current_size[])
        if cnt > 0
            println("Layer $lc: length = $cnt")
        end
    end
    return hnsw
end

function search(
    hnsw::HNSW,
    query::SVector{8,UInt64},
    k::Int;
    expansion_search::Int = 30,
    maximum_candidates = 1000,
)::Vector{Int}
    ctx = SearchContext(hnsw, expansion_search, maximum_candidates)
    result = Vector{Int}(undef, k)
    n = search!(result, hnsw, query, ctx)
    return result[1:n]
end

# In-place search for 100% allocation-free queries using SearchContext
function search!(
    result::Vector{Int},
    hnsw::HNSW,
    query::SVector{8,UInt64},
    ctx::SearchContext;
    k::Int = length(result)
)
    reset!(ctx.candidates)
    reset!(ctx.W)

    if ctx.epoch == typemax(Int)
        fill!(ctx.visited, 0)
        ctx.epoch = 1
    else
        ctx.epoch += 1
    end

    # Ensure visited buffer is large enough
    n_nodes = hnsw.current_size[]
    if length(ctx.visited) < n_nodes
        resize!(ctx.visited, max(n_nodes, length(ctx.visited) * 2))
        fill!(ctx.visited, 0)
        ctx.epoch = 1
    end

    L = length(hnsw.graphs)
    ep = hnsw.enter_point[]
    for level = L:-1:2
        ep = _greedy_search(hnsw, ctx, query, ep, level)
    end
    
    _search_layer!(hnsw, ctx, query, ep, 1)
    
    len_W = length(ctx.W)
    insertion_sort!(ctx.W.data, len_W)
    
    n = min(k, len_W, length(result))
    for i = 1:n
        @inbounds result[i] = ctx.W.data[i].second
    end
    return n
end

function approx_vs_exact(
    hnsw,
    query::SVector{8,UInt64},
    k::Int,
    expansion_search::Int,
    maximum_candidates::Int,
)
    inds = search(
        hnsw,
        query,
        k;
        expansion_search = expansion_search,
        maximum_candidates = maximum_candidates,
    )
    approx_dists = [hamming_distance(query, hnsw.data[i]) for i in inds]
    distances = [hamming_distance(query, hnsw.data[i]) for i = 1:hnsw.current_size[]]
    k_nearest_manual = sortperm(distances)[1:k]

    println("HNSW inds ", sort(inds))
    println("Manual inds ", sort(k_nearest_manual))
    println("HNSW distances ", sort(approx_dists))
    println(
        "Manual distances ",
        sort([hamming_distance(hnsw.data[i], query) for i in k_nearest_manual]),
    )
end

function approx_vs_exact(
    hnsw::HNSW,
    k::Int;
    expansion_search::Int = 30,
    maximum_candidates::Int = 1000,
)
    q = SVector{8,UInt64}(reinterpret(UInt64, rand(Int8, 64)))
    approx_vs_exact(hnsw, q, k, expansion_search, maximum_candidates)
end
