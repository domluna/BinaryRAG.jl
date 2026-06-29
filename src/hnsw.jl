mutable struct HNSW
    const connectivity::Int
    const connectivity0::Int
    const mL::Float64

    const graphs::Vector{Vector{Vector{Int}}}

    enter_point::Int
    data::Vector{SVector{8,UInt64}}

    function HNSW(
        connectivity::Int;
        connectivity0::Int = connectivity * 2,
        mL::Float64 = 1 / log(connectivity),
    )
        new(connectivity, connectivity0, mL, Vector{Vector{Int}}[], 1, SVector{8,UInt64}[])
    end
end

function rand_level(hnsw::HNSW)::Int
    floor(Int, (-log(rand()) * hnsw.mL) + 1)
end

# SearchContext encapsulating heaps and visited buffer for 100% allocation-free queries
mutable struct SearchContext
    const candidates::MinHeap
    const W::MaxHeap
    visited::Vector{Int}
    epoch::Int

    function SearchContext(hnsw::HNSW, expansion_search::Int = 30, maximum_candidates::Int = 1000)
        new(
            MinHeap(maximum_candidates),
            MaxHeap(expansion_search),
            zeros(Int, length(hnsw.data)),
            1
        )
    end
end

# In-place insertion sort for small vectors (0 allocations, extremely fast for size <= 100)
function insertion_sort!(data::Vector{Pair{Int,Int}}, len::Int)
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

# Fast greedy search for upper layers of the HNSW graph (0 allocations)
function _greedy_search(hnsw::HNSW, query::SVector{8,UInt64}, ep::Int, level::Int)::Int
    curr = ep
    curr_d = hamming_distance(query, hnsw.data[curr])
    changed = true
    while changed
        changed = false
        neighbors = hnsw.graphs[level][curr]
        for e in neighbors
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

# In-place search layer using SearchContext (0 allocations)
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

    while length(candidates) > 0
        d_c, c = pop!(candidates)

        if length(W) >= W.k && d_c > W.data[1].first
            break
        end

        neighbors = hnsw.graphs[level][c]
        for e in neighbors
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
    push!(hnsw.data, q)
    ind = length(hnsw.data)
    l = rand_level(hnsw)
    new_entry_point = l > length(hnsw.graphs)

    old_L = length(hnsw.graphs)
    # Ensure hnsw.graphs has at least l levels
    while length(hnsw.graphs) < l
        push!(hnsw.graphs, [sizehint!(Int[], hnsw.connectivity) for _ in 1:ind])
    end

    # Ensure each existing level has length ind
    for level = 1:old_L
        mx = level == 1 ? hnsw.connectivity0 : hnsw.connectivity
        push!(hnsw.graphs[level], sizehint!(Int[], mx))
    end

    if ind == 1
        return
    end

    L = length(hnsw.graphs)
    ep = hnsw.enter_point
    
    # 1. Greedy search on upper layers down to l+1
    for level = L:-1:l+1
        ep = _greedy_search(hnsw, q, ep, level)
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
        
        n_neighbors = min(hnsw.connectivity, len_W)
        neighbors = Int[ctx.W.data[i].second for i = 1:n_neighbors]
        hnsw.graphs[level][ind] = neighbors
        
        for n in neighbors
            n_neighbors_list = hnsw.graphs[level][n]
            if ind ∉ n_neighbors_list
                if length(n_neighbors_list) < mx
                    push!(n_neighbors_list, ind)
                else
                    # O(M) Max-Replacement: Find the neighbor with the maximum distance to n
                    max_d = -1
                    max_idx = -1
                    d_new = hamming_distance(hnsw.data[ind], hnsw.data[n])
                    for i in 1:length(n_neighbors_list)
                        @inbounds x = n_neighbors_list[i]
                        d_x = hamming_distance(hnsw.data[x], hnsw.data[n])
                        if d_x > max_d
                            max_d = d_x
                            max_idx = i
                        end
                    end
                    if d_new < max_d
                        @inbounds n_neighbors_list[max_idx] = ind
                    end
                end
            end
        end

        ep = neighbors[1]

        reset!(ctx.candidates)
        reset!(ctx.W)
        ctx.epoch += 1
    end

    if new_entry_point
        hnsw.enter_point = ep
    end
    return
end

function construct(
    n::Int;
    connectivity::Int = 16,
    expansion_factor = 100,
    maximum_candidates::Int = 1000,
)::HNSW
    hnsw = HNSW(connectivity; connectivity0 = connectivity * 2)
    ctx = SearchContext(hnsw, expansion_factor, maximum_candidates)
    for _ = 1:n
        q = SVector{8,UInt64}(reinterpret(UInt64, rand(Int8, 64)))
        insert!(hnsw, ctx, q)
    end
    for lc = 1:length(hnsw.graphs)
        cnt = count(i -> i == hnsw.enter_point || !isempty(hnsw.graphs[lc][i]), 1:length(hnsw.data))
        println("Layer $lc: length = $cnt")
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
    n_nodes = length(hnsw.data)
    if length(ctx.visited) < n_nodes
        resize!(ctx.visited, max(n_nodes, length(ctx.visited) * 2))
        fill!(ctx.visited, 0)
        ctx.epoch = 1
    end

    L = length(hnsw.graphs)
    ep = hnsw.enter_point
    for level = L:-1:2
        ep = _greedy_search(hnsw, query, ep, level)
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
    distances = [hamming_distance(query, hnsw.data[i]) for i = 1:length(hnsw.data)]
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
