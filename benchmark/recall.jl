using BinaryRAG
using StaticArrays
using Random
using Statistics
using Base.Threads

# Helper to read a subset of embeddings from the binary file
function read_embeddings_subset(filename::String, M::Int)
    if !isfile(filename)
        error("Data file '$filename' not found. Please ensure the PubMed embeddings are placed in the 'data/' folder.")
    end
    open(filename, "r") do f
        n = read(f, Int64)
        d = read(f, Int64)
        @assert d == 8 "Expected 8 UInt64s (64 bytes) per embedding, got $d"
        
        M_to_read = min(M, n)
        flat_data = Vector{UInt64}(undef, M_to_read * 8)
        read!(f, flat_data)
        
        return collect(reinterpret(SVector{8, UInt64}, flat_data))
    end
end

# Parallel index construction using the data passed in
function build_index_parallel(data; connectivity = 16, expansion_factor = 100, maximum_candidates = 1000)
    n = length(data)
    hnsw = HNSW(connectivity, n; connectivity0 = connectivity * 2)
    
    # 1. Insert the first node sequentially to establish the entry point
    ctx1 = SearchContext(hnsw, expansion_factor, maximum_candidates)
    insert!(hnsw, ctx1, data[1])
    
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
                @inbounds insert!(hnsw, ctx, data[i])
            end
        end
    end
    wait.(tasks)
    return hnsw
end

# Index-based recall (strict index matching)
function calculate_index_recall(hnsw_inds::Vector{Int}, exact_inds::Vector{Int})
    return length(intersect(hnsw_inds, exact_inds)) / length(exact_inds)
end

# Distance-based recall (accuracy of the retrieved distances)
function calculate_distance_recall(hnsw_dists::Vector{Int}, exact_dists::Vector{Int})
    k = length(exact_dists)
    max_acceptable_dist = exact_dists[k]
    return count(d -> d <= max_acceptable_dist, hnsw_dists) / k
end

function main()
    # Parse command line arguments
    # Usage: julia --project=. benchmark/recall.jl [N] [k] [efSearch]
    N = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
    k = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    expansion = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 100

    Random.seed!(42)
    
    println("=== HNSW Recall & Speed Benchmarking ===")
    println("Threads available: ", nthreads())
    println("Parameters: N = $N, k = $k, efSearch = $expansion")
    
    # Load embeddings
    println("\n1. Loading embeddings from data/data.bin...")
    t_load = @elapsed pubmed_data = read_embeddings_subset("data/data.bin", N)
    println("Successfully loaded $(length(pubmed_data)) embeddings in ", round(t_load, digits=3), " seconds.")
    
    # Construct HNSW index
    println("\n2. Constructing HNSW index in parallel (M=16, efConstruction=100)...")
    t_build = @elapsed hnsw = build_index_parallel(pubmed_data; connectivity=16, expansion_factor=100)
    println("HNSW index constructed in ", round(t_build, digits=3), " seconds.")
    
    # Evaluate recall on 100 random queries
    println("\n3. Evaluating recall on 100 random queries...")
    idx_recalls = Float64[]
    dist_recalls = Float64[]
    hnsw_query_times = Float64[]
    exact_query_times = Float64[]
    
    # Pre-allocate search context
    ctx = SearchContext(hnsw, expansion, 1000)
    res_buf = Vector{Int}(undef, k)
    
    for i in 1:100
        query = pubmed_data[rand(1:N)]
        
        # Exact search
        t_exact = @elapsed exact_results = k_closest(pubmed_data, query, k)
        exact_inds = [r.second for r in exact_results]
        exact_dists = [r.first for r in exact_results]
        
        # HNSW search
        t_hnsw = @elapsed begin
            n_found = search!(res_buf, hnsw, query, ctx)
            hnsw_inds = res_buf[1:n_found]
        end
        hnsw_dists = [hamming_distance(query, pubmed_data[idx]) for idx in hnsw_inds]
        
        push!(idx_recalls, calculate_index_recall(hnsw_inds, exact_inds))
        push!(dist_recalls, calculate_distance_recall(hnsw_dists, exact_dists))
        push!(hnsw_query_times, t_hnsw)
        push!(exact_query_times, t_exact)
    end
    
    println("\n=== Evaluation Summary ===")
    println("Average Index-based Recall:  ", round(mean(idx_recalls) * 100, digits=2), "%")
    println("Average Distance-based Recall: ", round(mean(dist_recalls) * 100, digits=2), "%")
    println("Average HNSW Query Latency:    ", round(mean(hnsw_query_times) * 1000 * 1000, digits=1), " μs")
    println("Average Exact Query Latency:   ", round(mean(exact_query_times) * 1000 * 1000, digits=1), " μs")
    println("Speedup:                       ", round(mean(exact_query_times) / mean(hnsw_query_times), digits=1), "x")
end

main()
