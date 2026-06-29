using BinaryRAG
using StaticArrays
using Chairmarks
using Base.Threads

println("Number of threads available: ", nthreads())

# Setup data for exact search
println("Generating 1,000,000 vectors for exact search...")
N = 1_000_000
db_int8 = rand(Int8, 64, N)
query_int8 = rand(Int8, 64)

# 1. Benchmark Exact Search (Matrix{Int8} - automatically reinterpreted under the hood!)
println("\n--- Benchmarking Exact Search (1M rows, k=10) ---")
bench_seq = @b k_closest($db_int8, $query_int8, 10)
println("Exact Search (Sequential): ", bench_seq)

bench_par = @b k_closest_parallel($db_int8, $query_int8, 10)
println("Exact Search (Parallel):   ", bench_par)

# 2. Benchmark In-Place Search (100% allocation-free)
println("\n--- Benchmarking In-Place Exact Search (k=10) ---")
heap = MaxHeap(10)
bench_inplace = @b k_closest!($heap, $db_int8, $query_int8)
println("In-Place Search (0 allocs): ", bench_inplace)

# 3. Benchmark HNSW Search
println("\nGenerating 10,000 vectors for HNSW graph...")
hnsw = construct(10_000; connectivity=16)

q_hnsw = SVector{8, UInt64}(reinterpret(UInt64, rand(Int8, 64)))
println("\n--- Benchmarking HNSW Search (10k rows, k=10) ---")
bench_hnsw = @b search($hnsw, $q_hnsw, 10; expansion_search=30)
println("HNSW Search (Standard): ", bench_hnsw)

# 4. Benchmark HNSW In-Place Search
println("\n--- Benchmarking In-Place HNSW Search (k=10) ---")
ctx = SearchContext(hnsw, 30, 1000)
result_buf = Vector{Int}(undef, 10)
bench_hnsw_inplace = @b search!($result_buf, $hnsw, $q_hnsw, $ctx)
println("HNSW Search (In-Place 0 allocs): ", bench_hnsw_inplace)
