module BinaryRAG

using Base.Threads
using StaticArrays

# Export Heap & Search APIs
export HNSW, MaxHeap, MinHeap, SearchContext, approx_vs_exact, construct,
       hamming_distance, heapify!, insert!, k_closest, k_closest!,
       k_closest_parallel, makeheap!, reset!, search, search!

include("heap.jl")
include("exact.jl")
include("hnsw.jl")

end # module
