module BinaryRAG

using Base.Threads
using StaticArrays

# Export Heap & Search APIs
export MaxHeap, MinHeap, reset!, insert!, heapify!, makeheap!
export k_closest, k_closest!, k_closest_parallel
export HNSW, construct, search, approx_vs_exact
export hamming_distance

include("heap.jl")
include("exact.jl")
include("hnsw.jl")

end # module
