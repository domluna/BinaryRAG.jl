# Changelog

All notable changes to the [BinaryRAG.jl](file:///Users/lunaticd/code/BinaryRAG.jl) project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] - 2026-06-28

### Added
- **`SearchContext` Struct**: Encapsulates heaps (`MinHeap`, `MaxHeap`) and the visited buffer. Allows pre-allocating state once and reusing it across queries.
- **`search!` Function**: An in-place search API (`search!(result, hnsw, query, ctx)`) that achieves **100% allocation-free** queries (0 allocations, 0 bytes) by writing results directly to a pre-allocated buffer.
- **`insertion_sort!` Helper**: Added a custom in-place sorting function for small heap arrays to avoid the overhead and view allocations of Julia's standard `sort!`.

### Changed
- **HNSW Graph Representation**: Converted `graphs` from a nested dictionary (`Dict{Int,Dict{Int,Vector{Int}}}`) to a flat nested vector (`Vector{Vector{Vector{Int}}}`). Retrieves neighbor lists via $O(1)$ flat array indexing, eliminating hash table overhead.
- **Greedy Upper-Layer Search**: Switched searches on upper layers ($L$ down to $2$) to a specialized `_greedy_search` function. Avoids allocating heaps and visited sets on layers where only a single entry point is needed.
- **Epoch-Based Visited Buffer**: Converted visited node tracking from a `Set{Int}` to a flat `Vector{Int}` inside `SearchContext`. Marks nodes using the current search `epoch`, allowing $O(1)$ resets by simply incrementing the epoch integer.
- **Dependency Cleanup**: Moved `JET.jl` to a development-only tool (completely removed from [Project.toml](file:///Users/lunaticd/code/BinaryRAG.jl/Project.toml) and [Manifest.toml](file:///Users/lunaticd/code/BinaryRAG.jl/Manifest.toml)).

### Fixed
- **Heap Insertion Performance Bug**: Fixed a bug in [src/heap.jl](file:///Users/lunaticd/code/BinaryRAG.jl/src/heap.jl) where `makeheap!` (an $O(N)$ operation) was called on every single insertion before the heap was full. Replaced with standard $O(\log N)$ sift-up/down operations.
- **Heap `sift_down!` Bug**: Fixed a bug in [src/heap.jl](file:///Users/lunaticd/code/BinaryRAG.jl/src/heap.jl) where the sift-down logic compared child values against the old root value rather than the new value being sifted down.
- **Heap Signature Collision**: Resolved a signature collision between min-heap and max-heap helper functions by renaming them to `sift_up_max!`/`sift_down_max!` and `sift_up_min!`/`sift_down_min!`.
- **HNSW Level Initialization**: Fixed a `BoundsError` in [src/hnsw.jl](file:///Users/lunaticd/code/BinaryRAG.jl/src/hnsw.jl) by ensuring newly created levels are initialized with empty neighbor lists for all existing nodes up to `ind`.
- **Parser SIMD Compatibility**: Removed a redundant `@simd` annotation from `hamming_distance` in [src/exact.jl](file:///Users/lunaticd/code/BinaryRAG.jl/src/exact.jl) to resolve a macro expansion error (`Base.SimdLoop.SimdError`) under the `JuliaLowering` parser used by `jetls`.

---

## Performance & Evaluation Benchmarks

### 1. Speed & Latency (100k PubMed Embeddings, k=10)
Measured using 4 CPU threads via `julia -t auto`:

| Search Method | Threads Used | Average Query Time | Speedup vs. Exact (Seq) | Speedup vs. Exact (Par) | Allocations |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Exact (Sequential)** | 1 | 386.04 $\mu$s | Baseline | — | 3 |
| **Exact (Parallel)** | 4 | 138.33 $\mu$s | 2.8x | Baseline | 52 |
| **HNSW (Standard)** | 1 | 31.04 $\mu$s | 12.4x | 4.5x | 15 |
| **HNSW (In-Place)** | 1 | **21.96 $\mu$s** | **17.6x** | **6.3x** | **0** |

*Note: The single-threaded in-place HNSW search is **6.3x faster** and **25x more resource-efficient** (query throughput per CPU core) than the 4-threaded parallel exact search.*

### 2. Setup vs. Query Trade-offs
Evaluating the trade-offs on the PubMed dataset (64-byte embeddings):

| Metric | Exact Method (100k) | HNSW Method (100k) | Exact Method (36.5M - Full) | HNSW Method (36.5M - Full) |
| :--- | :---: | :---: | :---: | :---: |
| **Data Loading Time** | **1.5 ms** | 1.5 ms | **2.39 seconds** | 2.39 seconds |
| **Index Build Time** | **0 seconds** | 6.08 seconds | **0 seconds** | ~55 minutes *(est.)* |
| **Query Latency ($k=10$)** | 386.0 $\mu$s | **21.9 $\mu$s** | ~140.0 ms *(est.)* | **< 100.0 $\mu$s** *(est.)* |
| **RAM Usage (Est.)** | **6.4 MB** | ~22.4 MB | **2.33 GB** | ~8.50 GB |

### 3. Recall & Accuracy on Binary Spaces
Evaluated on 10,000 vectors with $k=10$ and $efSearch=100$:

- **Uniform Random (512-bit)**: **72.9% Index-based Recall** / **81.1% Distance-based Recall**.
  - *Why*: The curse of dimensionality concentrates distances around 256 bits, removing any navigable topological structure or distance gradients.
- **Low-Dimensional Manifold (16-bit random)**: **67.0% Index-based Recall** / **100.0% Distance-based Recall**.
  - *Why*: Restricting randomness to a 16-bit subspace creates a steep, navigable distance gradient, allowing HNSW to find the true nearest neighbors with 100% accuracy.
- **Real-World PubMed Embeddings**: **94.4% Index-based Recall** / **99.1% Distance-based Recall** (scales to **92.8% / 96.9%** at 100k).
  - *Why*: Real-world embeddings naturally group into dense semantic topic clusters, providing excellent distance gradients and graph routing connectivity.
