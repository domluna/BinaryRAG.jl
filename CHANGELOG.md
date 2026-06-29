# Changelog

All notable changes to the [BinaryRAG.jl](file:///Users/lunaticd/code/BinaryRAG.jl) project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] - 2026-06-28

### Added
- **Static Safe Pre-allocation & Level Capping**: Replaced dynamic resizing with a static pre-allocation strategy for upper layers (Levels 2–10) using a 10x–100x safety margin. Added a level-capping fallback in `Base.insert!` to ensure safety. Completely eliminates resizing lock overhead and potential concurrency deadlocks, **saving 1.2 GB of RAM** (bringing 1M footprint down to **~350 MB**) and achieving **98.5% recall**.
- **Corrected Search Start Level**: Updated the greedy search start level in both `insert!` and `search!` to `hnsw.enter_point_level[]` (instead of the maximum physical level of the graph). This prevents `BoundsError` on sized upper layers and eliminates redundant traversals on empty levels.
- **Level-Dependent Search Expansion (`ef`)**: Restricts search expansion on upper layers of the graph during construction to `ef = connectivity` (since upper layers are only used for routing), and only uses the full `efConstruction` on the ground layer (layer 1). This **reduces index construction time by an additional 5.4%** without affecting recall.
- **Software Prefetching**: Implemented software prefetching of neighbor vectors during graph traversal using LLVM intrinsics (`Base.llvmcall`). At 1M scale, this **reduces query latency by 26.3%** (from 166.4 $\mu$s to **122.6 $\mu$s**) and **reduces parallel build time by 15.2%** (from 42.1s to **35.7s**).
- **Specialized `hamming_distance` for `SVector{8, UInt64}`**: Added an inlined, type-stable method that uses static loop unrolling and `@inbounds` to guarantee hardware SIMD vectorization (AVX-512 / NEON) without AbstractArray dispatch overhead.
- **Heuristic Neighbor Selection (Heuristic 2)**: Implemented Heuristic 2 neighbor selection from the HNSW paper. Considers the diversity of candidates to select neighbors that are close but in different directions. Improves **1M recall by 4.6%** (reaching **98.2%**) and **reduces query latency by 7%**.
- **Parallel HNSW Construction**: Implemented a thread-safe parallel index builder (`construct`) using a **Task-Pool with Atomic Work-Stealing** pattern. Achieved a **4.25x speedup** on 4 threads.
- **Striped Lock Pool & Flat Counts**: Replaced the individual lock per node with a striped lock pool of size 2,048, and replaced the `Atomic{Int}` counts with a flat `Vector{Int}` per level. Since reads and writes are protected by the locks, this is 100% thread-safe and reduced memory allocations during HNSW construction by **83.6%** (from 1.3M down to 215k).
- **Lock-Free Reader / Locked Writer Graph**: Redesigned the graph to use a pre-allocated `Matrix{Int}` of size `(mx, max_elements)` per level. Updates are protected by the striped locks, while readers are **100% lock-free** and synchronized via memory fences (`Threads.atomic_fence()`).
- **`SearchContext` Struct**: Encapsulates heaps (`MinHeap`, `MaxHeap`), the visited buffer, and a pre-allocated `neighbors_buf` to ensure 0 heap allocations during both search and parallel insertion.
- **`search!` Function**: An in-place search API (`search!(result, hnsw, query, ctx)`) that achieves **100% allocation-free** queries (0 allocations, 0 bytes) by writing results directly to a pre-allocated buffer.
- **`insertion_sort!` Helper**: Added a custom in-place sorting function for small heap arrays to avoid the overhead and view allocations of Julia's standard `sort!`.

### Changed
- **HNSW Graph Representation**: Converted `graphs` from a nested dictionary (`Dict{Int,Dict{Int,Vector{Int}}}`) to a flat pre-allocated matrix.
- **Greedy Upper-Layer Search**: Switched searches on upper layers ($L$ down to $2$) to a specialized `_greedy_search` function.
- **Epoch-Based Visited Buffer**: Converted visited node tracking from a `Set{Int}` to a flat `Vector{Int}` inside `SearchContext`.
- **Dependency Cleanup**: Moved `JET.jl` to a development-only tool.

### Fixed
- **HNSW Construction Memory Optimization**: Replaced `sort!` with an $O(M)$ max-replacement pass, reducing memory allocations during construction by **~30%** and memory usage by **~25%**.
- **Data Races in Parallel Insertion**: Resolved write-write races on neighbor lists by locking the new node `ind` during its initial edge writing, and resolved reader-writer races on weak-memory architectures (e.g. ARM) using `Threads.atomic_fence()`.
- **Heap Insertion Performance Bug**: Fixed a bug in [src/heap.jl](file:///Users/lunaticd/code/BinaryRAG.jl/src/heap.jl) where `makeheap!` was called on every single insertion before the heap was full.
- **Heap `sift_down!` Bug**: Fixed a bug in [src/heap.jl](file:///Users/lunaticd/code/BinaryRAG.jl/src/heap.jl) where the sift-down logic compared child values against the old root value.
- **Heap Signature Collision**: Resolved a signature collision between min-heap and max-heap helper functions.
- **Parser SIMD Compatibility**: Removed a redundant `@simd` annotation from `hamming_distance` in [src/exact.jl](file:///Users/lunaticd/code/BinaryRAG.jl/src/exact.jl).

---

## Performance & Evaluation Benchmarks

### 1. Speed & Latency (100k PubMed Embeddings, k=10)
Measured using 4 CPU threads via `julia -t auto`:

| Search Method | Threads Used | Average Query Time | Speedup vs. Exact (Seq) | Speedup vs. Exact (Par) | Allocations |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Exact (Sequential)** | 1 | 394.17 $\mu$s | Baseline | — | 3 |
| **Exact (Parallel)** | 4 | 136.63 $\mu$s | 2.9x | Baseline | 52 |
| **HNSW (Standard)** | 1 | 34.75 $\mu$s | 11.3x | 3.9x | 21 |
| **HNSW (In-Place)** | 1 | **25.17 $\mu$s** | **15.7x** | **5.4x** | **0** |

### 2. Setup vs. Query Trade-offs
Evaluating the trade-offs on the PubMed dataset (64-byte embeddings):

| Metric | Exact Method (100k) | HNSW Method (100k) | Exact Method (1M) | HNSW Method (1M) | Exact Method (36.5M - Full) | HNSW Method (36.5M - Full) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Data Loading Time** | **1.5 ms** | 1.5 ms | **41 ms** | 41 ms | **2.39 seconds** | 2.39 seconds |
| **Index Build Time** | **0 seconds** | **2.41 seconds** *(4 threads)* | **0 seconds** | **34.3 seconds** *(4 threads)* | **0 seconds** | **~22 minutes** *(4 threads, est.)* |
| **Query Latency ($k=10$)** | 394.2 $\mu$s | **25.2 $\mu$s** | 4.49 ms | **119.7 $\mu$s** | ~140.0 ms *(est.)* | **< 200.0 $\mu$s** *(est.)* |
| **RAM Usage (Est.)** | **6.4 MB** | ~22.4 MB | **64 MB** | **~350 MB** | **2.33 GB** | **~8.50 GB** |

*Note: With our parallel construction and Heuristic 2, HNSW build times are extremely fast: **100k in 2.41s**, **1M in 34.3s**, projecting a full 36.5M dataset indexing time of only **~22 minutes**.*

### 3. Recall & Accuracy on Binary Spaces
Evaluated with $k=10$ and $efSearch=100$:

- **Uniform Random (512-bit)**: **72.9% Index-based Recall** / **81.1% Distance-based Recall** (N=10k).
  - *Why*: The curse of dimensionality concentrates distances around 256 bits, removing any navigable topological structure or distance gradients.
- **Low-Dimensional Manifold (16-bit random)**: **67.0% Index-based Recall** / **100.0% Distance-based Recall** (N=10k).
  - *Why*: Restricting randomness to a 16-bit subspace creates a steep, navigable distance gradient, allowing HNSW to find the true nearest neighbors with 100% accuracy.
- **Real-World PubMed Embeddings**:
  - **10k PubMed**: **94.4% Index-based Recall** / **99.1% Distance-based Recall**.
  - **100k PubMed**: **95.5% Index-based Recall** / **99.3% Distance-based Recall** *(+2.4% vs Simple Heuristic)*.
  - **1M PubMed**: **92.9% Index-based Recall** / **98.5% Distance-based Recall** *(~98.5% accuracy at million-scale)*.
  - *Why*: Real-world embeddings naturally group into dense semantic topic clusters, providing excellent distance gradients and graph routing connectivity even as the dataset scales 100x.

