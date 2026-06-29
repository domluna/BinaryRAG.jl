# BinaryRAG.jl

`BinaryRAG.jl` is a high-performance, concurrent Approximate Nearest Neighbor (ANN) search engine built in Julia, specifically optimized for binary spaces (Hamming distance) using the Hierarchical Navigable Small World (HNSW) graph algorithm.

It is designed to handle large-scale embedding datasets (e.g., 36M+ PubMed embeddings) with extremely low query latency and a minimal memory footprint.

---

## Performance Quick-Look (1M PubMed Embeddings)
- **Index Build Time**: **35.7 seconds** (parallel construction on 4 threads).
- **Query Latency ($k=10$)**: **122.6 $\mu$s** per query (single-threaded, 100% allocation-free).
- **Query Speedup**: **37.2x faster** than parallel exact search.
- **Distance Accuracy**: **98.3%** recall.

---

## Installation & Quick Start

```julia
using BinaryRAG
using StaticArrays

# 1. Initialize a dataset of 10,000 512-bit (64-byte) binary vectors
n = 10_000
data = [SVector{8, UInt64}(rand(UInt64, 8)) for _ in 1:n]

# 2. Build the HNSW index in parallel (uses Threads.nthreads())
# M = 16 (connectivity), efConstruction = 100
hnsw = HNSW(16, n)
for i in 1:n
    insert!(hnsw, data[i])
end

# 3. Perform a query (k = 10, efSearch = 100)
query = SVector{8, UInt64}(rand(UInt64, 8))
results = search(hnsw, query, 10; expansion_search=100)
```

For **100% allocation-free** queries in production:
```julia
# Pre-allocate the search context once per thread
ctx = SearchContext(hnsw, 100, 1000)
result_buffer = Vector{Int}(undef, 10)

# In-place search (0 heap allocations, 0 bytes allocated!)
n_found = search!(result_buffer, hnsw, query, ctx)
```

---

## Advanced Performance Architecture & Optimizations

This codebase implements several non-obvious, low-level optimization techniques to bypass Julia's garbage collector (GC), maximize CPU cache locality, and utilize hardware-level parallelism.

### 1. Software Prefetching via LLVM Intrinsics (`Base.llvmcall`)

#### The Problem
During HNSW graph traversal, we jump non-contiguously from node to node. The CPU's hardware prefetcher (which looks for simple sequential strides like `i, i+1, i+2`) cannot predict where the graph traversal will go next, leading to frequent L1/L2 cache misses. Since fetching data from DRAM takes **~100ns** (200+ CPU cycles), the CPU spends most of its time idling.

#### The Solution
We know the list of all neighbors `neighbors_buf[1:c_neighbors_count]` *before* we start computing their distances. We use an LLVM prefetch intrinsic to tell the CPU's Memory Management Unit (MMU) to start loading the 64-byte vector data from DRAM into the L1 cache **asynchronously** in the background:

```julia
@inline function prefetch(ptr::Ptr{Nothing})
    Base.llvmcall(
        ("""
         declare void @llvm.prefetch(i8*, i32, i32, i32)
         
         define void @entry(i8* %ptr) {
             call void @llvm.prefetch(i8* %ptr, i32 0, i32 3, i32 1)
             ret void
         }
         """, "entry"),
        Nothing, Tuple{Ptr{Nothing}}, ptr
    )
end
```

#### Why it is Safe
The hardware `prefetch` instruction is **non-faulting**. If you pass an invalid, null, or unmapped pointer, the CPU simply **ignores the instruction** and does nothing. It will never trigger a segmentation fault or CPU exception.

#### Impact
At the 1 million vector scale (where the dataset size exceeds the CPU's L2 cache), this optimization **reduced query latency by 26.3%** and **reduced index construction time by 15.2%** by hiding memory latency.

---

### 2. Striped Lock Pool & Flat Counts (83% Allocation Reduction)

#### The Problem
To make index construction thread-safe, we originally allocated one `SpinLock` per node, and wrapped neighbor counts in `Threads.Atomic{Int}`. For $N=100,000$ over 10 levels, this created **1.1 million individual heap-allocated objects** in the constructor, causing massive garbage collection (GC) overhead.

#### The Solution
1. **Flat Counts**: Since reads and writes to neighbor lists are already protected by locks, we replaced the atomic counts with a flat `Vector{Int}` per level (exactly 10 allocations total).
2. **Striped Locking**: Instead of one lock per node, we allocate a fixed pool of **2,048 locks**. Any node `n` maps to a lock index using a hash: `lock_idx = mod1(n, 2048)`.

With only 4–16 threads, the probability of two threads trying to lock different nodes that map to the same lock is extremely low ($<0.2\%$), resulting in virtually **zero lock contention** while reducing lock allocations from 100,000 to **exactly 2,048**.

#### Impact
This reduced the HNSW constructor allocations by **83.6%** (from 1.3M down to 215k) and saved **~18 MB** of RAM.

---

### 3. Parallel Index Construction (Atomic Work-Stealing)

Instead of spawning a task per node (which creates massive scheduling overhead), `BinaryRAG.jl` spawns exactly `T = Threads.nthreads()` long-running tasks. 

```
[Main Thread] ── Spawns T Tasks ──> [Task 1] [Task 2] ... [Task T]
                                       │        │            │
                                       ▼        ▼            ▼
                                 [   Atomic Work-Stealing Loop   ]
                                 [  Steals next index from counter  ]
```

Each task allocates exactly **one** `SearchContext` at startup and enters a loop, stealing the next node index to insert using an atomic counter:
```julia
next_idx = Threads.Atomic{Int}(2)
tasks = map(1:T) do _
    Threads.@spawn begin
        ctx = SearchContext(hnsw, expansion_factor, maximum_candidates)
        while true
            i = Threads.atomic_add!(next_idx, 1)
            i > n && break
            insert!(hnsw, ctx, data[i])
        end
    end
end
wait.(tasks)
```
This keeps memory allocations and garbage collection overhead during parallel construction identical to sequential construction.

---

### 4. Hardware SIMD Vectorization for Hamming Distance

Each 512-bit (64-byte) binary embedding fits perfectly in a single **512-bit CPU register** (e.g., ZMM on x86, or two/four registers on ARM).

We define a specialized signature for `SVector{8, UInt64}` that uses static loop bounds and `@inbounds`:
```julia
@inline function hamming_distance(a::SVector{8,UInt64}, b::SVector{8,UInt64})::Int
    s = 0
    @inbounds for i = 1:8
        s += count_ones(a[i] ⊻ b[i])
    end
    return s
end
```
Because the loop bounds are static, LLVM fully unrolls this loop and compiles it directly into native SIMD instructions:
- **ARM (NEON)**: Emits `eor.16b` (128-bit XOR) and `cnt.16b` (NEON bit count).
- **x86-64 (AVX-512)**: Emits `vpxor` and `vpopcntq`.

This allows the CPU to compute the 512-bit Hamming distance in **1–2 clock cycles** without any branching or loop overhead.

---

## Benchmarks & Verification

To run the recall and query speed benchmarks yourself:

```bash
# Run 100k PubMed evaluation
julia -t auto --project=. benchmark/recall.jl 100000

# Run 1M PubMed evaluation
julia -t auto --project=. benchmark/recall.jl 1000000 10 100
```
