function approx_vs_exact(
    hnsw::HNSW,
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
