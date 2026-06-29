using Test
using BinaryRAG
using StaticArrays
using Base.Threads

# A simple, obviously correct brute force implementation for verification
function brute_force_k_closest(db::AbstractMatrix{T}, query::AbstractVector{T}, k::Int) where {T}
    dists = [hamming_distance(view(db, :, i), query) for i in 1:size(db, 2)]
    sort!(dists)
    return dists[1:k]
end

function brute_force_k_closest(db::AbstractVector{V}, query::AbstractVector{T}, k::Int) where {T,V}
    dists = [hamming_distance(db[i], query) for i in eachindex(db)]
    sort!(dists)
    return dists[1:k]
end

@testset "BinaryRAG.jl Tests" begin
    # Setup test data
    N = 1000
    db_matrix = rand(Int8, 64, N)
    query = rand(Int8, 64)

    db_vector = [db_matrix[:, i] for i in 1:N]
    db_svector = [SVector{64, Int8}(db_matrix[:, i]) for i in 1:N]
    query_svector = SVector{64, Int8}(query)

    @testset "Brute Force Verification" begin
        r_mat = brute_force_k_closest(db_matrix, query, 5)
        r_vec = brute_force_k_closest(db_vector, query, 5)
        @test r_mat == r_vec
    end

    @testset "Exact Search (k = 1)" begin
        expected = brute_force_k_closest(db_matrix, query, 1)

        # Test Matrix
        @test [r.first for r in k_closest(db_matrix, query, 1)] == expected
        @test [r.first for r in k_closest_parallel(db_matrix, query, 1)] == expected

        # Test in-place Matrix
        heap_mat = MaxHeap(1)
        @test [r.first for r in k_closest!(heap_mat, db_matrix, query)] == expected

        # Test Vector
        @test [r.first for r in k_closest(db_vector, query, 1)] == expected
        @test [r.first for r in k_closest_parallel(db_vector, query, 1)] == expected

        # Test in-place Vector
        heap_vec = MaxHeap(1)
        @test [r.first for r in k_closest!(heap_vec, db_vector, query)] == expected
    end

    @testset "Exact Search (k = 10)" begin
        expected = brute_force_k_closest(db_matrix, query, 10)

        # Test Matrix
        @test [r.first for r in k_closest(db_matrix, query, 10)] == expected
        @test [r.first for r in k_closest_parallel(db_matrix, query, 10)] == expected

        # Test in-place Matrix
        heap_mat = MaxHeap(10)
        @test [r.first for r in k_closest!(heap_mat, db_matrix, query)] == expected
        @test [r.first for r in k_closest!(heap_mat, db_matrix, query)] == expected # Test reset!

        # Test Vector
        @test [r.first for r in k_closest(db_vector, query, 10)] == expected
        @test [r.first for r in k_closest_parallel(db_vector, query, 10)] == expected

        # Test in-place Vector
        heap_vec = MaxHeap(10)
        @test [r.first for r in k_closest!(heap_vec, db_vector, query)] == expected
    end

    @testset "HNSW (Approximate Search)" begin
        # Construct HNSW graph with 500 nodes
        hnsw = construct(500; connectivity=16)
        @test length(hnsw.data) == 500
        @test haskey(hnsw.graphs, 1)

        # Search for closest node
        q = SVector{8, UInt64}(reinterpret(UInt64, rand(Int8, 64)))
        results = search(hnsw, q, 5; expansion_search=30)
        @test length(results) <= 5
        @test all(1 <= idx <= 500 for idx in results)
    end

    @testset "Edge Cases" begin
        @test_throws ArgumentError k_closest(db_matrix, query, N + 1)
        heap = MaxHeap(N + 1)
        @test_throws ArgumentError k_closest!(heap, db_matrix, query)
    end
end
