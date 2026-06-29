@inline function hamming_distance(s1::AbstractString, s2::AbstractString)::Int
    s = 0
    for (c1, c2) in zip(s1, s2)
        if c1 != c2
            s += 1
        end
    end
    s
end

@inline function hamming_distance(x1::T, x2::T)::Int where {T<:Integer}
    return Int(count_ones(x1 ⊻ x2))
end

@inline function hamming_distance1(
    x1::AbstractArray{T},
    x2::AbstractArray{T},
)::Int where {T<:Integer}
    s = 0
    for i in eachindex(x1, x2)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

@inline function hamming_distance(
    x1::AbstractArray{T},
    x2::AbstractArray{T},
)::Int where {T<:Integer}
    s = 0
    @inbounds for i in eachindex(x1)
        s += hamming_distance(x1[i], x2[i])
    end
    s
end

function _k_closest!(
    heap::MaxHeap,
    db::AbstractVector{V},
    query::AbstractVector{T};
    startind::Int = 1,
) where {T<:Integer,V<:AbstractVector{T}}
    if heap.k == 1
        min_d = typemax(Int)
        min_idx = -1
        @inbounds for i in eachindex(db)
            d = hamming_distance(db[i], query)
            if d < min_d
                min_d = d
                min_idx = startind + i - 1
            end
        end
        heap.data[1] = min_d => min_idx
        heap.current_idx = 2
        return heap.data
    end
    reset!(heap)
    @inbounds for i in eachindex(db)
        d = hamming_distance(db[i], query)
        insert!(heap, d => startind + i - 1)
    end
    return heap.data
end

function k_closest!(
    heap::MaxHeap,
    db::AbstractVector{V},
    query::AbstractVector{T};
    startind::Int = 1,
) where {T<:Integer,V<:AbstractVector{T}}
    if heap.k > length(db)
        throw(ArgumentError("k ($(heap.k)) cannot be larger than the database size ($(length(db)))"))
    end
    data = _k_closest!(heap, db, query; startind = startind)
    return sort!(data, by = x -> x.first)
end

function k_closest(
    db::AbstractVector{V},
    query::AbstractVector{T},
    k::Int;
    startind::Int = 1,
) where {T<:Integer,V<:AbstractVector{T}}
    heap = MaxHeap(k)
    return k_closest!(heap, db, query; startind = startind)
end

function k_closest_parallel(
    db::AbstractArray{V},
    query::AbstractVector{T},
    k::Int,
) where {T<:Integer,V<:AbstractVector{T}}
    n = length(db)
    if k > n
        throw(ArgumentError("k ($k) cannot be larger than the database size ($n)"))
    end
    t = nthreads()
    if n < 10_000 || t == 1
        return k_closest(db, query, k)
    end
    task_ranges = [(i:min(i + n ÷ t - 1, n)) for i = 1:n÷t:n]
    tasks = map(task_ranges) do r
        Threads.@spawn begin
            h = MaxHeap(k)
            _k_closest!(h, view(db, r), query; startind = r[1])
        end
    end
    results = fetch.(tasks)
    sort!(vcat(results...), by = x -> x.first)[1:k]
end

function _k_closest!(
    heap::MaxHeap,
    db::AbstractMatrix{T},
    query::AbstractVector{T};
    startind::Int = 1,
) where {T<:Integer}
    if heap.k == 1
        min_d = typemax(Int)
        min_idx = -1
        @inbounds for i = 1:size(db, 2)
            d = hamming_distance(view(db, :, i), query)
            if d < min_d
                min_d = d
                min_idx = startind + i - 1
            end
        end
        heap.data[1] = min_d => min_idx
        heap.current_idx = 2
        return heap.data
    end
    reset!(heap)
    @inbounds for i = 1:size(db, 2)
        d = hamming_distance(view(db, :, i), query)
        insert!(heap, d => startind + i - 1)
    end
    return heap.data
end

function k_closest!(
    heap::MaxHeap,
    db::AbstractMatrix{T},
    query::AbstractVector{T};
    startind::Int = 1,
) where {T<:Integer}
    if heap.k > size(db, 2)
        throw(ArgumentError("k ($(heap.k)) cannot be larger than the database size ($(size(db, 2)))"))
    end
    data = _k_closest!(heap, db, query; startind = startind)
    return sort!(data, by = x -> x.first)
end

function k_closest(
    db::AbstractMatrix{T},
    query::AbstractVector{T},
    k::Int;
    startind::Int = 1,
) where {T<:Integer}
    heap = MaxHeap(k)
    return k_closest!(heap, db, query; startind = startind)
end

function k_closest_parallel(
    db::AbstractMatrix{T},
    query::AbstractVector{T},
    k::Int,
) where {T<:Integer}
    n = size(db, 2)
    if k > n
        throw(ArgumentError("k ($k) cannot be larger than the database size ($n)"))
    end
    t = nthreads()
    if n < 10_000 || t == 1
        return k_closest(db, query, k)
    end
    task_ranges = [(i:min(i + n ÷ t - 1, n)) for i = 1:n÷t:n]
    tasks = map(task_ranges) do r
        Threads.@spawn begin
            h = MaxHeap(k)
            _k_closest!(h, view(db, :, r), query; startind = r[1])
        end
    end
    results = fetch.(tasks)
    sort!(vcat(results...), by = x -> x.first)[1:k]
end

# Specialized methods for 8-bit integer matrices to trigger SIMD reinterpretation
function k_closest!(
    heap::MaxHeap,
    db::AbstractMatrix{T},
    query::AbstractVector{T};
    startind::Int = 1,
) where {T<:Union{Int8, UInt8}}
    if size(db, 1) % 8 == 0
        db_u64 = reinterpret(UInt64, db)
        N = div(length(query), 8)
        query_u64 = SVector{N, UInt64}(reinterpret(UInt64, query))
        return k_closest!(heap, db_u64, query_u64; startind = startind)
    end
    if heap.k > size(db, 2)
        throw(ArgumentError("k ($(heap.k)) cannot be larger than the database size ($(size(db, 2)))"))
    end
    data = _k_closest!(heap, db, query; startind = startind)
    return sort!(data, by = x -> x.first)
end

function k_closest(
    db::AbstractMatrix{T},
    query::AbstractVector{T},
    k::Int;
    startind::Int = 1,
) where {T<:Union{Int8, UInt8}}
    heap = MaxHeap(k)
    return k_closest!(heap, db, query; startind = startind)
end

function k_closest_parallel(
    db::AbstractMatrix{T},
    query::AbstractVector{T},
    k::Int,
) where {T<:Union{Int8, UInt8}}
    if size(db, 1) % 8 == 0
        db_u64 = reinterpret(UInt64, db)
        N = div(length(query), 8)
        query_u64 = SVector{N, UInt64}(reinterpret(UInt64, query))
        return k_closest_parallel(db_u64, query_u64, k)
    end
    n = size(db, 2)
    if k > n
        throw(ArgumentError("k ($k) cannot be larger than the database size ($n)"))
    end
    t = nthreads()
    if n < 10_000 || t == 1
        return k_closest(db, query, k)
    end
    task_ranges = [(i:min(i + n ÷ t - 1, n)) for i = 1:n÷t:n]
    tasks = map(task_ranges) do r
        Threads.@spawn begin
            h = MaxHeap(k)
            _k_closest!(h, view(db, :, r), query; startind = r[1])
        end
    end
    results = fetch.(tasks)
    sort!(vcat(results...), by = x -> x.first)[1:k]
end
