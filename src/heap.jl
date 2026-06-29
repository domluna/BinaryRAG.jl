# MaxHeap Implementation
mutable struct MaxHeap
    const data::Vector{Pair{Int,Int}}
    current_idx::Int
    const k::Int

    function MaxHeap(k::Int)
        new(fill(typemax(Int) => -1, k), 1, k)
    end
end

function Base.length(h::MaxHeap)
    return h.current_idx - 1
end

function Base.getindex(h::MaxHeap, inds...)
    h.data[inds...]
end

function reset!(heap::MaxHeap)
    heap.current_idx = 1
end

function sift_up_max!(data::Vector{Pair{Int,Int}}, val::Pair{Int,Int}, curr::Int)
    @inbounds while curr > 1
        parent = div(curr, 2)
        if val.first > data[parent].first
            data[curr] = data[parent]
            curr = parent
        else
            break
        end
    end
    @inbounds data[curr] = val
end

function sift_down_max!(data::Vector{Pair{Int,Int}}, val::Pair{Int,Int}, curr::Int, len::Int)
    @inbounds while true
        left = 2 * curr
        right = 2 * curr + 1
        largest = curr

        if left <= len && data[left].first > val.first
            largest = left
        end
        if right <= len
            val_to_compare = largest == left ? data[left].first : val.first
            if data[right].first > val_to_compare
                largest = right
            end
        end

        if largest != curr
            data[curr] = data[largest]
            curr = largest
        else
            break
        end
    end
    @inbounds data[curr] = val
end

function Base.insert!(heap::MaxHeap, value::Pair{Int,Int})
    if heap.current_idx <= heap.k
        curr = heap.current_idx
        heap.current_idx += 1
        sift_up_max!(heap.data, value, curr)
    elseif value.first < heap.data[1].first
        sift_down_max!(heap.data, value, 1, heap.k)
    end
end

function insert_limit!(heap::MaxHeap, value::Pair{Int,Int}, ef::Int)
    if heap.current_idx <= ef
        curr = heap.current_idx
        heap.current_idx += 1
        sift_up_max!(heap.data, value, curr)
    elseif value.first < heap.data[1].first
        sift_down_max!(heap.data, value, 1, ef)
    end
end



# MinHeap Implementation
mutable struct MinHeap
    data::Vector{Pair{Int,Int}}
    current_idx::Int

    function MinHeap(k::Int)
        new(fill(typemax(Int) => -1, k), 1)
    end
end

function Base.length(h::MinHeap)
    return h.current_idx - 1
end

function Base.getindex(h::MinHeap, inds...)
    h.data[inds...]
end

function reset!(heap::MinHeap)
    heap.current_idx = 1
end

function sift_up_min!(data::Vector{Pair{Int,Int}}, val::Pair{Int,Int}, curr::Int)
    @inbounds while curr > 1
        parent = div(curr, 2)
        if val.first < data[parent].first
            data[curr] = data[parent]
            curr = parent
        else
            break
        end
    end
    @inbounds data[curr] = val
end

function sift_down_min!(data::Vector{Pair{Int,Int}}, val::Pair{Int,Int}, curr::Int, len::Int)
    @inbounds while true
        left = 2 * curr
        right = 2 * curr + 1
        smallest = curr

        if left <= len && data[left].first < val.first
            smallest = left
        end
        if right <= len
            val_to_compare = smallest == left ? data[left].first : val.first
            if data[right].first < val_to_compare
                smallest = right
            end
        end

        if smallest != curr
            data[curr] = data[smallest]
            curr = smallest
        else
            break
        end
    end
    @inbounds data[curr] = val
end

function Base.insert!(heap::MinHeap, value::Pair{Int,Int})
    n = length(heap.data)
    if heap.current_idx > n
        old_n = n
        resize!(heap.data, max(16, n * 2))
        @inbounds for i in old_n+1:length(heap.data)
            heap.data[i] = typemax(Int) => -1
        end
    end
    curr = heap.current_idx
    heap.current_idx += 1
    sift_up_min!(heap.data, value, curr)
end

function Base.pop!(heap::MinHeap)::Pair{Int,Int}
    len = length(heap)
    if len == 0
        throw(ArgumentError("min-heap is empty"))
    end
    @inbounds min_value = heap.data[1]
    if len > 1
        @inbounds last_value = heap.data[len]
        sift_down_min!(heap.data, last_value, 1, len - 1)
    end
    heap.current_idx -= 1
    return min_value
end

function Base.iterate(heap::MinHeap, state=1)
    if state > length(heap)
        return nothing
    end
    return heap.data[state], state + 1
end
