module Batches

abstract type AbstractBatch{T,batch_size,n} end

ndata(::AbstractBatch{T,batch_size,n})       where {T,batch_size,n} = n
nbatch(::AbstractBatch{T,batch_size,n})      where {T,batch_size,n} = batch_size
Base.eltype(::AbstractBatch{T,batch_size,n}) where {T,batch_size,n} = T

# this line shamelessly comes from iterators.jl
Base.length(b::AbstractBatch)  = div(ndata(b), nbatch(b)) + ((mod(ndata(b), nbatch(b)) > 0) ? 1 : 0) 
Base.endof(b::AbstractBatch)   = length(b)
Base.start(b::AbstractBatch)   = 1
Base.done(b::AbstractBatch, i) = i < length(b)
Base.next(b::AbstractBatch, i) = b[i], i + 1

struct SimpleBatch{T, batch_size, n} <: AbstractBatch{T, batch_size, n}
    data::T
    size::Tuple # size of returned batches

    function SimpleBatch(data::T, batch_size::Int) where T <: AbstractArray
        n  = last(size(data)) # number of data points
        sz = ntuple(ndims(data)) do i
            # replace the last dimension with batch_size
            i == ndims(data) ? batch_size : size(data, i)
        end
        new{T, batch_size, n}(data, sz)
    end
end

Base.size(b::SimpleBatch) = b.size
Base.size(b::SimpleBatch, i) = b.size[i]

@generated function _get_lastdim(arr::AbstractArray{T,N}, inds) where {T,N}
    get_batch = :(arr[])
    push!(get_batch.args, ntuple(i->:(:), N-1)..., :inds)
    return get_batch
end

function Base.getindex(b::SimpleBatch{T}, i::Int)::T where T
    @boundscheck i <= length(b) || throw(BoundsError())
    l = (i - 1)*nbatch(b) + 1
    r = min(i*nbatch(b), ndata(b))
    _get_lastdim(b.data, l:r)
end

struct PreallocBatch{U, batch_size, n} <: AbstractBatch{U, batch_size, n}
    b::AbstractBatch{T, batch_size, n} where T
    prealloc::U

    function PreallocBatch(b::AbstractBatch{T, batch_size, n} where T, ::Type{U}) where {batch_size, n, U<:AbstractArray} 
        prealloc = U(size(b))
        return new{U, nbatch(b), ndata(b)}(b, prealloc)
    end

    function PreallocBatch(b::AbstractBatch{T}) where T<:AbstractArray
        return PreallocBatch(b, T)
    end
end

function Base.getindex(b::PreallocBatch{T}, i::Int)::T where T 
    @boundscheck i <= length(b) || throw(BoundsError())
    b.prealloc .= b.b[i]
    return b.prealloc
end

struct BatchTuple{U, batch_size, n}  <: AbstractBatch{U, batch_size, n}
    batches::NTuple{N, <:AbstractBatch{<:Any, batch_size, n}} where N

    function BatchTuple(batches::Vararg{B}) where B <: AbstractBatch{<:Any, batch_size, n} where {batch_size, n}
        new{Tuple{eltype.(batches)...}, batch_size, n}(batches)
    end
end

Base.getindex{T}(bt::BatchTuple{T}, i::Int)::T = map(b->b[i], bt.batches)

function batches(v::AbstractArray{T,N}, batch_size::Int; prealloc = false, shuffle = false) where {T,N}
    data = !shuffle ? v : _get_lastdim(v, randperm(size(v, N)))
    batch_size <= size(v, ndims(v)) || error("batch size is larger than the number of data points")

    sb = SimpleBatch(data, batch_size) 

    if prealloc
        return PreallocBatch(sb)
    else
        return sb
    end
end

function batches(tv::NTuple{N,<:AbstractArray} where N, batch_size::Int; prealloc = false, shuffle = false)
    ndatas = last.(size.(tv))
    
    all((n = ndatas[1]) .== ndatas) || throw(DimensionMismatch(
        "incompatibly sized inputs along last dimension, got inputs with $ndatas data points $(length(tv) > 1? "respectively": "")"
    ))

    return BatchTuple(Tuple(batches(v, batch_size, prealloc = prealloc) for v in tv)...)
end

export ndata, nbatch, batches, 
    SimpleBatch, PreallocBatch, BatchTuple

end