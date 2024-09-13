using LinearAlgebra, Random
using TensorRefinement.Auxiliary

import Base: length, deepcopy, reverse!, ndims, size, ones, zeros
import Base: fill!
import Random: rand!
import LinearAlgebra: rank
import Base: permutedims!
import Base: getindex
import Base: append!, prepend!, push!, pushfirst!, pop!, popfirst!, insert!, deleteat!
import Base: vcat, hcat
import LinearAlgebra: lmul!, rmul!
import Base: *, kron, +
import LinearAlgebra: qr!, svd!

export TT
export length, ndims, size, rank, ranks
export deepcopy, reverse!, permutedims!
export fill!, rand!
export getfirstfactor, getlastfactor, getfactor, setfirstfactor!, setlastfactor!, setfactor!
export rankselect!, rankselect
export getindex
export append!, prepend!, push!, pushfirst!, pop!, popfirst!, insert!, deleteat!
export compose!, compose, composecore!, composecore, composeblock!, composeblock
export vcat, hcat, dcat
export lmul!, rmul!, mul, had, *, kron, ⊗, add, +
export qr!, svd!

"""
    struct TT{T<:Number, N}

A tensor train (TT) structure that represents a given tensor in the TT format. 

# Fields
- `factors::Dec{T,N}`: A decomposition that contains the factors of the tensor train. 

# Constructor
- `TT(factors::Dec{T,N})`: Construct a TT object with the specified factors. 

Throw an `ArgumentError` if `N < 2`, since no factor should have less than two rank indices.
"""
struct TT{T<:Number,N}
	factors::Dec{T,N}

	function TT(factors::Dec{T,N}) where {T<:Number,N}
		if N < 2
			throw(ArgumentError("each factor is required to have two rank indices"))
		end
		U = new{T,N}(factors)
		return U
	end

end

"""
    Path

Type alias for a wide variety of index or range types.

# Possible Types
- `Vector{Int}`: A vector of integers representing specific indices.
- `Int`: A single integer representing a specific index.
- `UnitRange{Int}`: A continuous range of integers, e.g., `1:10`.
- `StepRange{Int,K}`: A range of integers with a step size, e.g., `1:2:10`, where `K` represents the step size type.
- `NTuple{M,Int} where M`: A tuple of integers with `M` elements, representing a fixed number of indices.
- `Vector{Any}`: A vector containing elements of any type.
- `Nothing`: A special case indicating no selection.
- `Colon`: A colon operator (`:`) used to represent all indices.
"""
const Path = Union{Vector{Int},Int,UnitRange{Int},StepRange{Int,K} where K,NTuple{M,Int} where M,Vector{Any},Nothing,Colon}

"""
    Permutation

Type alias for representing permutations of mode dimensions or indices in tensor operations.

# Possible Types
- `NTuple{K,Int}`: A tuple of `K` integers representing a permutation of indices.
- `Vector{Int}`: A vector of integers representing a permutation of indices.
"""
const Permutation = Union{NTuple{K,Int},Vector{Int}} where K

"""
    TT(::Type{T}, d::Int, L::Int) where {T<:Number}

Construct an empty `TT` object with a specified number type, number of dimensions and number of factors.

# Arguments
- `T`: The type of the numbers in the tensor.
- `d::Int`: The number of dimensions in each factor (excluding the rank dimensions).
- `L::Int`: The number of factors.

# Returns
- `TT` object with uninitialized factors.

# Throws
- `ArgumentError`: If `d` is not a valid number of dimensions (negative) or `L` is not a valid number of factors (negative).
"""
function TT(::Type{T}, d::Int, L::Int) where {T<:Number}
	checkndims(d)
	checklength(L)
	factors = Vector{Array{T,d+2}}(undef, L)
	U = TT(factors)
	return U
end

"""
    TT(d::Int, L::Int)

Construct an empty `TT` object with `Float64` as number type, and specified number of dimensions and number of factors.

# Arguments
- `d::Int`: The number of dimensions in each factor (excluding the rank dimensions).
- `L::Int`: The number of factors.

# Returns
- `TT` object with uninitialized factors and `Float64` as number type.

# Throws
- `ArgumentError`: If `d` is not a valid number of dimensions (negative) or `L` is not a valid number of factors (negative).
"""
TT(d::Int, L::Int) = TT(Float64, d, L)

"""
    TT(::Type{T}, d::Int) where {T<:Number}

Construct an empty `TT` object with a specified number type and number of dimensions, but no factors.

# Arguments
- `T`: The type of the entries in the tensor.
- `d::Int`: The number of dimensions in each factor (excluding the rank dimensions).

# Returns
- Empty `TT` object with no factors.

# Throws
- `ArgumentError`: If `d` is not a valid number of dimensions (negative)
"""
function TT(::Type{T}, d::Int) where {T<:Number}
	checkndims(d)
	return TT(Vector{Array{T,d+2}}(undef, 0))
end

"""
    TT(d::Int)

Construct an empty `TT` object with `Float64` as default number type and a specified number of dimensions, but no factors.

# Arguments
- `d::Int`: The number of dimensions in each factor (excluding the rank dimensions).

# Returns
- An empty `TT` object with no factors.

# Throws
- `ArgumentError`: If `d` is not a valid number of dimensions (negative)
"""
TT(d::Int) = TT(Float64, d)

"""
    TT(U::TT{T,N}) where {T<:Number,N}

Return the input `TT` object as is.

# Arguments
- `U::TT{T,N}`: Tensor Train of type `TT` object to return.

# Returns
- The `TT` object `U` as is.
"""
function TT(U::TT{T,N}) where {T<:Number,N}
	return U
end

"""
    TT(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Construct a tensor train (TT) object with specified factor sizes and ranks.

# Arguments
- `T`: Type of the entries in the tensor (e.g., `Float64`).
- `n::Union{DecSize, FactorSize}`: Matrix specifying the sizes of each mode (dimension) for the factors. If passed as `DecSize`, the number of rows should match the number of dimensions.
- `r::Union{Int, DecRank}`: Specifies the ranks between consecutive factors. Can be an integer (for single rank across all factors) or a vector (`DecRank`), which determines ranks for each connection between factors.
- `first::Int=0`: Optional. Specifies the first rank. Should be positive if specified.
- `last::Int=0`: Optional. Specifies the last rank. Should be positive if specified.
- `len::Int=0`: Optional. Specifies the number of factors in the TT object. Should be positive if specified.

# Returns
- `TT` object initialized with the specified type, factor sizes and ranks.

# Throws
- `ArgumentError`: If the rank is not a nonnegative integer or a vector of such
- `ArgumentError`: If the rank parameter does not contain at least two entries, when the rank parameter is specified as a vector.
- `ArgumentError`: If the rank parameter does not contain L+1 entries, when the size parameter is specified as a matrix with L columns and the rank parameter is specified as a vector.
- `ArgumentError`: If the first rank, when specified separately, is not positive.
- `ArgumentError`: If the last rank, when specified separately, is not positive.
- `ArgumentError`: If the first and last ranks are specified separately, even though the rank parameter is not specified as an integer.
- `ArgumentError`: If the number of factors, when specified, is not positive.
- `ArgumentError`: If the number of rows in the size matrix and the number of factors are not equal, when the number of rows in the size matrix is larger than one and the number of factors is specified.
- `ArgumentError`: If `len` is not equal to L, when the rank parameter is specified as a vector with L+1 entries and the number of factors len is specified.
"""
function TT(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}
	checksize(n[:,:])
	nmat = isa(n, DecSize)
	nlen = size(n, 2)
	rvec = isa(r, DecRank)
	rlen = length(r)
	if any(r .≤ 0)
		throw(ArgumentError("the rank should be a positive integer or a vector of such"))
	end
	if rvec && (rlen < 2)
		throw(ArgumentError("when the rank parameter is specified as a vector, it should contain at least two entries"))
	end
	if rvec && nmat && rlen ≠ nlen+1
		throw(ArgumentError("when the size parameter is specified as a matrix with L columns and the rank parameter is specified as a vector, the latter should contain L+1 entries"))
	end
	if first < 0
		throw(ArgumentError("the first rank, when specified separately, should be positive"))
	end
	if last < 0
		throw(ArgumentError("the last rank, when specified separately, should be positive"))
	end
	if rvec && (first > 0 || last > 0)
		throw(ArgumentError("the first and last ranks are allowed to be specified separately only if the rank parameter is specified as an integer"))
	end
	if len < 0
		throw(ArgumentError("the number of factors, when specified, should be positive"))
	end
	if len > 0
		if nmat && len ≠ nlen
			throw(ArgumentError("when the number of rows in the size matrix is larger than one and the number of factors is specified, the two should be equal"))
		end
		if rvec && len+1 ≠ rlen
			throw(ArgumentError("when the rank parameter is specified as a vector with L+1 entries and the number of factors len is specified, it is required that len=L"))
		end
	end
	L = max(nlen, rlen-1, len)
	if !nmat
		n = repeat(n[:,:], outer=(1,L))
	end
	if !rvec
		r = r*ones(Int, L+1)
		if first > 0
			r[1] = first
		end
		if last > 0
			r[L+1] = last
		end
	end
	checkrank(r; len=L)
	factors = [ Array{T}(undef, r[ℓ], n[:,ℓ]..., r[ℓ+1]) for ℓ ∈ 1:L ]
	U = TT(factors)
	return U
end

"""
    TT(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Construct a tensor train (TT) object with entry data type `Float64`, exhibting specified factor sizes and ranks.

# Arguments
- `n::Union{DecSize, FactorSize}`: Matrix specifying the sizes of each mode (dimension) for the factors. If passed as `DecSize`, the number of rows should match the number of dimensions.
- `r::Union{Int, DecRank}`: Specifies the ranks between consecutive factors. Can be an integer (for single rank across all factors) or a vector (`DecRank`), which determines ranks for each connection between factors.
- `first::Int=0`: Optional. Specifies the first rank. Should be positive if specified.
- `last::Int=0`: Optional. Specifies the last rank. Should be positive if specified.
- `len::Int=0`: Optional. Specifies the number of factors in the TT object. Should be positive if specified.

# Returns
- `TT` object initialized with the type `Float64` for entries, and the specified factor sizes and ranks.

# Throws
- `ArgumentError`: If the rank is not a nonnegative integer or a vector of such
- `ArgumentError`: If the rank parameter does not contain at least two entries, when the rank parameter is specified as a vector.
- `ArgumentError`: If the rank parameter does not contain L+1 entries, when the size parameter is specified as a matrix with L columns and the rank parameter is specified as a vector.
- `ArgumentError`: If the first rank, when specified separately, is not positive.
- `ArgumentError`: If the last rank, when specified separately, is not positive.
- `ArgumentError`: If the first and last ranks are specified separately, even though the rank parameter is not specified as an integer.
- `ArgumentError`: If the number of factors, when specified, is not positive.
- `ArgumentError`: If the number of rows in the size matrix and the number of factors are not equal, when the number of rows in the size matrix is larger than one and the number of factors is specified.
- `ArgumentError`: If `len` is not equal to L, when the rank parameter is specified as a vector with L+1 entries and the number of factors len is specified.
"""
TT(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = TT(Float64, n, r; first=first, last=last, len=len)

"""
    TT!(V::Factor{T,N}) where {T<:Number,N}

Create a tensor train (TT) object with a single factor `V` of type `Factor{T,N}`.

# Arguments
- `V::Factor{T,N}`: Tensor factor of type `factor` to initialize the TT object with.

# Returns
- `TT{T,N}` object containing the passed on factor.
"""
function TT!(V::Factor{T,N}) where {T<:Number,N}
	factors = [V]
	U = TT(factors)
	return U
end

"""
    TT(V::Factor{T,N}; len::Int=1)

Construct a tensor train (TT) decomposition with a specified length, each factor being a copy of the factor `V`.

# Arguments
- `V::Factor{T,N}`: Factor, whose entries are in `T` and with `N` dimensions, to use for creating the `TT` type.
- `len::Int=1`: Length of the created tensor train; defaults to 1.

# Returns
- Tensor train type whose factors consist of copies of `V`.

# Throws
- `ArgumentError`: If the number of factors `len`, when specified, is not positive.
- `ArgumentError`: If the two ranks of `V` are not equal when the number of factors `len` is specified as larger than one.
"""
function TT(V::Factor{T,N}; len::Int=1) where {T<:Number,N}
	if len < 0
		throw(ArgumentError("the number of factors, when specified, should be positive"))
	end
	if len == 0
		return TT(T, N-2)
	end
	if len > 1
		p,q = factorranks(V)
		if p ≠ q
			throw(ArgumentError("the two ranks of V should be equal when the number of factors is specified as larger than one"))
		end
	end
	factors = [ copy(V) for ℓ ∈ 1:len ]
	U = TT(factors)
	return U
end

"""
    length(U::TT{T,N})

Return the number of factors in the tensor train `U`.

# Arguments
- `U::TT{T,N}`: Tensor train object of type `TT`.

# Returns
- `Int`: Number of factors in `U`.

# Throws
- `ArgumentError`: If the lenght `L` is negative.
"""
function length(U::TT{T,N}) where {T<:Number,N}
	L = length(U.factors)
	checklength(L)
	return L
end

"""
    ndims(U::TT{T,N})

Return the number of mode dimensions of the tensor train object `U`.

# Arguments
- `U::TT{T,N}`: Tensor train object of type `TT`.

# Returns
- `Int`: Number of mode dimensions.
"""
function ndims(U::TT{T,N}) where {T<:Number,N}
	d = [ factorndims(V) for V ∈ U.factors ]
	checkndims(d)
	return d[1]
end

"""
    size(U::TT{T,N})

Return the sizes of each mode of the tensor train object `U`.

# Arguments
- `U::TT{T,N}`: Tensor train object of type `TT`.

# Returns
- A matrix of integers, where each column represents the different mode sizes for each factor contained in the tensor train.
"""
function size(U::TT{T,N}) where {T<:Number,N}
	L = length(U)
	d = ndims(U)
	n = [ size(U.factors[ℓ], 1+k) for k ∈ 1:d, ℓ ∈ 1:L ]
	n = n[:,:]
	checksize(n)
	return n
end

"""
    ranks(U::TT{T,N})

Return the ranks of the tensor train object `U`.

# Arguments
- `U::TT{T,N}`: Tensor train object of type `TT`.

# Returns
Two vectors of length L: `p` and `q` where `p[ℓ]` is the first rank and `q[ℓ]` is the last rank of the ℓ-th factor in the tensor train `U`.
"""
function ranks(U::TT{T,N}) where {T<:Number,N}
	L = length(U)
	d = ndims(U)
	p = [ size(U.factors[ℓ], 1) for ℓ ∈ 1:L ]
	q = [ size(U.factors[ℓ], d+2) for ℓ ∈ 1:L ]
	return p,q
end

"""
    rank(U::TT{T,N}) where {T<:Number,N}

Return a vector of ranks, which consists of all first ranks of the factors in the given tensor train and the last rank of the last factor.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.

# Returns
- A vector containing the first rank of each factor in `U` as well as the last rank of the last factor in the decomposition.

# Throws
- `DimensionMismatch`: If the factors in `U` have inconsistent ranks.
"""
function rank(U::TT{T,N}) where {T<:Number,N}
	p,q = ranks(U)
	try checkranks(p,q) catch e
		isa(e, DimensionMismatch) && throw(DimensionMismatch("the factors have inconsistent ranks"))
	end
	return [p..., q[end]]
end

"""
    deepcopy(U::TT{T,N})

Return a deep copy of the tensor train object `U`.

# Arguments
- `U::TT{T,N}`: Tensor train object.

# Returns
- `TT{T,N}`: A deep copy of `U`.
"""
deepcopy(U::TT{T,N}) where {T<:Number,N} = TT(deepcopy(U.factors))

"""
    reverse!(W::TT{T,N})

Reverse the order of factors in the tensor train `W` in place and transpose their ranks.

# Arguments
- `W::TT{T,N}`: Tensor train of type `TT`.

# Returns
- `W`: The reversed tensor train of type `TT` with transposed ranks.
"""
function reverse!(W::TT{T,N}) where {T<:Number,N}
	L = length(W)
	reverse!(W.factors)
	for ℓ ∈ 1:L
		W.factors[ℓ] = factorranktranspose(W.factors[ℓ])
	end
	return W
end

"""
    permutedims!(U::TT{T,N}, τ::Permutation)

Permute the mode dimensions of each factor in the tensor train `U` in place.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `τ::Permutation`: Permutation representing the new ordering of mode dimensions.

# Returns
- `TT{T,N}`: The tensor train `U` with permuted dimensions.

# Throws
- `ArgumentError`: If the decomposition exhibits zero mode dimensions.
- `ArgumentError`: If τ is not a valid permutation of the mode dimensions of U.
"""
function permutedims!(U::TT{T,N}, τ::Permutation) where {T<:Number,N}
	d = N-2
	if d == 0
		throw(ArgumentError("the decomposition should have at least one mode dimension"))
	end
	if length(τ) ≠ d || !isperm(τ)
		throw(ArgumentError("τ is not a valid permutation of the mode dimensions of U"))
	end
	isa(τ, Vector{Int}) || (τ = collect(τ))
	L = length(U)
	for ℓ ∈ 1:L
		U.factors[ℓ] = factormodetranspose(U.fators[ℓ], τ)
	end
	return U
end

"""
    fill!(U::TT{T,N}, v::T)

Fill each factor contained in the tensor train `U` with the value `v` in place.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `v::T`: Value to fill each factor with.

# Returns
- `U`: Tensor train of type `TT` with each factor filled with value `v`.
"""
function fill!(U::TT{T,N}, v::T) where {T<:Number,N}
	L = length(U)
	for ℓ ∈ 1:L
		fill!(U.factors[ℓ], v)
	end
	return U
end

"""
    rand!(rng::AbstractRNG, U::TT{T,N})

Fill each factor in the tensor train `U` in place with random values using the provided random number generator `rng`.

# Arguments
- `rng::AbstractRNG`: Random number generator to utilize for generating random values.
- `U::TT{T,N}`: Tensor train of type `TT`, whose factors are to be filled with random numbers.

# Returns
- `U`: Tensor train of type `TT` with each factor filled with random values.
"""
function rand!(rng::AbstractRNG, U::TT{T,N}) where {T<:Number,N}
	L = length(U)
	for ℓ ∈ 1:L
		rand!(rng, U.factors[ℓ])
	end
	return U
end

"""
    rand!(U::TT{T,N}) where {T<:Number,N}

Fill each factor in the tensor train `U` with random values in place, using the default global random number generator (`Random.GLOBAL_RNG`).

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`, whose factors are to be filled with random values.

# Returns
- `U`: Tensor train of type `TT` with random values filled in its factors.
"""
rand!(U::TT{T,N}) where {T<:Number,N} = rand!(Random.GLOBAL_RNG, U)

"""
    zeros(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a tensor train `TT` with all factors filled with zeros of the specified type.

# Arguments
- `::Type{T}`: Data type of the zeros.
- `n::Union{DecSize,FactorSize}`: Specifies the size or factor size of the tensor train.
- `r::Union{Int,DecRank}`: Specifies the rank or rank vector for the tensor train.
- `first::Int=0`: Optional. Specifies the rank of the first mode dimension.
- `last::Int=0`: Optional. Specifies the rank of the last mode dimension.
- `len::Int=0`: Optional. Specifies the number of factors in the tensor train.

# Returns
- `U`: Tensor train `TT` with all factors filled with zeros of specified type.
"""
zeros(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = fill!(TT(T, n, r; first=first, last=last, len=len), convert(T, 0))

"""
    zeros(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a tensor train with all factors filled with zeros of type `Float64`.

# Arguments
- `n::Union{DecSize,FactorSize}`: Specifies the size or factor size of the tensor train.
- `r::Union{Int,DecRank}`: Specifies the rank or rank vector for the tensor train.
- `first::Int=0`: Optional. Specifies the rank of the first mode dimension.
- `last::Int=0`: Optional. Specifies the rank of the last mode dimension.
- `len::Int=0`: Optional. Specifies the number of factors in the tensor train.

# Returns
- `U`: Tensor train with all factors filled with zeros of type `Float64`.
"""
zeros(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = ttzeros(Float64, n, r; first=first, last=last, len=len)

"""
    ones(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a tensor train `TT` with all factors filled with ones of the specified type.

# Arguments
- `::Type{T}`: Data type of the ones.
- `n::Union{DecSize,FactorSize}`: Specifies the size or factor size of the tensor train.
- `r::Union{Int,DecRank}`: Specifies the rank or rank vector for the tensor train.
- `first::Int=0`: Optional. Specifies the rank of the first mode dimension.
- `last::Int=0`: Optional. Specifies the rank of the last mode dimension.
- `len::Int=0`: Optional. Specifies the number of factors in the tensor train.

# Returns
- `U`: Tensor train `TT` with all factors filled with ones of the specified type.
"""
ones(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = fill!(TT(T, n, r; first=first, last=last, len=len), convert(T, 1))

"""
    ones(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a tensor train with all factors filled with ones of type `Float64`.

# Arguments
- `n::Union{DecSize,FactorSize}`: Specifies the size or factor size of the tensor train.
- `r::Union{Int,DecRank}`: Specifies the rank or rank vector for the tensor train.
- `first::Int=0`: Optional. Specifies the rank of the first mode dimension.
- `last::Int=0`: Optional. Specifies the rank of the last mode dimension.
- `len::Int=0`: Optional. Specifies the number of factors in the tensor train.

# Returns
- `U`: Tensor train with all factors filled with ones of type `Float64`.
"""
ones(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = ttzeros(Float64, n, r; first=first, last=last, len=len)

"""
    rand(rng::AbstractRNG, ::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a tensor train of type `TT` filled with random values, using the provided random number generator `rng` and specifications regarding size, ranks, and length.

# Arguments
- `rng::AbstractRNG`: Random number generator used to generate random values.
- `::Type{T}`: Element type of the tensor train.
- `n::Union{DecSize,FactorSize}`: Mode sizes for the tensor train.
- `r::Union{Int,DecRank}`: Ranks for the tensor train.
- `first::Int=0`: Optional first rank (default is 0).
- `last::Int=0`: Optional last rank (default is 0).
- `len::Int=0`: Optional number of factors (default is 0).

# Returns
- A tensor train of type `TT` with random values generated for each factor.
"""
rand(rng::AbstractRNG, ::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = rand!(rng, TT(T, n, r; first=first, last=last, len=len))

"""
    rand(rng::AbstractRNG, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a tensor train of type `TT` filled with random values of type `Float64`, using the provided random number generator `rng`, with mode sizes `n` and ranks `r`.

# Arguments
- `rng::AbstractRNG`: Random number generator used to generate random values.
- `n::Union{DecSize,FactorSize}`: Mode sizes for the tensor train.
- `r::Union{Int,DecRank}`: Ranks for the tensor train.
- `first::Int=0`: Optional first rank (default is 0).
- `last::Int=0`: Optional last rank (default is 0).
- `len::Int=0`: Optional number of factors (default is 0).

# Returns
- A tensor train of type `TT` with random values of type `Float64` generated for each factor.
"""
rand(rng::AbstractRNG, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = ttrand(rng, Float64, n, r; first=first, last=last, len=len)

"""
    rand(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a tensor train of type `TT` filled with random values of given element type, using the random number generator `Random.GLOBAL_RNG` and considering mode sizes `n`, ranks `r`, and other optional parameters.

# Arguments
- `::Type{T}`: Element type of the entries of the tensor train.
- `n::Union{DecSize,FactorSize}`: Mode sizes for the tensor train.
- `r::Union{Int,DecRank}`: Ranks for the tensor train.
- `first::Int=0`: Optional first rank (default is 0).
- `last::Int=0`: Optional last rank (default is 0).
- `len::Int=0`: Optional number of factors (default is 0).

# Returns
- A tensor train of type `TT` with random values generated by `Random.GLOBAL_RNG` of specified type for each factor.
"""
rand(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = ttrand(Random.GLOBAL_RNG, T, n, r; first=first, last=last, len=len)

"""
    rand(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a tensor train of type `TT` filled with random values of type `Float64`, using the random number generator `Random.GLOBAL_RNG` and specified mode sizes `n`, ranks `r`, and other optional parameters.

# Arguments
- `n::Union{DecSize,FactorSize}`: Mode sizes for the tensor train.
- `r::Union{Int,DecRank}`: Ranks for the tensor train.
- `first::Int=0`: Optional first rank (default is 0).
- `last::Int=0`: Optional last rank (default is 0).
- `len::Int=0`: Optional number of factors (default is 0).

# Returns
- A tensor train of type `TT` with random values of type `Float64` generated by `Random.GLOBAL_RNG` for each factor.
"""
rand(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = ttrand(Random.GLOBAL_RNG, Float64, n, r; first=first, last=last, len=len)

"""
    getfirstfactor(U::TT{T,N}) where {T<:Number,N}

Retrieve the first factor of the tensor train `U`.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`, where `T` is the element type and `N` is the number of dimensions.

# Returns
- The first factor in the tensor train `U`.
"""
getfirstfactor(U::TT{T,N}) where {T<:Number,N} = U.factors[1]

"""
    getlastfactor(U::TT{T,N}) where {T<:Number,N}

Retrieve the last factor of the tensor train `U`.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`, where `T` is the element type and `N` is the number of dimensions.

# Returns
- The last factor in the tensor train `U`.
"""
getlastfactor(U::TT{T,N}) where {T<:Number,N} = U.factors[end]

"""
    getfactor(U::TT{T,N}, ℓ::Int)

Return the factor at position `ℓ` in the tensor train `U`.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `ℓ::Int`: Index of the factor to retrieve.

# Returns
- `Factor{T,N}`: The factor at the specified index `ℓ` of the tensor train.

# Throws
- `ArgumentError`: If `ℓ` is out of valid range.
"""
function getfactor(U::TT{T,N}, ℓ::Int) where {T<:Number,N}
	L = length(U)
	if ℓ ⊈ 1:L
		throw(ArgumentError("ℓ is out of range"))
	end
	return U.factors[ℓ]
end

"""
    setfirstfactor!(U::TT{T,N}, F::Array{T,N})

Set the first factor of the tensor train `U` to `F` in place.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `F::Array{T,N}`: New factor to replace the current first factor with.

# Returns
- `TT{T,N}`: Modified tensor train `U` with the first factor set to `F`.
"""
function setfirstfactor!(U::TT{T,N}, F::Array{T,N}) where {T<:Number,N}
	U.factors[1] = F
	return U
end

"""
    setlastfactor!(U::TT{T,N}, F::Array{T,N})

Set the last factor of the tensor train `U` to `F` in place.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `F::Array{T,N}`: New factor to replace the current last factor with.

# Returns
- `TT{T,N}`: Modified tensor train `U` with the last factor set to `F`.
"""
function setlastfactor!(U::TT{T,N}, F::Array{T,N}) where {T<:Number,N}
	U.factors[end] = F
	return U
end

"""
    setfactor!(U::TT{T,N}, F::Array{T,N}, ℓ::Int)

Set the factor at position `ℓ` in the tensor train `U` to `F` in place.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `F::Array{T,N}`: New factor to replace the current factor at index `ℓ` with.
- `ℓ::Int`: The index of the factor to set.

# Returns
- `TT{T,N}`: Modified tensor train `U` with the factor at index `ℓ` set to `F`.

# Throws
- `ArgumentError`: If `ℓ` is out of valid range.
"""
function setfactor!(U::TT{T,N}, F::Array{T,N}, ℓ::Int) where {T<:Number,N}
	L = length(U)
	if ℓ ⊈ 1:L
		throw(ArgumentError("ℓ is out of range"))
	end
	U.factors[ℓ] = F
	return U
end

"""
    rankselect!(U::TT{T,N}, α::Indices, β::Indices)

Select the first rank dimensions of the first factor and the last rank dimensions of the last factor in the tensor train `U` based on the provided indices `α` and `β`, modifying `U` in place.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `α::Indices`: Reference numbers for selecting the first rank dimensions of the first factor of `U`.
- `β::Indices`: Reference numbers for selecting the last rank dimensions of the last factor of `U`.

# Returns
- `U`: Tensor train type `TT`, with the selected ranks. 

# Throws
Summarized Error list:
- `ArgumentError`: If the tensor train is empty, or if `α` or `β` contain invalid or empty ranges.

Extended Error list:
- `ArgumentError`: If the range for the first rank is empty.
- `ArgumentError`: If the range for the first rank is incorrect.
- `ArgumentError`: If the range for the second rank is empty.
- `ArgumentError`: If the range for the second rank is incorrect.
"""
function rankselect!(U::TT{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	# if isa(α, Int) || isa(β, Int)
	# 	throw(ArgumentError("for consistency with Base.selectdim, scalar α and β are not accepted; use α:α or β:β instead of α or β to select a subtensro of the factor whose first or second rank is one"))
	# end
	L = length(U)
	r = rank(U); p,q = r[1],r[L+1]
	α = indvec(α; min=1, max=p)
	β = indvec(β; min=1, max=q)
	if length(α) == 0
		throw(ArgumentError("the range for the first rank is empty"))
	end
	if α ⊈ 1:p
		throw(ArgumentError("the range for the first rank is incorrect"))
	end
	if length(β) == 0
		throw(ArgumentError("the range for the second rank is empty"))
	end
	if β ⊈ 1:q
		throw(ArgumentError("the range for the second rank is incorrect"))
	end
	setlastfactor!(U, factorrankselect(getlastfactor(U), :, β))
	setfirstfactor!(U, factorrankselect(getfirstfactor(U), α, :))
	return U
end

"""
    rankselect(U::TT{T,N}, α::Indices, β::Indices)

Create a new tensor train object by selecting the first rank dimensions of the first factor and the last rank dimensions of the last factor in the tensor train `U` based on the provided indices `α` and `β`.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `α::Indices`: Reference numbers for selecting the first rank dimensions of the first factor of `U`.
- `β::Indices`: Reference numbers for selecting the last rank dimensions of the last factor of `U`.

# Returns
- `V`: New tensor train object, with the selected ranks.

# Throws
Summarized Error list:
- `ArgumentError`: If the decomposition is empty, or if `α` or `β` contain invalid or empty ranges.

Extended Error list:
- `ArgumentError`: If the range for the first rank is empty.
- `ArgumentError`: If the range for the first rank is incorrect.
- `ArgumentError`: If the range for the second rank is empty.
- `ArgumentError`: If the range for the second rank is incorrect.
"""
function rankselect(U::TT{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	V = deepcopy(U)
	rankselect!(V, α, β)
	return V
end

"""
    getindex(U::TT{T,N}, α::Indices, β::Indices) where {T<:Number,N}

Create a new tensor train object by selecting the first rank dimensions of the first factor and the last rank dimensions of the last factor in the tensor train `U` based on the provided indices `α` and `β`.
Function acts as an alias for [`rankselect`](@ref).

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `α::Indices`: Reference numbers for selecting the first rank dimensions of the first factor of `U`.
- `β::Indices`: Reference numbers for selecting the last rank dimensions of the last factor of `U`.

# Returns
- `V`: New tensor train object, with the selected ranks.

# Throws
Summarized Error list:
- `ArgumentError`: If the decomposition is empty, or if `α` or `β` contain invalid or empty ranges.

Extended Error list:
- `ArgumentError`: If the range for the first rank is empty.
- `ArgumentError`: If the range for the first rank is incorrect.
- `ArgumentError`: If the range for the second rank is empty.
- `ArgumentError`: If the range for the second rank is incorrect.
"""
getindex(U::TT{T,N}, α::Indices, β::Indices) where {T<:Number,N} = rankselect(U, α, β)

"""
    append!(U::TT{T,N}, V::TT{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}

Append the tensor train type `V` to the end of the tensor train type `U` in place. Optionally, the ranks are checked before and after the operation.

# Arguments
- `U::TT{T,N}`: Target tensor train of type `TT` to which `V` will be appended.
- `V::TT{T,N}`: Tensor Train of type `TT`, which to append to `U`.
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` and `V` before appending.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the combined result after appending.

# Returns
- `U`: Modified tensor train type `U` after appending `V`.

# Throws
Summarized Error list:
- `DimensionMismatch`: If `U` and `V` have different numbers of dimensions, or if the ranks of their factors are incorrect or inconsistent with the operation.

Extended Error list:
- `DimensionMismatch`: If U and V are inconsistent in the number of dimensions.
- `ArgumentError`:If the factors of U have incorrect or inconsistent ranks.
- `ArgumentError`: If the factors of V have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function append!(U::TT{T,N}, V::TT{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if ndims(U) ≠ ndims(V)
		throw(DimensionMismatch("U and V are inconsistent in the number of dimensions"))
	end
	p,q = ranks(U)
	r,s = ranks(V)
	if rankprecheck
		try checkranks(p,q) catch
			throw(ArgumentError("the factors of U have incorrect or inconsistent ranks"))
		end
		try checkranks(r,s) catch
			throw(ArgumentError("the factors of V have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		append!(p, r); append!(q, s);
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return append!(U.factors, V.factors)
end

"""
    prepend!(U::TT{T,N}, V::TT{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}

Prepend the tensor train type `V` to the beginnning of the tensor train type `U` in place. Optionally, the ranks are checked before and after the operation.

# Arguments
- `U::TT{T,N}`: Target tensor train of type `TT`, to which `V` will be prepended.
- `V::TT{T,N}`: Tensor train of type `TT`, which to append to `U`.
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` and `V` before prepending.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the combined result after prepending.

# Returns
- `U`: Modified tensor train type `U` after prepending `V`.

# Throws
Summarized Error list:
- `DimensionMismatch`: If `U` and `V` have different numbers of dimensions, or if the ranks of their factors are incorrect or inconsistent with the operation.

Extended Error list:
- `DimensionMismatch`: If U and V have different numbers of dimensions.
- `ArgumentError`: If the factors of U have incorrect or inconsistent ranks.
- `ArgumentError`: If the factors of V have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function prepend!(U::TT{T,N}, V::TT{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if ndims(U) ≠ ndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = ranks(U); r,s = ranks(V)
	if rankprecheck
		try checkranks(p,q) catch
			throw(ArgumentError("the factors of U have incorrect or inconsistent ranks"))
		end
		try checkranks(r,s) catch
			throw(ArgumentError("the factors of V have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		prepend!(p, r); prepend!(q, s);
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return prepend!(U.factors, V.factors)
end

"""
    push!(U::TT{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}

Push a factor `V` to the end of the tensor train type `U` in place. Optionally, the ranks before and after the operation are checked.

# Arguments
- `U::TT{T,N}`: Target tensor train of type `TT`, to which the factor `V` will be pushed.
- `V::Factor{T,N}`: The factor to push to the tensor train type `U`.
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` before pushing.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the result after pushing.

# Returns
- `U`: Modified tensor train type `U` after pushing `V`.

# Throws
Summarized Error list:
- `DimensionMismatch`: If `U` and `V` have different numbers of dimensions, or if their ranks (in case of `U`, the ranks of its factors) are inconsistent with the operation.

Extended Error list:
- `DimensionMismatch`: If U and V have different numbers of dimensions.
- `ArgumentError`: If the factors of U have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function push!(U::TT{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if ndims(U) ≠ factorndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = ranks(U)
	r,s = factorranks(V)
	if rankprecheck
		try checkranks(p,q) catch
			throw(ArgumentError("the factors of U have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		r,s = factorranks(V); push!(p, r); push!(q, s)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return push!(U.factors, V)
end

"""
    pushfirst!(U::TT{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}

Push a factor `V` to the beginning of the tensor train type `U` in place. Optionally, the ranks before and after the operation are checked.

# Arguments
- `U::TT{T,N}`: Target tensor train of type `TT`, to which the factor `V` will be pushed at the beginning.
- `V::Factor{T,N}`: The factor to push to the beginning of `U`.
- `rankprecheck::Bool=true`: If `true` (by default), the ranks of `U` are checked before pushing.
- `rankpostcheck::Bool=true`: If `true` (by default), the ranks of the result are checked after pushing.

# Returns
- `U`: Modified tensor train type `U` after pushing `V` at the beginning.

# Throws
Summarized Error list:
- `DimensionMismatch`: If `U` and `V` have different numbers of dimensions, or if their ranks (in case of `U`, the ranks of its factors) are inconsistent with the operation.

Extended Error list:
- `DimensionMismatch`: If U and V have different numbers of dimensions.
- `DimensionMismatch`: If the factors of U have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function pushfirst!(U::TT{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if ndims(U) ≠ factorndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = ranks(U)
	r,s = factorranks(V)
	if rankprecheck
		try
			checkranks(p,q)
		catch e
			throw(ArgumentError("the factors of U have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		r,s = factorranks(V); pushfirst!(p, r); pushfirst!(q, s)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return pushfirst!(U.factors, V)
end

"""
    pop!(U::TT{T,N}) where {T<:Number,N}

Pop the last factor from the tensor train type `U` and return it.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`, from which to pop the last factor.

# Returns
- `V`: The last factor that was removed from `U`.
"""
function pop!(U::TT{T,N}) where {T<:Number,N}
	return pop!(U.factors)
end

"""
    popfirst!(U::TT{T,N}) where {T<:Number,N}

Pop the first factor from the tensor train type `U` and return it.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`, from which to pop the first factor.

# Returns
- `V`: The first factor that was removed from `U`.
"""
function popfirst!(U::TT{T,N}) where {T<:Number,N}
	return popfirst!(U.factors)
end

"""
    insert!(U::TT{T,N}, ℓ::Int, V::Factor{T,N}; path::String="", rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}

Insert a factor `V` into the tensor train type `U` at the specified index `ℓ`. Optionally, the ranks are checked before and after the operation.

# Arguments
- `(U::TT{T,N}`: Tensor train of type `TT`, in which to insert the factor `V`.
- `ℓ::Int`: The index at which to insert the factor `V`.
- `V::Factor{T,N}`: The factor to insert into the decomposition `U`.
- `path::String=""`: The path direction for insertion; should be "forward" or "backward".
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` before insertion.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the result after insertion.

# Returns
- `U`: Modified tensor train type `U` after inserting `V`.

# Throws
Summarized Error list:
- `ArgumentError`: If `path` is not "forward" or "backward".
- `DimensionMismatch`: If `ℓ` is out of range or if the ranks (of `U`, `V` or the factors of `U`) are inconsistent for the operation.

Extended Error list:
- `ArgumentError`:  If path is neither \"forward\" nor \"backward\"
- `ArgumentError`: If ℓ is not from 1:L, where L is the number of factors in `U`.
- `DimensionMismatch`: If the factors of U have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function insert!(U::TT{T,N}, ℓ::Int, V::Factor{T,N}; path::String="", rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	L = length(U)
	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end
	if ℓ ∉ 1:L
		throw(ArgumentError("ℓ is required to be from 1:L, where L is the number of factors in U"))
	end
	(path == "forward") && (ℓ = ℓ+1)
	(ℓ == 1) && return pushfirst!(U, V; rankprecheck=rankprecheck, rankpostcheck=rankpostcheck)
	(ℓ == L+1) && return push!(U, V; rankprecheck=rankprecheck, rankpostcheck=rankpostcheck)
	p,q = ranks(U)
	if rankprecheck
		try checkranks(p,q) catch
			throw(DimensionMismatch("the factors of U have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		r,s = factorranks(V); insert!(p, ℓ, r); insert!(q, ℓ, s)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return insert!(U.factors, ℓ, V)
end

"""
    deleteat!(U::TT{T,N}, Λ::Path; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}

Delete factors from the tensor train type `U` at the specified indices `Λ` in place. Optionally, the ranks are checked before and after the operation.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`, from which factors will be deleted.
- `Λ::Path`: Indices of factors to delete. Can be a single integer, a vector of integers, or a tuple of integers.
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` before deletion.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the result after deletion.

# Returns
- `U`: Modified tensor train type `U` after deleting the specified factors.

# Throws
Summarized Error list:
- `ArgumentError`: If the entries of `Λ` are not unique or are not within the valid range.
- `DimensionMismatch`: If the ranks of `U` are inconsistent for this operation.

Extended Error list:
- `ArgumentError`:  If the entries of Λ are not unique.
- `ArgumentError`: If Λ is not an element or a subset of 1:L with unique entries, where L is the number of factors in U.
- `DimensionMismatch`: If the factors of U have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U are inconsistent for this operation.
"""
function deleteat!(U::TT{T,N}, Λ::Path; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	L = length(U)
	isa(Λ, Colon) && (Λ = collect(2:L); path = "backward")
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	(length(Λ) == 0) && return W
	if unique(Λ) ≠ Λ
		throw(ArgumentError("the entries of Λ should be unique"))
	end
	if Λ ⊈ 1:L
		throw(ArgumentError("Λ should be an element or a subset of 1:L with unique entries, where L is the number of factors in U"))
	end
	p,q = ranks(U)
	if rankprecheck
		try checkranks(p,q) catch
			throw(DimensionMismatch("the factors of U have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		deleteat!(p, Λ); deleteat!(q, Λ)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U are inconsistent for this operation"))
		end
	end
	return deleteat!(U.factors, Λ)
end

"""
    compose!(W::TT{T,N}, Λ::Path; path::String="", major::String="last")

Contract the factors of the tensor train `W` along the path specified by `Λ`, modifying `W` in place.

# Arguments
- `W::TT{T,N}`: Tensor train of type `TT`.
- `Λ::Path`: Indices along which to perform contractions. Can be a `Colon` to select all factors or a vector of integers.
- `path::String=""`: Direction of contraction, either `"forward"` or `"backward"`. Defaults to `""`, which deduces the path from `Λ`.
- `major::String="last"`: The major ordering of dimensions for the contraction. Either `"first"` or `"last"`.

# Returns
- `W`: Modified tensor train `W` with factors contracted along the specified path.

# Throws
Summarized Error list:
- `ArgumentError`: If `Λ` has duplicate entries, is out of the valid range, or if `path` or `major` are set to invalid values.

Extended Error list:
- `ArgumentError`: If path is none of the following: \"\",  \"forward\" or \"backward\".
- `ArgumentError`: If major is none of the following: \"last\" (default) or \"first\".
- `ArgumentError`: If path is neither \"forward\" nor \"backward\", when Λ is neither empty nor a colon.
- `ArgumentError`: If Λ has duplicate entries.
- `ArgumentError`: If non-empty Λ is neither a colon nor a Vector/NTuple/UnitRange of Int (from 1:L-1 for path=\"forward\" and from 2:L for path=\"backward\", where L is the number of factors in W)
"""
function compose!(W::TT{T,N}, Λ::Path; path::String="", major::String="last") where {T<:Number,N}
	if path ∉ ("","forward","backward")
		throw(ArgumentError("the value of the keyword argument path should be \"\" (default, accepted only for empty Λ and for Λ=:), \"forward\" or \"backward\""))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	L = length(W)
	rank(W)
	isa(Λ, Colon) && (Λ = collect(2:L); path = "backward")
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	(length(Λ) == 0) && return W
	if path ∉ ("forward","backward")
		throw(ArgumentError("when Λ is neither empty nor a colon, path should be either \"forward\" or \"backward\""))
	end
	(path == "backward") || (Λ .+= 1; path = "backward")
	if unique(Λ) ≠ Λ
		throw(ArgumentError("Λ has duplicate entries"))
	end
	if Λ ⊈ 2:L
		throw(ArgumentError("Λ, when nonempty, should be a colon or a Vector/NTuple/UnitRange of Int, from 1:L-1 for path=\"forward\" and from 2:L for path=\"backward\", where L is the number of factors in W"))
	end
	sort!(Λ; rev=true)
	for ℓ ∈ Λ
		F = getfactor(W, ℓ-1)
		F = factorcontract(F, getfactor(W, ℓ); major=major)
		setfactor!(W, F, ℓ-1)
		deleteat!(W, ℓ; rankprecheck=false, rankpostcheck=false)
	end
	return W
end

"""
    compose!(W::TT{T,N}; path::String="", major::String="last") where {T<:Number,N}

Contract all factors of the tensor train `W` in place along the specified path.

# Arguments
- `W::TT{T,N}`: Tensor train of type `TT`.
- `path::String=""`: Direction of contraction.
- `major::String="last"`: The major ordering of dimensions for the contraction. Either `"first"` or `"last"`.

# Returns
- `W`: The modified tensor train after performing the contraction of all factors along the specified path.

# Throws
- `ArgumentError`: If path is none of the following: \"\",  \"forward\" or \"backward\".
- `ArgumentError`: If major is none of the following: \"last\" (default) or \"first\".
"""
compose!(W::TT{T,N}; path::String="", major::String="last") where {T<:Number,N} = compose!(W, :; path=path, major=major)

"""
    compose(W::TT{T,N}, Λ::Path; path::String="", major::String="last")

Create a new tensor train that is a deepcopy of `W`, with the factors along the path specified by `Λ` contracted.

# Arguments
- `W::TT{T,N}`: Tensor train of type `TT`.
- `Λ::Path`: Indices along which to perform contractions. Can be a `Colon` to select all factors or a vector of integers.
- `path::String=""`: Direction of contraction, either `"forward"` or `"backward"`. Defaults to `""`, which deduces the path from `Λ`.
- `major::String="last"`: The major ordering of dimensions for the contraction. Either `"first"` or `"last"`.

# Returns
- `U`: New tensor train object of type `TT`, obtained by contracting a deepcopy of `W` along the specified path.

# Throws
Summarized Error list:
- `ArgumentError`: If `Λ` has duplicate entries, is out of the valid range, or if `path` or `major` are set to invalid values.

Extended Error list:
- `ArgumentError`: If path is none of the following: \"\",  \"forward\" or \"backward\".
- `ArgumentError`: If major is none of the following: \"last\" (default) or \"first\".
- `ArgumentError`: If path is neither \"forward\" nor \"backward\", when Λ is neither empty nor a colon.
- `ArgumentError`: If Λ has duplicate entries.
- `ArgumentError`: If non-empty Λ is neither a colon nor a Vector/NTuple/UnitRange of Int (from 1:L-1 for path=\"forward\" and from 2:L for path=\"backward\", where L is the number of factors in W)
"""
function compose(W::TT{T,N}, Λ::Path; path::String="", major::String="last") where {T<:Number,N}
	U = deepcopy(W)
	return compose!(U, Λ; path=path, major=major)
end

"""
    compose(W::TT{T,N}; path::String="", major::String="last") where {T<:Number,N}

Contract all the factors of the tensor train `W` along the specified path and return a new tensor train.

# Arguments
- `W::TT{T,N}`: Tensor train of type `TT`.
- `path::String=""`: Direction of contraction.
- `major::String="last"`: The major ordering of dimensions for the contraction. Either `"first"` or `"last"`.

# Returns
- A new tensor train with all the factors contracted along the specified path.

# Throws
- `ArgumentError`: If path is none of the following: \"\",  \"forward\" or \"backward\".
- `ArgumentError`: If major is none of the following: \"last\" (default) or \"first\".
"""
compose(W::TT{T,N}; path::String="", major::String="last") where {T<:Number,N} = compose(W, :; path=path, major=major)

"""
    composecore(W::TT{T,N}; major::String="last")

Compute the tensor of a tensor train `W`, which results from contracting all its factors.

# Arguments
- `W::TT{T,N}`: Tensor train of type `TT`.
- `major::String="last"`: The major ordering of dimensions for the contraction. Either `"first"` or `"last"`. Defaults to `"last"`.

# Returns
- `Factor{T,N}`: Final tensor obtained by contracting all the factors of `W`.

# Throws
- `ArgumentError`: If `major` is set to a value different from \"last\" (default) or \"first\"".
- `ArgumentError`: If the decomposition is empty.
"""
function composecore(W::TT{T,N}; major::String="last") where {T<:Number,N}
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	L = length(W); rank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	F = copy(getfactor(W, 1))
	for ℓ ∈ 2:L
		F = factorcontract(F, getfactor(W, ℓ); major=major)
	end
	return F
end

"""
    composecore!(U::TT{T,N}; major::String="last")

Contract all factors of the tensor train `U` in place and return the resulting tensor.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `major::String="last"`: The major ordering of dimensions for the contraction. Either `"first"` or `"last"`. Defaults to `"last"`.

# Returns
- `Factor{T,N}`: Final tensor obtained by contracting all the factors of `U`.

# Throws
- `ArgumentError`: If `major` is set to a value different from \"last\" (default) or \"first\"".
- `ArgumentError`: If the decomposition is empty.
"""
function composecore!(U::TT{T,N}; major::String="last") where {T<:Number,N}
	compose!(U; major=major)
	return getfirstfactor(U)
end

"""
    composeblock!(U::TT{T,N}, α::Int, β::Int; major::String="last")

Contract the factors of the tensor train `U` after having selected the first and last rank of the tensor train via indices `α` and `β`, modifying `U` in place.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `α::Int`: Index for the first rank of the block.
- `β::Int`: Index for the second rank of the block.
- `major::String="last"`: The major ordering of dimensions for the contraction. Either `"first"` or `"last"`. Defaults to `"last"`.

# Returns
- `Factor{T,N}`: Final tensor after contracting the factors of `U` exhibiting the selected first and last rank.

# Throws
- `ArgumentError`: If `α` or `β` are out of range.
"""
function composeblock!(U::TT{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}
	r = rank(U); p,q = r[1],r[end]
	if α ∉ 1:p
		throw(ArgumentError("the first rank index is out of range"))
	end
	if β ∉ 1:q
		throw(ArgumentError("the second rank index is out of range"))
	end
	rankselect!(U, α:α, β:β)
	return compose(U; major=major)
end

"""
    composeblock(U::TT{T,N}, α::Int, β::Int; major::String="last")

Create a new tensor obtained by contracting the factors of a deep copy of the tensor train `U` after having selected the first and last rank of the tensor train via indices `α` and `β`.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `α::Int`: Index for the first rank of the block.
- `β::Int`: Index for the second rank of the block.
- `major::String="last"`: The major ordering of dimensions for the contraction. Either `"first"` or `"last"`. Defaults to `"last"`.

# Returns
- `Factor{T,N}`: Final tensor after contracting the factors of the deepcopy of `U` exhibiting the selected first and last rank.

# Throws
- `ArgumentError`: If `α` or `β` are out of range.
"""
function composeblock(U::Dec{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}
	V = deepcopy(U)
	V = composeblock!(V, α, β, major=major)
	return V
end

"""
    vcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M}) -> TT{T,N}

Vertically concatenate the given tensor trains `U`, `V` and any additional tensor trains `W...`, to form a new tensor train `Z`.

# Arguments
- `U::TT{T,N}`: First tensor train of type `TT`.
- `V::TT{T,N}`: Second tensor train of type `TT`.
- `W::Vararg{TT{T,N},M}`: Additional tensor trains to include in the concatenation.

# Returns
- `TT{T,N}`: New tensor train `Z` that is the vertical concatenation of `U`, `V`, and any additional tensor trains.

# Throws
- `ArgumentError`: If the number of factors, mode size, or ranks of the tensor trains are incompatible.

- `ArgumentError`: If the decompositions are incompatible in the number of factors.
- `ArgumentError`: If the decompositions are incompatible in mode size.
- `ArgumentError`: If the decompositions are incompatible in the last rank.
"""
function vcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M}) where {T<:Number,N,M}
	L = length(U); d = ndims(U); m = size(U); p = rank(U)
	W = (V,W...)
	for V ∈ W
		if length(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if size(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = rank(V)
		if q[L+1] ≠ p[L+1]
			throw(ArgumentError("the decompositions are incompatible in the last rank"))
		end
	end
	Z = TT(T, d, 0)
	for ℓ ∈ 1:L-1
		push!(Z, factordcat(getfactor(U, ℓ), [ getfactor(V, ℓ) for V ∈ W ]...))
	end
	push!(Z, factorvcat(getlastfactor(U), [ getlastfactor(V) for V ∈ W ]...))
	return Z
end

"""
    hcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M})

Horizontally concatenate the given tensor trains `U`, `V` and any additional tensor trains `W...`, to form a new tensor train `Z`.

# Arguments
- `U::TT{T,N}`: First tensor train of type `TT`.
- `V::TT{T,N}`: Second tensor train of type `TT`.
- `W::Vararg{TT{T,N},M}`: Additional tensor trains to concatenate.

# Returns
- `TT{T,N}`: New tensor train `Z` that is the horizontal concatenation of `U`, `V`, and any additional tensor trains.

# Throws
- `ArgumentError`: If the number of factors, mode size, or ranks of the tensor trains are incompatible.

- `ArgumentError`: If the decompositions are incompatible in the number of factors.
- `ArgumentError`: If the decompositions are incompatible in mode size.
- `ArgumentError`: If the decompositions are incompatible in the first rank.
"""
function hcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M}) where {T<:Number,N,M}
	L = length(U); m = size(U); p = rank(U)
	W = (V,W...)
	for V ∈ W
		if length(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if size(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = rank(V)
		if q[1] ≠ p[1]
			throw(ArgumentError("the decompositions are incompatible in the first rank"))
		end
	end
	Z = TT(T, d, 0)
	push!(Z, factorhcat(getfirstfactor(U), [ getfirstfactor(V) for V ∈ W ]...))
	for ℓ ∈ 2:L
		push!(Z, factordcat(getfactor(U, ℓ), [ getfactor(V, ℓ) for V ∈ W ]...))
	end
	return Z
end

"""
    dcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M})

Diagonally concatenate the given tensor trains `U`, `V` and any additional tensor trains `W...`, to form a new tensor train `Z`.

# Arguments
- `U::TT{T,N}`: First tensor train of type `TT`.
- `V::TT{T,N}`: Second tensor train of type `TT`.
- `W::Vararg{TT{T,N},M}`: Additional tensor trains to concatenate.

# Returns
- `TT{T,N}`: New tensor train `Z` that is the diagonal concatenation of `U`, `V` and any additional tensor trains.

# Throws
- `ArgumentError`: If the decompositions are incompatible in the number of factors.
- `ArgumentError`: If the decompositions are incompatible in mode size.
"""
function dcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M}) where {T<:Number,N,M}
	L = length(U); m = decsize(U)
	W = (V,W...)
	for V ∈ W
		if length(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if size(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
	end
	Z = TT(T, d, 0)
	for ℓ ∈ 1:L
		push!(Z, factordcat(getfactor(U, ℓ), [ getfactor(V, ℓ) for V ∈ W ]...))
	end
	return Z
end

"""
    lmul!(α::T, U::TT{T,N})

Scale the tensor train `U` in place by multiplying its last factor by a value `α`.

# Arguments
- `α::T`: Value of type `T` to multiply with.
- `U::TT{T,N}`: Tensor train of type `TT`.

# Returns
- `TT{T,N}`: Modified tensor train `U` after scaling its last factor by `α`.
"""
function lmul!(α::T, U::TT{T,N}) where {T<:Number,N}
	F = getlastfactor(U)
	F .= α * F
	return U
end

"""
    rmul!(U::TT{T,N}, α::T) where {T<:Number,N}

Scale the tensor train `U` in place by multiplying its last factor by a value `α`.
Function acts as a wrapper around [`lmul!`](@ref) to allow for the arguments' order to be changed.

# Arguments
- `α::T`: Value of type `T` to multiply with.
- `U::TT{T,N}`: Tensor train of type `TT`.

# Returns
- `TT{T,N}`: Modified tensor train `U` after scaling its last factor by `α`.
"""
rmul!(U::TT{T,N}, α::T) where {T<:Number,N} = lmul!(α, U)

"""
    *(α::T, U::TT{T,N})

Create a new tensor train by multiplying the last factor of the deepcopy of `U` by a value `α`.

# Arguments
- `α::T`: Value of type `T` to multiply with.
- `U::TT{T,N}`: Tensor train of type `TT`.

# Returns
- `TT{T,N}`: New tensor train `V` that is a deepcopy of `U` with its last factor scaled by `α`.
"""
function *(α::T, U::TT{T,N}) where {T<:Number,N}
	V = deepcopy(U)
	return lmul!(α, U)
end

"""
    *(U::TT{T,N}, α::T) where {T<:Number,N}

Create a new tensor train by multiplying the last factor of the deepcopy of `U` by a value `α`.
Function acts as a wrapper around the default function [`*`](@ref) to allow for the arguments' order to be interchanged.

# Arguments
- `U::TT{T,N}`: Tensor train of type `TT`.
- `α::T`: Value of type `T` to multiply with.

# Returns
- `TT{T,N}`: New tensor train `V` that is a deepcopy of `U` with its last factor scaled by `α`.
"""
*(U::TT{T,N}, α::T) where {T<:Number,N} = *(α, U)

"""
    mul(U₁::TT{T,N₁}, σ₁::Indices, U₂::TT{T,N₂}, σ₂::Indices)

Multiply two tensor trains `U₁` and `U₂` along the specified sets of modes `σ₁` and `σ₂`, respectively, and return a new tensor train.

# Arguments
- `U₁::TT{T,N₁}`: First tensor train of type `TT`.
- `σ₁::Indices`: Indices of modes for the first tensor train to be contracted along.
- `U₂::TT{T,N₂}`: Second tensor train of type `TT`.
- `σ₂::Indices`: Indices of modes for the second tensor train to be contracted along.

# Returns
- `TT{T,N}`: New tensor train obtaied by the multiplication of `U₁` and `U₂` along the specified modes.

# Throws
- `ArgumentError`: If `U₁` and `U₂` differ in the number of factors.
- `ArgumentError`: If σ₁ is passed as a vector, and it is not a vector of type Vector{Int} or an empty vector of type Vector{Any}.
- `ArgumentError`: If σ₂ is passed as a vector, and it is not a vector of type Vector{Int} or an empty vector of type Vector{Any}.
- `ArgumentError`: If the specified sets of modes of σ₁ and σ₂ are inconsistent.
- `ArgumentError`: If the set of modes of U₁ is specified incorrectly.
- `ArgumentError`: If the set of modes of U₂ is specified incorrectly.
- `ArgumentError`: If U₁ and U₂ are inconsistent with respect to the specified modes.
"""
function mul(U₁::TT{T,N₁}, σ₁::Indices, U₂::TT{T,N₂}, σ₂::Indices) where {T<:Number,N₁,N₂}
	n₁ = size(U₁); d₁ = ndims(U₁)
	n₂ = size(U₂); d₂ = ndims(U₂)
	L = length(U₁)
	if length(U₂) ≠ L
		throw(ArgumentError("U₁ and U₂ differ in the number of factors"))
	end
	if isa(σ₁, Vector{Any}) && length(σ₁) > 0
		throw(ArgumentError("if σ₁ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	if isa(σ₂, Vector{Any}) && length(σ₂) > 0
		throw(ArgumentError("if σ₂ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	isa(σ₁, Vector{Int}) || (σ₁ = indvec(σ₁; max=d₁))
	isa(σ₂, Vector{Int}) || (σ₂ = indvec(σ₂; max=d₂))
	if length(σ₁) ≠ length(σ₂)
		throw(ArgumentError("the specified sets of modes of σ₁ and σ₂ are inconsistent"))
	end
	if σ₁ ⊈ 1:d₁ || unique(σ₁) ≠ σ₁
		throw(ArgumentError("the set of modes of U₁ is specified incorrectly"))
	end
	if σ₂ ⊈ 1:d₂ || unique(σ₂) ≠ σ₂
		throw(ArgumentError("the set of modes of U₂ is specified incorrectly"))
	end
	if n₁[σ₁,:] ≠ n₂[σ₂,:]
		throw(ArgumentError("U₁ and U₂ are inconsistent with respect to the specified modes"))
	end
	factors = [ factormp(getfactor(U₁, ℓ), σ₁, getfactor(U₂, ℓ), σ₂) for ℓ ∈ 1:L ]
	return TT(factors)
end

"""
    had(U::TT{T,N}, V::TT{T,N})

Perform the Hadamard product of two tensor trains `U` and `V`, and yield the new tensor train.

# Arguments
- `U::TT{T,N}`: First tensor train of type `TT`.
- `V::TT{T,N}`: Second tensor train of type `TT`.

# Returns
- `TT{T,N}`: New tensor train obtained by the element-wise multiplication of the factors of tensor trains `U` and `V`.

# Throws
- `ArgumentError`: If `U` and `V` differ in the number of factors.
- `ArgumentError`: If `U` and `V` are inconsistent in mode size.
"""
function had(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N}
	L = length(U)
	if length(V) ≠ L
		throw(ArgumentError("U and V differ in the number of factors"))
	end
	if size(U) ≠ size(V)
		throw(ArgumentError("U and V are inconsistent in mode size"))
	end
	factors = [ factorhp(getfactor(U, ℓ), getfactor(V, ℓ)) for ℓ ∈ 1:L ]
	return TT(factors)
end

"""
    *(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N}

Perform the Hadamard product of two tensor trains `U` and `V`, and yield the new tensor train.
Funcion is an alias for [`had`](@ref).

# Arguments
- `U::TT{T,N}`: First tensor train of type `TT`.
- `V::TT{T,N}`: Second tensor train of type `TT`.

# Returns
- `TT{T,N}`: New tensor train obtained by the element-wise multiplication of the factors of tensor trains `U` and `V`.

# Throws
- `ArgumentError`: If `U` and `V` differ in the number of factors.
- `ArgumentError`: If `U` and `V` are inconsistent in mode size.
"""
*(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N} = had(U, V)

"""
    kron(U₁::Dec{T,N}, U₂::Dec{T,N}) -> TT{T,N}

Compute the Kronecker product of two decompositions `U₁` and `U₂` of type `Dec`, and return the result as a new tensor train.

# Arguments
- `U₁::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors with elements of type `T` and with `N` as the number dimensions.
- `U₂::Dec{T,N}`: Second decomposition of type `Dec`, , which represents a vector of factors with elements of type `T` and with `N` as the number dimensions.

# Returns
- `TT{T,N}`: New tensor train of type `TT` obtained as the Kronecker product of `U₁` and `U₂`.

# Throws
- `ArgumentError`: If `U₁` and `U₂` differ in the number of factors. 
- `ArgumentError`: If `U₁` and `U₂` are inconsistent in the number of mode dimensions.
"""
function kron(U₁::Dec{T,N}, U₂::Dec{T,N}) where {T<:Number,N}
	L = length(U₁)
	if length(U₂) ≠ L
		throw(ArgumentError("U₁ and U₂ differ in the number of factors"))
	end
	if ndims(U₁) ≠ ndims(U₂)
		throw(ArgumentError("U₁ and U₂ are inconsistent in the number of mode dimensions"))
	end
	factors = [ factorkp(U₁[ℓ], U₂[ℓ]) for ℓ ∈ 1:L ]
	return TT(factors)
end

"""
    ⊗(U₁::Dec{T,N}, U₂::Dec{T,N})

An alias for `kron(U₁, U₂)`. Compute the Kronecker product of two decompositions `U₁` and `U₂`.

# Arguments
- `U₁::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors with elements of type `T` and with `N` as the number dimensions.
- `U₂::Dec{T,N}`: Second decomposition of type `Dec`, , which represents a vector of factors with elements of type `T` and with `N` as the number dimensions.

# Returns
- `TT{T,N}`: New tensor train of type `TT` obtained as the Kronecker product of `U₁` and `U₂`.
"""
⊗(U₁::Dec{T,N}, U₂::Dec{T,N}) where {T<:Number,N} = kron(U₁, U₂)

"""
    add(U::TT{T,N}, V::TT{T,N})

Add two tensor trains `U` and `V` and return a new tensor train object that represents their sum.

# Arguments
- `U::TT{T,N}`: First tensor train of type `TT`.
- `V::TT{T,N}`: Second tensor train of type `TT`.

# Returns
- `TT{T,N}`: New tensor train `W` obtained as the sum of `U` and `V`.

# Throws
- `ArgumentError`: If U and V differ in the number of factors.
- `ArgumentError`: If U and V are inconsistent in mode size.
- `ArgumentError`: If the decompositions are incompatible in the first rank.
- `ArgumentError`: If the decompositions are incompatible in the last rank.
"""
function add(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N}
	m = size(U); L = length(U)
	if length(V) ≠ L
		throw(ArgumentError("U and V differ in the number of factors"))
	end
	(L == 0) && return TT(T, d, 0)
	p = rank(U); q = rank(V)
	if size(V) ≠ m
		throw(ArgumentError("U and V are inconsistent in mode size"))
	end
	if q[1] ≠ p[1]
		throw(ArgumentError("the decompositions are incompatible in the first rank"))
	end
	if q[L+1] ≠ p[L+1]
		throw(ArgumentError("the decompositions are incompatible in the last rank"))
	end
	W = TT(T, d, 0)
	if L == 1
		push!(W, getfirstfactor(U) + getfirstfactor(V))
	elseif L > 1
		push!(W, factorhcat(getfirstfactor(U), getfirstfactor(V)))
		for ℓ ∈ 2:L-1
			push!(W, factordcat(getfactor(U, ℓ), getfactor(V, ℓ)))
		end
		push!(W, factorvcat(getlastfactor(U), getlastfactor(V)))
	end
	return W
end

"""
    +(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N}

Add two tensor trains `U` and `V` and return a new tensor train object that represents their sum.
Function is an alias for [`add`](@ref).

# Arguments
- `U::TT{T,N}`: First tensor train of type `TT`.
- `V::TT{T,N}`: Second tensor train of type `TT`.

# Returns
- `TT{T,N}`: New tensor train `W` obtained as the sum of `U` and `V`.

# Throws
- `ArgumentError`: If U and V differ in the number of factors.
- `ArgumentError`: If U and V are inconsistent in mode size.
- `ArgumentError`: If the decompositions are incompatible in the first rank.
- `ArgumentError`: If the decompositions are incompatible in the last rank.
"""
+(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N} = add(U, V)

"""
    qr!(W::TT{T,N}, Λ::Path; path::String="") where {T<:FloatRC,N}

Perform an in-place QR decomposition on the specified factors of the tensor train `W`. 

# Arguments
- `W::TT{T,N}`: Tensor train of type `TT`.
- `Λ::Path`: Path along which to perform the QR decomposition. Can be specified as a range, a vector of integers, or `Colon`.
- `path::String=""`: Specifies the direction of the decomposition. Can be `"forward"` for left-to-right decomposition, `"backward"` for right-to-left decomposition, or an empty string `""` to automatically deduce the path based on `Λ`.

# Returns
- Modified tensor train `W` after performing the QR decomposition along the specified path.

# Throws
- `ArgumentError`: If the tensor train `W` is empty.
- `ArgumentError`: If `path` is none of the following: `"forward"`, `"backward"`, or an empty string.
- `ArgumentError`: If `path` cannot be deduced from Λ or path is not specified when `Λ` is a colon.
- `ArgumentError`: If `Λ` has duplicate entries.
- `ArgumentError`: If the entries of `Λ` do not form a set of contiguous integers.
- `ArgumentError`: If `Λ` is out of the valid range `1:L` where `L` is the length of the tensor train `W`.
- `ArgumentError`: If `Λ` is neither sorted in ascending nor in descending order, leading to inconsitency with either \"forward\" or \"backward\" `path`.
- `ArgumentError`: If `Λ` is not sorted in ascending order, rendering it inconsistent with any forward path.
- `ArgumentError`: If `Λ` is not sorted in descending order, rendering it inconsistent with any backward path.
"""
function qr!(W::TT{T,N}, Λ::Path; path::String="") where {T<:FloatRC,N}
	L = length(W); rank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if path ∉ ("","forward","backward")
		throw(ArgumentError("path should be either \"\" (default, accepted only when path can be deduced from Λ), \"forward\" or \"backward\""))
	end
	if path == "" && (isa(Λ, Colon) || length(unique(Λ)) == 1)
		throw(ArgumentError("path cannot be deduced from Λ and should be specified as either \"forward\" or \"backward\""))
	end
	isa(Λ, Colon) && (Λ = (path == "forward") ? (1:L) : (L:-1:1))
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	(length(Λ) == 0) && return W
	if unique(Λ) ≠ Λ
		throw(ArgumentError("Λ has duplicate entries"))
	end
	if (minimum(Λ):maximum(Λ)) ⊈ Λ
		throw(ArgumentError("the entries of Λ should form a set of contiguous integers"))
	end
	if Λ ⊈ 1:L
		throw(ArgumentError("Λ is out of range"))
	end
	fw,bw = issorted(Λ, rev=false),issorted(Λ, rev=true)
	if !fw && !bw
		throw(ArgumentError("Λ is not sorted in ascending or descending order, so it is inconsistent with any forward or backward path"))
	end
	@assert !(fw && bw && path == "")
	fw && path == "" && (path = "forward")
	bw && path == "" && (path = "backward")
	if path == "forward" && !fw
		throw(ArgumentError("Λ is not sorted in ascending order, so it is inconsistent with any forward path"))
	end
	if path == "backward" && !bw
		throw(ArgumentError("Λ is not sorted in descending order, so it is inconsistent with any backward path"))
	end
	M = length(Λ)
	for λ ∈ 1:M
		ℓ = Λ[λ]
		Q,R = factorqr!(getfactor(W, ℓ); rev=(path == "backward"))
		setfactor!(W, Q, ℓ)
		if λ < M
			ν = Λ[λ+1]
			F = factorcontract(R, getfactor(W ,ν), rev=(path == "backward"))
			setfactor!(W, F, ν)
		else
			insert!(W, ℓ, R; path=path, rankprecheck=false, rankpostcheck=true)
		end
	end
	return W
end

"""
    qr!(W::TT{T,N}; path::String="") where {T<:FloatRC,N}

Perform an in-place QR decomposition on all the factors of the tensor train `W`.

# Arguments
- `W::TT{T,N}`: Tensor train of type `TT`.
- `path::String=""`: Specifies the direction of the decomposition. Can be `"forward"` for left-to-right decomposition or `"backward"` for right-to-left decomposition.

# Returns
- Modified tensor train `W` after performing the QR decomposition along the specified path.

# Throws
- `ArgumentError`: If the tensor train `W` is empty.
- `ArgumentError`: If `path` is not specified as `"forward"` or `"backward"` (Notice: Λ is a colon, thus path must be specified)
"""
qr!(W::TT{T,N}; path::String="") where {T<:FloatRC,N} = qr!(W, :; path=path)

"""
    svd!(W::TT{T,N}, Λ::Path, n::Union{Colon,DecSize}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") -> (TT{T,N}, Vector{Float64}, Vector{Float64}, Float64, Vector{Int}, Vector{Vector{Float64}})

Perform an in-place singular value decomposition (SVD) with specified path and tolerances on the tensor train `W`. Assume that the decomposition `W` is orthogonal.

# Arguments
- `W::TT{T,N}`: Tensor train to be decomposed in-place.
- `Λ::Path`: Path along which to perform the SVD. Can be a colon (`:`) or a vector of integers.
- `n::Union{Colon,DecSize}`: Specifies target dimensions of the factors after decomposition. Can be a colon (`:`) or a size matrix.
- `path::String`: Specifies the direction of the decomposition. Accepts `""` (default), `"forward"`, or `"backward"`.
- `aTol::Float2`: Absolute tolerance for truncation. Can be a single float or a vector of floats.
- `aTolDistr::Float2`: Absolute tolerance for distributed truncation. Can be a single float or a vector of floats.
- `rTol::Float2`: Relative tolerance for truncation. Can be a single float or a vector of floats.
- `rTolDistr::Float2`: Relative tolerance for distributed truncation. Can be a single float or a vector of floats.
- `maxrank::Int2`: Maximum rank for truncation. Can be a single integer or a vector of integers.
- `major::String`: Specifies whether to prioritize the first or last indices during the operation. Acceptable values are `"last"` (default) or `"first"`.

# Returns
- `W::TT{T,N}`: Modified tensor train after performing the in-place SVD.
- `ε::Vector{Float64}`: Vector of absolute errors for each factor.
- `δ::Vector{Float64}`: Vector of relative errors for each factor.
- `μ::Float64`: Norm of the core tensor after the decomposition.
- `ρ::Vector{Int}`: Vector of ranks for each factor.
- `σ::Vector{Vector{Float64}}`: Vector of singular values for each factor.

# Throws
Summarized Error List:
- `ArgumentError`: If any input arguments are invalid, for example incorrect dimensions, negative values where non-negative are expected or invalid strings for path or major.

Extended Error List:
- `ArgumentError`: If the decomposition is empty.
- `ArgumentError`: If path is none of the following: \"\" (default, accepted only when path can be deduced from Λ), \"forward\" or \"backward\"".
- `ArgumentError`: If path cannot be deduced from Λ or is the path is neither \"forward\" nor \"backward\" when Λ is a colon.
- `ArgumentError`: If the entries of Λ should form a set of contiguous integers")
- `ArgumentError`: If Λ is out of range.
- `ArgumentError`: If Λ is not sorted in ascending or descending order, rendering it inconsistent with any forward or backward path.
- `ArgumentError`: If Λ is not sorted in ascending order, rendering it inconsistent with any forward path.
- `ArgumentError`: If Λ is not sorted in descending order, rendering it inconsistent with any backward path.
- `ArgumentError`: If the number of rows in n is not equal to the number of dimensions in each factor of W.
- `ArgumentError`: If the number of columns in n is not equal to the number of elements in Λ.
- `ArgumentError`: If n and Λ are incompatible with the size of the factors of W.
- `ArgumentError`: If aTol is not a nonnegative Float64 or a vector of such.
- `ArgumentError`: If aTol, passed as a vector, has incorrect length.
- `ArgumentError`: If aTolDistr is not a nonnegative Float64 or a vector of such.
- `ArgumentError`: If aTolDistr, passed as a vector, has incorrect length.
- `ArgumentError`: If rTol is not a nonnegative Float64 or a vector of such.
- `ArgumentError`: If rTol, passed as a vector, has incorrect length.
- `ArgumentError`: If rTolDistr is not a nonnegative Float64 or a vector of such.
- `ArgumentError`: If rTolDistr, passed as a vector, has incorrect length.
- `ArgumentError`: If maxrank is not a nonnegative Int or a vector of such.
- `ArgumentError`: If maxrank, passed as a vector, has incorrect length.
- `ArgumentError`: If major is neither \"last\" (default) nor \"first\"".
"""
function svd!(W::TT{T,N}, Λ::Path, n::Union{Colon,DecSize}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") where {T<:FloatRC,N}
	# the decomposition is assumed to be orthogonal
	L = length(W); rank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if path ∉ ("","forward","backward")
		throw(ArgumentError("path should be either \"\" (default, accepted only when path can be deduced from Λ), \"forward\" or \"backward\""))
	end
	if path == "" && (isa(Λ, Colon) || length(unique(Λ)) == 1)
		throw(ArgumentError("path cannot be deduced from Λ and should be specified as either \"forward\" or \"backward\""))
	end
	isa(Λ, Colon) && (Λ = (path == "forward") ? (1:L) : (L:-1:1))
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	(length(Λ) == 0) && return W
	if (minimum(Λ):maximum(Λ)) ⊈ Λ
		throw(ArgumentError("the entries of Λ should form a set of contiguous integers"))
	end
	if Λ ⊈ 1:L
		throw(ArgumentError("Λ is out of range"))
	end
	fw,bw = issorted(Λ, rev=false),issorted(Λ, rev=true)
	if !fw && !bw
		throw(ArgumentError("Λ is not sorted in ascending or descending order, so it is inconsistent with any forward or backward path"))
	end
	fw && path == "" && (path = "forward")
	bw && path == "" && (path = "backward")
	if path == "forward" && !fw
		throw(ArgumentError("Λ is not sorted in ascending order, so it is inconsistent with any forward path"))
	end
	if path == "backward" && !bw
		throw(ArgumentError("Λ is not sorted in descending order, so it is inconsistent with any backward path"))
	end
	m = size(W)
	isa(n, Colon) && (n = m[:,Λ])
	if size(n,1) ≠ size(m,1)
		throw(ArgumentError("the number of rows in n should be equal to the number of dimensions in each factor of W"))
	end
	if size(n,2) ≠ length(Λ)
		throw(ArgumentError("the number of columns in n should be equal to the number of elements in Λ"))
	end
	for ℓ ∈ unique(Λ)
		λ = (Λ .== ℓ)
		if m[:,ℓ] ≠ prod(n[:,λ]; dims=2)[:]
			throw(ArgumentError("n and Λ are incompatible with the size of the factors of W"))
		end
	end
	K = length(Λ)

	if any(aTol .< 0)
		throw(ArgumentError("aTol should be a nonnegative Float64 or a vector of such"))
	end
	if isa(aTol, Float64)
		aTol = aTol*ones(K)/sqrt(K)
	elseif length(aTol) ≠ K
		throw(ArgumentError("aTol, passed as a vector, has incorrect length"))
	end

	if any(aTolDistr .< 0)
		throw(ArgumentError("aTolDistr should be a nonnegative Float64 or a vector of such"))
	end
	if isa(aTolDistr, Float64)
		aTolDistr = aTolDistr*ones(K)/sqrt(K)
	elseif length(aTolDistr) ≠ K
		throw(ArgumentError("aTolDistr, passed as a vector, has incorrect length"))
	end

	if any(rTol .< 0)
		throw(ArgumentError("rTol should be a nonnegative Float64 or a vector of such"))
	end
	if isa(rTol, Float64)
		rTol = rTol*ones(K)/sqrt(K)
	elseif length(rTol) ≠ K
		throw(ArgumentError("rTol, passed as a vector, has incorrect length"))
	end

	if any(rTolDistr .< 0)
		throw(ArgumentError("rTolDistr should be a nonnegative Float64 or a vector of such"))
	end
	if isa(rTolDistr, Float64)
		rTolDistr = rTolDistr*ones(K)/sqrt(K)
	elseif length(rTolDistr) ≠ K
		throw(ArgumentError("rTolDistr, passed as a vector, has incorrect length"))
	end

	if any(maxrank .< 0)
		throw(ArgumentError("maxrank should be a nonnegative Int or a vector of such"))
	end
	if isa(maxrank, Int)
		maxrank = maxrank*ones(Int, K)
	elseif length(maxrank) ≠ K
		throw(ArgumentError("maxrank, passed as a vector, has incorrect length"))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end

	ε = zeros(Float64, K); δ = zeros(Float64, K)
	σ = Vector{Vector{Float64}}(undef, K)
	ρ = zeros(Int, K)
	μ = 0.0
	aTolAcc = 0.0; rTolAcc = 0.0
	for λ ∈ 1:K
		F = getfactor(W, Λ[λ])
		if λ == 1
			ε₁ = [aTol[λ],aTolDistr[λ]]; ε₁ = ε₁[ε₁ .> 0]; ε₁ = isempty(ε₁) ? 0.0 : minimum(ε₁)
			δ₁ = [rTol[λ],rTolDistr[λ]]; δ₁ = δ₁[δ₁ .> 0]; δ₁ = isempty(δ₁) ? 0.0 : minimum(δ₁)
			U,V,ε[1],δ[1],μ,ρ[1],σ[1] = factorsvd!(F, n[:,λ], :; atol=ε₁, rtol=δ₁, rank=maxrank[λ], rev=(path == "backward"), major=major)
		else
			(aTolDistr[λ] > 0) && (aTolDistr[λ] = sqrt(aTolDistr[λ]^2+aTolAcc^2); aTolAcc = 0.0)
			(rTolDistr[λ] > 0) && (rTolDistr[λ] = sqrt(rTolDistr[λ]^2+rTolAcc^2); rTolAcc = 0.0)
			ε₁ = [aTol[λ],aTolDistr[λ],μ*rTol[λ],μ*rTolDistr[λ]]; ε₁ = ε₁[ε₁ .> 0]
			ε₁ = isempty(ε₁) ? 0.0 : minimum(ε₁)
			U,V,ε[λ],_,_,ρ[λ],σ[λ] = factorsvd!(F, n[:,λ], :; atol=ε₁, rank=maxrank[λ], rev=(path == "backward"), major=major)
			δ[λ] = (μ > 0) ? ε[λ]/μ : 0.0
		end
		setfactor!(W, U, Λ[λ])
		if λ < K && Λ[λ+1] ≠ Λ[λ]
			F = getfactor(W, Λ[λ+1])
			F = factorcontract(V, F, rev=(path == "backward"), major=major)
			setfactor!(W, F, Λ[λ+1])
		else
			insert!(W, Λ[λ], V; path=path, rankprecheck=false, rankpostcheck=true)
			(path == "forward") && (Λ[λ:end] .+= 1)
		end
	end
	return W,ε,δ,μ,ρ,σ
end

"""
    svd!(W::TT{T,N}, Λ::Path, n::Union{Colon,DecSize}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") -> (TT{T,N}, Vector{Float64}, Vector{Float64}, Float64, Vector{Int}, Vector{Vector{Float64}})

Perform an in-place singular value decomposition (SVD) of the in `Λ` specified factors with automatically deduced target dimensions of the factors.
Consider specified path and tolerances on the tensor train `W`. Assume that the decomposition `W` is orthogonal.

# Arguments
- `W::TT{T,N}`: Tensor train to be decomposed in-place.
- `Λ::Path`: Path along which to perform the SVD. Can be a colon (`:`) or a vector of integers.
- `path::String`: Specifies the direction of the decomposition. Accepts `""` (default), `"forward"`, or `"backward"`.
- `aTol::Float2`: Absolute tolerance for truncation. Can be a single float or a vector of floats.
- `aTolDistr::Float2`: Absolute tolerance for distributed truncation. Can be a single float or a vector of floats.
- `rTol::Float2`: Relative tolerance for truncation. Can be a single float or a vector of floats.
- `rTolDistr::Float2`: Relative tolerance for distributed truncation. Can be a single float or a vector of floats.
- `maxrank::Int2`: Maximum rank for truncation. Can be a single integer or a vector of integers.
- `major::String`: Specifies whether to prioritize the first or last indices during the operation. Acceptable values are `"last"` (default) or `"first"`.

# Returns
- `W::TT{T,N}`: Modified tensor train after performing the in-place SVD.
- `ε::Vector{Float64}`: Vector of absolute errors for each factor.
- `δ::Vector{Float64}`: Vector of relative errors for each factor.
- `μ::Float64`: Norm of the core tensor after the decomposition.
- `ρ::Vector{Int}`: Vector of ranks for each factor.
- `σ::Vector{Vector{Float64}}`: Vector of singular values for each factor.

# Throws
- `ArgumentError`: If any input arguments are invalid, for example incorrect dimensions, negative values where non-negative are expected or invalid strings for path or major.
- For full error control, see default [`decsvd!`](@ref).
"""
svd!(W::TT{T,N}, Λ::Path; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") where {T<:FloatRC,N} = svd!(W, Λ, :; path=path, aTol=aTol, aTolDistr=aTolDistr, rTol=rTol, rTolDistr=rTolDistr, maxrank=maxrank, major=major)

"""
    svd!(W::TT{T,N}, Λ::Path, n::Union{Colon,DecSize}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") -> (TT{T,N}, Vector{Float64}, Vector{Float64}, Float64, Vector{Int}, Vector{Vector{Float64}})

Perform an in-place singular value decomposition (SVD) of all factors with automatically deduced target dimensions of the factors.
Consider specified path and tolerances on the tensor train `W`. Assume that the decomposition `W` is orthogonal.

# Arguments
- `W::TT{T,N}`: Tensor train to be decomposed in-place.
- `path::String`: Specifies the direction of the decomposition. Accepts `""` (default), `"forward"`, or `"backward"`.
- `aTol::Float2`: Absolute tolerance for truncation. Can be a single float or a vector of floats.
- `aTolDistr::Float2`: Absolute tolerance for distributed truncation. Can be a single float or a vector of floats.
- `rTol::Float2`: Relative tolerance for truncation. Can be a single float or a vector of floats.
- `rTolDistr::Float2`: Relative tolerance for distributed truncation. Can be a single float or a vector of floats.
- `maxrank::Int2`: Maximum rank for truncation. Can be a single integer or a vector of integers.
- `major::String`: Specifies whether to prioritize the first or last indices during the operation. Acceptable values are `"last"` (default) or `"first"`.

# Returns
- `W::TT{T,N}`: Modified tensor train after performing the in-place SVD.
- `ε::Vector{Float64}`: Vector of absolute errors for each factor.
- `δ::Vector{Float64}`: Vector of relative errors for each factor.
- `μ::Float64`: Norm of the core tensor after the decomposition.
- `ρ::Vector{Int}`: Vector of ranks for each factor.
- `σ::Vector{Vector{Float64}}`: Vector of singular values for each factor.

# Throws
- `ArgumentError`: If any input arguments are invalid, for example incorrect dimensions, negative values where non-negative are expected or invalid strings for path or major.
- For full error control, see default [`decsvd!`](@ref).
"""
svd!(W::TT{T,N}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") where {T<:FloatRC,N} = svd!(W, :, :; path=path, aTol=aTol, aTolDistr=aTolDistr, rTol=rTol, rTolDistr=rTolDistr, maxrank=maxrank, major=major)
