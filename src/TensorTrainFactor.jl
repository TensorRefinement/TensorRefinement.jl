export FactorSize, Factor, VectorFactor, MatrixFactor
export factorsize, factorranks, factorndims, factornumentries, factorstorage
export factor, factormatrix, factorrankselect, block
export factorvcat, factorhcat, factordcat, factorltcat, factorutcat
export factorranktranspose, factormodetranspose
export factormodereshape
export factordiagm
export factorcontract, factormp, factorkp, factorhp
export factorqr!, factorqradd, factorsvd!


"""
`Factorsize` is an alias for a `Vector{Int}`.
"""
const FactorSize = Vector{Int}

"""
`Factor{T,N}` is an alias for `Array{T,N}`.
"""
const Factor{T,N} = Array{T,N} where {T<:Number,N}
"""
`VectorFactor{T}` is a 3D tensor with entries of type `T`.
"""
const VectorFactor{T} = Factor{T,3} where T<:Number
"""
`MatrixFactor{T}` is a 4D tensor with entries of type `T`.
"""
const MatrixFactor{T} = Factor{T,4} where T<:Number


"""
    factorsize(U::Factor{T,N}) where {T<:Number,N}

Determine the sizes of the mode dimensions of the given factor `U` in a vector, excluding the first and last dimensions.

# Arguments
- `U::Factor{T, N}`: factor of type `Factor` with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` as the number dimensions.

# Returns
- `Factorsize(n)`: custom type `Factorsize` that holds the mode dimensions of `U` excluding the first and last dimension as a vector.

# Throws
- `ArgumentError`: If factor exhibits only one or no rank dimension.
- `ArgumentError`: If factor exhibits negative mode size.

# Example
```julia
U = Factor(rand(3, 4, 5, 6, 7))  # Initialization of random Factor
size_vector = factorsize(U)  # Returns vector containing all but the first and last dimensions
println(size_vector)  # Outputs FactorSize([4, 5, 6]) where Factorsize = Vector{Int}
```
"""
function factorsize(U::Factor{T,N}) where {T<:Number,N}
	sz = size(U)
	if length(sz) < 2
		throw(ArgumentError("the factor should have, at least, two rank dimensions"))
	end
	n = collect(sz[2:end-1])
	if any(n .== 0)
		throw(ArgumentError("the mode sizes should be positive"))
	end
	FactorSize(n)
end

"""
    factorranks(U::Factor{T,N}) where {T<:Number,N}

Determine the first and last rank dimension of a given factor `U`.

# Arguments
- `U::Factor{T, N}`: factor of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` as the number dimensions.

# Returns
- Tuple `(sz[1], sz[end])` where `sz[1]` is the first and `sz[end]` is the last dimension of the input tensor.

# Throws
- `ArgumentError`: If factor exhibits only one or no rank dimension.
- `ArgumentError`: If factor exhibits negative mode size.

"""
function factorranks(U::Factor{T,N}) where {T<:Number,N}
	sz = size(U)
	if length(sz) < 2
		throw(ArgumentError("the factor should have, at least, two rank dimensions"))
	end
	if any(sz[2:end-1] .== 0)
		throw(ArgumentError("the mode sizes should be positive"))
	end
	sz[1],sz[end]
end

"""
    factornumentries(U::Factor{T,N}) where {T<:Number,N}

Determine the number of entries in the given factor.

# Arguments
- `U::Factor{T, N}`: factor of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` as the number dimensions.

# Returns
- An integer specifiying the number of entries in the given [`Factor`].
"""
factornumentries(U::Factor{T,N}) where {T<:Number,N} = length(U)

"""
An alias for the function factornumentries.
"""
factorstorage(U::Factor{T,N}) where {T<:Number,N} = factornumentries(U)

"""
    factorndims(::Factor{T,N}) where {T<:Number,N}

Return the number of mode dimensions excluding the first and last dimension.

# Arguments
- `::Factor{T, N}`: factor of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` as the number dimensions.

# Returns
- An integer representing the number of mode dimensions, which is `N - 2` (number of entries in [`factorsize`](@ref)).

# Throws
- `ArgumentError`: If factor exhibits only one or no rank dimension.
"""
function factorndims(::Factor{T,N}) where {T<:Number,N}
	if N < 2
		throw(ArgumentError("the factor should have, at least, two rank dimensions"))
	end
	N-2
end

"""
    factor(U::Array{T,N}) where {T<:Number,N}

Reshape a given tensor `U` into a factor that can be used in further TT-operations.

# Arguments
- `U::Array{T, N}`: Multi-dimensional array with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` as the number dimensions.

# Returns
- A reshaped version of the tensor `U` with a single first and last dimension to facilitate TT-operations.
"""
function factor(U::Array{T,N}) where {T<:Number,N}
	reshape(U, 1, size(U)..., 1)
end

"""
    factor(U::Matrix{T}, m::Union{Int, NTuple{M,Int}, Vector{Int}, Vector{Any}}, 
           n::Union{Int, NTuple{N,Int}, Vector{Int}, Vector{Any}}, 
           π::Union{NTuple{K,Int}, Vector{Int}}) where {T<:Number, K, M, N}

Reshape a matrix `U` into a multi-dimensional array with specified mode sizes `m` and `n`, 
and permute the mode dimensions according to the permutation `π`.

# Arguments
- `U::Matrix{T}`: Input matrix with elements of type `T`.
- `m::Union{Int, NTuple{M,Int}, Vector{Int}, Vector{Any}}`: Mode sizes for the first dimension of `U`.
- `n::Union{Int, NTuple{N,Int}, Vector{Int}, Vector{Any}}`: Mode sizes for the second dimension of `U`.
- `π::Union{NTuple{K,Int}, Vector{Int}}`: Permutation of the mode dimensions.

# Returns
- A reshaped and permuted tensor based on the provided mode sizes `m`, `n`, and permutation `π`.

# Throws
- `ArgumentError`: If `π` is empty or not a valid permutation of `1:length(m) + length(n)`.
- `ArgumentError`: If `m` or `n` is none of the following: integer, vector or tuple of integers, empty vector or tuple. 
- `DimensionMismatch`: If the dimensions of `U` are not divisible by the products of `m` or `n`, respectively.
"""
function factor(U::Matrix{T}, m::Union{Int,NTuple{M,Int},Vector{Int},Vector{Any}}, n::Union{Int,NTuple{N,Int},Vector{Int},Vector{Any}}, π::Union{NTuple{K,Int},Vector{Int}}) where {T<:Number,K,M,N}
	d = length(π)
	if d == 0
		throw(ArgumentError("π is empty"))
	end
	if isa(m, Vector{Any}) && length(m) > 0
		throw(ArgumentError("m should be an integer, a vector or tuple of integers or an empty vector or tuple"))
	end
	isa(m, Int) && (m = [m])
	isa(m, Vector{Any}) && (m = Vector{Int}(undef, 0))
	isa(m, Tuple{}) && (m = Vector{Int}(undef, 0))
	isa(m, Vector{Int}) || (m = collect(m))
	dm = length(m)
	if isa(n, Vector{Any}) && length(n) > 0
		throw(ArgumentError("n should be an integer, a vector or tuple of integers or an empty vector or tuple"))
	end
	isa(n, Int) && (n = [n])
	isa(n, Vector{Any}) && (n = Vector{Int}(undef, 0))
	isa(n, Tuple{}) && (n = Vector{Int}(undef, 0))
	isa(n, Vector{Int}) || (n = collect(n))
	dn = length(n)
	if dm+dn ≠ d || !isperm(π)
		throw(ArgumentError("π is not a valid permutation of 1:length(m)+length(n)"))
	end
	sz = size(U); p = sz[1]÷prod(m); q = sz[2]÷prod(n)
	if sz[1] ≠ prod(m)*p
		throw(DimensionMismatch("the first dimension of U is not divisible by prod(m)"))
	end
	if sz[2] ≠ prod(n)*q
		throw(DimensionMismatch("the second dimension of U is not divisible by prod(n)"))
	end
	U = reshape(U, (m...,p,n...,q))
	U = permutedims(U, (dm+1,π[1:dm]...,(π[dm+1:d].+1)...,d+2))
	U
end

"""
    factor(U::Matrix{T}, m::Union{Int, NTuple{M,Int}, Vector{Int}, Vector{Any}}, 
           n::Union{Int, NTuple{N,Int}, Vector{Int}, Vector{Any}}) where {T<:Number, M, N}

Reshape a matrix `U` into a multi-dimensional array with mode sizes `m` and `n` and use `1:length(m)+length(n)` as permutation of dimensions.

Function is a variant of the more general `factor` function with an automatic permutation sequence `1:length(m)+length(n)` applied to the reshaped matrix.

# Arguments
- `U::Matrix{T}`: Input matrix with elements of type `T`.
- `m::Union{Int, NTuple{M,Int}, Vector{Int}, Vector{Any}}`: Mode sizes for the first dimension of `U`.
- `n::Union{Int, NTuple{N,Int}, Vector{Int}, Vector{Any}}`: Mode sizes for the second dimension of `U`.

# Returns
- A reshaped and permuted factor based on the mode sizes `m` and `n` with a permutation `1:length(m)+length(n)`.

# Throws
- `ArgumentError`: If `m` or `n` is not an integer, a vector or tuple of integers, or an empty vector or tuple.
- `DimensionMismatch`: If the dimensions of `U` are not divisible by the products of `m` or `n`.
"""
factor(U::Matrix{T}, m::Union{Int,NTuple{M,Int},Vector{Int},Vector{Any}}, n::Union{Int,NTuple{N,Int},Vector{Int},Vector{Any}}) where {T<:Number,M,N} = factor(U, m, n, collect(1:length(m)+length(n)))

"""
    factor(U::Vector{T}, m::Union{Int, NTuple{M,Int}, Vector{Int}, Vector{Any}}, 
           π::Union{NTuple{M,Int}, Vector{Int}}) where {T<:Number, M}

Reshape a vector `U` into a two-dimensional array, then reshape it further into a multi-dimensional array based on the mode sizes `m` and apply the permutation `π`.

Function is a variant of the more general `factor` function, which allows reshaping a vector `U` into a matrix and then applying mode sizes and a custom permutation.

# Arguments
- `U::Vector{T}`: Input vector with elements of type `T`.
- `m::Union{Int, NTuple{M,Int}, Vector{Int}, Vector{Any}}`: Mode sizes for the reshaped array.
- `π::Union{NTuple{M,Int}, Vector{Int}}`: Permutation of the mode dimensions.

# Returns
- A reshaped and permuted factor based on the mode sizes `m` and permutation `π`.

# Throws
- `ArgumentError`: If `m` is not an integer, a vector or tuple of integers, or an empty vector or tuple.
"""
factor(U::Vector{T}, m::Union{Int,NTuple{M,Int},Vector{Int},Vector{Any}}, π::Union{NTuple{M,Int},Vector{Int}}) where {T<:Number,M} = factor(U[:,:], m, (), π)

"""
    factor(U::Vector{T}, m::Union{Int, NTuple{M,Int}, Vector{Int}, Vector{Any}}) where {T<:Number, M}

Reshape a vector `U` into a two-dimensional array, then reshape it into a multi-dimensional array based on the mode sizes `m` with a natural permutation.

Function is a variant of the more general `factor` function, which reshapes a vector `U` into a matrix and then into an array with mode sizes `m`. It also applies a natural permutation `1:length(m)`.

# Arguments
- `U::Vector{T}`: Input vector with elements of type `T`.
- `m::Union{Int, NTuple{M,Int}, Vector{Int}, Vector{Any}}`: Mode sizes for the reshaped array.

# Returns
- A reshaped and permuted factor based on the mode sizes `m` with a natural permutation.

# Throws
- `ArgumentError`: If `m` is not an integer, a vector or tuple of integers, or an empty vector or tuple.
"""
factor(U::Vector{T}, m::Union{Int,NTuple{M,Int},Vector{Int},Vector{Any}}) where {T<:Number,M} = factor(U[:,:], m, (), collect(1:length(m)))

"""
    factormatrix(U::Factor{T,K}, π::Indices, σ::Indices) where {T<:Number, K}

Convert a factor `U` into a matrix by permuting and reshaping the mode dimensions
according to the specified indices `π` and `σ`.

# Arguments
- `U::Factor{T,K}`: Input factor of type [`Factor`](@ref) with elements of type `T` and `K` dimensions.
- `π::Indices`: Reference numbers specifying the permutation of mode dimensions for the first dimension of the resulting matrix.
- `σ::Indices`: Reference numbers specifying the permutation of mode dimensions for the second dimension of the resulting matrix.

# Returns
- A matrix obtained by permuting and reshaping `U` according to the specified indices `π` and `σ`.

# Throws
- `ArgumentError`: If `U` does not have at least one mode dimension.
- `ArgumentError`: If `π` or `σ` are not valid indices or do not constitute a valid permutation of mode dimensions.
"""
function factormatrix(U::Factor{T,K}, π::Indices, σ::Indices) where {T<:Number,K}
	d = factorndims(U)
	if d == 0
		throw(ArgumentError("U should have at least one mode dimension"))
	end
	if isa(π, Vector{Any}) && length(π) > 0
		throw(ArgumentError("π should be an integer, a vector or tuple of integers, a colon or an empty vector or tuple"))
	end
	π = indvec(π; min=1, max=d); dπ = length(π)
	if isa(σ, Vector{Any}) && length(σ) > 0
		throw(ArgumentError("σ should be an integer, a vector or tuple of integers, a colon or an empty vector or tuple"))
	end
	σ = indvec(σ; min=1, max=d); dσ = length(σ)
	τ = vcat(π,σ)
	if dπ+dσ ≠ d || !isperm(τ)
		throw(ArgumentError("π and σ do not constitute a valid permutation of the mode dimensions of U"))
	end
	n = factorsize(U); p,q = factorranks(U)
	U = permutedims(U, ((π.+1)...,1,(σ.+1)...,d+2))
	U = reshape(U, p*prod(n[π]), q*prod(n[σ]))
	U
end

"""
    factorrankselect(U::Factor{T,N}, α::Indices, β::Indices) where {T<:Number, N}

Select specific slices from a factor `U` based on the provided rank indices `α` and `β`.

# Arguments
- `U::Factor{T,N}`: Input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `α::Indices`: Reference numbers for selecting slices along the first rank dimension.
- `β::Indices`: Reference numbers for selecting slices along the second rank dimension.

# Returns
- A sub-array of `U` corresponding to the specified slices.

# Throws
- `ArgumentError`: If the indices `α` or `β` are incorrect or out of range for the respective rank dimensions.
"""
function factorrankselect(U::Factor{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	isa(α, Int) && (α = [α])
	isa(β, Int) && (β = [β])
	p,q = factorranks(U)
	n = factorsize(U)
	α = indvec(α; min=1, max=p)
	β = indvec(β; min=1, max=q)
	if α ⊈ 1:p
		throw(ArgumentError("the index or range for the first rank is incorrect"))
	end
	if β ⊈ 1:q
		throw(ArgumentError("the index or range for the second rank is incorrect"))
	end
	U[α,ntuple(k -> Colon(), Val(N-2))...,β]
end

"""
    block(U::Factor{T,N}, α::Int, β::Int) where {T<:Number, N}

Extracts a specific block from the factor `U` based on the provided rank indices `α` and `β`.

Detailed description:
Function reshapes given factor into 3D array, selects all entries along second dimension, where `α` and `β` specify the first
and third dimension, and reshapes selected entries into a block with sizes equivalent to the mode dimensions of the initial array.

# Arguments
- `U::Factor{T,N}`: Input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `α::Int`: Reference numbers for selecting the slice along the first rank dimension.
- `β::Int`: Reference numbers for selecting the slice along the second rank dimension.

# Returns
- A sub-array representing the block corresponding to the specified rank indices.

# Throws
- `ArgumentError`: If the indices `α` or `β` are out of range for the respective rank dimensions.
"""
function block(U::Factor{T,N}, α::Int, β::Int) where {T<:Number,N}
	p,q = factorranks(U)
	n = factorsize(U)
	if α ∉ 1:p
		throw(ArgumentError("the first rank index is out of range"))
	end
	if β ∉ 1:q
		throw(ArgumentError("the second rank index is out of range"))
	end
	U = reshape(U, p, :, q)
	V = U[α,:,β]
	reshape(V, n...)
end

"""
    factorvcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number, N, M}

Vertically concatenate the factors `U`, `V`, and additional factors in `W` along the first rank dimension.

# Arguments
- `U::Factor{T,N}`: First factor to be concatenated. Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `V::Factor{T,N}`: Second factor to be concatenated. Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `W::Vararg{Factor{T,N},M}`: Additional factors to be concatenated, each of Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.

# Returns
- A new factor resulting from the vertical concatenation of `U`, `V`, and the additional factors in `W`.

# Throws
- `ArgumentError`: If any of the factors have incompatible mode sizes or second ranks.
"""
function factorvcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number,N,M}
	m = factorsize(U); _,p = factorranks(U)
	W = (V,W...)
	for V ∈ W
		if factorsize(V) ≠ m
			throw(ArgumentError("the factors are incompatible in mode size"))
		end
		_,q = factorranks(V)
		if q ≠ p
			throw(ArgumentError("the factors are incompatible in the second rank"))
		end
	end
	cat(U, W...; dims=1)
end

"""
    factorhcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number, N, M}

Horizontally concatenate the factors `U`, `V`, and additional factors in `W` along the second rank dimension.

# Arguments
- `U::Factor{T,N}`: First factor to be concatenated. Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `V::Factor{T,N}`: Second factor to be concatenated. Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `W::Vararg{Factor{T,N},M}`: Additional factors to be concatenated, each of Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.

# Returns
- A new factor resulting from the horizontal concatenation of `U`, `V`, and the additional factors in `W`.

# Throws
- `ArgumentError`: If any of the factors have incompatible mode sizes or first ranks.
"""
function factorhcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number,N,M}
	m = factorsize(U); p,_ = factorranks(U)
	W = (V,W...)
	for V ∈ W
		if factorsize(V) ≠ m
			throw(ArgumentError("the factors are incompatible in mode size"))
		end
		q,_ = factorranks(V)
		if q ≠ p
			throw(ArgumentError("the factors are incompatible in the first rank"))
		end
	end
	d = N-2
	cat(U, W...; dims=d+2)
end

"""
    factordcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number, N, M}

Concatenate the factors `U`, `V`, and additional factors in `W` along the mode dimensions and the second rank dimension.

# Arguments
- `U::Factor{T,N}`: First factor to be concatenated. Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `V::Factor{T,N}`: Second factor to be concatenated. Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `W::Vararg{Factor{T,N},M}`: Additional factors to be concatenated, each of Of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.

# Returns
- A new factor resulting from the concatenation of `U`, `V`, and the additional factors in `W` along the mode dimensions and the second rank dimension.

# Throws
- `ArgumentError`: If any of the factors have incompatible mode sizes.
"""
function factordcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number,N,M}
	m = factorsize(U)
	W = (V,W...)
	for V ∈ W
		if factorsize(V) ≠ m
			throw(ArgumentError("the factors are incompatible in mode size"))
		end
	end
	d = N-2
	cat(U, W...; dims=(1,d+2))
end

""" 
    factorutcat(U₁₁::Factor{T,N}, U₁₂::Factor{T,N}, U₂₂::Factor{T,N}) where {T<:Number, N}

Concatenate three factors `U₁₁`, `U₁₂`, and `U₂₂` in an upper triangular block matrix form.

# Arguments
- `U₁₁::Factor{T,N}`: First factor for the upper left block, with elements of type `T` and `N` dimensions.
- `U₁₂::Factor{T,N}`: Second factor for the upper right block, with elements of type `T` and `N` dimensions.
- `U₂₂::Factor{T,N}`: Third factor for the lower right block, with elements of type `T` and `N` dimensions.

# Returns
- A new factor representing the upper triangular concatenation of `U₁₁`, `U₁₂`, and `U₂₂`.

# Throws
- `ArgumentError`: If the factors are incompatible in mode size or rank.
"""
function factorutcat(U₁₁::Factor{T,N}, U₁₂::Factor{T,N}, U₂₂::Factor{T,N}) where {T<:Number,N}
	n₁₁ = factorsize(U₁₁); p₁₁,q₁₁ = factorranks(U₁₁)
	n₁₂ = factorsize(U₁₂); p₁₂,q₁₂ = factorranks(U₁₂)
	n₂₂ = factorsize(U₂₂); p₂₂,q₂₂ = factorranks(U₂₂)
	if n₁₂ ≠ n₁₁ || n₂₂ ≠ n₁₁
		throw(ArgumentError("the factors are incompatible in mode size"))
	end
	if p₁₁ ≠ p₁₂
		throw(ArgumentError("U₁₁ and U₁₂ are incompatible in the first rank"))
	end
	if q₁₂ ≠ q₂₂
		throw(ArgumentError("U₁₂ and U₂₂ are incompatible in the second rank"))
	end
	U₂₁ = zeros(T, (p₂₂,n₁₁...,q₁₁))
	d = N-2
	cat(cat(U₁₁, U₂₁; dims=1), cat(U₁₂, U₂₂; dims=1); dims=d+2)
end

"""
    factorltcat(U₁₁::Factor{T,N}, U₂₁::Factor{T,N}, U₂₂::Factor{T,N}) where {T<:Number, N}

Concatenate three factors `U₁₁`, `U₂₁`, and `U₂₂` in a lower triangular block matrix form.

# Arguments
- `U₁₁::Factor{T,N}`: First factor for the upper left block, with elements of type `T` and `N` dimensions.
- `U₂₁::Factor{T,N}`: Second factor for the lower left block, with elements of type `T` and `N` dimensions.
- `U₂₂::Factor{T,N}`: Third factor for the lower right block, with elements of type `T` and `N` dimensions.

# Returns
- A new factor representing the lower triangular concatenation of `U₁₁`, `U₂₁`, and `U₂₂`.

# Throws
- `ArgumentError`: If the factors are incompatible in mode size or rank.
"""
function factorltcat(U₁₁::Factor{T,N}, U₂₁::Factor{T,N}, U₂₂::Factor{T,N}) where {T<:Number,N}
	n₁₁ = factorsize(U₁₁); p₁₁,q₁₁ = factorranks(U₁₁)
	n₂₁ = factorsize(U₂₁); p₂₁,q₂₁ = factorranks(U₂₁)
	n₂₂ = factorsize(U₂₂); p₂₂,q₂₂ = factorranks(U₂₂)
	if n₂₁ ≠ n₁₁ || n₂₂ ≠ n₁₁
		throw(ArgumentError("the factors are incompatible in mode size"))
	end
	if p₂₁ ≠ p₂₂
		throw(ArgumentError("U₂₁ and U₂₂ are incompatible in the first rank"))
	end
	if q₁₁ ≠ q₂₁
		throw(ArgumentError("U₁₁ and U₂₁ are incompatible in the second rank"))
	end
	U₁₂ = zeros(T, (p₁₁,n₁₁...,q₂₂))
	d = N-2
	cat(cat(U₁₁, U₂₁; dims=1), cat(U₁₂, U₂₂; dims=1); dims=d+2)
end

"""
    factorranktranspose(U::Factor{T,N}) where {T<:Number, N}

Transpose the rank dimensions of the factor `U`.

# Arguments
- `U::Factor{T,N}`: Input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.

# Returns
- A new factor with the rank dimensions of `U` transposed.
"""
function factorranktranspose(U::Factor{T,N}) where {T<:Number,N}
	d = N-2
	prm = (d+2,ntuple(k -> k+1, Val(d))...,1)
	permutedims(U, prm)
end

"""
    factormodetranspose(U::Factor{T,N}, π::NTuple{K,Int}) where {T<:Number, N, K}

Permute the mode dimensions of the factor `U` according to the permutation `π`.

# Arguments
- `U::Factor{T,N}`: Input factor with elements of type `T` and `N` dimensions.
- `π::NTuple{K,Int}`: Permutation of the mode dimensions of `U`. Should contain exactly `K = N - 2` elements, which are a valid permutation of `1:d` where `d` is the number of mode dimensions.

# Returns
- A new factor resulting from permuting the mode dimensions of `U` according to `π`.

# Throws
- `ArgumentError`: If `U` has fewer than one mode dimension.
- `ArgumentError`: If `π` is not a valid permutation of the mode dimensions of `U`.
"""
function factormodetranspose(U::Factor{T,N}, π::NTuple{K,Int}) where {T<:Number,N,K}
	d = N-2
	if d == 0
		throw(ArgumentError("the factor should have at least one mode dimension"))
	end
	if K ≠ d
		throw(ArgumentError("π is not a valid permutation of the mode dimensions of U: π contains $K elements, while U has $N mode dimensions"))
	end
	if !isperm(π)
		throw(ArgumentError("π is not a valid permutation"))
	end
	prm = (1,(π.+1)...,d+2)
	permutedims(U, prm)
end

"""
    factormodetranspose(U::Factor{T,N}, π::Vector{Int}) where {T<:Number,N}

Permute the mode dimensions of the factor `U` according to the permutation `π`, where `π` is passed as a vector instead of a tuple.

Function acts as a wrapper around `factormodetranspose` to convert the vector `π` to a tuple and then call the main method.

# Arguments
- `U::Factor{T,N}`: Input factor with elements of type `T` and `N` dimensions.
- `π::Vector{Int}`: Permutation of the mode dimensions of `U`. Should contain exactly `N - 2` elements to allow for permutation of the mode dimensions.

# Returns
- New factor resulting from permuting the mode dimensions of `U` according to `π`.

# Throws
- `ArgumentError`: If `U` has fewer than one mode dimension.
- `ArgumentError`: If `π` is not a valid permutation of the mode dimensions of `U`.
"""
factormodetranspose(U::Factor{T,N}, π::Vector{Int}) where {T<:Number,N} = factormodetranspose(U, Tuple(π))

"""
    factormodetranspose(U::Factor{T,2}) where {T<:Number}

Transpose the two dimensions of the factor `U`, assuming `U` has exactly two dimensions.

Function is a specialized version of `factormodetranspose` for the case when `U` has two dimensions.
It transposes `U` by swapping the two dimensions, equivalent to the permutation `(2, 1)`.

# Arguments
- `U::Factor{T,2}`: Two-dimensional factor with elements of type `T`.

# Returns
- A new factor obtained by transposing the two dimensions of `U`.

# Throws
- `ArgumentError`: If `U` is not two-dimensional.
"""
factormodetranspose(U::Factor{T,2}) where {T<:Number} = factormodetranspose(U, (2,1))

"""
    factormodereshape(U::Factor{T,N}, n::FactorSize) where {T<:Number, N}

Reshape the mode dimensions of the factor `U` to the specified sizes in `n`.

# Arguments
- `U::Factor{T,N}`: Input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `n::FactorSize`: Vector specifying the new sizes for the mode dimensions. Product of `n` must equal the product of current mode sizes of `U`.

# Returns
- A new factor resulting from reshaping the mode dimensions of `U` to `n`.

# Throws
- `DimensionMismatch`: If the product of `n` does not equal the product of the current mode sizes of `U`.
"""
function factormodereshape(U::Factor{T,N}, n::FactorSize) where {T<:Number,N}
	d = N-2
	p,q = factorranks(U)
	if prod(n) ≠ prod(factorsize(U))
		throw(DimensionMismatch("n is inconsistent with U"))
	end
	reshape(U, p, n..., q)
end

"""
    factormodereshape(U::Factor{T,N}, n::Vector{Any}) where {T<:Number,N}

Reshape the mode dimensions of the factor `U` to the specified sizes in `n`, which defaults to an empty vector if `n` is of type `Vector{Any}`.

Function acts as a specialized version of [`factormodereshape`](@ref) where the input vector `n` is replaced with an empty `Vector{Int}()`. 

# Arguments
- `U::Factor{T,N}`: Input factor of type `Factor` with elements of type `T` and `N` dimensions.
- `n::Vector{Any}`: Vector specifying the new sizes for the mode dimensions.

# Returns
- A reshaped factor where the mode dimensions are modified, which defaults to use an empty vector of integers for mode reshaping.

# Throws
- `DimensionMismatch`: If the product of `n` does not equal the product of the current mode sizes of `U`.
"""
factormodereshape(U::Factor{T,N}, n::Vector{Any}) where {T<:Number,N} = factormodereshape(U, Vector{Int}())

""" 
    factordiagm(U::Factor{T,N}) where {T<:Number, N}

Create a tensor with diagonal properties from the factor `U` by placing the elements of `U` along the diagonal of a larger-dimensional tensor.

# Arguments
- `U::Factor{T,N}`: Input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.

# Returns
- A new factor that represents a tensor with diagonal properties constructed from `U`, dimensions expanded.

# Throws
- `ArgumentError`: If `U` has fewer than one mode dimension.
"""
function factordiagm(U::Factor{T,N}) where {T<:Number,N}
	d = N-2
	if d == 0
		throw(ArgumentError("the factor should have at least one mode dimension"))
	end
	p,q = factorranks(U)
	n = factorsize(U)
	U = reshape(U, p, prod(n), q)
	V = zeros(T, p, prod(n), prod(n), q)
	for β ∈ 1:q, i ∈ 1:prod(n), α ∈ 1:p
		V[α,i,i,β] = U[α,i,β]
	end
	reshape(V, p, n..., n..., q)
end

"""
    factorcontract(U::Factor{T,N}, V::Factor{T,N}; rev::Bool=false, major::String="last") where {T<:Number, N}

Contract two factors `U` and `V` along their shared rank dimensions. Function supports contracting in different orientations and orders based on parameters `rev` and `major`.

# Arguments
- `U::Factor{T,N}`: First input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `V::Factor{T,N}`: Second input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `rev::Bool=false`: If `true`, reverse the roles of `U` and `V` in the contraction.
- `major::String="last"`: Specifies major contraction order. Must be either `"last"` (default) or `"first"`.

# Returns
- A new factor resulting from contracting `U` and `V` along their shared rank dimensions.

# Throws
- `ArgumentError`: If `U` and `V` have inconsistent ranks
- `ArgumentError`: If `major` is not `"last"` or `"first"`.
"""
function factorcontract(U::Factor{T,N}, V::Factor{T,N}; rev::Bool=false, major::String="last") where {T<:Number,N}
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	rev && ((U,V) = (V,U))
	m = factorsize(U); p,r = factorranks(U)
	n = factorsize(V); s,q = factorranks(V)
	if r ≠ s
		throw(ArgumentError("U and V have inconsistent ranks"))
	end
	d = length(m)
	if d == 0
		rev && return V*U
		return U*V
	end
	U = reshape(U,(p*prod(m),r)); V = reshape(V,(r,prod(n)*q))
	W = U*V; W = reshape(W, (p,m...,n...,q))
	prm = collect(1:2d); prm = reshape(prm, (d,2))
	(major == "first") && (prm = reverse(prm; dims=2))
	prm = prm'; prm = prm[:]
	W = permutedims(W, [1,(prm.+1)...,2*d+2])
	k = [m..., n...]; k = k[prm]; k = reshape(k, (2,d)); k = prod(k; dims=1)
	reshape(W, (p,k...,q))
end

"""
    factorcontract(U::Factor{T,N}, V::S) where {T<:Number, N, S<:AbstractMatrix{T}}

Contract a factor `U` with a matrix `V`, aligning the rank dimensions and performing matrix multiplication.

# Arguments
- `U::Factor{T,N}`: Input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.
- `V::S`: Matrix of type `S` and elements of type `T` to contract with `U`.

# Returns
- A new factor resulting from the contraction of `U` with `V`.

# Throws
- `ArgumentError`: If `U` and `V` have inconsistent ranks.
"""
function factorcontract(U::Factor{T,N}, V::S) where {T<:Number,N,S<:AbstractMatrix{T}}
	n = factorsize(U)
	p,r = factorranks(U)
	s,q = size(V)
	if r ≠ s
		throw(ArgumentError("U and V have inconsistent ranks"))
	end
	U = reshape(U, p*prod(n), r)
	W = U*V
	reshape(W, p, n..., q)
end

"""
    factorcontract(U::S, V::Factor{T,N}) where {T<:Number, N, S<:AbstractMatrix{T}}

Contracts a matrix `U` with a factor `V`, aligning the rank dimensions and performing matrix multiplication.

# Arguments
- `U::S`: Matrix of type `S` and elements of type `T` to contract with `V`.
- `V::Factor{T,N}`: Input factor of type [`Factor`](@ref) with elements of type `T` and `N` dimensions.

# Returns
- A new factor resulting from the contraction of `U` with `V`.

# Throws
- `ArgumentError`: If `U` and `V` have inconsistent ranks.
"""
function factorcontract(U::S, V::Factor{T,N}) where {T<:Number,N,S<:AbstractMatrix{T}}
	n = factorsize(V)
	p,r = size(U)
	s,q = factorranks(V)
	if r ≠ s
		throw(ArgumentError("U and V have inconsistent ranks"))
	end
	V = reshape(V, r, prod(n)*q)
	W = U*V
	reshape(W, p, n..., q)
end

"""
    factorcontract(U::S, V::R) where {T<:Number, S<:AbstractMatrix{T}, R<:AbstractMatrix{T}}

Perform a matrix multiplication (contraction) of two factors `U` and `V`, which are matrices.

# Arguments
- `U::S`: First input matrix of type `AbstractMatrix{T}` with elements of type `T`.
- `V::R`: Second input matrix of type `AbstractMatrix{T}` with elements of type `T`.

# Returns
- The result of matrix multiplication `U * V`, which contracts the two input matrices.

# Throws
- `DimensionMismatch`: If the number of columns in `U` does not match the number of rows in `V`.
"""
factorcontract(U::S, V::R) where {T<:Number,S<:AbstractMatrix{T},R<:AbstractMatrix{T}} = U*V


"""
    factormp(U₁::Factor{T,N₁}, σ₁::Indices, U₂::Factor{T,N₂}, σ₂::Indices) where {T<:Number, N₁, N₂}

Perform a mode-wise multiplication (contraction) of two factors `U₁` and `U₂` along specified modes `σ₁` and `σ₂`. Operation contracts the specified modes, while the remaining dimensions of the factors are combined.

# Arguments
- `U₁::Factor{T,N₁}`: First factor of type [`Factor`](@ref) with elements of type `T` and `N₁` dimensions.
- `σ₁::Indices`: Reference numbers specifying which modes of `U₁` to use in the contraction. Can be a vector of integers or an empty vector.
- `U₂::Factor{T,N₂}`: Second factor of type [`Factor`](@ref) with elements of type `T` and `N₂` dimensions.
- `σ₂::Indices`: Reference numbers specifying which modes of `U₂` to use in the contraction. Can be a vector of integers or an empty vector.

# Returns
- A new factor resulting from the mode-wise multiplication of `U₁` and `U₂`, with contracted modes combined and the remaining dimensions preserved.

# Throws

- `ArgumentError`: If `σ₁` is passed as a vector, but it is not of type Vector{Int} and it is not an empty vector of the type Vector{Any}.
- `ArgumentError`: If `σ₂` is passed as a vector, but it is not of type Vector{Int} and it is not an empty vector of the type Vector{Any}.
- `ArgumentError`: If the specified sets of modes of σ₁ and σ₂ are inconsistent.
- `ArgumentError`: If the set of modes of `U₁` is specified incorrectly.
- `ArgumentError`: If the set of modes of U₂ is specified incorrectly.
- `ArgumentError`: If `U₁` and `U₂` are inconsistent with respect to the specified modes.
"""
function factormp(U₁::Factor{T,N₁}, σ₁::Indices, U₂::Factor{T,N₂}, σ₂::Indices) where {T<:Number,N₁,N₂}
	n₁ = factorsize(U₁); d₁ = factorndims(U₁)
	n₂ = factorsize(U₂); d₂ = factorndims(U₂)
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
	if n₁[σ₁] ≠ n₂[σ₂]
		throw(ArgumentError("U₁ and U₂ are inconsistent with respect to the specified modes"))
	end
	τ₁ = setdiff(1:d₁, σ₁)
	τ₂ = setdiff(1:d₂, σ₂)
	(p₁,q₁) = factorranks(U₁); (p₂,q₂) = factorranks(U₂)
	U₁ = permutedims(U₁, (1,d₁+2,(τ₁.+1)...,(σ₁.+1)...))
	nτ₁ = Vector{Int}(n₁[τ₁])
	nσ₁ = Vector{Int}(n₁[σ₁])
	nτ₂ = Vector{Int}(n₂[τ₂])
	nσ₂ = Vector{Int}(n₂[σ₂])
	U₁ = reshape(U₁, p₁*q₁*prod(nτ₁), prod(nσ₁))
	U₂ = permutedims(U₂, [(σ₂.+1)...,(τ₂.+1)...,1,d₂+2])
	U₂ = reshape(U₂, prod(nσ₂), prod(nτ₂)*p₂*q₂)
	U = U₁*U₂; n = [nτ₁..., nτ₂...]; d = length(τ₁) + length(τ₂)
	if d == 0
		n = Vector{Int}()
	end
	U = reshape(U, p₁, q₁, n..., p₂, q₂)
	U = permutedims(U, (1,d+3,(3:d+2)...,2,d+4))
	reshape(U, p₁*p₂, n..., q₁*q₂)
end

# function factorkp(U::Factor{T,N}, V::Vararg{Factor{T,N},M}) where {T<:Number,N,M}
# 	d = factorndims(U)
# 	nf = 1+length(V)
# 	n = Matrix{Int}(undef, d, nf)
# 	p = Vector{Int}(undef, nf)
# 	q = Vector{Int}(undef, nf)
# 	n[:,1] = factorsize(U); p[1],q[1] = factorranks(U)
# 	for k ∈ 2:nf
# 		n[:,k] = factorsize(V[k-1]); p[k],q[k] = factorranks(V[k-1])
# 		U = factormp(U, [], V[k-1], [])
# 	end
# 	prm = collect(1:nf*d); prm = reshape(prm, d, nf)
# 	prm = prm'; prm = prm[:]
# 	U = permutedims(U, [1,(prm.+1)...,nf*d+2])
# 	p,q = prod(p),prod(q); n = prod(n; dims=2)
# 	reshape(U, p, n..., q)
# end

# function factorkp2(U::Factor{T,N}, V::Factor{T,N}) where {T<:Number,N}
# 	d = factorndims(U)
# 	n = factorsize(U).*factorsize(V)
# 	p,q = factorranks(U).*factorranks(V)
# 	W = factormp(U, [], V, [])
# 	prm = collect(1:2*d); prm = reshape(prm, d, 2)
# 	prm = prm'; prm = prm[:]
# 	W = permutedims(W, [1,(prm.+1)...,2*d+2])
# 	reshape(W, p, n..., q)
# end

"""
    factorkp(U::Union{Factor{T,N}, Pair{Factor{T,N},Int}}, V::Vararg{Union{Factor{T,N}, Pair{Factor{T,N},Int}},M}) where {T<:Number,N,M}

Perform a Kronecker product of multiple factors (optionally raised to specified nonnegative integer exponents).

# Arguments
- `U::Union{Factor{T, N}, Pair{Factor{T, N}, Int}}`: first factor can either be a [`Factor`](@ref) type or a pair `(Factor, Int)`. If given as a pair, the integer is the exponent for the respective factor in the Kronecker product.
- `V::Vararg{Union{Factor{T, N}, Pair{Factor{T, N}, Int}}, M}`: variable number (denoted by M) of additional factors, each of which can also be either a `Factor` type or a pair `(Factor, Int)`. The same usage for the integer applies as in the above line.

# Returns
- `W`: resulting tensor (or matrix if d = 0) after the Kronecker products of all provided factors (with optionally some factors exponentiated). Final tensor is a result of a series of multiplications and reshaping operations.

# Throws
- `ArgumentError`: If a negative exponent is provided in a pair `(Factor, Int)`.
"""
function factorkp(U::Union{Factor{T,N},Pair{Factor{T,N},Int}}, V::Vararg{Union{Factor{T,N},Pair{Factor{T,N},Int}},M}) where {T<:Number,N,M}
	V = (U,V...)
	nf = length(V)
	U = Vector{Factor{T,N}}(undef, nf)
	s = Vector{Int}(undef, nf)
	for k ∈ 1:nf
		W = V[k]
		if isa(W, Pair)
			if W[2] < 0
				throw(ArgumentError("all the specified exponents should be nonnegative"))
			end
			U[k] = W[1]
			s[k] = W[2]
		else
			U[k] = W
			s[k] = 1
		end
	end
	m = findfirst(s .> 0)
	d = factorndims(U[m])
	W = U[m]
	n = factorsize(U[m])
	p,q = factorranks(U[m])
	for i ∈ 2:s[m]
		W = factormp(W, [], U[m], [])
		n = n.*factorsize(U[m])
		p,q = (p,q).*factorranks(U[m])
	end
	for k ∈ m+1:nf
		for i ∈ 1:s[k]
			W = factormp(W, [], U[k], [])
			n = n.*factorsize(U[k])
			p,q = (p,q).*factorranks(U[k])
		end
	end
	nf = sum(s)
	prm = collect(1:nf*d); prm = reshape(prm, d, nf)
	prm = prm'; prm = prm[:]
	if d > 0
		W = permutedims(W, [1,(prm.+1)...,nf*d+2])
		W = reshape(W, p, n..., q)
	else
		W = reshape(W, p, q)
	end
	W
end

"""
    factorhp(U::Factor{T,N}, V::Factor{T,N}) where {T<:Number, N}

Compute the higher-order tensor product of two factors `U` and `V`. Resulting factor has dimensions formed by multiplying the ranks and mode sizes of `U` and `V`.

# Arguments
- `U::Factor{T,N}`: First factor of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` dimensions.
- `V::Factor{T,N}`: Second factor of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` dimensions.

# Returns
- A reshaped tensor representing the higher-order product of `U` and `V`, with dimensions `(p * r, n..., q * s)` where:
  - `(p, q)` are the ranks of `U`.
  - `(r, s)` are the ranks of `V`.
  - `n` represents the mode sizes of `U` and `V`.

# Throws
- `ArgumentError`: If `U` and `V` do not have the same mode size.
"""
function factorhp(U::Factor{T,N}, V::Factor{T,N}) where {T<:Number,N}
	m = factorsize(U); (p,q) = factorranks(U)
	n = factorsize(V); (r,s) = factorranks(V)
	if m ≠ n
		throw(ArgumentError("U and V should have the same mode size"))
	end
	m = prod(n); U = reshape(U, (p,m,q)); V = reshape(V, (r,m,s))
	W = Array{T,5}(undef, p, r, m, q, s)
	for i ∈ 1:m
		W[:,:,i,:,:] = reshape(kron(V[:,i,:], U[:,i,:]), (p,r,q,s))
	end
	reshape(W, p*r, n..., q*s)
end

"""
    factorhp(U₁::Factor{T,N₁}, σ₁::Indices, U₂::Factor{T,N₂}, σ₂::Indices) where {T<:Number, N₁, N₂}

Compute the higher-order tensor product of two factors `U₁` and `U₂` along specified modes `σ₁` and `σ₂`. Function allows for a flexible contraction over selected modes.

# Arguments
- `U₁::Factor{T,N₁}`: First factor of type [`Factor`](@ref) with elements of type `T` and `N₁` dimensions.
- `σ₁::Indices`: Reference numbers specifying which modes of `U₁` to use in the contraction. Can be a vector of integers or an empty vector.
- `U₂::Factor{T,N₂}`: Second factor of type [`Factor`](@ref) with elements of type `T` and `N₂` dimensions.
- `σ₂::Indices`: Reference numbers specifying which modes of `U₂` to use in the contraction. Can be a vector of integers or an empty vector.

# Returns
- A reshaped tensor representing the higher-order product of `U₁` and `U₂`, contracted over the specified modes.

# Throws

Summarized Error list:
- `ArgumentError`: If `σ₁` or `σ₂` are incorrectly specified or inconsistent with the dimensions of `U₁` or `U₂`.
- `ArgumentError`: If `U₁` and `U₂` are inconsistent with respect to the specified modes.

Extended Error list:
- `ArgumentError`: If `σ₁` is passed as a vector, but it is not of type Vector{Int} and it is not an empty vector of the type Vector{Any}.
- `ArgumentError`: If `σ₂` is passed as a vector, but it is not of type Vector{Int} and it is not an empty vector of the type Vector{Any}.
- `ArgumentError`: If the specified sets of modes of `σ₁` and `σ₂` are inconsistent.
- `ArgumentError`: If the set of modes of `U₁` is specified incorrectly.
- `ArgumentError`: If the set of modes of `U₂` is specified incorrectly.
- `ArgumentError`: If `U₁` and `U₂` are inconsistent with respect to the specified modes.
"""
function factorhp(U₁::Factor{T,N₁}, σ₁::Indices, U₂::Factor{T,N₂}, σ₂::Indices) where {T<:Number,N₁,N₂}
	n₁ = factorsize(U₁); d₁ = factorndims(U₁)
	n₂ = factorsize(U₂); d₂ = factorndims(U₂)
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
	nσ₁ = n₁[σ₁]
	nσ₂ = n₂[σ₂]
	if nσ₁ ≠ nσ₂
		throw(ArgumentError("U₁ and U₂ are inconsistent with respect to the specified modes"))
	end
	m = prod(nσ₁)
	τ₁ = setdiff(1:d₁, σ₁); nτ₁ = n₁[τ₁]; m₁ = prod(nτ₁)
	τ₂ = setdiff(1:d₂, σ₂); nτ₂ = n₂[τ₂]; m₂ = prod(nτ₂)
	(p₁,q₁) = factorranks(U₁); (p₂,q₂) = factorranks(U₂)
	U₁ = permutedims(U₁, (1,(τ₁.+1)...,d₁+2,(σ₁.+1)...))
	U₁ = reshape(U₁, p₁*m₁*q₁, m)
	U₂ = permutedims(U₂, (1,(τ₂.+1)...,d₂+2,(σ₂.+1)...))
	U₂ = reshape(U₂, p₂*m₂*q₂, m)
	U = Array{T,3}(undef, p₁*m₁*q₁, p₂*m₂*q₂, m)
	@views for i ∈ 1:m
		U[:,:,i] .= U₁[:,i]*transpose(U₂[:,i])
	end
	U = reshape(U, p₁, m₁, q₁, p₂, m₂, q₂, m)
	U = permutedims(U, (1,4,2,7,5,3,6))
	U = reshape(U, p₁*p₂, nτ₁..., nσ₁..., m₂*q₁*q₂)
	U = permutedims(U, (1,(invperm((τ₁...,σ₁...)).+1)...,d₁+2))
	reshape(U, p₁*p₂, n₁..., nτ₂..., q₁*q₂)
end

"""
    factorproject!(V::Factor{T,N}, U::Factor{T,N}, W::Factor{T,N}; rev::Bool=false) where {T<:FloatRC, N}

Project the factor `U` onto the subspace spanned by the factor `W`, while storing the projection in `V`. The function modifies `U` and `V` based on the projection.

# Arguments
- `V::Factor{T,N}`: Resulting factor (of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` dimensions) where the projection will be stored. Should have mode size `1,...,1` and ranks consistent with `U` and `W`.
- `U::Factor{T,N}`: Factor (of aforementioned type) to be projected. Must have the same mode size as `W`.
- `W::Factor{T,N}`: Factor (also of aforementioned type) that defines the subspace for projection. Must have the same mode size as `U`.
- `rev::Bool=false`: Keyword argument that determines the orientation of the projection operation.
  - `false` (default): Project `U` onto the subspace defined by `W` with respect to the first rank.
  - `true`: Project `U` onto the subspace defined by `W` with respect to the second rank.

# Returns
- `V`: Factor `V` after storing the projection result. It is being reshaped accordingly.

# Throws
- `DimensionMismatch`: If `U` and `W` differ in mode size.
- `DimensionMismatch`: If `V` does not have mode size `1,...,1`.
- `DimensionMismatch`: If `U` and `W` have incompatible ranks for the specified `rev`.
- `DimensionMismatch`: If `V` has ranks inconsistent with `U` and `W`.
"""
function factorproject!(V::Factor{T,N}, U::Factor{T,N}, W::Factor{T,N}; rev::Bool=false) where {T<:FloatRC,N}
	n = factorsize(U)
	m = prod(n)
	if factorsize(W) ≠ n
		throw(DimensionMismatch("U and W differ in mode size"))
	end
	if any(factorsize(V) .≠ 1) 
		throw(DimensionMismatch("V should have mode size 1,…,1"))
	end
	p,q = factorranks(U)
	r,s = factorranks(W)
	if rev
		if q ≠ s
			throw(DimensionMismatch("U and W differ in the second rank"))
		end
		if factorranks(V) ≠ (p,r)
			throw(DimensionMismatch("V is inconsistant with U and W in rank"))
		end
		V = reshape(V, p, r)
		U = reshape(U, p, m*q)
		W = reshape(W, r, m*s)
		mul!(V, U, adjoint(W))
		U .-= V*W
		V = reshape(V, p, ones(Int, length(n))..., r)
	else
		if p ≠ r
			throw(DimensionMismatch("U and W differ in the first rank"))
		end
		if factorranks(V) ≠ (s,q)
			throw(DimensionMismatch("V is inconsistant with U and W in rank"))
		end
		V = reshape(V, s, q)
		U = reshape(U, p*m, q)
		W = reshape(W, r*m, s)
		mul!(V, adjoint(W), U)
		U .-= W*V
		V = reshape(V, s, ones(Int, length(n))..., q)
	end
	V
end

"""
    factorqr!(U::Factor{T,N}; rev::Bool=false, factf=(rev ? A -> LinearAlgebra.lq!(A) : A -> LinearAlgebra.qr!(A))) where {T<:FloatRC, N}

Perform a QR or LQ factorization of the tensor `U`, depending on the value of the keyword argument `rev` (reverse). 

# Arguments
- `U::Factor{T, N}`: mutable factor of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` as the number dimensions.
- `rev::Bool=false`: keyword argument that determines the type of factorization. If `false`, performs a QR factorization; if `true`, performs an LQ factorization.
- `factf`: keyword argument that specifies the chosen factorization function. By default, `LinearAlgebra.qr!` and `LinearAlgebra.lq!` are used respectively (depending on `rev`).

# Returns
- tuple `(U, R)`, where:
  - `U`: transformed tensor after applying the QR or LQ factorization. (N-dimensional)
  - `R`: factor tensor obtained by reshaping the factor matrix of the QR or LQ factorization (N-dimensional)
"""
function factorqr!(U::Factor{T,N}; rev::Bool=false, factf=(rev ? A -> LinearAlgebra.lq!(A) : A -> LinearAlgebra.qr!(A))) where {T<:FloatRC,N}
	n = factorsize(U); p,q = factorranks(U); m = ones(Int, length(n))
	if rev
		if p == 0 || q == 0
			R,U = zeros(T, p, m..., 0),zeros(T, 0, n..., q)
		else
			U = reshape(U, p, prod(n)*q)
			R,U = factf(U); R,U = Matrix(R),Matrix(U)
			ρ = size(R, 2)
			R,U = reshape(R, p, m..., ρ),reshape(U, ρ, n..., q)
		end
	else
		if p == 0 || q == 0
			U,R = zeros(T, p, n..., 0),zeros(T, 0, m..., q)
		else
			U = reshape(U, p*prod(n), q)
			U,R = factf(U); U,R = Matrix(U),Matrix(R)
			ρ = size(R, 1)
			U,R = reshape(U, p, n..., ρ),reshape(R, ρ, m..., q)
		end
	end
	U,R
end

"""
    factorqr!(U::Factor{T,N}, ::Val{false}; rev::Bool=false) where {T<:FloatRC{<:AbstractFloat},N}

Perform an in-place QR or LQ factorization of the factor `U`, depending on the value of the keyword argument `rev` (reverse). This variant uses the `Val{false}` signature for the default QR factorization.

# Arguments
- `U::Factor{T,N}`: Mutable factor of type `Factor` with elements of type `T` (a subtype of `FloatRC`) and `N` dimensions.
- `rev::Bool=false`: Keyword argument that specifies whether to perform an LQ factorization (`true`) or a QR factorization (`false`, default).

# Returns
- A tuple `(U, R)` where:
  - `U`: Transformed tensor after applying the QR or LQ factorization.
  - `R`: Factor tensor obtained by reshaping the factor matrix of the factorization.
"""
factorqr!(U::Factor{T,N}, ::Val{false}; rev::Bool=false) where {T<:FloatRC{<:AbstractFloat},N} = factorqr!(U;  rev=rev)

"""
    factorqr!(U::Factor{T,N}, ::Val{true}; rev::Bool=false, returnS::Bool=false,
              factf=(A -> LinearAlgebra.qr!(A, LinearAlgebra.ColumnNorm()))) where {T<:FloatRC, N}

Perform a QR factorization of the factor `U` with optional pivoting and the ability to return an additional factor `S`. The function can cope with different orientations (via `rev`) and supports custom factorization functions.

# Arguments
- `U::Factor{T,N}`: Factor to be decomposed, of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` dimensions.

- `::Val{true}`: Value type that indicates that this version of `factorqr!` should be used when `returnS` is needed.

- `rev::Bool=false`: Keyword argument that determines the orientation of the QR factorization.
  - `false` (default): Standard QR factorization is performed.
  - `true`: Roles of rows and columns are reversed, affecting the orientation of the factorization.

- `returnS::Bool=false`: Keyword argument that specifies whether to return an additional factor `S` such that `A ⨝ S = Q` if `rev == false` and `S ⨝ A = Q` if `rev == true`.

- `factf`: Function used to perform the QR factorization, default is set to `LinearAlgebra.qr!` with column pivoting.

# Returns
- `Q, R`: Factors `Q` (orthogonal) and `R` (upper triangular) from the QR decomposition.
- `Q, R, S`: If `returnS == true`, returns an additional factor `S` along with `Q` and `R`.
"""
function factorqr!(U::Factor{T,N}, ::Val{true}; rev::Bool=false, returnS::Bool=false,
	               factf=(A -> LinearAlgebra.qr!(A, LinearAlgebra.ColumnNorm()))) where {T<:FloatRC,N}
	# when returnS==true, a factor S satisfying A ⨝ S = Q if rev==false and S ⨝ A = Q if rev==true is returned
	n = factorsize(U); p,q = factorranks(U); m = ones(Int, length(n))
	if rev
		if p == 0 || q == 0
			R,Q = zeros(T, p, m..., 0),zeros(T, 0, n..., q)
			if returnS
				S = zeros(T, 0, m..., p)
			end
		else
			U = reshape(U, p, prod(n)*q)
			U = permutedims(U) # reallocation
			fact = factf(U)
			π = invperm(fact.p)
			R = permutedims(fact.R[:,π])
			ρ = size(R, 2)
			R = reshape(R, p, m..., ρ)
			Q = permutedims(fact.Q*Matrix{T}(I, ρ, ρ))
			Q = reshape(Q, ρ, n..., q)
			if returnS
				S = inv(fact.R[:,1:ρ])
				(ρ < p) && (S = [S; zeros(T, p-ρ, ρ)])
				S = permutedims(S[π,:])
				S = reshape(S, ρ, m..., p)
			end
		end
	else
		if p == 0 || q == 0
			Q,R = zeros(T, p, n..., 0),zeros(T, 0, m..., q)
			if returnS
				S = zeros(T, q, m..., 0)
			end
		else
			U = reshape(U, p*prod(n), q)
			fact = factf(U)
			π = invperm(fact.p)
			R = fact.R[:,π]
			ρ = size(R, 1)
			R = reshape(R, ρ, m..., q)
			Q = fact.Q*Matrix{T}(I, ρ, ρ)
			Q = reshape(Q, p, n..., ρ)
			if returnS
				S = inv(fact.R[:,1:ρ])
				(ρ < q) && (S = [S; zeros(T, q-ρ, ρ)])
				S = S[π,:]
				S = reshape(S, q, m..., ρ)
			end
		end
	end
	if returnS
		return Q,R,S
	end
	Q,R
end

"""
    factorqradd(Q::Factor{T,N}, R::Union{Factor{T,N},Nothing}, U::Factor{T,N}; rev::Bool=false) where {T<:FloatRC, N}

Add rows or columns to an orthogonal factor `Q` and its corresponding upper triangular factor `R`, depending on the value of `rev`. This function is an extension to the QR factorization by updating the factors to incorporate a new matrix `U`.

# Arguments
- `Q::Factor{T,N}`: Orthogonal factor of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and `N` dimensions.
  - If `rev == true`, `Q` is assumed to be orthogonal with respect to the first rank.
  - If `rev == false`, `Q` is assumed to be orthogonal with respect to the second rank.

- `R::Union{Factor{T,N},Nothing}`: Upper triangular factor of the QR factorization. Can be of type [`Factor`](@ref) or `Nothing`.
  - If `R` is `Nothing`, an identity matrix of appropriate dimensions will be used.

- `U::Factor{T,N}`: New matrix to be added to the existing factors (same mode dimensions as `Q`).

- `rev::Bool=false`: Keyword argument indicating the direction of orthogonality for the operation. 
  - `false` (default): Assumes `Q` to be orthogonal with respect to the second rank, and columns will be added.
  - `true`: Assumes `Q` to be orthogonal with respect to the first rank, and rows will be added.

# Returns
- `Q`: Updated orthogonal factor after incorporating `U`.
- `R`: Updated upper triangular factor after incorporating `U`.

# Throws

Summarized Error list:
- `ArgumentError`: If the mode sizes of `Q` and `U` are incompatible.
- `ArgumentError`: If `R` has incompatible rank or mode dimensions with `Q`.
- `ArgumentError`: If `Q` and `U` are incompatible in their respective ranks.
- `ArgumentError`: If `R` is specified and has an incompatible number of mode dimensions or rank.

Extended Error List:
- `ArgumentError`: If the mode sizes of `Q` and `U` are incompatible.
- `ArgumentError`: If `R` and `Q` have incompatible number of mode dimensions or incompatible number of unitary mode sizes.
- `ArgumentError`: If `R` and `Q` have incompatible rank.
- `ArgumentError`: If `Q` and `U` have incompatible second rank.
- `ArgumentError`: If `R`  has a different number of mode dimensions as Q or different unitary mode sizes.
- `ArgumentError`: If `Q` and `R` have incompatible rank.
- `ArgumentError`: If `Q` and `U` have incompatible first rank.
"""
function factorqradd(Q::Factor{T,N}, R::Union{Factor{T,N},Nothing}, U::Factor{T,N}; rev::Bool=false) where {T<:FloatRC,N}
	# assumes that Q is orthogonal w.r.t the first rank if rev==true and w.r.t the second rank if rev==false
	n = factorsize(Q); m = ones(Int, length(n))
	if factorsize(U) ≠ n
		throw(ArgumentError("Q and U are incompatible in mode size"))
	end
	if rev
		r,q = factorranks(Q)
		s,qq = factorranks(U)
		if isa(R, Nothing)
			R = reshape(Matrix{T}(I, r, r), r, m..., r)
			p = r
		else
			if factorsize(R) ≠ m
				throw(ArgumentError("R should have the same number of mode dimensions as Q and unitary mode sizes"))
			end
			p,rr = factorranks(R)
			if rr ≠ r
				throw(ArgumentError("R and Q are incompatible in rank"))
			end
		end
		if qq ≠ q
			throw(ArgumentError("Q and U are incompatible in the second rank"))
		end
		R = reshape(R, p, r)
		Q = reshape(Q, r, prod(n)*q)
		U = reshape(U, s, prod(n)*q)
		R,Q = lqaddrows(R, Q, U); r = size(R, 2)
		R,Q = reshape(R, p+s, m..., r),reshape(Q, r, n..., q)
	else
		p,r = factorranks(Q)
		pp,s = factorranks(U)
		if isa(R, Nothing)
			R = reshape(Matrix{T}(I, r, r), r, m..., r)
			q = r
		else
			if factorsize(R) ≠ m
				throw(ArgumentError("R should have the same number of mode dimensions as Q and unitary mode sizes"))
			end
			rr,q = factorranks(R)
			if rr ≠ r
				throw(ArgumentError("Q and R are incompatible in rank"))
			end
		end
		if pp ≠ p
			throw(ArgumentError("Q and U are incompatible in the first rank"))
		end
		R = reshape(R, r, q)
		Q = reshape(Q, p*prod(n), r)
		U = reshape(U, p*prod(n), s)
		Q,R = qraddcols(Q, R, U); r = size(R, 1)
		Q,R = reshape(Q, p, n..., r),reshape(R, r, m..., q+s)
	end
	Q,R
end

"""
    factorsvd!(W, m, n; atol=0, rtol=0, rank=0, major="last", rev=false)

produces U and V such that W ≈ U ⋈ V if rev == false and W ≈ V ⋈ U if rev == true
U is orthogonal — with respect to the second or first rank index
                  if rev == false or rev == true respectively
m and n are the mode-size vectors of U and V respectively (at most one of these two arguments may be replaced by ":")
major determines whether the "first" or "last" factor in the product U ⋈ V carries the major bits
rank=0 leads to no rank thresholding
"""

"""
    function factorsvd!(W::Factor{T,N},
                    m::Union{FactorSize,Colon},
                    n::Union{FactorSize,Colon};
					soft::S=zero(S),
					hard::S=zero(S),
                    atol::S=zero(S),
                    rtol::S=zero(S),
                    rank::Int=0,
                    major::String="last",
					rev::Bool=false,
					factf=(A -> LinearAlgebra.svd!(A; full=false, alg=LinearAlgebra.QRIteration())) ) where {S<:AbstractFloat,T<:FloatRC{S},N}

Perform a singular value decomposition (SVD) of the factor `W` where the factor dimensions are adjusted accordingly and optional thresholding is carried out based on specified parameters. 

# Arguments
- `W::Factor{T, N}`: Input factor of type [`Factor`](@ref) with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and `N` dimensions.
- `m::Union{FactorSize, Colon}`: Mode-size parameter for first dimension. Can be a [`FactorSize`](@ref) vector specifying the sizes of the mode dimensions or `Colon` to represent all indices.
- `n::Union{FactorSize, Colon}`: Mode-size parameter for second dimension. Can also be a [`FactorSize`](@ref) vector or `Colon`.
- `soft::S=zero(S)`: Keyword argument specifying the soft threshold for singular value truncation. Must be nonnegative and finite.
- `hard::S=zero(S)`: Keyword argument specifying the hard threshold for singular value truncation. Must be nonnegative and finite.
- `atol::S=zero(S)`: Absolute tolerance for singular values. Must be nonnegative and finite.
- `rtol::S=zero(S)`: Relative tolerance for singular values. Must be nonnegative and finite.
- `rank::Int=0`: Maximum allowable rank for the truncated SVD. Must be nonnegative.
- `major::String="last"`: Specifies the major contraction order. Must be either `"last"` (default) or `"first"`.
- `rev::Bool=false`: Boolean indicating whether to reverse the dimensions in the output.
- `factf`: Function used to perform the SVD. Default sets it to a function using `LinearAlgebra.svd!` with `QRIteration`.

# Returns
- `U`: The left singular vectors after applying SVD, reshaped based on the specified parameters (and potentially truncated).
- `V`: The right singular vectors after applying SVD, reshaped similarly (and also potentially truncated).
- `ε`: The effective noise level that stems from thresholding.
- `δ`: The relative noise level, calculated as `ε / μ`.
- `μ`: The norm of the input factor `W`.
- `ρ`: The rank of the truncated SVD.
- `σ`: A vector of the singular values (after thresholding).

# Throws

Summarized Error List:
- `ArgumentError`: If both `m` and `n` are specified as `Colon`, this causes ambiguity.
- `ArgumentError`: If either `m` or `n` is specified with non-positive or inconsistent dimensions.
- `ArgumentError`: If `major` is not `"first"` or `"last"`.
- `ArgumentError`: If `soft`, `hard`, `atol`, or `rtol` are negative or not finite.
- `ArgumentError`: If `rank` is negative.
- `DimensionMismatch`: If mode-size vectors `m` and `n` are inconsistent with dimensions of `W`.
- `ErrorException`: If numerical issues such as overflow or underflow occur during computations.

Full Error List:
- `ArgumentError`: If both `m` and `n` are specified as `Colon`, this causes ambiguity.
- `ArgumentError`: If the elements of the first mode-size vector are not positive.
- `ArgumentError`: If the elements of the second mode-size vector are not positive.
- `ArgumentError`: If `major` is neither "last" (default) nor "first".
- `ArgumentError`: If `soft` is negative or infinite.
- `ArgumentError`: If `hard` is negative or finite.
- `ArgumentError`: If `atol` is negative or infinte.
- `ArgumentError`:  If `rtol` is negative or infinte.
- `ArgumentError`: If optional argument rank is negative.

- `DimensionMismatch`: If the number of entries in the first mode-size vector is inconsistent with the specified factor.
- `DimensionMismatch`: If the number of entries in the second mode-size vector is inconsistent with the specified factor.
- `DimensionMismatch`: If not every mode dimension of W is divisible by the corresponding element of the specified mode-size vector
- `DimensionMismatch`: If specified mode-size vectors are inconsistent with the specified factor.

- `ErrorException`: If squaring `soft` leads to overflow and it was passed finite.
- `ErrorException`: If squaring `soft` leads to underflow and it was passed positive.
- `ErrorException`: If sqauring `hard` leads to overflow and it was passed finite.
- `ErrorException`: If squaring `hard` leads to underflow and it was passed positive.
- `ErrorException`: If squaring `atol` leads to overflow and it was passed finite.
- `ErrorException`: If squaring `atol` leads to underflow and it was passed positive.



# Example
```julia
# Assume Factor{T, N}, FactorSize, and FloatRC are properly defined and initialized

# Initialize W as a Factor{T, N} type, for example, a decomposed tensor
W = Factor(...)  # specific initialization needed

# Specify mode-size parameters
m = FactorSize([2, 3])
n = FactorSize([2, 2])

# Perform SVD with pre-specified thresholds and tolerances
U, V, ε, δ, μ, ρ, σ = factorsvd!(W, m, n; soft=0.5, hard=0.05, atol=1e-6, rtol=1e-3, rank=10, major="last")

# Output the resulting factors and singular values
println("Left singular vectors (U): ", U)
println("Right singular vectors (V): ", V)
println("Effective noise level (ε): ", ε)
println("Relative noise level (δ): ", δ)
println("Norm of the input factor (μ): ", μ)
println("Rank of the truncated SVD (ρ): ", ρ)
println("Singular values after thresholding (σ): ", σ)
"""
function factorsvd!(W::Factor{T,N},
                    m::Union{FactorSize,Colon},
                    n::Union{FactorSize,Colon};
					soft::S=zero(S),
					hard::S=zero(S),
                    atol::S=zero(S),
                    rtol::S=zero(S),
                    rank::Int=0,
                    major::String="last",
					rev::Bool=false,
					factf=(A -> LinearAlgebra.svd!(A; full=false, alg=LinearAlgebra.QRIteration())) ) where {S<:AbstractFloat,T<:FloatRC{S},N}
	d = factorndims(W)
	k = factorsize(W)
	if isa(m, Colon) && isa(n, Colon)
		throw(ArgumentError("to avoid ambiguity, at lease one of the two mode-size parameters should be specified as a vector"))
	end
	if isa(m, FactorSize)
		if any(m .≤ 0)
			throw(ArgumentError("the elements of the first mode-size parameter, when it is specified, should be positive"))
		end
		if length(m) ≠ d
			throw(DimensionMismatch("the number of entries in the first mode-size vector is inconsistent with the specified factor"))
		end
	else
		m = k.÷n
		if k ≠ m.*n
			throw(DimensionMismatch("not every mode dimension of W is divisible by the corresponding element of the specified mode-size vector"))
		end
	end
	if isa(n, FactorSize)
		if any(n .≤ 0)
			throw(ArgumentError("the elements of the second mode-size vector, when it is specified, should be positive"))
		end
		if length(n) ≠ d
			throw(DimensionMismatch("the number of entries in the second mode-size vector is inconsistent with the specified factor"))
		end
	else
		n = k.÷m
		if k ≠ m.*n
			throw(DimensionMismatch("not every mode dimension of W is divisible by the corresponding element of the specified mode-size vector"))
		end
	end
	if k ≠ m.*n
		throw(DimensionMismatch("the specified mode-size vectors are inconsistent with the specified factor"))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	#
	if soft < 0 || !isfinite(soft)
		throw(ArgumentError("soft, when specified, should be nonnegative and finite"))
	end
	#
	soft² = soft^2
	if !isfinite(soft²)
		throw(ErrorException("overflow encountered while squaring soft, which was passed finite"))
	end
	if soft > 0 && soft² == 0
		throw(ErrorException("underflow encountered while squaring soft, which was passed positive"))
	end
	#
	if hard < 0 || !isfinite(hard)
		throw(ArgumentError("hard, when specified, should be nonnegative and finite"))
	end
	#
	hard² = hard^2
	if !isfinite(hard²)
		throw(ErrorException("overflow encountered while squaring hard, which was passed finite"))
	end
	if hard > 0 && hard² == 0
		throw(ErrorException("underflow encountered while squaring hard, which was passed positive"))
	end
	#
	if atol < 0 || !isfinite(atol)
		throw(ArgumentError("atol, when specified, should be nonnegative and finite"))
	end
	#
	atol² = atol^2
	if !isfinite(atol²)
		throw(ErrorException("overflow encountered while squaring atol, which was passed finite"))
	end
	if atol > 0 && atol² == 0
		throw(ErrorException("underflow encountered while squaring atol, which was passed positive"))
	end
	#
	if rtol < 0 || !isfinite(rtol)
		throw(ArgumentError("rtol, when specified, should be nonnegative and finite"))
	end
	#
	if rank < 0
		throw(ArgumentError("the optional argument rank should be nonnegative"))
	end
	#
	p,q = factorranks(W); prm = collect(1:d)
	k = rev ? [n; m] : [m; n]
	prm = (major == "last") ? [2*prm.-1; 2*prm] : [2*prm; 2*prm.-1]
	W = reshape(W, (p,k[invperm(prm)]...,q))
	W = permutedims(W, vcat(1, prm.+1, 2d+2))
	W = reshape(W, p*prod(k[1:d]), prod(k[d+1:2d])*q)
	μ = norm(W)
	if μ == 0
		ε = zero(S); δ = zero(S); ρ = 0
		σ = Vector{S}()
		U = zeros(T, p, k[1:d]..., ρ)
		V = zeros(T, ρ, k[d+1:2d]..., q)
		rev && ((U,V) = (V,U))
	else
		fact = factf(W)
		U = fact.U
		σ = fact.S
		V = fact.Vt
		μ = norm(σ)
		if !isfinite(μ)
			throw(ErrorException("overflow encountered while computing the norm of the decomposition"))
		end
		rtol = min(rtol, one(S));
		tol = μ*rtol
		if μ > 0 && rtol > 0 && tol == 0
			throw(ErrorException("underflow encountered while computing the absolute accuracy threshold from the relative one"))
		end
		tol² = tol^2
		if μ > 0 && rtol > 0 && tol² == 0
			throw(ErrorException("underflow encountered while computing the squared absolute accuracy threshold from the squared relative one"))
		end
		tol = max(atol, tol)
		σσ,ε,ρ = Auxiliary.threshold(σ, soft, hard, tol, rank)
		δ = ε/μ
		U = U[:,1:ρ]; V = V[1:ρ,:]
		rev ? U = U*Diagonal(σσ) : V = Diagonal(σσ)*V
		U = reshape(U, p, k[1:d]..., ρ)
		V = reshape(V, ρ, k[d+1:2d]..., q)
		rev && ((U,V) = (V,U))
	end
	U,V,ε,δ,μ,ρ,σ
end