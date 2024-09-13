export DecSize, DecRank, Dec, VectorDec, MatrixDec
export checkndims, checklength, checkndims, checksize, checkrank, checkranks
export declength, decndims, decsize, decranks, decrank
export dec, dec!, vector, decrankselect!, decrankselect, factor!, factor, block!, block, decvcat, dechcat, decdcat
export decscale!, decreverse!, decmodetranspose!
export decmodereshape
export decfill!, decrand!
export deczeros, decones, decrand
export decappend!, decprepend!, decpush!, decpushfirst!, decpop!, decpopfirst!, decinsert!, decdeleteat!
export decinsertidentity!
export decskp!, decskp, decmp, deckp, decaxpby!, decadd!, decaxpby, decadd, dechp
export decqr!, decsvd!


"""
`DecSize` is an alias for a Matrix with integer entries of type `Matrix{Int}`.
"""
const DecSize = Matrix{Int}
"""
`DecRank` is an alias for a Vector with integer entries of type `Vector{Int}`.
"""
const DecRank = Vector{Int}

"""
`Dec` is an alias for a `Vector{Factor{T,N}}`.
"""
const Dec{T,N} = Vector{Factor{T,N}} where {T<:Number,N}

"""
`VectorDec{T}` is an alias for Vector{VectorFactor{T}}`.
"""
const VectorDec{T} = Vector{VectorFactor{T}} where T<:Number

"""
`MatrixDec{T}` is an alias for Vector{MatrixFactor{T}}`.
"""
const MatrixDec{T} = Vector{MatrixFactor{T}} where T<:Number

"""
    function checkndims(d::Int)

Check the correctness of the numbers of dimensions.

# Arguments
- `d::Int`: Input dimension.

# Returns
- No Return. Function merely flags negative dimensions.

# Throws
- `ArgumentError`: If `d` is negative.
"""
function checkndims(d::Int)
	if d < 0
		throw(ArgumentError("d should be nonnegative"))
	end
end

"""
    function checklength(L::Int)

Check the correctness of the length L.

# Arguments
- `L::Int`: Input length.

# Returns
- No Return. Function merely flags negative length.

# Throws
- `ArgumentError`: If `L` is negative.
"""
function checklength(L::Int)
	if L < 0
		throw(ArgumentError("the number of factors should be nonnegative"))
	end
end

"""
    function checkndims(d::Vector{Int})

Check whether a given vector exhibits uniform values.

# Arguments
- `d::Vector{Int}`: Input vector of integers.

# Returns
- No Return. Function merely flags incorrent vectors.

# Throws
- `ArgumentError`: If `d` does not contain elements.
- `BoundsError`: If the elements of `d` are negative or zero.
- `ArgumentError`: If the values in the dimension vector differ.
"""
function checkndims(d::Vector{Int})
	L = length(d)
	if L == 0
		throw(ArgumentError("d is empty"))
	end
	if any(d .≤ 0)
		throw(BoundsError("the elements of d should be positive"))
	end
	if length(unique(d)) ≠ 1
		throw(ArgumentError("the values in the dimension vector are not identical"))
	end
end

"""
    function checksize(n::DecSize; len::Int=0, dim::Int=0)

Check the correctness of a given size matrix with integer entries (Type: Decsize).

# Arguments
- `n::DecSize`: Input size matrix with integer entries.
- `len::Int=0`: Length of the size matrix. By default, it is 0.
- `dim::Int=0`: Dimension of the size matrix. By default, it is 0.

# Returns
- No Return. Function merely flags incorrent size matrix.

# Throws
- `ArgumentError`: If `n` does not contain elements.
- `ArgumentError`: If `n` contains negative or zero elements.
- `ArgumentError`: If the number of factors is negative or zero.
- `ArgumentError`: If the number of mode dimensions is negative or zero.
- `ArgumentError`: If the number of rows is incorrect.
- `ArgumentError`: If the number of columns is incorrect.
"""
function checksize(n::DecSize; len::Int=0, dim::Int=0)
	if length(n) == 0
		throw(ArgumentError("the size matrix is empty"))
	end
	if any(n .≤ 0)
		throw(ArgumentError("the elements of the size matrix should be positive"))
	end
	if len < 0
		throw(ArgumentError("the number of factors should be positive"))
	end
	if dim < 0
		throw(ArgumentError("the number of mode dimensions should be positive"))
	end
	if dim > 0 && size(n, 1) ≠ dim
		throw(ArgumentError("the number of rows in n is incorrect"))
	end
	if len > 0 && size(n, 2) ≠ len
		throw(ArgumentError("the number of columns in n is incorrect"))
	end
end

"""
    checkrank(r::DecRank; len::Int=0)

Check the correctness of a given rank vector with integer entries (Type: DecRank).

# Arguments
- `r::DecRank`: Input rank vector with integer entries.
- `len::Int=0`: Length of the size matrix. By default, it is 0.

# Returns
- No Return. Function merely flags incorrent rank vectors.

# Throws
- `ArgumentError`: If `r` does not contain at least two elements.
- `ArgumentError`: If `r` contains negative elements.
- `ArgumentError`: If the number of elements in the rank vector is incorrect.
"""
function checkrank(r::DecRank; len::Int=0)
	if length(r) < 2
		throw(ArgumentError("the rank vector should contain at least two elements"))
	end
	if any(r .< 0)
		throw(ArgumentError("the elements of the rank vector should be nonnegative"))
	end
	if len > 0 && length(r) ≠ len+1
		throw(ArgumentError("the number of elements in the rank vector is incorrect"))
	end
end

"""
    checkranks(p::Vector{Int}, q::Vector{Int}; len::Int=0)

Check the correctness of two given rank vectors with integer entries.

# Arguments
- `p::Vector{Int}`: First input vector with integer entries.
- `q::Vector{Int}`: Second input vector with integer entries.
- `len::Int=0`: Length of the vectors. By default, it is 0.

# Returns
- No Return. Function merely flags incorrent rank vectors.

# Throws
- `ArgumentError`: If `p` and `q` do not have the same length.
- `ArgumentError`: If `p` or `q` contain negative elements.
- `ArgumentError`: If the number of elements in the rank vectors is incorrect.
"""
function checkranks(p::Vector{Int}, q::Vector{Int}; len::Int=0)
	if length(p) ≠ length(q)
		throw(ArgumentError("the rank vectors should have the same length"))
	end
	if any([p q] .< 0)
		throw(ArgumentError("the elements of p and q should be nonnegative"))
	end
	if p[2:end] ≠ q[1:end-1]
		throw(DimensionMismatch("the ranks are inconsistent"))
	end
	if len > 0 && length(p) ≠ len+1
		throw(ArgumentError("the number of elements in the rank vectors is incorrect"))
	end
end

"""
    function declength(U::Dec{T,N}) where {T<:Number,N}

Check the correctness of two given rank vectors with integer entries.

# Arguments
- `U::Dec{T,N}`: Decomposition of `N` Factors of type `Factor`.

# Returns
- An integer: the length of the decomposition.

# Throws
- `ArgumentError`: If `L` is negative.
"""
function declength(U::Dec{T,N}) where {T<:Number,N}
	L = length(U)
	checklength(L)
	return L
end

"""
    decndims(U::Dec{T,N}) where {T<:Number,N}

Return the number of contraction dimensions of the decomposition.

# Arguments
- `U::Dec{T,N}`: Decomposition of type [`Dec`](@ref) with elements of type `T` and with `N` as the number of factors.

# Returns
- An integer representing the number of contraction dimensions, which is `N - 2`.
"""
decndims(U::Dec{T,N}) where {T<:Number,N} = N-2

"""
    decsize(U::Dec{T,N}) where {T<:Number,N}

Return a matrix representing the mode sizes of each factor in the decomposition `U`.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- A matrix of integers, where each column represents the different mode sizes for each factor contained in the decomposition `U`.
"""
function decsize(U::Dec{T,N}) where {T<:Number,N}
	L = length(U)
	d = decndims(U)
	n = Matrix{Int}(undef, d, L)
	for ℓ ∈ 1:L, k ∈ 1:d
		n[k,ℓ] = size(U[ℓ], 1+k)
	end
	checksize(n)
	return n
end

"""
    decranks(U::Dec{T,N}) where {T<:Number,N}

Return two vectors, which represent the first and last ranks, respectively, of each factor in the decomposition type `U`.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- Two vectors of length L: `p` and `q` where `p[ℓ]` is the first rank and `q[ℓ]` is the last rank of the ℓ-th factor in `U`.
"""
function decranks(U::Dec{T,N}) where {T<:Number,N}
	L = length(U)
	d = decndims(U)
	p = Vector{Int}(undef, L)
	q = Vector{Int}(undef, L)
	for ℓ ∈ 1:L
		p[ℓ] = size(U[ℓ], 1)
		q[ℓ] = size(U[ℓ], d+2)
	end
	return p,q
end

"""
    decrank(U::Dec{T,N}) where {T<:Number,N}

Return a vector of ranks by combining all first ranks of the factors of `U` and appending the last rank of the last factor in the decomposition at the end of the vector.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- A vector containing the first rank of each factor in `U` as well as the last rank of the last factor in the decomposition.

# Throws
- `DimensionMismatch`: If the factors in `U` have inconsistent ranks.
"""
function decrank(U::Dec{T,N}) where {T<:Number,N}
	p,q = decranks(U)
	try checkranks(p,q) catch e
		isa(e, DimensionMismatch) && throw(DimensionMismatch("the factors of the decomposition have inconsistent ranks"))
	end
	return [p..., q[end]]
end

"""
    dec(::Type{T}, d::Int, L::Int) where {T<:Number}

Create an empty decomposition with `L` factors, with each factor exhibiting `d` mode dimensions and entries of data type `T`.

# Arguments
- `T::Type{T}`: Data type of the elements in each factor.
- `d::Int`: Number of mode dimensions.
- `L::Int`: Number of factors.

# Returns
- Vector of uninitialized arrays representing the decomposition.

# Throws
- `ArgumentError`: If `d` or `L` is negative.
"""
function dec(::Type{T}, d::Int, L::Int) where {T<:Number}
	checkndims(d)
	checklength(L)
	U = Vector{Array{T,d+2}}(undef, L)
	return U
end

"""
    dec(d::Int, L::Int)

Create an empty decomposition with `L` factors, with each factor exhibiting `d` mode dimensions and entries of default data type `Float64`.

# Arguments
- `d::Int`: Number of mode dimensions.
- `L::Int`: Number of factors.

# Returns
- Vector of uninitialized arrays representing the decomposition, with entries of default data type `Float64`.

# Throws
- `ArgumentError`: If `d` or `L` is negative.
"""
dec(d::Int, L::Int) = dec(Float64, d, L)

"""
    dec(::Type{T}, d::Int) where {T<:Number}

Create an empty decomposition with zero factors, each having `d` mode dimensions and data type `T`.

# Arguments
- `T::Type{T}`: Data type of the elements in each factor.
- `d::Int`: Number of mode dimensions.

# Returns
- Vector of uninitialized arrays representing an empty decomposition.

# Throws
- `ArgumentError`: If `d` is invalid (i.e., if `d` is negative).
"""
function dec(::Type{T}, d::Int) where {T<:Number}
	checkndims(d)
	return Vector{Array{T,d+2}}(undef, 0)
end

"""
    dec(d::Int)

Create an empty decomposition with zero factors, each having `d` mode dimensions and default data type `Float64`.

# Arguments
- `d::Int`: Number of mode dimensions.

# Returns
- Vector of uninitialized arrays representing an empty decomposition with default data type `Float64`.
"""
dec(d::Int) = dec(Float64, d, L)

"""
    dec(U::Dec{T,N}) where {T<:Number,N}

Return the decomposition `U` as is.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- The decomposition object `U` as is.
"""
function dec(U::Dec{T,N}) where {T<:Number,N}
	return U
end

"""
    dec(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a decomposition with specified size, rank, and further optional parameters.

# Arguments
- `T::Type{T}`: Data type of the elements in each factor.
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` was not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` was not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A vector of arrays representing the decomposition with each factor exibiting the size and rank specified in advance.

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
function dec(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}
	checksize(n[:,:])
	nmat = isa(n, DecSize)
	nlen = size(n, 2)
	rvec = isa(r, DecRank)
	rlen = length(r)
	if any(r .< 0)
		throw(ArgumentError("the rank should be a nonnegative integer or a vector of such"))
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
	U = [ Array{T}(undef, r[ℓ], n[:,ℓ]..., r[ℓ+1]) for ℓ ∈ 1:L ]
	return U
end

"""
    dec(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a decomposition with default type `Float64` using the specified size, rank, and optional parameters.

# Arguments
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` with `Float64` elements and the specified sizes, ranks, and parameters.

# Throws
- `ArgumentError`: If invalid size, rank, or other parameters are provided (see detailed error control in the [`dec`](@ref) function with type specification).
"""
dec(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = dec(Float64, n, r; first=first, last=last, len=len)

"""
    dec!(U::Factor{T,N}) where {T<:Number,N}

Convert a single factor `U` into a decomposition object containing merely that factor.

# Arguments
- `U::Factor{T,N}`: Factor of type `factor` to be turned into a decomposition.

# Returns
- Decomposition object `Dec{T,N}` containing the single factor `U`.
"""
function dec!(U::Factor{T,N}) where {T<:Number,N}
	return Dec{T,N}([U])
end

"""
    dec(U::Factor{T,N}; len::Int=1) where {T<:Number,N}

Create a decomposition type `Dec` (vector of factors) with a specified length, each factor being a copy of the factor `U`.

# Arguments
- `U::Factor{T,N}`: Factor, whose entries are in `T` and with `N` dimensions, to use for creating the decomposition type.
- `len::Int=1`: Length of the created decomposition, whose factors are copies of `U`.

# Returns
- Decomposition type whose factors consist of copies of `U`.

# Throws
- `ArgumentError`: If the number of factors `len`, when specified, is not positive.
- `ArgumentError`: If the two ranks of `U` are not equal when the number of factors `len` is specified as larger than one.
"""
function dec(U::Factor{T,N}; len::Int=1) where {T<:Number,N}
	if len < 0
		throw(ArgumentError("the number of factors, when specified, should be positive"))
	end
	if len == 0
		return dec(T, N-2)
	end
	if len > 1
		p,q = factorranks(U)
		if p ≠ q
			throw(ArgumentError("the two ranks of U should be equal when the number of factors is specified as larger than one"))
		end
	end
	return Dec{T,N}([ copy(U) for ℓ ∈ 1:len ])
end

"""
    vector(U::Dec{T,N}) where {T<:Number,N}

Return the decomposition `U` as is; since a decomposition type (vector of factors) is already a vector. 

# Arguments
- `U::Dec{T,N}`: Decomposition, whose entries are in `T` and with `N` dimensions.

# Returns
- The input decomposition object `U` as is. 
"""
function vector(U::Dec{T,N}) where {T<:Number,N}
	return U
end

"""
    decrankselect!(U::Dec{T,N}, α::Indices, β::Indices) where {T<:Number,N}

Select the first rank dimensions of the first factor and the last rank dimensions of the last factor in the decomposition `U` based on the provided indices `α` and `β`, modifying `U` in place.

# Arguments
- `U::Dec{T,N}`: Decomposition, whose entries are in `T` and with `N` dimensions, to select ranks from.
- `α::Indices`: Reference numbers for selecting the first rank dimensions of the first factor of `U`.
- `β::Indices`: Reference numbers for selecting the last rank dimensions of the last factor of `U`.

# Returns
- `U`: Decomposition type `Dec`, with the selected ranks. 

# Throws
Summarized Error list:
- `ArgumentError`: If the decomposition is empty, or if `α` or `β` contain invalid or empty ranges.

Extended Error list:
- `ArgumentError`: If the range for the first rank is empty.
- `ArgumentError`: If the range for the first rank is incorrect.
- `ArgumentError`: If the range for the second rank is empty.
- `ArgumentError`: If the range for the second rank is incorrect.
"""
function decrankselect!(U::Dec{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	# if isa(α, Int) || isa(β, Int)
	# 	throw(ArgumentError("for consistency with Base.selectdim, scalar α and β are not accepted; use α:α or β:β instead of α or β to select a subtensor of the factor whose first or second rank is one"))
	# end
	L = declength(U)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	r = decrank(U); p,q = r[1],r[L+1]
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
	U[L] = factorrankselect(U[L], :, β)
	U[1] = factorrankselect(U[1], α, :)
	return U
end

"""
    decrankselect(U::Dec{T,N}, α::Indices, β::Indices) where {T<:Number,N}

Create a new decomposition object by selecting the first rank dimensions of the first factor and the last rank dimensions of the last factor in the decomposition `U` based on the provided indices `α` and `β`.

# Arguments
- `U::Dec{T,N}`: Decomposition, whose entries are in `T` and with `N` dimensions, to select ranks from.
- `α::Indices`: Reference numbers for selecting the first rank dimensions of the first factor of `U`.
- `β::Indices`: Reference numbers for selecting the last rank dimensions of the last factor of `U`.

# Returns
- `V`: Decomposition type `Dec`, with the selected ranks. 

# Throws
Summarized Error list:
- `ArgumentError`: If the decomposition is empty, or if `α` or `β` contain invalid or empty ranges.

Extended Error list:
- `ArgumentError`: If the range for the first rank is empty.
- `ArgumentError`: If the range for the first rank is incorrect.
- `ArgumentError`: If the range for the second rank is empty.
- `ArgumentError`: If the range for the second rank is incorrect.
"""
function decrankselect(U::Dec{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	V = deepcopy(U)
	decrankselect!(V, α, β)
	return V
end

"""
    factor!(U::Dec{T,N}; major::String="last") where {T<:Number,N}

Factorize the decomposition `U` in place to a single factor (by contracting along all mode dimensions).

# Arguments
- `U::Dec{T,N}`: Decomposition, whose entries are in `T` and with `N` dimensions, to factorize.
- `major::String="last"`: Specifies the major ordering of the contraction. Can be "first" or "last".

# Returns
- Single factor object obtained by contracting all factors in the input decomposition object `U`.
"""
function factor!(U::Dec{T,N}; major::String="last") where {T<:Number,N}
	decskp!(U; major=major); U = U[1]
	return U
end

"""
    factor(U::Dec{T,N}; major::String="last") where {T<:Number,N}

Create a new decomposition by factorizing the decomposition `U` to a single factor (contraction along all mode dimmensions).

# Arguments
- `U::Dec{T,N}`: Decomposition, whose entries are in `T` and with `N` dimensions, to factorize.
- `major::String="last"`: Reference numbers for selecting slices along the first rank dimension of the first factor of `U`.

# Returns
- `V`: New factor object type [`Factor`](@ref) obtained by contracting the entire decomposition. 
"""
function factor(U::Dec{T,N}; major::String="last") where {T<:Number,N}
	V = deepcopy(U)
	V = factor!(V; major=major)
	return V
end

""""
    block!(U::Dec{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}

Selects first ranks of the first and last ranks of the last factor of the decomposition `U` specified by the indices `α` and `β`, contract all factors together in place and yield the resulting block.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, which to modify.
- `α::Int`: The index to select from the first rank.
- `β::Int`: The index to select from the last rank.
- `major::String="last"`: Specifies the major ordering of the contraction. Can be "first" or "last".

# Returns
- Resulting factor type `U` after rank selection and contraction to a block.
"""
function block!(U::Dec{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}
	L = declength(U)
	r = decrank(U); p = r[1]; q = r[L+1]
	if α ∉ 1:p
		throw(ArgumentError("the first rank index is out of range"))
	end
	if β ∉ 1:q
		throw(ArgumentError("the second rank index is out of range"))
	end
	decrankselect!(U, α:α, β:β)
	decskp!(U; major=major); U = U[1]
	U = block(U, 1, 1)
	return U
end

"""
    block(U::Dec{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}

Create a new block by selecting the first ranks of the first and last ranks of the last factor of the decomposition `U` specified by the indices `α` and `β`, contract all factors together and yield the resulting block.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, to copy and modify.
- `α::Int`: The index to select from the first rank.
- `β::Int`: The index to select from the last rank.
- `major::String="last"`: Specifies the major order for the operation; default is "last".

# Returns
- `V`: New factor type obtained by rank selection and contraction to a block.
"""
function block(U::Dec{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}
	V = deepcopy(U)
	V = block!(V, α, β, major=major)
	return V
end

"""
    decvcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}

Vertically concatenate the factors of multiple decomposition types (vectors of factors).

# Arguments
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `W::Vararg{Dec{T,N},M}`: Additional decomposition types to concatenate.

# Returns
- New decomposition type obtained by vertically concatenating the given decomposition types (vectors of factors).

# Throws
Summarized Error list:
- `ArgumentError`: If the decompositions are incompatible in the number of factors, mode size, or last rank.

Extended Error list:
- `ArgumentError`:  If the decompositions are incompatible in the number of factors.
- `ArgumentError`: If the decompositions are incompatible in mode size.
- `ArgumentError`: If the decompositions are incompatible in the last rank.
"""
function decvcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
	L = declength(U); m = decsize(U); p = decrank(U)
	W = (V,W...)
	for V ∈ W
		if declength(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if decsize(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = decrank(V)
		if q[L+1] ≠ p[L+1]
			throw(ArgumentError("the decompositions are incompatible in the last rank"))
		end
	end
	return [ [ factordcat(U[ℓ], [ V[ℓ] for V ∈ W ]...) for ℓ ∈ 1:L-1 ]..., factorvcat(U[L], [ V[L] for V ∈ W ]...) ]
end


"""
    dechcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}

Horizontally concatenate the factors of multiple decomposition types (vectors of factors).

# Arguments
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `W::Vararg{Dec{T,N},M}`: Additional decompositions types to concatenate.

# Returns
- New decomposition type obtained by horizontally concatenating the given decomposition types (vectors of factors).

# Throws
Summarized Error list:
- `ArgumentError`: If the decompositions are incompatible in the number of factors, mode size, or first rank.

Extended Error list:
- `ArgumentError`:  If the decompositions are incompatible in the number of factors.
- `ArgumentError`: If the decompositions are incompatible in mode size.
- `ArgumentError`: If the decompositions are incompatible in the first rank.
"""
function dechcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
	L = declength(U); m = decsize(U); p = decrank(U)
	W = (V,W...)
	for V ∈ W
		if declength(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if decsize(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = decrank(V)
		if q[1] ≠ p[1]
			throw(ArgumentError("the decompositions are incompatible in the first rank"))
		end
	end
	return [ factorhcat(U[1], [ V[1] for V ∈ W ]...), [ factordcat(U[ℓ], [ V[ℓ] for V ∈ W ]...) for ℓ ∈ 2:L ]...]
end

"""
    decdcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}

Concatenates the factors of multiple decomposition types (vectors of factors) along a diagonal mode.

# Arguments
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `W::Vararg{Dec{T,N},M}`: Additional decomposition types to concatenate.

# Returns
- New decomposition obtained by diagonally concatenating the factors of the given decomposition types.

# Throws
Summarized Error list:
- `ArgumentError`: If the decompositions are incompatible in the number of factors or mode size.

Extended Error list:
- `ArgumentError`:  If the decompositions are incompatible in the number of factors.
- `ArgumentError`: If the decompositions are incompatible in mode size.
"""
function decdcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
	L = declength(U); m = decsize(U)
	W = (V,W...)
	for V ∈ W
		if declength(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if decsize(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
	end
	return [ factordcat(U[ℓ], [ V[ℓ] for V ∈ W ]...) for ℓ ∈ 1:L ]
end

"""
    decscale!(U::Dec{T,N}, α::T) where {T<:Number,N}

Scale the last factor of the decomposition type `U` in place by a scalar `α`.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, which is to be scaled.
- `α::T`: The scalar factor to multiply the last factor by.

# Returns
- `U`: Decomposition type with the last factor scaled by α.
"""
function decscale!(U::Dec{T,N}, α::T) where {T<:Number,N}
	L = declength(U)
	U[L] *= α
	return U
end

"""
    decreverse!(W::Dec{T,N}) where {T<:Number,N}

Reverse the order of factors in the decomposition `W` in place and transposes their ranks.

# Arguments
- `W::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, which is to be reversed.

# Returns
- `W`: The reversed decomposition of type `Dec` with transposed ranks.
"""
function decreverse!(W::Dec{T,N}) where {T<:Number,N}
	L = declength(W)
	reverse!(W)
	for ℓ ∈ 1:L
		W[ℓ] = factorranktranspose(W[ℓ])
	end
	return W
end
"""
    decmodetranspose!(U::Dec{T,N}, τ::Union{NTuple{K,Int},Vector{Int}}) where {T<:Number,N,K}

Transpose the mode dimensions of each factor in the decomposition type `U` according to the permutation `τ`.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, whose factors' modes are to be transposed.
- `τ::Union{NTuple{K,Int},Vector{Int}}`: A tuple or vector of integers specifiying the permutation of the mode dimensions.

# Returns
- `U`: Decomposition type, whose factors exhibit transposed mode dimensions.

# Throws
Summarized Error list:
- `ArgumentError`: If the decomposition has no mode dimensions or if `τ` is not a valid permutation.

Extended Error list:
- `ArgumentError`:  If the decomposition has no mode dimensions.
- `ArgumentError`: If `τ` is not a valid permutation of the mode dimensions of `U`.
"""
function decmodetranspose!(U::Dec{T,N}, τ::Union{NTuple{K,Int},Vector{Int}}) where {T<:Number,N,K}
	d = N-2
	if d == 0
		throw(ArgumentError("the decomposition should have at least one mode dimension"))
	end
	if length(τ) ≠ d || !isperm(τ)
		throw(ArgumentError("τ is not a valid permutation of the mode dimensions of U"))
	end
	isa(τ, Vector{Int}) || (τ = collect(τ))
	L = declength(U)
	for ℓ ∈ 1:L
		U[ℓ] = factormodetranspose(U[ℓ], τ)
	end
	return U
end

"""
    decmodetranspose!(U::Dec{T,N}) where {T<:Number,N}

Transpose the mode dimensions of each factor in the decomposition `U` by reversing the mode dimensions.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec` whose factors' mode dimensions are to be reversed.

# Returns
- `U`: Decomposition of type `Dec` with the mode dimensions of each factor reversed.

# Throws
- `ArgumentError`: If the decomposition has no mode dimensions.
- `ArgumentError`: If the transpose operation is not applicable.
"""
decmodetranspose!(U::Dec{T,N}) where {T<:Number,N} = decmodetranspose!(U, collect(decndims(U):-1:1))

"""
    decmodereshape(U::Dec{T,N}, n::DecSize) where {T<:Number,N}

Reshape each factor of the decomposition type (vector of factors) `U` to have new mode sizes specified by `n`.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, which is to be reshaped.
- `n::DecSize`: Size matrix specifying the new sizes for each mode of the factors of `U`.

# Returns
- New decomposition with reshaped factors.

# Throws
- `ArgumentError`: If the decomposition does not have at least one mode dimension
- `ArgumentError`: If the number of columns in `n`is inconsistent with `U`.
- `DimensionMismatch`: If `n` is inconsistent with `U`.
"""
function decmodereshape(U::Dec{T,N}, n::DecSize) where {T<:Number,N}
	d = N-2
	if d == 0
		throw(ArgumentError("the decomposition should have at least one mode dimension"))
	end
	ℓ = declength(U)
	if size(n, 2) ≠ ℓ
		throw(ArgumentError("the number of columns in n is inconsistent with U"))
	end
	if prod(n; dims=1) ≠ prod(decsize(U); dims=1)
		throw(DimensionMismatch("n is inconsistent with U"))
	end
	U = [ factormodereshape(U[k], n[:,k]) for k ∈ 1:ℓ ]
	d = size(n, 1)
	U = Dec{T,d+2}(U)
	return U
end

"""
    decfill!(U::Dec{T,N}, v::T) where {T<:Number,N}

Fill each factor of the decomposition type `U` with the value `v` in place.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec` whose factors are to be filled.
- `v::T`: The value to fill each element of the factors with.

# Returns
- `U`: Decomposition of type `Dec` with each factor filled with value `v`.
"""
function decfill!(U::Dec{T,N}, v::T) where {T<:Number,N}
	L = declength(U)
	for ℓ ∈ 1:L
		fill!(U[ℓ], v)
	end
	return U
end

"""
    decrand!(rng::AbstractRNG, U::Dec{T,N}) where {T<:Number,N}

Fill each factor of the decomposition type `U` with random numbers generated from the provided random number generator `rng`.

# Arguments
- `rng::AbstractRNG`: Random number generator to utilize for generating random values.
- `U::Dec{T,N}`: Decomposition of type `Dec`, whose factors are to be filled with random numbers.

# Returns
- `U`: Decomposition of type `Dec` with each factor filled with random numbers.
"""
function decrand!(rng::AbstractRNG, U::Dec{T,N}) where {T<:Number,N}
	L = declength(U)
	for ℓ ∈ 1:L
		rand!(rng, U[ℓ])
	end
	return U
end

"""
    decrand!(U::Dec{T,N}) where {T<:Number,N}

Fill each factor of the decomposition `U` with random numbers using the global random number generator.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec` whose factors are to be filled with random numbers generated from the global RNG (`Random.GLOBAL_RNG`).

# Returns
- `U`: Decomposition of type `Dec` with each factor filled with random numbers.
"""
decrand!(U::Dec{T,N}) where {T<:Number,N} = decrand!(Random.GLOBAL_RNG, U)

"""
    deczeros(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a decomposition of the specified type `T` where each factor is initialized to contain only zeros.

# Arguments
- `T::Type{T}`: Data type of the elements in each factor.
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` where each factor is filled with zeros.
"""
deczeros(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = decfill!(dec(T, n, r; first=first, last=last, len=len), zero(T))

"""
    deczeros(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a decomposition of type `Float64` where each factor is initialized to contain only zeros.

# Arguments
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` (with `Float64` elements) where each factor is filled with zeros.
"""
deczeros(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = deczeros(Float64, n, r; first=first, last=last, len=len)

"""
    decones(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a decomposition of the specified type `T` where each factor is initialized to contain only ones.

# Arguments
- `T::Type{T}`: Data type of the elements in each factor.
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` with the specified data type `T`, where each factor is filled with ones.
"""
decones(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = decfill!(dec(T, n, r; first=first, last=last, len=len), one(T))

"""
    decones(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a decomposition of type `Float64` where each factor is initialized to contain only ones.

# Arguments
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` with `Float64` elements, where each factor is filled with ones.
"""
decones(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = decones(Float64, n, r; first=first, last=last, len=len)

"""
    decrand(rng::AbstractRNG, ::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a decomposition of the specified type `T` and fill each factor with random numbers generated by the provided random number generator `rng`.

# Arguments
- `rng::AbstractRNG`: Random number generator to use for filling the decomposition with random numbers.
- `T::Type{T}`: Data type of the elements in each factor.
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` with the specified data type `T`, where each factor is filled with random numbers generated by the provided `rng`.
"""
decrand(rng::AbstractRNG, ::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = decrand!(rng, dec(T, n, r; first=first, last=last, len=len))
"""
    decrand(rng::AbstractRNG, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a decomposition of type `Float64` and fill each factor with random numbers generated by the provided random number generator `rng`.

# Arguments
- `rng::AbstractRNG`: Random number generator to use for filling the decomposition with random numbers.
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` with `Float64` elements, where each factor is filled with random numbers generated by the provided `rng`.
"""
decrand(rng::AbstractRNG, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = decrand(rng, Float64, n, r; first=first, last=last, len=len)

"""
    decrand(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}

Create a decomposition of the specified type `T` and fill each factor with random numbers using the global random number generator (`Random.GLOBAL_RNG`).

# Arguments
- `T::Type{T}`: Data type of the elements in each factor.
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` with the specified data type `T`, where each factor is filled with random numbers generated by `Random.GLOBAL_RNG`.
"""
decrand(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = decrand(Random.GLOBAL_RNG, T, n, r; first=first, last=last, len=len)

"""
    decrand(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0)

Create a decomposition of type `Float64` and fill each factor with random numbers using the global random number generator (`Random.GLOBAL_RNG`).

# Arguments
- `n::Union{DecSize,FactorSize}`: Size matrix or factor size, representing the dimensions of the modes in each factor of the decomposition.
- `r::Union{Int,DecRank}`: Rank or rank vector, representing the ranks of the decomposition.
- `first::Int=0`: Optional first rank. Specifies the rank of the first mode dimension if `r` is not expressed using a vector.
- `last::Int=0`: Optional last rank. Specifies the rank of the last mode dimension if `r` is not expressed using a vector.
- `len::Int=0`: Number of factors. Specifies the number of factors in the decomposition.

# Returns
- A decomposition of type `Dec` with `Float64` elements, where each factor is filled with random numbers generated by `Random.GLOBAL_RNG`.
"""
decrand(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = decrand(Random.GLOBAL_RNG, Float64, n, r; first=first, last=last, len=len)


"""
    decappend!(U::Dec{T,N}, V::Dec{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number, N}

Append the decomposition type (vector of factors) `V` to the end of the decomposition type `U` in place. Optionally, the ranks are checked before and after the operation.

# Arguments
- `U::Dec{T,N}`: Target decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, to which `V` will be appended.
- `V::Dec{T,N}`: Decomposition of type `Dec`, which to append to `U`.
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` and `V` before appending.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the combined result after appending.

# Returns
- `U`: Modified decomposition type `U` after appending `V`.

# Throws
Summarized Error list:
- `DimensionMismatch`: If `U` and `V` have different numbers of dimensions, or if the ranks of their factors are incorrect or inconsistent with the operation.

Extended Error list:
- `DimensionMismatch`: If U and V are inconsistent in the number of dimensions.
- `ArgumentError`:If the factors of U have incorrect or inconsistent ranks.
- `ArgumentError`: If the factors of V have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function decappend!(U::Dec{T,N}, V::Dec{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if decndims(U) ≠ decndims(V)
		throw(DimensionMismatch("U and V are inconsistent in the number of dimensions"))
	end
	(declength(V) == 0) && return U
	p,q = decranks(U)
	r,s = decranks(V)
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
	return append!(U, V)
end

"""
    decprepend!(U::Dec{T,N}, V::Dec{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number, N}

Prepend the decomposition type (vector of factors) `V` to the beginning of the decomposition type (vector of factors) `U` in place. Optionally, the ranks are checked before and after the operation.

# Arguments
- `U::Dec{T,N}`: Target decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, to which `V` will be prepended.
- `V::Dec{T,N}`: Decomposition of type `Dec`, which to append to `U`.
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` and `V` before prepending.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the combined result after prepending.

# Returns
- `U`: Modified decomposition type `U` after prepending `V`.

# Throws
Summarized Error list:
- `DimensionMismatch`: If `U` and `V` have different numbers of dimensions, or if the ranks of their factors are incorrect or inconsistent with the operation.

Extended Error list:
- `DimensionMismatch`: If U and V have different numbers of dimensions.
- `ArgumentError`: If the factors of U have incorrect or inconsistent ranks.
- `ArgumentError`: If the factors of V have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function decprepend!(U::Dec{T,N}, V::Dec{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if decndims(U) ≠ decndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	(declength(V) == 0) && return U
	p,q = decranks(U); r,s = decranks(V)
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
	return prepend!(U, V)
end


"""
    decpush!(U::Dec{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number, N}

Push a factor `V` to the end of the decomposition type (vector of factors) `U` in place. Optionally, the ranks before and after the operation are checked.

# Arguments
- `U::Dec{T,N}`: Target decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, to which the factor `V` will be pushed.
- `V::Factor{T,N}`: The factor to push to the decomposition type (vector of factors) `U`.
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` before pushing.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the result after pushing.

# Returns
- `U`: Modified decomposition type (vector of factors) `U` after pushing `V`.

# Throws
Summarized Error list:
- `DimensionMismatch`: If `U` and `V` have different numbers of dimensions, or if their ranks (in case of `U`, the ranks of its factors) are inconsistent with the operation.

Extended Error list:
- `DimensionMismatch`: If U and V have different numbers of dimensions.
- `ArgumentError`: If the factors of U have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function decpush!(U::Dec{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if decndims(U) ≠ factorndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = decranks(U)
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
	return push!(U, V)
end

"""
    decpushfirst!(U::Dec{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number, N}

Push a factor `V` to the beginning of the decomposition type (vector of factors) `U` in place. Optionally, the ranks before and after the operation are checked.

# Arguments
- `U::Dec{T,N}`: Target decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, to which the factor `V` will be pushed at the beginning.
- `V::Factor{T,N}`: The factor to push to the beginning of `U`.
- `rankprecheck::Bool=true`: If `true` (by default), the ranks of `U` are checked before pushing.
- `rankpostcheck::Bool=true`: If `true` (by default), the ranks of the result are checked after pushing.

# Returns
- `U`: Modified decomposition type (vector of factors) `U` after pushing `V` at the beginning.

# Throws
Summarized Error list:
- `DimensionMismatch`: If `U` and `V` have different numbers of dimensions, or if their ranks (in case of `U`, the ranks of its factors) are inconsistent with the operation.

Extended Error list:
- `DimensionMismatch`: If U and V have different numbers of dimensions.
- `DimensionMismatch`: If the factors of U have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function decpushfirst!(U::Dec{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if decndims(U) ≠ factorndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = decranks(U)
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
	return pushfirst!(U, V)
end

"""
    decpop!(U::Dec{T,N}) where {T<:Number, N}

Pop the last factor from the decomposition type (vector of factors) `U` and return it.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, from which to pop the last factor.

# Returns
- `V`: The last factor that was removed from `U`.
"""
function decpop!(U::Dec{T,N}) where {T<:Number,N}
	return pop!(U)
end

"""
    decpopfirst!(U::Dec{T,N}) where {T<:Number,N}

Pop the first factor from the decomposition type (vector of factors) `U` and return it.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, from which to pop the first factor.

# Returns
- `V`: The first factor that was removed from `U`.
"""
function decpopfirst!(U::Dec{T,N}) where {T<:Number,N}
	return popfirst!(U)
end

"""
    decinsert!(U::Dec{T,N}, ℓ::Int, V::Factor{T,N}; path::String="", rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number, N}

Insert a factor `V` into the decomposition type (vector of factors) `U` at the specified index `ℓ`. Optionally, the ranks are checked before and after the operation.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, in which to insert the factor `V`.
- `ℓ::Int`: The index at which to insert the factor `V`.
- `V::Factor{T,N}`: The factor to insert into the decomposition `U`.
- `path::String=""`: The path direction for insertion; should be "forward" or "backward".
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` before insertion.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the result after insertion.

# Returns
- `U`: Modified decomposition type `U` after inserting `V`.

# Throws
Summarized Error list:
- `ArgumentError`: If `path` is not "forward" or "backward".
- `DimensionMismatch`: If `ℓ` is out of range or if the ranks (of `U`, `V` or the factors of `U`) are inconsistent for the operation.

Extended Error list:
- `ArgumentError`:  If path is neither \"forward\" nor \"backward\"
- `ArgumentError`: If ℓ is not from 1:L, where L is the number of factors in `U
- `DimensionMismatch`: If the factors of U have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U and V are inconsistent for this operation.
"""
function decinsert!(U::Dec{T,N}, ℓ::Int, V::Factor{T,N}; path::String="", rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	L = declength(U)
	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end
	if ℓ ∉ 1:L
		throw(ArgumentError("ℓ is required to be from 1:L, where L is the number of factors in U"))
	end
	(path == "forward") && (ℓ = ℓ+1)
	(ℓ == 1) && return decpushfirst!(U, V; rankprecheck=rankprecheck, rankpostcheck=rankpostcheck)
	(ℓ == L+1) && return decpush!(U, V; rankprecheck=rankprecheck, rankpostcheck=rankpostcheck)
	p,q = decranks(U)
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
	return insert!(U, ℓ, V)
end
"""
    decdeleteat!(U::Dec{T,N}, Λ::Union{Int,Vector{Int},NTuple{M,Int} where M}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number, N}

Delete factors from the decomposition type (vector of factors) `U` at the specified indices `Λ` in place. Optionally, the ranks are checked before and after the operation.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, from which factors will be deleted.
- `Λ::Union{Int,Vector{Int},NTuple{M,Int} where M}`: Indices of factors to delete. Can be a single integer, a vector of integers, or a tuple of integers.
- `rankprecheck::Bool=true`: If `true` (by default), checks the ranks of `U` before deletion.
- `rankpostcheck::Bool=true`: If `true` (by default), checks the ranks of the result after deletion.

# Returns
- `U`: Modified decomposition type `U` after deleting the specified factors.

# Throws
Summarized Error list:
- `ArgumentError`: If the entries of `Λ` are not unique or are not within the valid range.
- `DimensionMismatch`: If the ranks of `U` are inconsistent for this operation.

Extended Error list:
- `ArgumentError`:  If the entries of Λ are not unique.
- `ArgumentError`: If Λ is not an element or a subset of 1:L with unique entries, where L is the number of factors in `U`.
- `DimensionMismatch`: If the factors of U have incorrect or inconsistent ranks.
- `DimensionMismatch`: If the ranks of U are inconsistent for this operation.
"""
function decdeleteat!(U::Dec{T,N}, Λ::Union{Int,Vector{Int},NTuple{M,Int} where M}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	L = declength(U)
	if isa(Λ, Vector{Int}) && unique(Λ) ≠ Λ
		throw(ArgumentError("the entries of Λ should be unique"))
	end
	isa(Λ, Vector{Int}) || (Λ = collect(Λ))
	if Λ ⊈ 1:L
		throw(ArgumentError("Λ should be an element or a subset of 1:L with unique entries, where L is the number of factors in U"))
	end
	p,q = decranks(U)
	if rankprecheck
		try checkranks(p,q) catch
			throw(DimensionMismatch("the factors of U have inconsistent ranks"))
		end
	end
	if rankpostcheck
		deleteat!(p, Λ); deleteat!(q, Λ)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U are inconsistent for this operation"))
		end
	end
	return deleteat!(U, Λ)
end


"""
    decinsertidentity!(U::Dec{T,N}, ℓ::Int; path::String="", rankprecheck::Bool=true) where {T<:Number, N}

Insert an identity factor into the decomposition type (vector of factors) `U` at the specified index `ℓ`. Optionally, the ranks are checked before the operation.

# Arguments
- `U::Dec{T,N}`: Decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions, in which to insert the identity factor.
- `ℓ::Int`: The index at which to insert the identity factor.
- `path::String=""`: The path direction for insertion (should be "forward" or "backward").
- `rankprecheck::Bool=true`: If `true` (by default), the ranks of `U` are checked before insertion.

# Returns
- `U`: Modified decomposition type `U` after inserting the identity factor.

# Throws
Summarized Error list:
- `ArgumentError`: If `path` is not "forward" or "backward".
- `DimensionMismatch`: If `ℓ` is out of range or if the ranks are inconsistent for this operation.

Extended Error list:
- `ArgumentError`:  If the is neither \"forward\" nor \"backward\".
- `ArgumentError`: If ℓ is not from 1:L, where L is the number of factors in U.
- `DimensionMismatch`: If the factors of U have inconsistent ranks.
"""
function decinsertidentity!(U::Dec{T,N}, ℓ::Int; path::String="", rankprecheck::Bool=true) where {T<:Number,N}
	L = declength(U)
	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end
	if ℓ ∉ 1:L
		throw(ArgumentError("ℓ is required to be from 1:L, where L is the number of factors in U"))
	end
	p,q = decranks(U)
	d = decndims(U)
	r = (path == "forward") ? q[ℓ] : p[ℓ]
	V = Matrix{T}(I, r, r); V = factor(V, ones(Int, d), [])
	(path == "forward") && (ℓ = ℓ+1)
	(ℓ == 1) && return decpushfirst!(U, V; rankprecheck=rankprecheck, rankpostcheck=false)
	(ℓ == L+1) && return decpush!(U, V; rankprecheck=rankprecheck, rankpostcheck=false)
	p,q = decranks(U)
	if rankprecheck
		try checkranks(p,q) catch
			throw(DimensionMismatch("the factors of U have inconsistent ranks"))
		end
	end
	return insert!(U, ℓ, V)
end

"""
    decskp!(W::Dec{T,N}, Λ::Indices; path::String="", major::String="last") where {T<:Number, N}

Perform sequential contraction of components of a vector of factors `W` based on specified indices `Λ`, following a specified path and contraction order.

# Arguments
- `W::Dec{T, N}`: Decomposition object of type `Dec`, which represents a vector of factors with elements of type `T` and with `N` as the number dimensions.
- `Λ::Indices`: reference numbers specifying which components of (the vector of factors) `W` to contract. Can be a colon `Colon` (indicating all indices) or a `Vector{Int}` specifying particular indices.
- `path::String=""`: keyword argument specifying the order of contraction. 
  - `""` (default): path can be deduced from `Λ` if `Λ` is a colon or empty.
  - `"forward"`: Contraction of components in a forward sequence.
  - `"backward"`: Contraction of components in a backward sequence.
- `major::String="last"`: keyword argument indicating the primary direction for the contraction operation.
  - `"last"` (default): Contraction focuses on the last dimension.
  - `"first"`: Contraction focuses on the first dimension.

# Returns
- `W`: modified decomposition object (vector of factors) after performing the contractions on the specified components.

# Throws
- `ArgumentError`: If `path` is not one of `""`, `"forward"`, or `"backward"`.
- `ArgumentError`: If  `major` is neither `"first"` nor `"last"`.
- `ArgumentError`: If the decomposition object `W` is empty (`L == 0`).
- `ArgumentError`: If `Λ` is a colon, but `path` is specified as non-empty.
- `ArgumentError`: If `Λ` is neither empty nor a colon and path is neither `"forward"` nor `"backward"`.
- `ArgumentError`: If `Λ` has duplicate entries.
- `ArgumentError`: If `Λ` contains invalid indices that do not match the expected range based on `path` and the number of factors in `W`.
"""
function decskp!(W::Dec{T,N}, Λ::Indices; path::String="", major::String="last") where {T<:Number,N}
	if path ∉ ("","forward","backward")
		throw(ArgumentError("the value of the keyword argument path should be \"\" (default, accepted only for empty Λ and for Λ=:), \"forward\" or \"backward\""))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	L = declength(W); decrank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if isa(Λ, Colon) && path ≠ ""
		throw(ArgumentError("when Λ is a colon, path should be omitted or specfied as \"\" (default)"))
	end
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
		throw(ArgumentError("Λ, when nonempty, should be a colon or an Int/Vector/NTuple/UnitRange of Int, from 1:L-1 for path=\"forward\" and from 2:L for path=\"backward\", where L is the number of factors in W"))
	end
	sort!(Λ; rev=true)
	for ℓ ∈ Λ
		W[ℓ-1] = factorcontract(W[ℓ-1], W[ℓ]; major=major)
		decdeleteat!(W, ℓ; rankprecheck=false, rankpostcheck=false)
	end
	return W
end

"""
    decskp!(W::Dec{T,N}; path::String="", major::String="last") where {T<:Number,N}

Perform sequential contraction of all components of a decomposition `W` following the specified contraction path and direction.

# Arguments
- `W::Dec{T, N}`: Decomposition object of type `Dec`, which represents a vector of factors with elements of type `T` and with `N` as the number of dimensions.
- `path::String=""`: Keyword argument specifying the order of contraction.
  - `""` (default): Path can be deduced if all indices are used.
  - `"forward"`: Contract components in a forward sequence.
  - `"backward"`: Contract components in a backward sequence.
- `major::String="last"`: Keyword argument indicating the primary direction for the contraction.
  - `"last"` (default): Focuses on the last dimension.
  - `"first"`: Focuses on the first dimension.

# Returns
- `W`: The modified decomposition object after performing the contractions.

# Throws
- `ArgumentError`: If `path` is not one of `""`, `"forward"`, or `"backward"`.
- `ArgumentError`: If  `major` is neither `"first"` nor `"last"`.
- `ArgumentError`: If the decomposition object `W` is empty (`L == 0`).
"""
decskp!(W::Dec{T,N}; path::String="", major::String="last") where {T<:Number,N} = decskp!(W, :; path=path, major=major)

"""
    decskp(W::Dec{T,N}, Λ::Indices; path::String="", major::String="last") where {T<:Number,N}

Perform a sequential contraction of selected components of a decomposition object (vector of factors) `W` based on specified indices `Λ`, a path (`"forward"`, `"backward"`, or `""`), and a major contraction direction.

# Arguments
- `W::Dec{T, N}`: Decomposition object of type `Dec`, which represents a vector of factors with elements of type `T` and with `N` as the number dimensions.
- `Λ::Indices`: reference numbers specifying which components of the decomposition obeject `W` to contract. Can be a colon `Colon` (indicating all indices), a `Vector{Int}`, or other types convertible to an index vector.
- `path::String=""`: keyword argument specifying the order of contraction. 
  - `""` (default): Approved only if `Λ` is empty or `Λ` is a colon.
  - `"forward"`: Contraction of components in a forward sequence.
  - `"backward"`: Contraction of components in a backward sequence.
- `major::String="last"`: keyword argument determining the primary direction for the contraction operation.
  - `"last"` (default): Contraction focuses on last dimension.
  - `"first"`: Contraction focuses on first dimension.

# Returns
- `U`: result of contracting the selected components of `W` based on the provided indices `Λ` and the specified path and major direction.

# Throws
- `ArgumentError`: If `path` is not one of `""`, `"forward"`, or `"backward"`.
- `ArgumentError`: If `major` is not `"first"` or `"last"`.
- `ArgumentError`: If the decomposition object `W` is empty (`L == 0`).
- `ArgumentError`: If `Λ` is empty.
- `ArgumentError`: If `Λ` is not a contiguous set of integers.
- `ArgumentError`: If `Λ` is a colon, but `path` is specified as non-empty.
- `ArgumentError`: If `Λ` is neither empty nor a colon and path is not specified (`"forward"`, or `"backward"`) .
- `ArgumentError`: If `Λ` has duplicate entries.
- `ArgumentError`: If `Λ` contains invalid indices that do not match the expected range based on `path` and the number of factors in `W`.
"""
function decskp(W::Dec{T,N}, Λ::Indices; path::String="", major::String="last") where {T<:Number,N}
	if path ∉ ("","forward","backward")
		throw(ArgumentError("the value of the keyword argument path should be \"\" (default, accepted only for empty Λ and for Λ=:), \"forward\" or \"backward\""))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	L = declength(W); decrank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if isa(Λ, Colon) && path ≠ ""
		throw(ArgumentError("when Λ is a colon, path should be omitted or specfied as \"\" (default)"))
	end
	isa(Λ, Colon) && (Λ = collect(2:L); path = "backward")
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	if length(Λ) == 0
		throw(ArgumentError("Λ is empty"))
	end
	if (minimum(Λ):maximum(Λ)) ⊈ Λ
		throw(ArgumentError("the entries of Λ should form a set of contiguous integers"))
	end
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
	M = length(Λ)
	sort!(Λ; rev=true)
	U = W[Λ[1]]
	for λ ∈ 2:M
		U = factorcontract(W[Λ[λ]], U; major=major)
	end
	return U
end

"""
    decskp(W::Dec{T,N}; path::String="", major::String="last") where {T<:Number,N}

Perform sequential contraction of all components of a decomposition `W` in a non-mutating way, following the specified contraction path and direction.

# Arguments
- `W::Dec{T, N}`: Decomposition object of type `Dec`, which represents a vector of factors with elements of type `T` and with `N` as the number of dimensions.
- `path::String=""`: Keyword argument specifying the order of contraction.
  - `""` (default): Path can be deduced if all indices are used.
  - `"forward"`: Contract components in a forward sequence.
  - `"backward"`: Contract components in a backward sequence.
- `major::String="last"`: Keyword argument indicating the primary direction for the contraction.
  - `"last"` (default): Focuses on the last dimension.
  - `"first"`: Focuses on the first dimension.

# Returns
- The contracted decomposition of type `Dec`.

# Throws
- `ArgumentError`: If `path` is not one of `""`, `"forward"`, or `"backward"`.
- `ArgumentError`: If  `major` is neither `"first"` nor `"last"`.
- `ArgumentError`: If the decomposition object `W` is empty (`L == 0`).
"""
decskp(W::Dec{T,N}; path::String="", major::String="last") where {T<:Number,N} = decskp(W, :; path=path, major=major)

"""
    decmp(U₁::Dec{T,N₁}, σ₁::Indices, U₂::Dec{T,N₂}, σ₂::Indices) where {T<:Number, N₁, N₂}

Perform the mode product of two decomposition objects `U₁` and `U₂` along specified sets of modes and returns a new decomposition object resulting from the multiplication.

# Arguments
- `U₁::Dec{T,N₁}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N₁` dimensions.
- `σ₁::Indices`: A set of mode indices for `U₁`. Specifies which modes of `U₁` to multiply.
- `U₂::Dec{T,N₂}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N₂` dimensions.
- `σ₂::Indices`: A set of mode indices for `U₂`. Specifies which modes of `U₂` to multiply.

# Returns
- `U`: A new decomposition object resulting from the mode product of `U₁` and `U₂` along the specified modes.

# Throws
- `ArgumentError`: If σ₁ is passed as a vector, and it is not a vector of the type Vector{Int} and not an empty vector of the type Vector{Any}.
- `ArgumentError`: If σ₂ is passed as a vector, and it is not a vector of the type Vector{Int} and not an empty vector of the type Vector{Any}.
- `ArgumentError`: If U₁ and U₂ differ in the number of factors.
- `ArgumentError`: If the specified sets of modes of σ₁ and σ₂ are inconsistent.
- `ArgumentError`: If the set of modes of U₁ is specified incorrectly.
- `ArgumentError`: If the set of modes of U₂ is specified incorrectly.
- `ArgumentError`: If U₁ and U₂ are inconsistent with respect to the specified modes.
"""
function decmp(U₁::Dec{T,N₁}, σ₁::Indices, U₂::Dec{T,N₂}, σ₂::Indices) where {T<:Number,N₁,N₂}
	n₁ = decsize(U₁); d₁ = decndims(U₁); L₁ = declength(U₁)
	n₂ = decsize(U₂); d₂ = decndims(U₂); L₂ = declength(U₂)
	if isa(σ₁, Vector{Any}) && length(σ₁) > 0
		throw(ArgumentError("if σ₁ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	if isa(σ₂, Vector{Any}) && length(σ₂) > 0
		throw(ArgumentError("if σ₂ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	isa(σ₁, Vector{Int}) || (σ₁ = indvec(σ₁; max=d₁))
	isa(σ₂, Vector{Int}) || (σ₂ = indvec(σ₂; max=d₂))
	if L₁ ≠ L₂
		throw(ArgumentError("U₁ and U₂ differ in the number of factors"))
	end
	L = L₁
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
	U = [ factormp(U₁[ℓ], σ₁, U₂[ℓ], σ₂) for ℓ ∈ 1:L ]
	return U
end


# function deckp(U::Dec{T,N}, V::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
# 	n = decsize(U); L = declength(U)
# 	d = 1+length(V)
# 	for k ∈ 2:d
# 		if declength(V[k-1]) ≠ L
# 			throw(ArgumentError("the input decompositions differ in the number of factors"))
# 		end
# 	end
# 	return [ factorkp(U[ℓ], [ V[k-1][ℓ] for k ∈ 2:d ]...) for ℓ ∈ 1:L ]
# end


"""
    deckp(U::Union{Dec{T,N}, Pair{Dec{T,N}, Int}}, V::Vararg{Union{Dec{T,N}, Pair{Dec{T,N}, Int}}, M}) where {T<:Number, N, M}

Perform the Kronecker product on multiple decomposition objects, optionally with specified exponents for each decomposition object, and return a new decomposition object resulting from the product.

# Arguments
- `U::Union{Dec{T,N}, Pair{Dec{T,N}, Int}}`: The first decomposition object or a pair of a decomposition object and an exponent. If a pair, the exponent specifies how many times the decomposition object is repeated in the Kronecker product.
- `V::Vararg{Union{Dec{T,N}, Pair{Dec{T,N}, Int}}, M}`: Additional decomposition objects or pairs of decomposition objects and exponents to include in the Kronecker product.

# Returns
- `W`: A new decomposition object resulting from the Kronecker product of the input decompositions, each raised to its corresponding exponent.

# Throws
- `ArgumentError`: If any exponent specified is negative.
- `ArgumentError`: If the input decompositions differ in the number of factors.
"""
function deckp(U::Union{Dec{T,N},Pair{Dec{T,N},Int}}, V::Vararg{Union{Dec{T,N},Pair{Dec{T,N},Int}},M}) where {T<:Number,N,M}
	V = (U,V...)
	nf = length(V)
	U = Vector{Tuple{Dec{T,N},Int}}(undef, nf)
	for k ∈ 1:nf
		W = V[k]
		if isa(W, Tuple)
			if W[2] < 0
				throw(ArgumentError("all the specified exponents should be nonnegative"))
			end
			U[k] = W[1],W[2]
		else
			U[k] = W,1
		end
	end
	n = decsize(U[1][1]); L = declength(U[1][1])
	for k ∈ 2:nf
		if declength(U[k][1]) ≠ L
			throw(ArgumentError("the input decompositions differ in the number of factors"))
		end
	end
	return [ factorkp([ U[k][1][ℓ] => U[k][2] for k ∈ 1:nf ]...) for ℓ ∈ 1:L ]
end


# function decaxpby(α::Vector{T}, U::Dec{T,N}, β::Vector{T}, V::Dec{T,N}) where {T<:Number,N}
# 	m = decsize(U); L = declength(U)
# 	p = decrank(U); q = decrank(V)
# 	if declength(V) ≠ L
# 		throw(ArgumentError("U and V differ in the number of factors"))
# 	end
# 	if decsize(V) ≠ m
# 		throw(ArgumentError("U and V are inconsistent in mode size"))
# 	end
# 	if length(α) ≠ L
# 		throw(ArgumentError("α should have the same length as U and V"))
# 	end
# 	if length(β) ≠ L
# 		throw(ArgumentError("β should have the same length as U and V"))
# 	end
# 	if q[1] ≠ p[1]
# 		throw(ArgumentError("the decompositions are incompatible in the first rank"))
# 	end
# 	if q[L+1] ≠ p[L+1]
# 		throw(ArgumentError("the decompositions are incompatible in the last rank"))
# 	end
# 	(L == 1) && return U .+ V
# 	W = [ factorhcat(α[1]*U[1], β[1]*V[1]), [ factordcat(α[ℓ]*U[ℓ], β[ℓ]*V[ℓ]) for ℓ ∈ 2:L-1 ]..., factorvcat(α[L]*U[L], β[L]*V[L]) ]
# 	return W
# end


"""
    decaxpby!(α::Vector{T}, U::Dec{T,N}, β::Vector{T}, V::Dec{T,N}) where {T<:Number,N}

Perform the operation `U = α .* U + β .* V` for decompositions `U` and `V`, where `α` and `β` are vectors of scalars that scale the corresponding factors of `U` and `V` respectively. The operation modifies `U` directly.

# Arguments
- `α::Vector{T}`: Vector of scaling factors for the decomposition object `U`. Must have the same length as the number of factors in `U`.
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `β::Vector{T}`: Vector of scaling factors for the decomposition object `V`. Must have the same length as the number of factors in `V`.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- `U`: Modified decomposition object `U` after applying the operation.

# Throws
Summarized Error List:
- `ArgumentError`: If `U` and `V` have a different number of factors, differ in mode sizes, are incompatible in ranks, or if `α` and `β` do not have the correct lengths.

Extended Error List:
- `ArgumentError`: If U and V differ in the number of factors
- `ArgumentError`: If U and V are inconsistent in mode size
- `ArgumentError`: If α does not have the same length as U and V
- `ArgumentError`: If β does not have the same length as U and V
- `ArgumentError`: If the decompositions is incompatible in the first rank.
- `ArgumentError`: If the decompositions is incompatible in the last rank.
"""
function decaxpby!(α::Vector{T}, U::Dec{T,N}, β::Vector{T}, V::Dec{T,N}) where {T<:Number,N}
	m = decsize(U); L = declength(U)
	p = decrank(U); q = decrank(V)
	if declength(V) ≠ L
		throw(ArgumentError("U and V differ in the number of factors"))
	end
	if decsize(V) ≠ m
		throw(ArgumentError("U and V are inconsistent in mode size"))
	end
	if length(α) ≠ L
		throw(ArgumentError("α should have the same length as U and V"))
	end
	if length(β) ≠ L
		throw(ArgumentError("β should have the same length as U and V"))
	end
	if q[1] ≠ p[1]
		throw(ArgumentError("the decompositions are incompatible in the first rank"))
	end
	if q[L+1] ≠ p[L+1]
		throw(ArgumentError("the decompositions are incompatible in the last rank"))
	end
	if L == 1
		U[1] = α[1]*U[1]+β[1]*V[1]
	else
		U[1] = factorhcat(α[1]*U[1], β[1]*V[1])
		for ℓ ∈ 2:L-1
			U[ℓ] = factordcat(α[ℓ]*U[ℓ], β[ℓ]*V[ℓ])
		end
		U[L] = factorvcat(α[L]*U[L], β[L]*V[L])
	end
	return U
end

"""
    decaxpby!(α::T, U::Dec{T,N}, β::T, V::Dec{T,N}) where {T<:Number,N}

Perform the operation `U = α .* U + β .* V` for decompositions `U` and `V`, where `α` and `β` are scalar values that scale all factors of `U` and `V` respectively. The operation modifies `U` directly.

# Arguments
- `α::T`: Scalar value to scale all factors of `U`.
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `β::T`: Scalar value to scale all factors of `V`.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- `U`: Modified decomposition object `U` after applying the operation.

# Throws
- `ArgumentError`: If `U` and `V` have a different number of factors, differ in mode sizes, or are incompatible in ranks.
"""
decaxpby!(α::T, U::Dec{T,N}, β::T, V::Dec{T,N}) where {T<:Number,N} = decaxpby!([ones(T, declength(U)-1); α], U, [ones(T, declength(V)-1); β], V)

"""
    decadd!(U::Dec{T,N}, V::Dec{T,N}) where {T<:Number,N}

Perform the operation `U = U + V` , adding decomposition `V` to `U` and modifying `U` directly.

# Arguments
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- `U`: The modified decomposition object `U` after the addition.

# Throws
Summarized Error List:
- `ArgumentError`: If `U` and `V` are incompatible in the number of factors, mode size, the first or the last rank.

Extended Error List:
- `ArgumentError`: If the decompositions are incompatible in the number of factors.
- `ArgumentError`: If the decompositions are incompatible in mode size.
- `ArgumentError`: If the decompositions are incompatible in the first rank.
- `ArgumentError`: If the decompositions are incompatible in the last rank.
"""
decadd!(U::Dec{T,N}, V::Dec{T,N}) where {T<:Number,N} = decaxpby!(one(T), U, one(T), V)

"""
    decaxpby(α::Vector{T}, U::Dec{T,N}, β::Vector{T}, V::Dec{T,N}) where {T<:Number,N}

Perform the operation `U = α .* U + β .* V` for decompositions `U` and `V`, where `α` and `β` are vectors of scalars that scale the corresponding factors of `U` and `V` respectively. The result is returned as a new decomposition.

# Arguments
- `α::Vector{T}`: Vector of scaling factors for the decomposition object `U`. Must have the same length as the number of factors in `U`.
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `β::Vector{T}`: Vector of scaling factors for the decomposition object `V`. Must have the same length as the number of factors in `V`.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- A new decomposition resulting from the operation `α .* U + β .* V`.

# Throws
- `ArgumentError`: If `U` and `V` have a different number of factors, differ in mode sizes, or are incompatible in ranks, or if `α` and `β` do not have the correct lengths.
"""
decaxpby(α::Vector{T}, U::Dec{T,N}, β::Vector{T}, V::Dec{T,N}) where {T<:Number,N} = decaxpby!(α, copy(U), β, V)

"""
    decaxpby(α::T, U::Dec{T,N}, β::T, V::Dec{T,N}) where {T<:Number,N}

Perform the operation `U = α .* U + β .* V` for decompositions `U` and `V`, where `α` and `β` are scalar values that scale all factors of `U` and `V` respectively. The result is returned as a new decomposition.

# Arguments
- `α::T`: Scalar value to scale all factors of `U`.
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `β::T`: Scalar value to scale all factors of `V`.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- A new decomposition resulting from the operation `α .* U + β .* V`.

# Throws
- `ArgumentError`: If `U` and `V` have a different number of factors, differ in mode sizes, or are incompatible in ranks.
"""
decaxpby(α::T, U::Dec{T,N}, β::T, V::Dec{T,N}) where {T<:Number,N} = decaxpby([ones(T, declength(U)-1); α], U, [ones(T, declength(V)-1); β], V)

"""
    decadd(U::Dec{T,N}, V::Dec{T,N}) where {T<:Number,N}

Perform the element-wise addition of two decompositions `U` and `V`. The result is returned as a new decomposition.

# Arguments
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- A new decomposition resulting from the element-wise addition of `U` and `V`.

# Throws
- `ArgumentError`: If `U` and `V` have a different number of factors, differ in mode sizes, or are incompatible in ranks.
"""
decadd(U::Dec{T,N}, V::Dec{T,N}) where {T<:Number,N} = decaxpby(one(T), U, one(T), V)

"""
    decadd(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}

Add multiple decompositions `U`, `V` and `W...` by stacking their factors and return a new decomposition object that represents their sum.

# Arguments
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `W::Vararg{Dec{T,N},M}`: Additional decomposition objects to add.

# Returns
- A new decomposition object obtained by horizontal and vertical concatenation of the factors of all given decompositions.

# Throws
Summarized error list:
- `ArgumentError`: Thrown if any of the decompositions have incompatible dimensions or ranks.

Extended error list:
- `ArgumentError`: If the decompositions are incompatible in the number of factors.
- `ArgumentError`: If the decompositions are incompatible in mode size.
- `ArgumentError`: If the decompositions are incompatible in the first rank.
- `ArgumentError`: If the decompositions are incompatible in the last rank.

# Details
- The factors of each decomposition are stacked together as follows:
  - The first factor is concatenated horizontally.
  - The middle factors are concatenated diagonally.
  - The last factor is concatenated vertically.
"""
function decadd(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
	L = declength(U); m = decsize(U); p = decrank(U)
	W = (V,W...)
	for V ∈ W
		if declength(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if decsize(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = decrank(V)
		if q[1] ≠ p[1]
			throw(ArgumentError("the decompositions are incompatible in the first rank"))
		end
		if q[L+1] ≠ p[L+1]
			throw(ArgumentError("the decompositions are incompatible in the last rank"))
		end
	end
	if L == 1
		return U + sum(W)
	end
	[ factorhcat(U[1], [ V[1] for V ∈ W ]...), [ factordcat(U[ℓ], [ V[ℓ] for V ∈ W ]...) for ℓ ∈ 2:L-1 ]..., factorvcat(U[L], [ V[L] for V ∈ W ]...) ]
end


"""
    dechp(U::Dec{T,N}, V::Dec{T,N}) where {T<:Number,N}

Perform the Hadamard product (element-wise multiplication) of two decompositions `U` and `V` and return a new decomposition object `W`.

# Arguments
- `U::Dec{T,N}`: First decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.
- `V::Dec{T,N}`: Second decomposition of type `Dec`, which represents a vector of factors whose entries are in `T` and with `N` dimensions.

# Returns
- `W`: A new decomposition  of type `Dec`, which results from the Hadamard product of `U` and `V`.

# Throws
- `ArgumentError`: Thrown if `U` and `V` have a different number of factors or are inconsistent in mode size.
"""
function dechp(U::Dec{T,N}, V::Dec{T,N}) where {T<:Number,N}
	m = decsize(U); L = declength(U)
	if declength(V) ≠ L
		throw(ArgumentError("U and V differ in the number of factors"))
	end
	if decsize(V) ≠ m
		throw(ArgumentError("U and V are inconsistent in mode size"))
	end
	W = [ factorhp(U[ℓ], V[ℓ]) for ℓ ∈ 1:L ]
	return W
end

"""
    decqr!(W::Dec{T,N}, Λ::Indices; pivot::Bool=false, path::String="", returnRfactors::Bool=false) where {T<:FloatRC, N}

Perform a QR decomposition on the components of a decomposition object (vector of factors) `W` with specified indices `Λ`, following a given path, and optional pivoting.

# Arguments
- `W::Dec{T, N}`: Decomposition object of type `Dec`, which represents a vector of factors with elements of type `T` and with `N` as the number dimensions.
- `Λ::Indices`: reference numbers specifying which components of the decomposition object `W` to apply the QR decomposition to. Can be a colon `Colon` (indicating all indices) or a `Vector{Int}`.
- `pivot::Bool=false`: keyword argument indicating whether pivoting should be used in the QR decomposition.
- `path::String=""`: keyword argument specifying the order of decomposition. 
  - `""` (default): The path is deduced from `Λ` if possible.
  - `"forward"`: Performs decomposition in a forward sequence.
  - `"backward"`: Performs decomposition in a backward sequence.
- `returnRfactors::Bool=false`: keyword argument indicating whether to return the `R` factors from the QR decomposition along with the modified decomposition object `W`.

# Returns
- `W`: modified decomposition object after performing the QR decomposition on the specified components.
- `Rfactors` (optional): vector of `Matrix{T}` containing the `R` factors from the QR decompositions; returned if `returnRfactors` is `true`.

# Throws
- `ArgumentError`: If the decomposition object `W` is empty (`L == 0`).
- `ArgumentError`: If `path` is not one of `""`, `"forward"`, or `"backward"`.
- `ArgumentError`: If `path` cannot be deduced from `Λ` when `path` is `""`.
- `ArgumentError`: If `Λ` contains duplicate entries.
- `ArgumentError`: If `Λ` is not a contiguous set of integers.
- `ArgumentError`: If `Λ` is out of the valid range for the indices of `W`.
- `ArgumentError`: If `Λ` is neither sorted in ascending nor in descending order, leading to inconsitency with either \"forward\" or \"backward\" `path`.
- `ArgumentError`: If `Λ` is not sorted in a consistent order with the `path` (ascending order matches "forward"` `path`; descending order matches `"backward"` `path`).
"""
function decqr!(W::Dec{T,N}, Λ::Indices; pivot::Bool=false, path::String="", returnRfactors::Bool=false) where {T<:FloatRC,N}
	L = declength(W); decrank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if path ∉ ("","forward","backward")
		throw(ArgumentError("path should be \"\" (default, accepted only when path can be deduced from Λ), \"forward\" or \"backward\""))
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
	if returnRfactors
		Rfactors = Vector{Matrix{T}}()
	end
	for λ ∈ 1:M
		ℓ = Λ[λ]
		W[ℓ],R = factorqr!(W[ℓ], Val(pivot); rev=(path == "backward"))
		if returnRfactors
			Rfactor = (λ < M) ? R : copy(R)
			if path == "backward"
				pushfirst!(Rfactors, reshape(Rfactor, size(Rfactor, 1), size(Rfactor, N)))
			else
				push!(Rfactors, reshape(Rfactor, size(Rfactor, 1), size(Rfactor, N)))
			end
		end
		if λ < M
			ν = Λ[λ+1]
			W[ν] = factorcontract(R, W[ν], rev=(path == "backward"))
		else
			decinsert!(W, ℓ, R; path=path, rankprecheck=false, rankpostcheck=true)
		end
	end
	if returnRfactors
		return W,Rfactors
	end
	return W
end

"""
    decqr!(W::Dec{T,N}; pivot::Bool=false, path::String="", returnRfactors::Bool=false) where {T<:FloatRC,N}

Perform a QR decomposition on all factors of a decomposition object (vector of factors) `W`, using optional pivoting and following the specified contraction path.

# Arguments
- `W::Dec{T, N}`: Decomposition object of type `Dec`, which represents a vector of factors with elements of type `T` and `N` dimensions.
- `pivot::Bool=false`: Keyword argument indicating whether to apply pivoting during the QR decomposition.
- `path::String=""`: Specifies the contraction path for the decomposition, either `""` (default), `"forward"`, or `"backward"`. 
  - `""`: The path is deduced from the indices if possible.
  - `"forward"`: Decomposes in a forward sequence.
  - `"backward"`: Decomposes in a backward sequence.
- `returnRfactors::Bool=false`: Whether to return the `R` factors from the QR decompositions.

# Returns
- `W`: Modified decomposition object after applying the QR decomposition.
- `Rfactors` (optional): If `returnRfactors` is set to `true`, a vector of `Matrix{T}` containing the `R` factors from the QR decomposition is returned.

# Throws
- `ArgumentError`: If the decomposition object `W` is empty (`L == 0`).
- `ArgumentError`: If `path` is neither `"forward"` nor `"backward"`.
- `ArgumentError`: If path is `""`, because the path cannot be deduced when `Λ` is a colon.
"""
decqr!(W::Dec{T,N}; pivot::Bool=false, path::String="", returnRfactors::Bool=false) where {T<:FloatRC,N} = decqr!(W, :; pivot=pivot, path=path, returnRfactors=returnRfactors)



"""
    decsvd!(W::Dec{T,N}, Λ::Indices, n::Union{Colon,DecSize}; 
            path::String="", 
            soft::Float2{AbstractFloat}=zero(S), 
            hard::Float2{AbstractFloat}=zero(S), 
            aTol::Float2{AbstractFloat}=zero(S), 
            aTolDistr::Float2{AbstractFloat}=zero(S), 
            rTol::Float2{AbstractFloat}=zero(S), 
            rTolDistr::Float2{AbstractFloat}=zero(S), 
            rank::Int2=0, 
            major::String="last")
	where {S<:AbstractFloat,T<:FloatRC{S},N}

Perform a truncated Singular Value Decomposition (SVD) on a decomposition object (vector of factors) `W`.

# Arguments
- `W::Dec{T,N}`: Decomposition object of type `Dec`, which represents a vector of factors with elements of type `T` and with `N` as the number dimensions.
- `Λ::Indices`: Reference number or a vector of reference numbers specifying the factors of `W` to be decomposed.
- `n::Union{Colon,DecSize}`: Specifies the mode sizes for the factors to be decomposed. If `Colon`, the mode sizes of the factors are deduced from `W`.
- `path::String=""`: Specifies the path for decomposition, either `"forward"` or `"backward"`. Defaults to `""`, which means the path will be automatically deduced.
- `soft::Float2{AbstractFloat}=zero(S)`: Soft threshold for truncation, either non-negative scalar or vector of such (one per decomposition step).
- `hard::Float2{AbstractFloat}=zero(S)`: Hard threshold for truncation, either non-negative scalar or vector of such, (one per decomposition step).
- `aTol::Float2{AbstractFloat}=zero(S)`: Absolute tolerance for SVD, either non-negative scalar or vector of such, (one per decomposition step).
- `aTolDistr::Float2{AbstractFloat}=zero(S)`: Distributed absolute tolerance for SVD, either a non-negative scalar or a vector of such, (one per decomposition step).
- `rTol::Float2{AbstractFloat}=zero(S)`: Relative tolerance for SVD, either a non-negative scalar or a vector of such, (one per decomposition step).
- `rTolDistr::Float2{AbstractFloat}=zero(S)`: Distributed relative tolerance for SVD, either a non-negative scalar or a vector of such, (one per decomposition step).
- `rank::Int2=0`: Rank for truncation, either a non-negative integer or a vector of such, (one per decomposition step).
- `major::String="last"`: Specifies the major ordering of the modes in the vector of factors. Must be either `"first"` or `"last"`. Default is set to `"last"`.

# Returns
- `W`: Updated decomposition object with SVD applied.
- `ε`: Vector of float numbers representing the computed truncation error for each decomposition step.
- `δ`: Vector of float numbers representing the relative error for each decomposition step.
- `μ`: Norm of the decomposition object.
- `ρ`: Vector of integers representing the ranks after truncation for each decomposition step.
- `σ`: Vector of vectors containing the singular values for each decomposition step.

# Errors

Summarized Error List:
- `ArgumentError`: If any input arguments are invalid, for example incorrect dimensions, negative values where non-negative are expected or invalid strings for path or major.

Extended Error List:
- `ArgumentError`: If the decomposition is empty.
- `ArgumentError`: If path is none of the following: \"\" (default, accepted only when path can be deduced from Λ), \"forward\" or \"backward\".
- `ArgumentError`: If path cannot be deduced from Λ or is the path is neither \"forward\" nor \"backward\" when Λ is a colon.
- `ArgumentError`: If the entries of Λ do not form a set of contiguous integers.
- `ArgumentError`: If Λ is out of range.
- `ArgumentError`: If Λ is not sorted in ascending or descending order, rendering it inconsistent with any forward or backward path.
- `ArgumentError`: If Λ is not sorted in ascending order, rendering it inconsistent with any forward path.
- `ArgumentError`: If Λ is not sorted in descending order, rendering it inconsistent with any backward path.
- `ArgumentError`: If the number of rows in n is not equal to the number of dimensions in each factor of W.
- `ArgumentError`: If the number of columns in n is not equal to the number of elements in Λ.
- `ArgumentError`: If n and Λ are incompatible with the size of the factors of W.
- `ArgumentError`: If soft is not a nonnegative Float or a vector of such.
- `ArgumentError`: If soft, passed as a vector, has incorrect length.
- `ArgumentError`: If hard is not a nonnegative Float or a vector of such.
- `ArgumentError`: If hard, passed as a vector, has incorrect length.
- `ArgumentError`: If aTol is not a nonnegative Float64 or a vector of such.
- `ArgumentError`: If aTol, passed as a vector, has incorrect length.
- `ArgumentError`: If aTolDistr is not a nonnegative Float64 or a vector of such.
- `ArgumentError`: If aTolDistr, passed as a vector, has incorrect length.
- `ArgumentError`: If rTol is not a nonnegative Float or a vector of such.
- `ArgumentError`: If rTol, passed as a vector, has incorrect length.
- `ArgumentError`: If rTolDistr is not a nonnegative Float or a vector of such.
- `ArgumentError`: If rTolDistr, passed as a vector, has incorrect length.
- `ArgumentError`: If rank is not a nonnegative Int or a vector of such.
- `ArgumentError`: If rank, passed as a vector, has incorrect length.
- `ArgumentError`: If major is neither \"last\" (default) nor \"first\".

# Example

```julia
W = Dec([...])  # Initialize a Dec object with appropriate tensor decomposition
Λ = [1, 3, 5]  # Indices specifying factors to be decomposed (here: assume that there are at least five factors in our initialized decomposition)

# Perform truncated SVD on the decomposition object W (mode sizes shall be deduced from `Dec` object `W`, so specify `n` as a colon)
W, ε, δ, μ, ρ, σ = decsvd!(W, Λ, :; path="forward", soft=0.1, hard=0.01, aTol=1e-5)
"""
function decsvd!(W::Dec{T,N}, Λ::Indices, n::Union{Colon,DecSize}; path::String="", soft::Float2{AbstractFloat}=zero(S), hard::Float2{AbstractFloat}=zero(S), aTol::Float2{AbstractFloat}=zero(S), aTolDistr::Float2{AbstractFloat}=zero(S), rTol::Float2{AbstractFloat}=zero(S), rTolDistr::Float2{AbstractFloat}=zero(S), rank::Int2=0, major::String="last") where {S<:AbstractFloat,T<:FloatRC{S},N}
	# assumes that the decomposition is orthogonal
	L = declength(W); decrank(W)
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
	m = decsize(W)
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
	#
	if any(soft .< 0)
		throw(ArgumentError("soft should be a nonnegative Float or a vector of such"))
	end
	if isa(soft, S)
		soft = soft*ones(S, K)
	elseif length(soft) ≠ K
		throw(ArgumentError("soft, passed as a vector, has incorrect length"))
	end

	if any(hard .< 0)
		throw(ArgumentError("hard should be a nonnegative Float or a vector of such"))
	end
	if isa(hard, S)
		hard = hard*ones(S, K)
	elseif length(hard) ≠ K
		throw(ArgumentError("hard, passed as a vector, has incorrect length"))
	end

	if any(aTol .< 0)
		throw(ArgumentError("aTol should be a nonnegative Float or a vector of such"))
	end
	if isa(aTol, S)
		aTol = aTol*ones(S, K)/sqrt(one(S)*K)
	elseif length(aTol) ≠ K
		throw(ArgumentError("aTol, passed as a vector, has incorrect length"))
	end

	if any(aTolDistr .< 0)
		throw(ArgumentError("aTolDistr should be a nonnegative Float64 or a vector of such"))
	end
	if isa(aTolDistr, S)
		aTolDistr = aTolDistr*ones(S, K)/sqrt(one(S)*K)
	elseif length(aTolDistr) ≠ K
		throw(ArgumentError("aTolDistr, passed as a vector, has incorrect length"))
	end

	if any(rTol .< 0)
		throw(ArgumentError("rTol should be a nonnegative Float or a vector of such"))
	end
	if isa(rTol, S)
		rTol = rTol*ones(S, K)/sqrt(one(S)*K)
	elseif length(rTol) ≠ K
		throw(ArgumentError("rTol, passed as a vector, has incorrect length"))
	end

	if any(rTolDistr .< 0)
		throw(ArgumentError("rTolDistr should be a nonnegative Float or a vector of such"))
	end
	if isa(rTolDistr, S)
		rTolDistr = rTolDistr*ones(S, K)/sqrt(one(S)*K)
	elseif length(rTolDistr) ≠ K
		throw(ArgumentError("rTolDistr, passed as a vector, has incorrect length"))
	end

	if any(rank .< 0)
		throw(ArgumentError("rank should be a nonnegative Int or a vector of such"))
	end
	if isa(rank, Int)
		rank = rank*ones(Int, K)
	elseif length(rank) ≠ K
		throw(ArgumentError("rank, passed as a vector, has incorrect length"))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end

	ε = zeros(S, K); δ = zeros(S, K)
	σ = Vector{Vector{S}}(undef, K)
	ρ = zeros(Int, K)
	μ = zero(S)
	aTolAcc = zero(S); rTolAcc = zero(S)
	for λ ∈ 1:K
		ℓ = Λ[λ]
		if λ == 1
			ε₁ = [aTol[λ],aTolDistr[λ]]; ε₁ = ε₁[ε₁ .> 0]; ε₁ = isempty(ε₁) ? zero(S) : minimum(ε₁)
			δ₁ = [rTol[λ],rTolDistr[λ]]; δ₁ = δ₁[δ₁ .> 0]; δ₁ = isempty(δ₁) ? zero(S) : minimum(δ₁)
			U,V,ε[1],δ[1],μ,ρ[1],σ[1] = factorsvd!(W[ℓ], n[:,λ], :; soft=soft[λ], hard=hard[λ], atol=ε₁, rtol=δ₁, rank=rank[λ], rev=(path == "backward"), major=major)
		else
			(aTolDistr[λ] > 0) && (aTolDistr[λ] = sqrt(aTolDistr[λ]^2+aTolAcc^2); aTolAcc = zero(S))
			(rTolDistr[λ] > 0) && (rTolDistr[λ] = sqrt(rTolDistr[λ]^2+rTolAcc^2); rTolAcc = zero(S))
			ε₁ = [aTol[λ],aTolDistr[λ],μ*rTol[λ],μ*rTolDistr[λ]]; ε₁ = ε₁[ε₁ .> 0]
			ε₁ = isempty(ε₁) ? zero(S) : minimum(ε₁)
			U,V,ε[λ],_,_,ρ[λ],σ[λ] = factorsvd!(W[ℓ], n[:,λ], :; soft=soft[λ], hard=hard[λ], atol=ε₁, rank=rank[λ], rev=(path == "backward"), major=major)
			δ[λ] = (μ > 0) ? ε[λ]/μ : zero(S)
		end
		W[ℓ] = U
		if λ < K && Λ[λ+1] ≠ ℓ
			ν = Λ[λ+1]
			W[ν] = factorcontract(V, W[ν], rev=(path == "backward"), major=major)
		else
			decinsert!(W, ℓ, V; path=path, rankprecheck=false, rankpostcheck=true)
			(path == "forward") && (Λ[λ:end] .+= 1)
		end
	end
	return W,ε,δ,μ,ρ,σ
end

"""
    decsvd!(W::Dec{T,N}, Λ::Indices; path::String="", soft::Float2{AbstractFloat}=zero(S), 
            hard::Float2{AbstractFloat}=zero(S), aTol::Float2{AbstractFloat}=zero(S), 
            aTolDistr::Float2{AbstractFloat}=zero(S), rTol::Float2{AbstractFloat}=zero(S), 
            rTolDistr::Float2{AbstractFloat}=zero(S), rank::Int2=0, major::String="last") 
    where {S<:AbstractFloat, T<:FloatRC{S}, N}

Perform a truncated Singular Value Decomposition (SVD) on the components of a decomposition object (vector of factors) `W` with specified indices `Λ`, applying truncation based on thresholds, tolerances, and ranks.
Function deduces mode sizes from the `Dec` object `W`.

# Arguments
- `W::Dec{T, N}`: Decomposition object of type `Dec`, representing a vector of factors with elements of type `T` and `N` dimensions.
- `Λ::Indices`: Indices specifying which factors of `W` to apply the SVD to.
- `path::String=""`: Specifies the contraction path for the decomposition. Can be `"forward"`, `"backward"`, or `""` (default, where the path is deduced).
- `soft::Float2{AbstractFloat}=zero(S)`: Soft threshold for truncation.
- `hard::Float2{AbstractFloat}=zero(S)`: Hard threshold for truncation.
- `aTol::Float2{AbstractFloat}=zero(S)`: Absolute tolerance for SVD.
- `aTolDistr::Float2{AbstractFloat}=zero(S)`: Distributed absolute tolerance for SVD.
- `rTol::Float2{AbstractFloat}=zero(S)`: Relative tolerance for SVD.
- `rTolDistr::Float2{AbstractFloat}=zero(S)`: Distributed relative tolerance for SVD.
- `rank::Int2=0`: Rank for truncation.
- `major::String="last"`: Specifies whether to focus on the "first" or "last" mode during truncation. Defaults to `"last"`.

# Returns
- `W`: Modified decomposition object after performing the truncated SVD.
- `ε`: Vector of truncation errors for each step.
- `δ`: Vector of relative errors for each step.
- `μ`: Norm of the decomposition object.
- `ρ`: Vector of post-truncation ranks for each step.
- `σ`: Vector of vectors containing the singular values for each step.

# Throws
- `ArgumentError`: If any arguments are invalid, for example incorrect dimensions or invalid thresholds.
- For an extended error list, see the default version of [`decsvd!`](@ref).
"""
decsvd!(W::Dec{T,N}, Λ::Indices; path::String="", soft::Float2{AbstractFloat}=zero(S), hard::Float2{AbstractFloat}=zero(S), aTol::Float2{AbstractFloat}=zero(S), aTolDistr::Float2{AbstractFloat}=zero(S), rTol::Float2{AbstractFloat}=zero(S), rTolDistr::Float2{AbstractFloat}=zero(S), rank::Int2=0, major::String="last") where {S<:AbstractFloat,T<:FloatRC{S},N} = decsvd!(W, Λ, :; path=path, soft=soft, hard=hard, aTol=aTol, aTolDistr=aTolDistr, rTol=rTol, rTolDistr=rTolDistr, rank=rank, major=major)

"""
    decsvd!(W::Dec{T,N}; path::String="", soft::Float2{AbstractFloat}=zero(S), 
            hard::Float2{AbstractFloat}=zero(S), aTol::Float2{AbstractFloat}=zero(S), 
            aTolDistr::Float2{AbstractFloat}=zero(S), rTol::Float2{AbstractFloat}=zero(S), 
            rTolDistr::Float2{AbstractFloat}=zero(S), rank::Int2=0, major::String="last") 
    where {S<:AbstractFloat, T<:FloatRC{S}, N}

Perform a truncated Singular Value Decomposition (SVD) on all components of a decomposition object (vector of factors) `W` with optional truncation based on thresholds, tolerances, and ranks.

# Arguments
- `W::Dec{T, N}`: Decomposition object of type `Dec`, representing a vector of factors with elements of type `T` and `N` dimensions.
- `path::String=""`: Specifies the contraction path for the decomposition. Can be `"forward"`, `"backward"`, or `""` (default, where the path is deduced).
- `soft::Float2{AbstractFloat}=zero(S)`: Soft threshold for truncation.
- `hard::Float2{AbstractFloat}=zero(S)`: Hard threshold for truncation.
- `aTol::Float2{AbstractFloat}=zero(S)`: Absolute tolerance for SVD.
- `aTolDistr::Float2{AbstractFloat}=zero(S)`: Distributed absolute tolerance for SVD.
- `rTol::Float2{AbstractFloat}=zero(S)`: Relative tolerance for SVD.
- `rTolDistr::Float2{AbstractFloat}=zero(S)`: Distributed relative tolerance for SVD.
- `rank::Int2=0`: Rank for truncation.
- `major::String="last"`: Specifies whether to focus on the "first" or "last" mode during truncation. Defaults to `"last"`.

# Returns
- `W`: Modified decomposition object after performing the truncated SVD.
- `ε`: Vector of truncation errors for each step.
- `δ`: Vector of relative errors for each step.
- `μ`: Norm of the decomposition object.
- `ρ`: Vector of post-truncation ranks for each step.
- `σ`: Vector of vectors containing the singular values for each step.

# Throws
- `ArgumentError`: If any arguments are invalid, such as incorrect dimensions, invalid thresholds, or mismatched sizes for paths or ranks.
- For an extended error list, see the default version of [`decsvd!`](@ref).
"""
decsvd!(W::Dec{T,N}; path::String="", soft::Float2{AbstractFloat}=zero(S), hard::Float2{AbstractFloat}=zero(S), aTol::Float2{AbstractFloat}=zero(S), aTolDistr::Float2{AbstractFloat}=zero(S), rTol::Float2{AbstractFloat}=zero(S), rTolDistr::Float2{AbstractFloat}=zero(S), rank::Int2=0, major::String="last") where {S<:AbstractFloat,T<:FloatRC{S},N} = decsvd!(W, :, :; path=path, soft=soft, hard=hard, aTol=aTol, aTolDistr=aTolDistr, rTol=rTol, rTolDistr=rTolDistr, rank=rank, major=major)
