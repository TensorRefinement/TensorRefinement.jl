module Auxiliary

using LinearAlgebra

export Indices, FloatRC, Float2, Float3, Int2, Int3
export indvec, threshold, compfloateps, modemul
export qraddcols!, qraddcols, lqaddrows


"""
`Indices` is an alias for an object of type `Vector{Int}`, `Int`, `UnitRange{Int}`, `StepRange{Int}` or `NTuple{M,Int}`.
"""
const Indices = Union{Vector{Int},Int,UnitRange{Int},StepRange{Int},NTuple{M,Int} where M,Vector{Any},Nothing,Colon}

"""
`FloatRC{T}` is an alias for an object of type `T` or `Complex{T}`, where T is an `AbstractFloat`.
"""
const FloatRC{T} = Union{T,Complex{T}} where T<:AbstractFloat

"""
`Float2{T}` is an alias for an object of type `T` or `Vector{T}`, where T is an `AbstractFloat`.
"""
const Float2{T} = Union{T,Vector{T}} where T<:AbstractFloat

"""
`Float3{T}` is an alias for an object of type `T`, `Vector{T}` or `Vector{Vector{T}}`, where T is an `AbstractFloat`.
"""
const Float3{T} = Union{T,Vector{T},Vector{Vector{T}}} where T<:AbstractFloat

"""
`Int2` is an alias for an object of type `Int` or `Vector{Int}`.
"""
const Int2 = Union{Int,Vector{Int}}

"""
`Int3` is an alias for an object of type `Int`, `Vector{Int}` or `Vector{Vector{Int}}`.
"""
const Int3 = Union{Int,Vector{Int},Vector{Vector{Int}}}

"""
    compfloateps(::Type{S}) where {T<:AbstractFloat, S<:FloatRC{T}}

Return the machine epsilon (smallest difference between distinct floating point numbers) for the floating-point type `T`.

# Arguments
- `S::Type{S}`: A type that is a subtype of `FloatRC{T}`, where `T` is a floating-point type (`AbstractFloat`).

# Returns
- `T`: The machine epsilon of the floating-point type `T`.
"""
compfloateps(::Type{S}) where {T<:AbstractFloat,S<:FloatRC{T}} = eps(T)

"""
    two(::Type{T}) where T<:Number

Return the numeric value `2` as the same type as `one(T)`. Useful for consistent numeric operations involving the number two.

# Arguments
- `::Type{T}`: Numeric type (`Number`).

# Returns
- `T`: Number `2` as type `T`.
"""
two(::Type{T}) where T<:Number = 2*one(T)

"""
    indvec(σ::Indices; min::Int=1, max::Int=0)

Convert several index specifications to a vector of integers. Function supports Julia indexing types like `Colon`, `Int`, `UnitRange`, `StepRange`, `Tuple`, and `Nothing`.

# Arguments
- `σ::Indices`: Index specification. Can be of various types such as `Colon`, `Int`, `UnitRange`, `StepRange`, `Tuple`, `Vector{Any}`, or `Nothing`.
- `min::Int=1`: Minimum value for the range (applicable when `σ` is `Colon`).
- `max::Int=0`: Maximum value for the range (applicable when `σ` is `Colon`).

# Returns
- `Vector{Int}`: Vector of integers derived from the specified index.
"""
function indvec(σ::Indices; min::Int=1, max::Int=0)
	isa(σ, Colon) && return collect(min:max)
	isa(σ, Vector{Int}) && return copy(σ)
	isa(σ, Int) && return [σ]
	isa(σ, UnitRange{Int}) && return collect(σ)
	isa(σ, StepRange{Int,S} where S) && return collect(σ)
	isa(σ, Tuple{}) && return Vector{Int}(undef, 0)
	isa(σ, NTuple{N,Int} where N) && return collect(σ)
	isa(σ, Vector{Any}) && return Vector{Int}(undef, 0)
	isa(σ, Nothing) && return Vector{Int}(undef, 0)
end

"""
    threshold(c::Vector{T}, τsoft::T, τhard::T, τ::T, r::Int)

Apply thresholding to the vector `c` based on soft, hard, and overall thresholds (`τsoft`, `τhard`, and `τ`). Optional truncation of the vector to maintain a specified rank `r` available.

# Arguments
- `c::Vector{T}`: Vector of elements of Type `T`.
- `τsoft::T`: Soft threshold value. All elements less than or equal to this value are set to zero, all above are shrunken towards zero.
- `τhard::T`: Hard threshold value. All elements less than or equal to this value are set to zero.
- `τ::T`: Overall tolerance threshold. Elements are truncated based on this value.
- `r::Int`: Desired rank of the vector. Only the first `r` elements are kept, all others are being set to zero.

# Returns
- `Tuple{Vector{T}, T, Int}`: Tuple containing the thresholded vector, the error (ε), and the new rank (ρ).

# Throws
- `ArgumentError`: If `τsoft` is negative.
- `ArgumentError`: If `τhard` is negative.
- `ArgumentError`: If `τ` is negative.
- `ArgumentError`: If `r` is negative.
"""
function threshold(c::Vector{T}, τsoft::T, τhard::T, τ::T, r::Int) where {T<:Real}
	if τsoft < 0
		throw(ArgumentError("τsoft is required to be nonnegative"))
	end
	if τhard < 0
		throw(ArgumentError("τhard is required to be nonnegative"))
	end
	if τ < 0
		throw(ArgumentError("τ is required to be nonnegative"))
	end
	if r < 0
		throw(ArgumentError("r is required to be nonnegative"))
	end
	n = length(c)
	ε = zero(T)
	ρ = n
	if n > 0
		@inbounds begin
			τ² = τ^2
			soft = τsoft > 0
			hard = τhard > 0
			tol = τ > 0
			if 0 < r < n
				for k ∈ n:-1:r+1
					ε += c[k]^2
				end
				ρ = r
			end
			for k ∈ ρ:-1:1
				s = ε+c[k]^2
				if tol
					if s ≤ τ²
						ε = s; ρ = k-1
						continue
					else
						tol = false
					end
				end
				if hard
					if c[k] ≤ τhard
						ε = s; ρ = k-1
						continue
					else
						hard = false
					end
				end
				if soft
					if c[k] ≤ τsoft
						ε = s; ρ = k-1
						continue
					else
						ε += k*τsoft^2
						soft = false
						break
					end
				end
			end
		end
		c = c[1:ρ]
		if τsoft > 0
			c .-= τsoft
		end
		ε = sqrt(ε)
	end
	c,ε,ρ
end

"""
    qraddcols(Q::Matrix{T}, R::Matrix{T}, U::Matrix{T}) where T<:FloatRC

Extend the QR decomposition by adding additional columns. Given an existing QR decomposition
`Q * R = A` and a new matrix `U`, this function computes the QR decomposition of `[A U]`.

# Arguments
- `Q::Matrix{T}`: Orthogonal matrix from the QR decomposition of type `T`.
- `R::Matrix{T}`: Upper triangular matrix from the QR decomposition of type `T`.
- `U::Matrix{T}`: Matrix to be added to the existing QR decomposition of type `T`.

# Returns
- `Q`: Updated orthogonal matrix after adding new columns.
- `R`: Updated upper triangular matrix after adding new columns.

# Throws
- `ArgumentError`: If `Q` and `R` are incompatible in rank.
- `ArgumentError`: If `Q` and `U` are incompatible in the first rank.
"""
function qraddcols(Q::Matrix{T}, R::Matrix{T}, U::Matrix{T}) where T<:FloatRC
	p,r = size(Q); rR,q = size(R); pU,s = size(U)
	if rR ≠ r
		throw(ArgumentError("Q and R are incompatible in rank"))
	end
	if pU ≠ p
		throw(ArgumentError("Q and U are incompatible in the first rank"))
	end
	Q = [Q U]
	Q,Z = qr!(Q); Q = Matrix(Q); ρ = size(Z, 1)
	d = diag(view(Z, 1:r, 1:r))
	Q[:,1:r] = Q[:,1:r]*Diagonal(d)
	d = conj(d)
	R = [R Diagonal(d)*Z[1:r,r+1:r+s]; zeros(T, ρ-r, q) Z[r+1:ρ,r+1:r+s]]
	return Q,R
end

"""
    lqaddrows(L::Matrix{T}, Q::Matrix{T}, U::Matrix{T}) where T<:FloatRC

Extend the LQ decomposition by adding additional rows. Given an existing LQ decomposition
`L * Q = A` and a new matrix `U`, this function computes the LQ decomposition of `[L U]'`.
Currently, this function is not implemented.

# Arguments
- `L::Matrix{T}`: Lower triangular matrix from the LQ decomposition of type `T`.
- `Q::Matrix{T}`: Orthogonal matrix from the LQ decomposition of type `T`.
- `U::Matrix{T}`: Matrix to be added to the existing QR decomposition of type `T`.

# Returns
- `L`: Updated lower triangular matrix after adding new rows. 
- `Q`: Updated orthogonal matrix after adding new rows.

# Throws
- `ArgumentError`: If `L` and `Q` are incompatible in size.
- `ArgumentError`: If `Q` and `U` are incompatible in size.
"""
function lqaddrows(L::Matrix{T}, Q::Matrix{T}, U::Matrix{T}) where T<:FloatRC
	p,rr = size(L); r,q = size(Q); s,qq = size(U)
	if rr ≠ r
		throw(ArgumentError("L and Q are incompatible in size"))
	end
	if qq ≠ q
		throw(ArgumentError("Q and U are incompatible in size"))
	end
	throw(ErrorException("NOT IMPLEMENTED"))
	return Q,R
end

"""
    _mode2mul(A::AbstractArray{<:Number,3}, B::AbstractMatrix{<:Number})

Perform a mode-2 multiplication of a 3-dimensional array `A` with a matrix `B`.
Effectively, this operation multiplies `B` with each "slice" of the 3D array `A` along its second dimension (mode-2).

# Arguments
- `A::AbstractArray{<:Number,3}`: 3-dimensional array (tensor) of type `Number`.
- `B::AbstractMatrix{<:Number}`: Matrix of type `Number` to multiply with the unfolded tensor `A`.

# Returns
- `A`: Result of the mode-2 multiplication, reshaped back to a 3-dimensional array.
- `mm`: Number of rows in matrix `B`, which eventually becomes the size of the second dimension of the returned `A`.
"""
function _mode2mul(A::AbstractArray{<:Number,3}, B::AbstractMatrix{<:Number})
	k,m,n = size(A)
	A = permutedims(A, [2,1,3])
	mm = size(B, 1)
	A = reshape(B*reshape(A, m, k*n), mm, k, n)
	A = permutedims(A, [2,1,3])
	A,mm
end

"""
    modemul(A::AbstractArray{<:Number,N}, B::Vararg{Pair{Int,<:AbstractMatrix{<:Number}},M}) where {N,M}

Perform a mode multiplication of an N-dimensional array `A` with multiple matrices, each matrix corresponding to a specific mode (dimension) of `A`.
Function applies several mode multiplications to the tensor `A`, such that it is transformed according to the provided matrices.

# Arguments
- `A::AbstractArray{<:Number,N}`: N-dimensional array of type `Number`.
- `B::Vararg{Pair{Int,<:AbstractMatrix{<:Number}},M}`: Variable number of pairs `(k, Bk)`, where `k` is the dimension index and `Bk` is a matrix to multiply `A` with along its `k`-th dimension.

# Returns
- `A`: The result of applying the mode multiplications, eventually, reshaped back to an N-dimensional array.

# Throws
- `ArgumentError`: If any dimension index is out of bounds 1:ndims(A), or if any dimension indices are not distinct.
- `ArgumentError`: If any matrix does not match the corresponding dimension size of the tensor.
- `ArgumentError`: If the dimenion indices are not distinct.
"""
function modemul(A::AbstractArray{<:Number,N}, B::Vararg{Pair{Int,<:AbstractMatrix{<:Number}},M}) where {N,M}
	n = collect(size(A))
	# if any(n .≤ 0)
	# 	throw(ArgumentError("the dimensions of the input tensor should all be positive"))
	# end
	d = length(n)
	m = length(B)
	dims = collect(B[i][1] for i ∈ 1:m)
	factors = collect(B[i][2] for i ∈ 1:m)
	for i ∈ 1:m
		k,Bk = dims[i],factors[i]
		if k ∉ 1:d
			throw(ArgumentError("each dimension index should be between 1 and ndims(A); the $(i)th one is $k while ndims(A)=$d"))
		end
		if size(Bk, 2) ≠ n[k]
			throw(ArgumentError("factor $i (with dimension index $k) is inconsistent with the tensor in size"))
		end
	end
	if length(unique(dims)) < m
		throw(ArgumentError("the dimension indices should be distinct"))
	end
	for (k,Bk) ∈ B
		A = reshape(A, prod(n[1:k-1]), n[k], prod(n[k+1:end]))
		A,n[k] = _mode2mul(A, Bk)
	end
	reshape(A, n...)
end


end
