module Legendre

using TensorRefinement.Auxiliary, ..TensorTrain, LinearAlgebra

export legeval, legneval, legtolegn, legntoleg, legdiff, legndiff, legref, legnref, legdec, legdeceval!

"""
    legtolegn(::Type{T}, r::Int) where {T<:AbstractFloat}

Construct a diagonal matrix that transforms coefficients in the Legendre polynomial basis 
to coefficients in the normalized Legendre polynomial basis.

# Arguments
- `::Type{T}`: Numeric type (subtype of `AbstractFloat`) for the matrix elements.
- `r::Int`: Size of the matrix.

# Returns
- `Diagonal{T}`: Diagonal matrix of size `r` x `r` containing the scaling factors 
  to convert from Legendre polynomials to normalized Legendre polynomials.

# Throws
- `ArgumentError`: If `r` is negative.
"""
function legtolegn(::Type{T}, r::Int) where {T<:AbstractFloat}
	u = [ 1/sqrt(j+one(T)/2) for j ∈ 0:r-1 ]
	Diagonal(u)
end

"""
    legntoleg(::Type{T}, r::Int) where {T<:AbstractFloat}

Construct a diagonal matrix that transforms coefficients in the normalized Legendre polynomial basis 
to coefficients in the standard Legendre polynomial basis.

# Arguments
- `::Type{T}`: Numeric type (subtype of `AbstractFloat`) for the matrix elements.
- `r::Int`: Size of the matrix.

# Returns
- `Diagonal{T}`: Diagonal matrix of size `r` x `r` containing the scaling factors 
  to convert from normalized Legendre polynomials to Legendre polynomials.

# Throws
- `ArgumentError`: If `r` is negative.
"""
function legntoleg(::Type{T}, r::Int) where {T<:AbstractFloat}
	u = [ sqrt(j+one(T)/2) for j ∈ 0:r-1 ]
	Diagonal(u)
end

"""
    legeval(t::Vector{T}, r::Int) where {T<:AbstractFloat}

Evaluate the first `r` Legendre polynomials at the points specified in the vector `t`.

# Arguments
- `t::Vector{T}`: Vector of points at which to evaluate the Legendre polynomials.
- `r::Int`: Number of Legendre polynomials to evaluate.

# Returns
- `Matrix{T}`: An `n` x `r` matrix `V`, where `n` is the length of `t`. The entry `V[i,j]` 
  contains the value of the `(j-1)`-th Legendre polynomial evaluated at `t[i]`.

# Throws
- `ArgumentError`: If `r` is negative.
"""
function legeval(t::Vector{T}, r::Int) where {T<:AbstractFloat}
	n = length(t)
	V = zeros(T, n, r)
	V[:,1] .= 1
	if r > 1
		V[:,2] .= t
	end
	for k ∈ 1:r-2
		V[:,k+2] .= (2-one(T)/(k+1)).*t.*V[:,k+1].-(1-one(T)/(k+1)).*V[:,k]
	end
	V
end

"""
    legneval(t::Vector{T}, r::Int) where {T<:AbstractFloat}

Evaluate the first `r` normalized Legendre polynomials at the points specified in the vector `t`.

# Arguments
- `t::Vector{T}`: Vector of points at which to evaluate the normalized Legendre polynomials.
- `r::Int`: Number of normalized Legendre polynomials to evaluate.

# Returns
- `Matrix{T}`: An `n` x `r` matrix `V`, where `n` is the length of `t`. The entry `V[i,j]` 
  contains the value of the `(j-1)`-th normalized Legendre polynomial evaluated at `t[i]`.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
legneval(t::Vector{T}, r::Int) where {T<:AbstractFloat} = legeval(t, r)*legntoleg(T, r)

"""
    legdiff(::Type{T}, r::Int) where {T<:AbstractFloat}

Construct a differentiation matrix for Legendre polynomials of degree up to `r-1`.

# Arguments
- `::Type{T}`: Numeric type (subtype of `AbstractFloat`) for the matrix elements.
- `r::Int`: Number of degrees of freedom (DOFs), i.e., the size of the matrix.

# Returns
- `Matrix{T}`: Matrix of size `r` x `r` representing the differentiation operator for Legendre polynomials of degree up to `r-1`.

# Throws
- `ArgumentError`: If `r` is not positive.
"""
function legdiff(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	W = zeros(T, r, r)
	if r == 1
		return W
	end
	for β ∈ 0:r-1, γ ∈ 1:2:β
		W[1+β-γ,1+β] = 2*(β-γ)+1
	end
	W
end

"""
    legndiff(::Type{T}, r::Int) where {T<:AbstractFloat}

Construct a differentiation matrix for normalized Legendre polynomials of degree up to `r-1`.

# Arguments
- `::Type{T}`: Numeric type (subtype of `AbstractFloat`) for matrix elements.
- `r::Int`: Number of degrees of freedom (DOFs), i.e., the size of the matrix.

# Returns
- `Matrix{T}`: Matrix of size `r` x `r` representing the differentiation operator for normalized Legendre polynomials of degree up to `r-1`.

# Throws
- `ArgumentError`: If `r` is not positive.
"""
legndiff(::Type{T}, r::Int) where {T<:AbstractFloat} = legtolegn(T, r)*legdiff(T, r)*legntoleg(T, r)

"""
    legref(ξ::T, η::T, r::Int) where {T<:AbstractFloat}

Compute a matrix related to the Legendre polynomials over a reference interval, parameterized by `ξ` and `η`.

# Arguments
- `ξ::T`: Float representing one of the parameters for the transformation.
- `η::T`: Float representing the other parameter for the transformation.
- `r::Int`: Number of degrees of freedom (DOFs), i.e., the size of the matrix.

# Returns
- `Matrix{T}`: An `r` x `r` matrix `W` that encodes the transformation in the Legendre polynomial basis for given parameters `ξ` and `η`.

# Throws
- `ArgumentError`: If `r` is less than 1.
"""
function legref(ξ::T, η::T, r::Int) where {T<:AbstractFloat}
	W = zeros(T, r, r)
	W[1,1] = 1
	if r == 1
		return W
	end
	V = legeval([η-ξ, η+ξ], r+1)
	V = V[2,:]-V[1,:]
	W[1,2:r] = V[3:r+1]-V[1:r-1]
	W[1,2:r] = W[1,2:r]./(2*ξ*(2*collect(1:r-1).+1))
	for β ∈ 1:r-1
		W[1+β,1+β] = ξ^β
	end
	for γ ∈ 1:r-1, β ∈ 1:r-1-γ
		W[1+β,1+β+γ] = ξ*(β+γ-one(T)/2)/(β-one(T)/2)*β/(β+γ)*W[1+β-1,1+β-1+γ] + ξ*(β+γ-one(T)/2)/(β+γ)*(β+1)/(β+3*one(T)/2)*W[1+β+1,1+β+1+γ-2] - (β+γ-one(T))/(β+γ)*W[1+β,1+β+γ-2] + 2*η*(β+γ-one(T)/2)/(β+γ)*W[1+β,1+β+γ-1]
	end
	W
end

"""
    legref(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute a matrix based on the Legendre polynomials that encodes specific transformations or relations.

# Arguments
- `::Type{T}`: Numeric type (subtype of `AbstractFloat`) for matrix elements.
- `r::Int`: Number of degrees of freedom (DOFs), i.e., the size of the matrix.

# Returns
- `Array{T,3}`: An `r` x `r` x `2` array `W` where the first slice (`W[:,:,1]`) represents the transformation matrix in a standard basis 
   and the second slice (`W[:,:,2]`) ...

# Throws
- `ArgumentError`: If `r` is less than 1.
"""
function legref(::Type{T}, r::Int) where {T<:AbstractFloat}
	W = zeros(T, r, r)
	if r ≥ 1
		W[1,1] = 2
	end
	if r ≥ 2
		W[1,2] = 1
	end
	for k ∈ 1:r÷2-1
		W[1,2+2*k] = (-1)*(k-one(T)/2)/(k+1)*W[1,2*k]
	end
	if r ≥ 1
		W[1,:] = W[1,:]/2
	end
	for β ∈ 1:r-1
		W[1+β,1+β] = (one(T)/2)^β
	end
	for γ ∈ 1:r-1, β ∈ 1:r-1-γ
		W[1+β,1+β+γ] = (β+γ-one(T)/2)/(β-one(T)/2)*β/(β+γ)*W[1+β-1,1+β-1+γ]/2 + (β+γ-one(T)/2)/(β+γ)*(β+1)/(β+3*one(T)/2)*W[1+β+1,1+β+1+γ-2]/2 - (β+γ-one(T))/(β+γ)*W[1+β,1+β+γ-2] + (β+γ-one(T)/2)/(β+γ)*W[1+β,1+β+γ-1]
	end
	W = [W W]
	W = reshape(W, r, r, 2)
	for α ∈ 0:r-1, β ∈ 0:r-1
		W[1+α,1+β,1] = W[1+α,1+β,1]*(-1)^(α+β)
	end
	W
end

"""
    lengref(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute a matrix based on the normalized Legendre polynomials that encodes specific transformations or relations.

# Arguments
- `::Type{T}`: Numeric type (subtype of `AbstractFloat`) for matrix elements.
- `r::Int`: Number of degrees of freedom (DOFs), i.e., the size of the matrix.

# Returns
- `Array{T,3}`: An `r` x `r` x `2` array `W` where the first slice (`W[:,:,1]`) represents the transformation matrix in a standard basis 
   and the second slice (`W[:,:,2]`) ...

# Throws
- `ArgumentError`: If `r` is less than 1.
"""
legnref(::Type{T}, r::Int) where {T<:AbstractFloat} = modemul(legref(T, r), Pair(1,legtolegn(T, r)), Pair(2,transpose(legntoleg(T, r))))

for fname ∈ (:legeval,:legneval)
	@eval begin
		($fname)(t::Vector{T}, c::Vector{<:FloatRC{T}}) where {T<:AbstractFloat} = ($fname)(t, length(c))*c
	end
end

for fname ∈ (:legtolegn,:legntoleg)
	@eval begin
		function ($fname)(C::AbstractArray{<:FloatRC{T}}; dims=1) where {T<:AbstractFloat}
			n = size(C)
			if any(n .≤ 0)
				throw(ArgumentError("the dimensions of the coefficient should all be positive"))
			end
			dims = Int.([collect(dims)...])
			if dims ⊈ 1:ndims(C)
				throw(ArgumentError("each element of dims should be between 1 and ndims(C)"))
			end
			if length(unique(dims)) < length(dims)
				throw(ArgumentError("the elements of dims should be unique"))
			end
			modemul(C, (Pair(k,($fname)(T, n[k])) for k ∈ dims)...)
		end
	end
end

for fname ∈ (:legdiff,:legndiff)
	@eval begin
		function ($fname)(C::AbstractArray{<:FloatRC{T}}; dims=1) where {T<:AbstractFloat}
			n = size(C)
			if any(n .≤ 0)
				throw(ArgumentError("the dimensions of the coefficient should all be positive"))
			end
			dims = Int.([collect(dims)...])
			if dims ⊈ 1:ndims(C)
				throw(ArgumentError("each element of dims should be between 1 and ndims(C)"))
			end
			modemul(C, (Pair(k,($fname)(T, n[k])) for k ∈ dims)...)
		end
	end
end

"""
    legdec(c::Vector{T}, L::Int; major::String="last") where {T<:AbstractFloat}

Construct a decomposition based on the Legendre polynomials for a given vector of coefficients `c` and a specified number of factors `L`.

# Arguments
- `c::Vector{T}`: Vector of coefficients for the Legendre polynomials. Length of this vector (`r`) determines the degree of the polynomial (maximal `r-1`).
- `L::Int`: Number of factors in the decomposition.
- `major::String`: Specifies the ordering of the decomposition, either "last" (default) or "first".

# Returns
- `Dec{T,N}`: Decomposition object based on the Legendre polynomials and the given vector of coefficients `c`.

# Throws
- `ArgumentError`: If `major` is neither "first" nor "last".
"""
function legdec(c::Vector{T}, L::Int; major::String="last") where {T<:AbstractFloat}
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	r = length(c)
	W = legref(T, r)
	W = permutedims(W, (2,3,1))
	U = dec(W; len=L)
	C = Matrix{T}(c'); C = factor(C, 1, [])
	decpushfirst!(U, C)
	(major == "last") && decreverse!(U)
	U
end

"""
    legdeceval!(U::Dec{T,N}, t::Vector{T}; major::String="last") where {T<:AbstractFloat,N}

Evaluate a Legendre polynomial decomposition `U` at a set of points `t`, adding the reshaped evaluation matrix as a factor to `U` in place.

# Arguments
- `U::Dec{T,N}`: Decomposition representing a Legendre polynomial; will be evaluated and modified in place.
- `t::Vector{T}`: Vector of points at which to evaluate the polynomial.
- `major::String`: Specifies the ordering of the decomposition, either "last" (default) or "first". 

# Returns
- `Dec{T,N}`: Modified decomposition object `U`, now including a factor based on the reshaped evaluation matrix at points `t`.

# Throws
- `ArgumentError`: If `major` is neither "first" nor "last".
"""
function legdeceval!(U::Dec{T,N}, t::Vector{T}; major::String="last") where {T<:AbstractFloat,N}
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	n = length(t)
	L = declength(U)
	p = decrank(U)
	p = (major == "last") ? p[1] : p[L+1]
	W = legeval(t, p)
	if major == "last"
		W = reshape(W, 1, n, p)
		W = Array{T,N}(W)
		decpushfirst!(U, W)
	else
		W = reshape(permutedims(W), p, n, 1)
		W = Array{T,N}(W)
		decpush!(U, W)
	end
	U
end



end
