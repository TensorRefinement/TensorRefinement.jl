module Chebyshev

using TensorRefinement.Auxiliary, TensorRefinement.Legendre, ..TensorTrain, LinearAlgebra

export chebeval, chebexnodes, chebtochebex, chebextocheb, chebextoleg, chebextolegn, chebrtnodes, chebtochebrt, chebrttocheb, chebrttoleg, chebrttolegn, chebtoleg, chebtolegn, legtocheb, legntocheb, chebref, chebdiff, chebexdiff, chebdec, chebdeceval!

"""
    chebeval(t::Vector{T}, r::Int) where {T<:AbstractFloat}

Evaluate the first `r` Chebyshev polynomials at the points specified in the vector `t`.

# Arguments
- `t::Vector{T}`: Vector of points at which to evaluate the Chebyshev polynomials.
- `r::Int`: Number of Chebyshev polynomials to evaluate.

# Returns
- `Matrix{T}`: An `n` x `r` matrix `V`, where `n` is the length of `t`. The entry `V[i,j]` 
  contains the value of the `(j-1)`-th Chebyshev polynomial evaluated at `t[i]`.

# Throws
- `ArgumentError`: If `r` is negative.
"""
function chebeval(t::Vector{T}, r::Int) where {T<:AbstractFloat}
	n = length(t)
	V = zeros(T, n, r)
	V[:,1] .= 1
	if r > 1
		V[:,2] = t
	end
	for k ∈ 1:r-2
		V[:,k+2] .= 2 .*t.*V[:,k+1].-V[:,k]
	end
	return V
end

"""
    chebeval(t::Vector{T}, c::Vector{T}) where {T<:AbstractFloat}

Evaluate a Chebyshev series for a given coefficient vector `c` at the points in `t`.

# Arguments
- `t::Vector{T}`: Input vector of points specifying where the series is evaluated.
- `c::Vector{T}`: Coefficients of the Chebyshev series.

# Returns
Vector `u` containing the evaluation of the Chebyshev series at each point in `t`. That means, u[i] contains the evalutation of the Chebyshev series at point t[i] with coefficients from `c`.
"""
function chebeval(t::Vector{T}, c::Vector{T}) where {T<:AbstractFloat}
	r = length(c)
	V = chebeval(t, r)
	u = V*c
	return u
end

"""
    chebexnodes(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the Chebyshev nodes of the second kind (also called « Chebyshev extrema ») on the interval [-1, 1].

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev nodes to compute.

# Returns
Vector of `r` Chebyshev nodes of the second kind.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebexnodes(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	t = Vector{T}(undef, r) 
	if r == 1
		t[1] = 0
	else 
		t .= ( sinpi(convert(T, i)/(2*(r-1))) for i ∈ 1-r:2:r-1 )
	end
	t
end

"""
    chebtochebex(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Chebyshev coefficients to Chebyshev extrema coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev coefficients to Chebyshev extrema coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebtochebex(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	[ cospi(convert(T, i*j)/(r-1)) for i ∈ r-1:-1:0, j ∈ 0:r-1 ]
end

"""
    chebextocheb(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Chebyshev extrema coefficients to Chebyshev coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev extrema coefficients to Chebyshev coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebextocheb(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	U = [ cospi(convert(T, i*j)/(r-1)) for j ∈ 0:r-1, i ∈ r-1:-1:0 ]
	U[:, 2:r] .*= 2
	w = [1; 2*ones(Int, r-2); 1]
	B = Diagonal(w).*(one(T)/(2*(r-1))) - w*w'.*(one(T)/(8*(r-1)^2))
	B*U
end

"""
    chebextoleg(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Chebyshev extrema coefficients to Legendre coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev extrema coefficients to Legendre coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
chebextoleg(::Type{T}, r::Int) where {T<:AbstractFloat} = chebtoleg(T, r)*chebextocheb(T, r)

"""
    chebextolegn(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Chebyshev extrema coefficients to normalized Legendre coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev extrema coefficients to normalized Legendre coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
chebextolegn(::Type{T}, r::Int) where {T<:AbstractFloat} = chebtolegn(T, r)*chebextocheb(T, r)

"""
    chebrtnodes(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the Chebyshev nodes of the first kind (also called « Chebyshev roots ») on the interval (-1, 1).

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev roots to compute.

# Returns
Vector of `r` Chebyshev nodes of the first kind.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebrtnodes(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	t = Vector{T}(undef, r) 
	if r == 1
		t[1] = 0
	else
		t .= ( sinpi(convert(T, i)/(2*r)) for i ∈ 1-r:2:r-1 )
	end
	t
end		

"""
    chebtochebrt(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix from Chebyshev polynomial coefficients to Chebyshev root polynomial coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev coefficients to Chebyshev root coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebtochebrt(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	# U = [ cospi(convert(T, (2*i+1)*j)/(2*r)) for i ∈ r-1:-1:0, j ∈ 0:r-1 ]
	U = [ cospi(((2*i+1)*j*one(T))/(2*r)) for i ∈ r-1:-1:0, j ∈ 0:r-1 ]
	U
end

"""
    chebrttocheb(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix from Chebyshev root polynomial coefficients to Chebyshev polynomial coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev root coefficients to Chebyshev coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebrttocheb(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	# U = [ cospi(convert(T, (2*i+1)*j)/(2*r))/r for j ∈ 0:r-1, i ∈ r-1:-1:0 ]
	U = [ cospi(((2*i+1)*j*one(T))/(2*r))/r for j ∈ 0:r-1, i ∈ r-1:-1:0 ]
	U[2:end,:] .*= 2
	U
end

"""
    chebrttoleg(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Chebyshev root coefficients to Legendre coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev root coefficients to Legendre coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
chebrttoleg(::Type{T}, r::Int) where {T<:AbstractFloat} = chebtoleg(T, r)*chebrttocheb(T, r)

"""
    chebrttolegn(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Chebyshev root coefficients to normalized Legendre coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev root coefficients to normalized Legendre coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
chebrttolegn(::Type{T}, r::Int) where {T<:AbstractFloat} = chebtolegn(T, r)*chebrttocheb(T, r)

"""
    chebtoleg(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Chebyshev coefficients to Legendre coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev coefficients to Legendre coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebtoleg(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	u = zeros(T, 2r-1)
	# u[1] = sqrt(convert(T, π))
	u[1] = sqrt(one(T)*π)
	u[2] = 2/u[1]
	for k ∈ 2:2r-2
		u[k+1] = u[k-1]*(1-one(T)/k)
	end
	W = zeros(T, r, r)
	W[1,1] = 1
	for j ∈	1:r-1
		W[j+1,j+1] = u[1]/u[2j+1]/2
	end
	for j ∈	0:r-1
		for i ∈	j-2:-2:0
			W[i+1,j+1] = -j*(i+one(T)/2)/(j+i+1)/(j-i)*u[j-i-1]*u[j+i];
		end
	end
	W
end

"""
    chebtolegn(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Chebyshev coefficients to normalized Legendre coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Chebyshev coefficients to normalized Legendre coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
chebtolegn(::Type{T}, r::Int) where {T<:AbstractFloat} = legtolegn(T, r)*chebtoleg(T, r)

"""
    legtocheb(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from Legendre coefficients to Chebyshev coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Legendre polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from Legendre coefficients to Chebyshev coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function legtocheb(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	u = zeros(T, 2r-1)
	u[1] = sqrt(convert(T, π))
	u[2] = 2/u[1]
	for k ∈ 2:2r-2
		u[k+1] = u[k-1]*(1-one(T)/k)
	end
	W = zeros(T, r, r)
	for j ∈	0:2:r-1
		W[1,j+1] = (u[j+1]/u[1])^2
	end
	for j ∈	0:r-1
		for i ∈ j:-2:1
			W[i+1,j+1] = 2*u[j-i+1]*u[j+i+1]/u[1]^2
		end
	end
	W
end

"""
    legntocheb(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the transformation matrix that converts from normalized Legendre coefficients to Chebyshev coefficients.

# Arguments
- `::Type{T}`: Type of the floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the transformation from normalized Legendre coefficients to Chebyshev coefficients.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
legntocheb(::Type{T}, r::Int) where {T<:AbstractFloat} = legtocheb(T, r)*legntoleg(T, r)

"""
    chebdiff(::Type{T}, r::Int) where {T<:AbstractFloat}

Computes the differentiation matrix for Chebyshev polynomials of degree up to `r-1`.

# Arguments
- `::Type{T}`: Numeric type for the matrix elements; subtype of `AbstractFloat`.
- `r::Int`: Number of Chebyshev polynomials (degree).

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the differentiation operator for Chebyshev polynomials of degree up to `r-1`.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebdiff(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	W = zeros(T, r, r)
	if r == 1
		return W
	end

	for s ∈ 1:2:r
		for j ∈ 1:r-s
			W[j,j+s] = 2(j+s-1)
		end
	end
	W[1,:] ./= 2
	W
end

"""
    chebexdiff(::Type{T}, r::Int) where {T<:AbstractFloat}

Computes the differentiation matrix for `r` Chebyshev extrema nodes.

# Arguments
- `::Type{T}`: Numeric type for the matrix elements; subtype of `AbstractFloat`.
- `r::Int`: Number of extrema nodes.

# Returns
- `Matrix{T}`: A matrix of size `r` x `r` representing the differentiation operator for `r` Chebyshev extrema nodes.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebexdiff(::Type{T}, r::Int) where {T<:AbstractFloat}
		if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	W = zeros(T, r, r)
	if r == 1
		return W
	end
	u = collect(1-r:2:r-1) .* (one(T)/(2r-2))
	v = u[2:r-1]
	v = -sinpi.(v)./(cospi.(v).^2)./2
	v = [-(2*(r-1)^2+1)*(one(T)/6); v; (2*(r-1)^2+1)*(one(T)/6)]
	a,b = sinpi.(u./2),cospi.(u./2)
	for i ∈ 1:r, j ∈ 1:r
		if i == j
			W[i,j] = v[i]
		else
			W[i,j] = 1/((a[i]*b[j]-b[i]*a[j])*(b[i]*b[j]-a[i]*a[j])*2)
		end
	end
	W[2:2:r,:] .*= -1
	W[:,2:2:r] .*= -1
	W[1,2:r-1] .*= 2
	W[r,2:r-1] .*= 2
	W[2:r-1,1] ./= 2
	W[2:r-1,r] ./= 2
	W
end

# function chebdiff(c::Vector{T}) where {T<:AbstractFloat}
# 	r = length(c)
# 	d = zeros(T, max(1, r-1))
# 	for β ∈ 0:r-1, γ ∈ 1:2:β
# 		d[1+β-γ] += (2*(β-γ)+1)*c[1+β]
# 	end
# 	d
# end


#"""
#    chebtochebex(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebextocheb(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebextoleg(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebextolegn(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebtochebrt(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebrttocheb(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebrttoleg(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebrttolegn(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebtoleg(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    legtocheb(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    chebtolegn(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#    legntocheb(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#
# Apply a transformation between different polynomial coefficient bases (Chebyshev, Legendre, etc.) along specified dimensions.

# Arguments
# - `C::AbstractArray{T}`: Input array of coefficients, where `T` is a subtype of `AbstractFloat`.
# - `dims=1`: Specifies the dimensions along which the transformation should be applied.

# Returns
# Array of coefficients after applying the transformation.

# Throws
# - `ArgumentError`: If any of the dimensions in `C` is non-positive.
# - `ArgumentError`: If any dimension in `dims` is not within the valid range of the dimensions of `C`.
# - `ArgumentError`: The elements in `dims` are not unique.
# """
for fname ∈ (:chebtochebex,:chebextocheb,:chebextoleg,:chebextolegn,:chebtochebrt,:chebrttocheb,:chebrttoleg,:chebrttolegn,:chebtoleg,:legtocheb,:chebtolegn,:legntocheb)
	@eval begin
		function ($fname)(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
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

#"""
#    chebdiff(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
#
#Apply a Chebyshev differentiation operator along the specified dimensions.

# Arguments
#- `C::AbstractArray{T}`: Input array of coefficients, where `T` is a subtype of `AbstractFloat`.
#- `dims=1`: Specifies the dimensions along which the differentiation should be applied.

# Returns
#An array after applying the Chebyshev differentiation operator.

# Throws
#- `ArgumentError`: If any of the dimensions in `C` is non-positive.
#- `ArgumentError`: If any dimension in `dims` is not within the valid range of the dimensions of `C`.
#- `ArgumentError`: The elements in `dims` are not unique.
#"""
for fname ∈ (:chebdiff,)
	@eval begin
		function ($fname)(C::AbstractArray{T}; dims=1) where {T<:AbstractFloat}
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
    chebref(ξ::T, η::T, r::Int) where {T<:AbstractFloat}

Compute the Chebyshev reference matrix for interpolation between two points `ξ` and `η`.
Currently, this function is not implemented.

# Arguments
- `ξ::T`: First one of the two points which are to be interpolated.
- `η::T`: Second one of the two points which are to be interpolated.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r x r` representing the reference for interpolation between `ξ` and `η` using the Chebyshev polynomials up to a degree of `r-1`.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebref(ξ::T, η::T, r::Int) where {T<:AbstractFloat}
	# TODO
	W = zeros(T, r, r)
	W
end

"""
    chebref(::Type{T}, r::Int) where {T<:AbstractFloat}

Compute the Chebyshev reference matrix (with entries of type `T`) for Chebyshev polynomials up to a degree of `r-1`.

# Arguments
- `::Type{T}`: Type of floating-point numbers.
- `r::Int`: Number of Chebyshev polynomials.

# Returns
- `Matrix{T}`: A matrix of size `r x r` representing the reference for the Chebyshev polynomials up to a degree of `r-1`.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
function chebref(::Type{T}, r::Int) where {T<:AbstractFloat}
	# TODO
	W = zeros(T, r, r)
	W
end

"""
    chebref(r::Int)

Convenience function to call `chebref(Float64, r)`.

# Arguments
- `r::Int`: Number of Chebyshev polynomials.

# Returns
Chebyshev reference matrix of size `r × r`, whose entries are of Type `Float64`.

# Throws
- `ArgumentError`: If the number of degrees of freedom `r` is not positive.
"""
chebref(r::Int) = chebref(Float64, r)

"""
    chebdec(c::Vector{T}, L::Int; major::String="last") where {T<:AbstractFloat}

Construct a Chebyshev decomposition based on the coefficients `c` and the number of terms `L`.

# Arguments
- `c::Vector{T}`: Vector of Chebyshev coefficients.
- `L::Int`: Number of terms in the decomposition.
- `major::String`: Order of decomposition, either `"first"` or `"last"` (default is `"last"`).

# Returns
- `Dec{T,N}`: A decomposition consisting of factors obtained by the reference matrices of Chebyshev polynomials.

# Throws
- `ArgumentError`: If major` is neither `"first"` nor `"last"`.
"""
function chebdec(c::Vector{T}, L::Int; major::String="last") where {T<:AbstractFloat}
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	r = length(c)
	W = chebref(T, r)
	W = permutedims(W, (2,3,1))
	U = dec(W; len=L)
	C = Matrix{T}(c'); C = factor(C, 1, [])
	decpushfirst!(U, C)
	(major == "last") && decreverse!(U)
	U
end

"""
    chebdeceval!(U::Dec{T,N}, t::Vector{T}; major::String="last") where {T<:AbstractFloat,N}

Evaluate a Chebyshev decomposition `U` at the points specified in `t`, and modify it in place.

# Arguments
- `U::Dec{T,N}`: Chebyshev decomposition to evaluate.
- `t::Vector{T}`: Vector of points at which to evaluate the decomposition.
- `major::String`: Order of decomposition, either `"first"` or `"last"` (default is `"last"`).

# Returns
- `Dec{T,N}`: The evaluated decomposition `U` at points `t`.

# Throws
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
function chebdeceval!(U::Dec{T,N}, t::Vector{T}; major::String="last") where {T<:AbstractFloat,N}
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	n = length(t)
	L = declength(U)
	p = decrank(U)
	p = (major == "last") ? p[1] : p[L+1]
	W = chebeval(t, p)
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
