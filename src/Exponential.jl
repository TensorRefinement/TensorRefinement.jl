module Exponential

using ..Auxiliary, ..TensorTrain

export trigevalmask, trigeval, trigrefmask, trigrefmask2, trigdiffmask, trigdiff, trigdec, trigdeceval!
export cosfactor, cosdec

"""
    trigevalmask(t::Vector{T}, ν::Vector{T}) where {T<:AbstractFloat}

Evaluate the trigonometric functions cosine and sine at a set of points `t` for a given set of frequencies `ν`.

# Arguments
- `t::Vector{T}`: Vector of points where the trigonometric functions are evaluated.
- `ν::Vector{T}`: Vector of frequencies for which the trigonometric functions are computed. Length of `ν` determines the number of cosine-sine pairs.

# Returns
- `Matrix{T}`: A matrix of size `(n, 2*r)`, where `n` is the length of `t` and `r` is the length of `ν`. Each row contains alternating cosine and sine values for the corresponding point in `t` and frequency in `ν`.

# Throws
- `ArgumentError`: If the number of frequencies `r` is not positive.
"""
function trigevalmask(t::Vector{T}, ν::Vector{T}) where {T<:AbstractFloat}
	n = length(t)
	r = length(ν)
	if r == 0
		throw(ArgumentError("the number of frequencies should be positive"))
	end
	V = zeros(T, n, 2*r)
	for α ∈ 1:r
		for i ∈ 1:n
			V[i,2*α-1] = cospi(ν[α]*t[i])
			V[i,2*α  ] = sinpi(ν[α]*t[i])
		end
	end
	V
end

"""
    trigeval(t::Vector{T}, ν::Vector{T}, c::Vector{T}) where {T<:AbstractFloat}

Evaluate a trigonometric series with given frequencies `ν` and coefficients `c` at a set of points `t`. Each element in `t` is evaluated as the sum of cosine and sine terms for 
each frequency in `ν` with the weight corresponding to the coefficients in `c`.

# Arguments
- `t::Vector{T}`: Vector of points where the trigonometric series is evaluated.
- `ν::Vector{T}`: Vector of frequencies for the trigonometric series. Length of `ν` determines the number of cosine-sine pairs.
- `c::Vector{T}`: Vector of coefficients, where each pair of consecutive elements corresponds to the cosine and sine weights for a particular frequency.

# Returns
- `Vector{T}`: Vector of length `n` (the length of `t`) which represents the values of the trigonometric series evaluated at each point in `t`.

# Throws
- `ArgumentError`: If the number of frequencies `r` is not positive.
- `ArgumentError`: If the length of `c` is not `2*r`, rendering `ν` and `c` incompatible in size.
"""
function trigeval(t::Vector{T}, ν::Vector{T}, c::Vector{T}) where {T<:AbstractFloat}
	n = length(t)
	r = length(ν)
	if r == 0
		throw(ArgumentError("the number of frequencies should be positive"))
	end
	if length(c) ≠ 2*r
		throw(ArgumentError("ν and c are incompatible in size"))
	end
	u = zeros(T, length(t))
	for i ∈ 1:n
		for α ∈ 1:r
			u[i] += c[2*α-1]*cospi(ν[α]*t[i])+c[2*α]*sinpi(ν[α]*t[i])
		end
	end
	u
end

"""
    trigrefmask(η::T, ν::Vector{T}) where {T<:AbstractFloat}

Construct a reference mask for trigonometric transformations, based on a shift `η` and a set of frequencies `ν`.

# Arguments
- `η::T`: Shift value for the trigonometric transformations.
- `ν::Vector{T}`: Vector of frequencies. Each frequency corresponds to a 2x2 block in the resulting matrix.

# Returns
- `Matrix{T}`: Square matrix of size `2*r × 2*r`, where `r` is the length of `ν`. Each 2x2 block contains the sine and cosine values for the corresponding frequency in `ν`, rotated by `η`.

# Throws
- `ArgumentError`: If the number of frequencies `r` is not positive.
"""
function trigrefmask(η::T, ν::Vector{T}) where {T<:AbstractFloat}
	r = length(ν)
	if r == 0
		throw(ArgumentError("the number of frequencies should be positive"))
	end
	W = zeros(T, 2*r, 2*r)
	for α ∈ 1:r
		s,c = sincospi(ν[α]*η)
		W[2*α-1,2*α-1] =  c
		W[2*α,  2*α-1] = -s
		W[2*α-1,2*α  ] =  s
		W[2*α,  2*α  ] =  c
	end
	W
end

"""
    trigrefmask2(ν::Vector{T}) where {T<:AbstractFloat}

Construct a reference mask for trigonometric interpolation using two points at `-1/2` and `1/2`.

# Arguments
- `ν::Vector{T}`: Vector of frequencies.

# Returns
- `Array{T,3}`: Three-dimensional array of size `2r × 2r × 2`, where `r` is the length of `ν`.

# Throws
- `ArgumentError`: If `ν` is empty (i.e., `r == 0`).
"""
function trigrefmask2(ν::Vector{T}) where {T<:AbstractFloat}
	r = length(ν)
	if r == 0
		throw(ArgumentError("the number of frequencies should be positive"))
	end
	W1 = trigrefmask(convert(T, -1//2), ν)
	W2 = trigrefmask(convert(T, 1//2), ν)
	W = [W1[:] W2[:]]
	W = reshape(W, 2*r, 2*r, 2)
	W
end

"""
    trigdiffmask(ν::Vector{T}) where {T<:AbstractFloat}

Constructs a differentiation mask for trigonometric functions based on the given frequencies.

# Arguments
- `ν::Vector{T}`: Vector of frequencies.

# Returns
- `Matrix{T}`: Square matrix of size `2r × 2r`, where `r` is the length of `ν`.

# Throws
- `ArgumentError`: If the number of frequencies `r` is not positive.
"""
function trigdiffmask(ν::Vector{T}) where {T<:AbstractFloat}
	r = length(ν)
	if r == 0
		throw(ArgumentError("the number of frequencies should be positive"))
	end
	W = zeros(T, 2*r, 2*r)
	for α ∈ 1:r
		W[2*α-1,2*α  ] =  π*ν[α]
		W[2*α,  2*α-1] = -π*ν[α]
	end
	W
end

"""
    trigdiff(ν::Vector{T}, c::Vector{T}) where {T<:AbstractFloat}

Perform differentiation on trigonometric functions with the given frequencies `ν` and coefficients `c`.

# Arguments
- `ν::Vector{T}`: Vector of frequencies.
- `c::Vector{T}`: Vector of coefficients, which must be of length `2r`, where `r` is the length of `ν`.

# Returns
- `Vector{T}`: Vector of differentiated coefficients of length `2r`.

# Throws
- `ArgumentError`: If the length of `c` does not match `2r`, rendering `v` and `c` incompatible in size.
"""
function trigdiff(ν::Vector{T}, c::Vector{T}) where {T<:AbstractFloat}
	r = length(ν)
	if length(c) ≠ 2*r
		throw(ArgumentError("ν and c are incompatible in size"))
	end
	d = zeros(T, 2*r)
	for α ∈ 1:r
		d[2*α-1] =  π*ν[α]*c[2*α]
		d[2*α]   = -π*ν[α]*c[2*α-1]
	end
	d
end

"""
    cosfactor(τ::Vector{T}, c::T) where {T<:AbstractFloat}

Construct a factorized cosine-sine matrix for each value of `τ` using the scaling value `c`.

# Arguments
- `τ::Vector{T}`: Vector of angles in radians.
- `c::T`: Value used in scaling the cosine and sine terms.

# Returns
- `Array{T,3}`: Three-dimensional array of size `2 × n × 2`, where `n` is the length of `τ`.

# Throws
- `ArgumentError`: If `τ` is empty (i.e., `n == 0`).
"""
function cosfactor(τ::Vector{T}, c::T) where {T<:AbstractFloat}
	n = length(τ)
	if n == 0
		throw(ArgumentError("τ should be nonempty"))
	end
	U = zeros(T, 2, n, 2)
	for i ∈ 1:n
		α = c*cos(τ[i])
		β = c*sin(τ[i])
		U[1,i,1] =  α
		U[2,i,1] =  β
		U[1,i,2] = -β
		U[2,i,2] =  α
	end
	U
end

"""
    cosdec(τ::Vector{Vector{T}}, c::Vector{T}) where {T<:AbstractFloat}

Construct a cosine-based decomposition given a set of angles `τ` and coefficients `c`.

# Arguments
- `τ::Vector{Vector{T}}`: Vector whose elements are vectors of angles in radians.
- `c::Vector{T}`: Vector of coefficients corresponding to the angles in `τ`.

# Returns
- `Dec{T,N}`: Decomposition `U` where each of the `L` elements is a cosine matrix turned into a factor.

# Throws
- `ArgumentError`: If `τ` and `c` are of different lengths.
- `ArgumentError`: If any element of `τ` is empty.
"""
function cosdec(τ::Vector{Vector{T}}, c::Vector{T}) where {T<:AbstractFloat}
	L = length(τ)
	if length(c) ≠ L
		throw(ArgumentError("τ and c should be of the same length"))
	end
	if any(τ .== 0)
		throw(ArgumentError("all elements of τ should be nonempty"))
	end
	U = [ cosfactor(τ[ℓ], c[ℓ]) for ℓ ∈ 1:L ]
	U[1] = U[1][1:1,:,:]
	U[L] = U[L][:,:,1:1]
	U = dec(U)
	U
end


"""
    trigdec(ν::Vector{T}, c::Vector{T}, L::Int; major::String="last") where {T<:AbstractFloat}

Construct a trigonometric decomposition based on frequencies `ν` and coefficients `c`.

# Arguments
- `ν::Vector{T}`: Vector of frequencies.
- `c::Vector{T}`: Vector of coefficients, which must be twice the length of `ν`.
- `L::Int`: Length of the decomposition.
- `major::String`: Either `"first"` or `"last"`, specifying the major dimension of the decomposition (default is `"last"`).

# Returns
- `Dec{T,N}`: Decomposition `U` of length `L` where the first factor is based on the coefficients `c`, and the remaining factors are based on trigonometric reference masks.

# Throws
- `ArgumentError`: If `ν` and `c` have incompatible sizes.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
function trigdec(ν::Vector{T}, c::Vector{T}, L::Int; major::String="last") where {T<:AbstractFloat}
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	r = length(ν)
	if length(c) ≠ 2*r
		throw(ArgumentError("ν and c are incompatible in size"))
	end
	U = dec(T, 1, L+1)
	U[1] = factor(Matrix{T}(c'), 1, [])
	for ℓ ∈ 1:L
		U[ℓ+1] = permutedims(trigrefmask2(ν/2^(ℓ-1)), (2,3,1))
	end
	(major == "last") && decreverse!(U)
	U
end

"""
    trigdeceval!(U::Dec{T,N}, t::Vector{T}, ν::Vector{T}; major::String="last") where {T<:AbstractFloat,N}

Evaluate a trigonometric decomposition `U` at a set of points `t` and frequencies `ν`, modifying the decomposition in place.

# Arguments
- `U::Dec{T,N}`: Decomposition to be evaluated.
- `t::Vector{T}`: Vector of points at which to evaluate the decomposition.
- `ν::Vector{T}`: Vector of frequencies.
- `major::String`: Either `"first"` or `"last"`, specifying the major dimension of the decomposition (default is `"last"`).

# Returns
- `Dec{T,N}`: Evaluated and accordingly updated decomposition `U`.

# Throws
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
- `ArgumentError`: If `N ≠ 3`, i.e., if there are factors whose number of mode indices is not 1.
- `ArgumentError` If the first or last rank of `U` and the size of `ν` are incompatible.
"""
function trigdeceval!(U::Dec{T,N}, t::Vector{T}, ν::Vector{T}; major::String="last") where {T<:AbstractFloat,N}
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	if N ≠ 3
		throw(ArgumentError("only one mode index per factor is currently supported"))
	end
	n = length(t)
	r = length(ν)
	L = declength(U)
	p = decrank(U)
	p = (major == "last") ? p[1] : p[L+1]
	if p ≠ 2*r
		throw(ArgumentError("the ", (major == "last") ? "first" : "last", " rank of U and the size of ν are incompatible"))
	end
	W = trigevalmask(t, ν)
	if major == "last"
		W = reshape(W, 1, n, 2*r)
		W = Array{T,N}(W)
		decpushfirst!(U, W)
	else
		W = reshape(transpose(W), 2*r, n, 1)
		W = Array{T,N}(W)
		decpush!(U, W)
	end
	U
end


end
