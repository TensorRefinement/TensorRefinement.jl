module Auxiliary

using LinearAlgebra

export Indices, FloatRC, Float2, Float3, Int2, Int3
export indvec, threshold, compfloateps, modemul
export qraddcols!, qraddcols, lqaddrows


const Indices = Union{Vector{Int},Int,UnitRange{Int},StepRange{Int,K} where K,NTuple{M,Int} where M,Vector{Any},Nothing,Colon}
const FloatRC{T} = Union{T,Complex{T}} where T<:AbstractFloat
const Float2{T} = Union{T,Vector{T}} where T<:AbstractFloat
const Float3{T} = Union{T,Vector{T},Vector{Vector{T}}} where T<:AbstractFloat
const Int2 = Union{Int,Vector{Int}}
const Int3 = Union{Int,Vector{Int},Vector{Vector{Int}}}

compfloateps(::Type{S}) where {T<:AbstractFloat,S<:FloatRC{T}} = eps(T)

two(::Type{T}) where T<:Number = 2*one(T)

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

function threshold(δ::Vector{T}, τ::T, ε::T, r::Int) where {T<:Real}
	if τ < 0
		throw(ArgumentError("τ is required to be nonnegative"))
	end
	if ε < 0
		throw(ArgumentError("ε is required to be nonnegative"))
	end
	if r < 0
		throw(ArgumentError("r is required to be nonnegative"))
	end
	n = length(δ)
	σ,ρ = zero(T),n
	if n > 0
		if r > 0
			ρ = min(r, ρ)
		end
		if τ > 0
			k = findlast(x -> x > τ, δ)
			if isa(k, Int)
				ρ = min(k, ρ)::Int
			else
				ρ = 0
			end
		end
		if ε > 0
			δ = cumsum(reverse(δ))
			k = findlast(x -> x ≤ ε, δ)
			if isa(k, Int)
				ρ = min(n-k, ρ)::Int
			end
			if ρ < n
				σ = δ[n-ρ]
			end
			return σ,ρ
		end
		σ = sum(δ[ρ+1:n])
	end
	σ,ρ
end


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


function _mode2mul(A::AbstractArray{<:Number,3}, B::AbstractMatrix{<:Number})
	k,m,n = size(A)
	A = permutedims(A, [2,1,3])
	mm = size(B, 1)
	A = reshape(B*reshape(A, m, k*n), mm, k, n)
	A = permutedims(A, [2,1,3])
	A,mm
end

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
