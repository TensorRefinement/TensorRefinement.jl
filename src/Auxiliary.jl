module Auxiliary

using LinearAlgebra

export Indices, FloatRC, Float2, Float3, Int2, Int3
export indvec, threshold, compfloateps
export qraddcols!, qraddcols, lqaddrows

const Indices = Union{Vector{Int},Int,UnitRange{Int},StepRange{Int,K} where K,NTuple{M,Int} where M,Vector{Any},Nothing,Colon}
const FloatRC = Union{T,Complex{T}} where T<:AbstractFloat
const Float2 = Union{Float64,Vector{Float64}}
const Float3 = Union{Float64,Vector{Float64},Vector{Vector{Float64}}}
const Int2 = Union{Int,Vector{Int}}
const Int3 = Union{Int,Vector{Int},Vector{Vector{Int}}}

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

function threshold(δ::Vector{T}, ε::T, ρ::Int) where {T<:Real}
	r = length(δ)
	if r == 0
		throw(ArgumentError("δ is required to be nonempty"))
	end
	if ε < 0
		throw(ArgumentError("ε is required to be nonnegative"))
	end
	if ρ < 0
		throw(ArgumentError("ρ is required to be nonnegative"))
	end
	if ρ == 0
		ρ = r
	end
	ρ = min(ρ, r)
	push!(δ, convert(T, 0))
	δ = reverse(δ)
	δ = cumsum(δ)
	k = findlast(x -> x ≤ ε, δ)
	k = max(k, r+1-ρ)
	ε = δ[k]
	ρ = r+1-k
	return ε, ρ
end

function compfloateps(::Type{T}) where T<:FloatRC
	if !isconcretetype(T)
		throw(ArgumentError("only concrete types are accepted"))
	end
	(T <: AbstractFloat) && return eps(T)
	(T <: Complex{S} where S<:AbstractFloat) && return eps(T.parameters[1])
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

end
