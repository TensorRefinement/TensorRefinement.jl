module Exponential

using ..Auxiliary, ..TensorTrain

export trigevalmask, trigeval, trigrefmask, trigrefmask2, trigdiffmask, trigdiff, trigdec, trigdeceval!
export cosfactor, cosdec

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
