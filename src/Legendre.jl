module Legendre

using TensorRefinement.Auxiliary, ..TensorTrain, LinearAlgebra

export legeval, legneval, legtolegn, legntoleg, legdiff, legndiff, legref, legnref, legdec, legdeceval!

function legtolegn(::Type{T}, r::Int) where {T<:AbstractFloat}
	u = [ 1/sqrt(j+one(T)/2) for j ∈ 0:r-1 ]
	Diagonal(u)
end

function legntoleg(::Type{T}, r::Int) where {T<:AbstractFloat}
	u = [ sqrt(j+one(T)/2) for j ∈ 0:r-1 ]
	Diagonal(u)
end

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

legneval(t::Vector{T}, r::Int) where {T<:AbstractFloat} = legeval(t, r)*legntoleg(T, r)

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

legndiff(::Type{T}, r::Int) where {T<:AbstractFloat} = legtolegn(T, r)*legdiff(T, r)*legntoleg(T, r)


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
