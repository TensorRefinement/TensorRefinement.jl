module Chebyshev

using TensorRefinement.Auxiliary, TensorRefinement.Legendre, ..TensorTrain, LinearAlgebra

export chebeval, chebexnodes, chebtochebex, chebextocheb, chebextoleg, chebextolegn, chebrtnodes, chebtochebrt, chebrttocheb, chebrttoleg, chebrttolegn, chebtoleg, chebtolegn, legtocheb, legntocheb, chebref, chebdiff, chebexdiff, chebdec, chebdeceval!

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

function chebeval(t::Vector{T}, c::Vector{T}) where {T<:AbstractFloat}
	r = length(c)
	V = chebeval(t, r)
	u = V*c
	return u
end

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

function chebtochebex(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	[ cospi(convert(T, i*j)/(r-1)) for i ∈ r-1:-1:0, j ∈ 0:r-1 ]
end

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

chebextoleg(::Type{T}, r::Int) where {T<:AbstractFloat} = chebtoleg(T, r)*chebextocheb(T, r)
chebextolegn(::Type{T}, r::Int) where {T<:AbstractFloat} = chebtolegn(T, r)*chebextocheb(T, r)

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

function chebtochebrt(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	# U = [ cospi(convert(T, (2*i+1)*j)/(2*r)) for i ∈ r-1:-1:0, j ∈ 0:r-1 ]
	U = [ cospi(((2*i+1)*j*one(T))/(2*r)) for i ∈ r-1:-1:0, j ∈ 0:r-1 ]
	U
end

function chebrttocheb(::Type{T}, r::Int) where {T<:AbstractFloat}
	if r ≤ 0
		throw(ArgumentError("the number of DOFs should be positive"))
	end
	# U = [ cospi(convert(T, (2*i+1)*j)/(2*r))/r for j ∈ 0:r-1, i ∈ r-1:-1:0 ]
	U = [ cospi(((2*i+1)*j*one(T))/(2*r))/r for j ∈ 0:r-1, i ∈ r-1:-1:0 ]
	U[2:end,:] .*= 2
	U
end

chebrttoleg(::Type{T}, r::Int) where {T<:AbstractFloat} = chebtoleg(T, r)*chebrttocheb(T, r)
chebrttolegn(::Type{T}, r::Int) where {T<:AbstractFloat} = chebtolegn(T, r)*chebrttocheb(T, r)

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

chebtolegn(::Type{T}, r::Int) where {T<:AbstractFloat} = legtolegn(T, r)*chebtoleg(T, r)

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

legntocheb(::Type{T}, r::Int) where {T<:AbstractFloat} = legtocheb(T, r)*legntoleg(T, r)

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


function chebref(ξ::T, η::T, r::Int) where {T<:AbstractFloat}
	# TODO
	W = zeros(T, r, r)
	W
end

function chebref(::Type{T}, r::Int) where {T<:AbstractFloat}
	# TODO
	W = zeros(T, r, r)
	W
end

chebref(r::Int) = chebref(Float64, r)


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
