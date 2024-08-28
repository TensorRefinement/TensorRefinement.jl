export FactorSize, Factor, VectorFactor, MatrixFactor
export factorsize, factorranks, factorndims, factornumentries, factorstorage
export factor, factormatrix, factorrankselect, block
export factorvcat, factorhcat, factordcat, factorltcat, factorutcat
export factorranktranspose, factormodetranspose
export factormodereshape
export factordiagm
export factorcontract, factormp, factorkp, factorhp
export factorqr!, factorqradd, factorsvd!

const FactorSize = Vector{Int}
const Factor{T,N} = Array{T,N} where {T<:Number,N}
const VectorFactor{T} = Factor{T,3} where T<:Number
const MatrixFactor{T} = Factor{T,4} where T<:Number




function factorsize(U::Factor{T,N}) where {T<:Number,N}
	sz = size(U)
	if length(sz) < 2
		throw(ArgumentError("the factor should have, at least, two rank dimensions"))
	end
	n = collect(sz[2:end-1])
	if any(n .== 0)
		throw(ArgumentError("the mode sizes should be positive"))
	end
	FactorSize(n)
end

function factorranks(U::Factor{T,N}) where {T<:Number,N}
	sz = size(U)
	if length(sz) < 2
		throw(ArgumentError("the factor should have, at least, two rank dimensions"))
	end
	if any(sz[2:end-1] .== 0)
		throw(ArgumentError("the mode sizes should be positive"))
	end
	sz[1],sz[end]
end

factornumentries(U::Factor{T,N}) where {T<:Number,N} = length(U)

factorstorage(U::Factor{T,N}) where {T<:Number,N} = factornumentries(U)

function factorndims(::Factor{T,N}) where {T<:Number,N}
	if N < 2
		throw(ArgumentError("the factor should have, at least, two rank dimensions"))
	end
	N-2
end

function factor(U::Array{T,N}) where {T<:Number,N}
	reshape(U, 1, size(U)..., 1)
end

function factor(U::Matrix{T}, m::Union{Int,NTuple{M,Int},Vector{Int},Vector{Any}}, n::Union{Int,NTuple{N,Int},Vector{Int},Vector{Any}}, π::Union{NTuple{K,Int},Vector{Int}}) where {T<:Number,K,M,N}
	d = length(π)
	if d == 0
		throw(ArgumentError("π is empty"))
	end
	if isa(m, Vector{Any}) && length(m) > 0
		throw(ArgumentError("m should be an integer, a vector or tuple of integers or an empty vector or tuple"))
	end
	isa(m, Int) && (m = [m])
	isa(m, Vector{Any}) && (m = Vector{Int}(undef, 0))
	isa(m, Tuple{}) && (m = Vector{Int}(undef, 0))
	isa(m, Vector{Int}) || (m = collect(m))
	dm = length(m)
	if isa(n, Vector{Any}) && length(n) > 0
		throw(ArgumentError("n should be an integer, a vector or tuple of integers or an empty vector or tuple"))
	end
	isa(n, Int) && (n = [n])
	isa(n, Vector{Any}) && (n = Vector{Int}(undef, 0))
	isa(n, Tuple{}) && (n = Vector{Int}(undef, 0))
	isa(n, Vector{Int}) || (n = collect(n))
	dn = length(n)
	if dm+dn ≠ d || !isperm(π)
		throw(ArgumentError("π is not a valid permutation of 1:length(m)+length(n)"))
	end
	sz = size(U); p = sz[1]÷prod(m); q = sz[2]÷prod(n)
	if sz[1] ≠ prod(m)*p
		throw(DimensionMismatch("the first dimension of U is not divisible by prod(m)"))
	end
	if sz[2] ≠ prod(n)*q
		throw(DimensionMismatch("the second dimension of U is not divisible by prod(n)"))
	end
	U = reshape(U, (m...,p,n...,q))
	U = permutedims(U, (dm+1,π[1:dm]...,(π[dm+1:d].+1)...,d+2))
	U
end

factor(U::Matrix{T}, m::Union{Int,NTuple{M,Int},Vector{Int},Vector{Any}}, n::Union{Int,NTuple{N,Int},Vector{Int},Vector{Any}}) where {T<:Number,M,N} = factor(U, m, n, collect(1:length(m)+length(n)))
factor(U::Vector{T}, m::Union{Int,NTuple{M,Int},Vector{Int},Vector{Any}}, π::Union{NTuple{M,Int},Vector{Int}}) where {T<:Number,M} = factor(U[:,:], m, (), π)
factor(U::Vector{T}, m::Union{Int,NTuple{M,Int},Vector{Int},Vector{Any}}) where {T<:Number,M} = factor(U[:,:], m, (), collect(1:length(m)))

function factormatrix(U::Factor{T,K}, π::Indices, σ::Indices) where {T<:Number,K}
	d = factorndims(U)
	if d == 0
		throw(ArgumentError("U should have at least one mode dimension"))
	end
	if isa(π, Vector{Any}) && length(π) > 0
		throw(ArgumentError("π should be an integer, a vector or tuple of integers, a colon or an empty vector or tuple"))
	end
	π = indvec(π; min=1, max=d); dπ = length(π)
	if isa(σ, Vector{Any}) && length(σ) > 0
		throw(ArgumentError("σ should be an integer, a vector or tuple of integers, a colon or an empty vector or tuple"))
	end
	σ = indvec(σ; min=1, max=d); dσ = length(σ)
	τ = vcat(π,σ)
	if dπ+dσ ≠ d || !isperm(τ)
		throw(ArgumentError("π and σ do not constitute a valid permutation of the mode dimensions of U"))
	end
	n = factorsize(U); p,q = factorranks(U)
	U = permutedims(U, ((π.+1)...,1,(σ.+1)...,d+2))
	U = reshape(U, p*prod(n[π]), q*prod(n[σ]))
	U
end

function factorrankselect(U::Factor{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	isa(α, Int) && (α = [α])
	isa(β, Int) && (β = [β])
	p,q = factorranks(U)
	n = factorsize(U)
	α = indvec(α; min=1, max=p)
	β = indvec(β; min=1, max=q)
	if α ⊈ 1:p
		throw(ArgumentError("the index or range for the first rank is incorrect"))
	end
	if β ⊈ 1:q
		throw(ArgumentError("the index or range for the second rank is incorrect"))
	end
	U[α,ntuple(k -> Colon(), Val(N-2))...,β]
end

function block(U::Factor{T,N}, α::Int, β::Int) where {T<:Number,N}
	p,q = factorranks(U)
	n = factorsize(U)
	if α ∉ 1:p
		throw(ArgumentError("the first rank index is out of range"))
	end
	if β ∉ 1:q
		throw(ArgumentError("the second rank index is out of range"))
	end
	U = reshape(U, p, :, q)
	V = U[α,:,β]
	reshape(V, n...)
end

function factorvcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number,N,M}
	m = factorsize(U); _,p = factorranks(U)
	W = (V,W...)
	for V ∈ W
		if factorsize(V) ≠ m
			throw(ArgumentError("the factors are incompatible in mode size"))
		end
		_,q = factorranks(V)
		if q ≠ p
			throw(ArgumentError("the factors are incompatible in the second rank"))
		end
	end
	cat(U, W...; dims=1)
end

function factorhcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number,N,M}
	m = factorsize(U); p,_ = factorranks(U)
	W = (V,W...)
	for V ∈ W
		if factorsize(V) ≠ m
			throw(ArgumentError("the factors are incompatible in mode size"))
		end
		q,_ = factorranks(V)
		if q ≠ p
			throw(ArgumentError("the factors are incompatible in the first rank"))
		end
	end
	d = N-2
	cat(U, W...; dims=d+2)
end

function factordcat(U::Factor{T,N}, V::Factor{T,N}, W::Vararg{Factor{T,N},M}) where {T<:Number,N,M}
	m = factorsize(U)
	W = (V,W...)
	for V ∈ W
		if factorsize(V) ≠ m
			throw(ArgumentError("the factors are incompatible in mode size"))
		end
	end
	d = N-2
	cat(U, W...; dims=(1,d+2))
end

function factorutcat(U₁₁::Factor{T,N}, U₁₂::Factor{T,N}, U₂₂::Factor{T,N}) where {T<:Number,N}
	n₁₁ = factorsize(U₁₁); p₁₁,q₁₁ = factorranks(U₁₁)
	n₁₂ = factorsize(U₁₂); p₁₂,q₁₂ = factorranks(U₁₂)
	n₂₂ = factorsize(U₂₂); p₂₂,q₂₂ = factorranks(U₂₂)
	if n₁₂ ≠ n₁₁ || n₂₂ ≠ n₁₁
		throw(ArgumentError("the factors are incompatible in mode size"))
	end
	if p₁₁ ≠ p₁₂
		throw(ArgumentError("U₁₁ and U₁₂ are incompatible in the first rank"))
	end
	if q₁₂ ≠ q₂₂
		throw(ArgumentError("U₁₂ and U₂₂ are incompatible in the second rank"))
	end
	U₂₁ = zeros(T, (p₂₂,n₁₁...,q₁₁))
	d = N-2
	cat(cat(U₁₁, U₂₁; dims=1), cat(U₁₂, U₂₂; dims=1); dims=d+2)
end

function factorltcat(U₁₁::Factor{T,N}, U₂₁::Factor{T,N}, U₂₂::Factor{T,N}) where {T<:Number,N}
	n₁₁ = factorsize(U₁₁); p₁₁,q₁₁ = factorranks(U₁₁)
	n₂₁ = factorsize(U₂₁); p₂₁,q₂₁ = factorranks(U₂₁)
	n₂₂ = factorsize(U₂₂); p₂₂,q₂₂ = factorranks(U₂₂)
	if n₂₁ ≠ n₁₁ || n₂₂ ≠ n₁₁
		throw(ArgumentError("the factors are incompatible in mode size"))
	end
	if p₂₁ ≠ p₂₂
		throw(ArgumentError("U₂₁ and U₂₂ are incompatible in the first rank"))
	end
	if q₁₁ ≠ q₂₁
		throw(ArgumentError("U₁₁ and U₂₁ are incompatible in the second rank"))
	end
	U₁₂ = zeros(T, (p₁₁,n₁₁...,q₂₂))
	d = N-2
	cat(cat(U₁₁, U₂₁; dims=1), cat(U₁₂, U₂₂; dims=1); dims=d+2)
end

function factorranktranspose(U::Factor{T,N}) where {T<:Number,N}
	d = N-2
	prm = (d+2,ntuple(k -> k+1, Val(d))...,1)
	permutedims(U, prm)
end

function factormodetranspose(U::Factor{T,N}, π::NTuple{K,Int}) where {T<:Number,N,K}
	d = N-2
	if d == 0
		throw(ArgumentError("the factor should have at least one mode dimension"))
	end
	if K ≠ d
		throw(ArgumentError("π is not a valid permutation of the mode dimensions of U: π contains $K elements, while U has $N mode dimensions"))
	end
	if !isperm(π)
		throw(ArgumentError("π is not a valid permutation"))
	end
	prm = (1,(π.+1)...,d+2)
	permutedims(U, prm)
end

factormodetranspose(U::Factor{T,N}, π::Vector{Int}) where {T<:Number,N} = factormodetranspose(U, Tuple(π))
factormodetranspose(U::Factor{T,2}) where {T<:Number} = factormodetranspose(U, (2,1))

function factormodereshape(U::Factor{T,N}, n::FactorSize) where {T<:Number,N}
	d = N-2
	p,q = factorranks(U)
	if prod(n) ≠ prod(factorsize(U))
		throw(DimensionMismatch("n is inconsistent with U"))
	end
	reshape(U, p, n..., q)
end

factormodereshape(U::Factor{T,N}, n::Vector{Any}) where {T<:Number,N} = factormodereshape(U, Vector{Int}())

function factordiagm(U::Factor{T,N}) where {T<:Number,N}
	d = N-2
	if d == 0
		throw(ArgumentError("the factor should have at least one mode dimension"))
	end
	p,q = factorranks(U)
	n = factorsize(U)
	U = reshape(U, p, prod(n), q)
	V = zeros(T, p, prod(n), prod(n), q)
	for β ∈ 1:q, i ∈ 1:prod(n), α ∈ 1:p
		V[α,i,i,β] = U[α,i,β]
	end
	reshape(V, p, n..., n..., q)
end

function factorcontract(U::Factor{T,N}, V::Factor{T,N}; rev::Bool=false, major::String="last") where {T<:Number,N}
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	rev && ((U,V) = (V,U))
	m = factorsize(U); p,r = factorranks(U)
	n = factorsize(V); s,q = factorranks(V)
	if r ≠ s
		throw(ArgumentError("U and V have inconsistent ranks"))
	end
	d = length(m)
	if d == 0
		rev && return V*U
		return U*V
	end
	U = reshape(U,(p*prod(m),r)); V = reshape(V,(r,prod(n)*q))
	W = U*V; W = reshape(W, (p,m...,n...,q))
	prm = collect(1:2d); prm = reshape(prm, (d,2))
	(major == "first") && (prm = reverse(prm; dims=2))
	prm = prm'; prm = prm[:]
	W = permutedims(W, [1,(prm.+1)...,2*d+2])
	k = [m..., n...]; k = k[prm]; k = reshape(k, (2,d)); k = prod(k; dims=1)
	reshape(W, (p,k...,q))
end

function factorcontract(U::Factor{T,N}, V::S) where {T<:Number,N,S<:AbstractMatrix{T}}
	n = factorsize(U)
	p,r = factorranks(U)
	s,q = size(V)
	if r ≠ s
		throw(ArgumentError("U and V have inconsistent ranks"))
	end
	U = reshape(U, p*prod(n), r)
	W = U*V
	reshape(W, p, n..., q)
end

function factorcontract(U::S, V::Factor{T,N}) where {T<:Number,N,S<:AbstractMatrix{T}}
	n = factorsize(V)
	p,r = size(U)
	s,q = factorranks(V)
	if r ≠ s
		throw(ArgumentError("U and V have inconsistent ranks"))
	end
	V = reshape(V, r, prod(n)*q)
	W = U*V
	reshape(W, p, n..., q)
end

factorcontract(U::S, V::R) where {T<:Number,S<:AbstractMatrix{T},R<:AbstractMatrix{T}} = U*V


function factormp(U₁::Factor{T,N₁}, σ₁::Indices, U₂::Factor{T,N₂}, σ₂::Indices) where {T<:Number,N₁,N₂}
	n₁ = factorsize(U₁); d₁ = factorndims(U₁)
	n₂ = factorsize(U₂); d₂ = factorndims(U₂)
	if isa(σ₁, Vector{Any}) && length(σ₁) > 0
		throw(ArgumentError("if σ₁ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	if isa(σ₂, Vector{Any}) && length(σ₂) > 0
		throw(ArgumentError("if σ₂ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	isa(σ₁, Vector{Int}) || (σ₁ = indvec(σ₁; max=d₁))
	isa(σ₂, Vector{Int}) || (σ₂ = indvec(σ₂; max=d₂))
	if length(σ₁) ≠ length(σ₂)
		throw(ArgumentError("the specified sets of modes of σ₁ and σ₂ are inconsistent"))
	end
	if σ₁ ⊈ 1:d₁ || unique(σ₁) ≠ σ₁
		throw(ArgumentError("the set of modes of U₁ is specified incorrectly"))
	end
	if σ₂ ⊈ 1:d₂ || unique(σ₂) ≠ σ₂
		throw(ArgumentError("the set of modes of U₂ is specified incorrectly"))
	end
	if n₁[σ₁] ≠ n₂[σ₂]
		throw(ArgumentError("U₁ and U₂ are inconsistent with respect to the specified modes"))
	end
	τ₁ = setdiff(1:d₁, σ₁)
	τ₂ = setdiff(1:d₂, σ₂)
	(p₁,q₁) = factorranks(U₁); (p₂,q₂) = factorranks(U₂)
	U₁ = permutedims(U₁, (1,d₁+2,(τ₁.+1)...,(σ₁.+1)...))
	nτ₁ = Vector{Int}(n₁[τ₁])
	nσ₁ = Vector{Int}(n₁[σ₁])
	nτ₂ = Vector{Int}(n₂[τ₂])
	nσ₂ = Vector{Int}(n₂[σ₂])
	U₁ = reshape(U₁, p₁*q₁*prod(nτ₁), prod(nσ₁))
	U₂ = permutedims(U₂, [(σ₂.+1)...,(τ₂.+1)...,1,d₂+2])
	U₂ = reshape(U₂, prod(nσ₂), prod(nτ₂)*p₂*q₂)
	U = U₁*U₂; n = [nτ₁..., nτ₂...]; d = length(τ₁) + length(τ₂)
	if d == 0
		n = Vector{Int}()
	end
	U = reshape(U, p₁, q₁, n..., p₂, q₂)
	U = permutedims(U, (1,d+3,(3:d+2)...,2,d+4))
	reshape(U, p₁*p₂, n..., q₁*q₂)
end

# function factorkp(U::Factor{T,N}, V::Vararg{Factor{T,N},M}) where {T<:Number,N,M}
# 	d = factorndims(U)
# 	nf = 1+length(V)
# 	n = Matrix{Int}(undef, d, nf)
# 	p = Vector{Int}(undef, nf)
# 	q = Vector{Int}(undef, nf)
# 	n[:,1] = factorsize(U); p[1],q[1] = factorranks(U)
# 	for k ∈ 2:nf
# 		n[:,k] = factorsize(V[k-1]); p[k],q[k] = factorranks(V[k-1])
# 		U = factormp(U, [], V[k-1], [])
# 	end
# 	prm = collect(1:nf*d); prm = reshape(prm, d, nf)
# 	prm = prm'; prm = prm[:]
# 	U = permutedims(U, [1,(prm.+1)...,nf*d+2])
# 	p,q = prod(p),prod(q); n = prod(n; dims=2)
# 	reshape(U, p, n..., q)
# end

# function factorkp2(U::Factor{T,N}, V::Factor{T,N}) where {T<:Number,N}
# 	d = factorndims(U)
# 	n = factorsize(U).*factorsize(V)
# 	p,q = factorranks(U).*factorranks(V)
# 	W = factormp(U, [], V, [])
# 	prm = collect(1:2*d); prm = reshape(prm, d, 2)
# 	prm = prm'; prm = prm[:]
# 	W = permutedims(W, [1,(prm.+1)...,2*d+2])
# 	reshape(W, p, n..., q)
# end

"""
    factorkp(U::Union{Factor{T,N}, Pair{Factor{T,N},Int}}, V::Vararg{Union{Factor{T,N}, Pair{Factor{T,N},Int}},M}) where {T<:Number,N,M}

Perform a Kronecker product of multiple factors (optionally raised to specified nonnegative integer exponents).

# Arguments
- `U::Union{Factor{T, N}, Pair{Factor{T, N}, Int}}`: first factor can either be a `Factor` type or a pair `(Factor, Int)`. If given as a pair, the integer is the exponent for the respective factor in the Kronecker product.
- `V::Vararg{Union{Factor{T, N}, Pair{Factor{T, N}, Int}}, M}`: variable number of additional factors, each of which can also be either a `Factor` type or a pair `(Factor, Int)`. The same usage for the integer applies as in the above line.

# Returns
- `W`: resulting tensor (or matrix if d = 0) after the Kronecker products of all provided factors (with optionally some factors exponentiated). Final tensor is a result of a series of multiplications and reshaping operations.

# Throws
- `ArgumentError`: If a negative exponent is provided in a pair `(Factor, Int)`.

"""

function factorkp(U::Union{Factor{T,N},Pair{Factor{T,N},Int}}, V::Vararg{Union{Factor{T,N},Pair{Factor{T,N},Int}},M}) where {T<:Number,N,M}
	V = (U,V...)
	nf = length(V)
	U = Vector{Factor{T,N}}(undef, nf)
	s = Vector{Int}(undef, nf)
	for k ∈ 1:nf
		W = V[k]
		if isa(W, Pair)
			if W[2] < 0
				throw(ArgumentError("all the specified exponents should be nonnegative"))
			end
			U[k] = W[1]
			s[k] = W[2]
		else
			U[k] = W
			s[k] = 1
		end
	end
	m = findfirst(s .> 0)
	d = factorndims(U[m])
	W = U[m]
	n = factorsize(U[m])
	p,q = factorranks(U[m])
	for i ∈ 2:s[m]
		W = factormp(W, [], U[m], [])
		n = n.*factorsize(U[m])
		p,q = (p,q).*factorranks(U[m])
	end
	for k ∈ m+1:nf
		for i ∈ 1:s[k]
			W = factormp(W, [], U[k], [])
			n = n.*factorsize(U[k])
			p,q = (p,q).*factorranks(U[k])
		end
	end
	nf = sum(s)
	prm = collect(1:nf*d); prm = reshape(prm, d, nf)
	prm = prm'; prm = prm[:]
	if d > 0
		W = permutedims(W, [1,(prm.+1)...,nf*d+2])
		W = reshape(W, p, n..., q)
	else
		W = reshape(W, p, q)
	end
	W
end



function factorhp(U::Factor{T,N}, V::Factor{T,N}) where {T<:Number,N}
	m = factorsize(U); (p,q) = factorranks(U)
	n = factorsize(V); (r,s) = factorranks(V)
	if m ≠ n
		throw(ArgumentError("U and V should have the same mode size"))
	end
	m = prod(n); U = reshape(U, (p,m,q)); V = reshape(V, (r,m,s))
	W = Array{T,5}(undef, p, r, m, q, s)
	for i ∈ 1:m
		W[:,:,i,:,:] = reshape(kron(V[:,i,:], U[:,i,:]), (p,r,q,s))
	end
	reshape(W, p*r, n..., q*s)
end

function factorhp(U₁::Factor{T,N₁}, σ₁::Indices, U₂::Factor{T,N₂}, σ₂::Indices) where {T<:Number,N₁,N₂}
	n₁ = factorsize(U₁); d₁ = factorndims(U₁)
	n₂ = factorsize(U₂); d₂ = factorndims(U₂)
	if isa(σ₁, Vector{Any}) && length(σ₁) > 0
		throw(ArgumentError("if σ₁ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	if isa(σ₂, Vector{Any}) && length(σ₂) > 0
		throw(ArgumentError("if σ₂ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	isa(σ₁, Vector{Int}) || (σ₁ = indvec(σ₁; max=d₁))
	isa(σ₂, Vector{Int}) || (σ₂ = indvec(σ₂; max=d₂))
	if length(σ₁) ≠ length(σ₂)
		throw(ArgumentError("the specified sets of modes of σ₁ and σ₂ are inconsistent"))
	end
	if σ₁ ⊈ 1:d₁ || unique(σ₁) ≠ σ₁
		throw(ArgumentError("the set of modes of U₁ is specified incorrectly"))
	end
	if σ₂ ⊈ 1:d₂ || unique(σ₂) ≠ σ₂
		throw(ArgumentError("the set of modes of U₂ is specified incorrectly"))
	end
	nσ₁ = n₁[σ₁]
	nσ₂ = n₂[σ₂]
	if nσ₁ ≠ nσ₂
		throw(ArgumentError("U₁ and U₂ are inconsistent with respect to the specified modes"))
	end
	m = prod(nσ₁)
	τ₁ = setdiff(1:d₁, σ₁); nτ₁ = n₁[τ₁]; m₁ = prod(nτ₁)
	τ₂ = setdiff(1:d₂, σ₂); nτ₂ = n₂[τ₂]; m₂ = prod(nτ₂)
	(p₁,q₁) = factorranks(U₁); (p₂,q₂) = factorranks(U₂)
	U₁ = permutedims(U₁, (1,(τ₁.+1)...,d₁+2,(σ₁.+1)...))
	U₁ = reshape(U₁, p₁*m₁*q₁, m)
	U₂ = permutedims(U₂, (1,(τ₂.+1)...,d₂+2,(σ₂.+1)...))
	U₂ = reshape(U₂, p₂*m₂*q₂, m)
	U = Array{T,3}(undef, p₁*m₁*q₁, p₂*m₂*q₂, m)
	@views for i ∈ 1:m
		U[:,:,i] .= U₁[:,i]*transpose(U₂[:,i])
	end
	U = reshape(U, p₁, m₁, q₁, p₂, m₂, q₂, m)
	U = permutedims(U, (1,4,2,7,5,3,6))
	U = reshape(U, p₁*p₂, nτ₁..., nσ₁..., m₂*q₁*q₂)
	U = permutedims(U, (1,(invperm((τ₁...,σ₁...)).+1)...,d₁+2))
	reshape(U, p₁*p₂, n₁..., nτ₂..., q₁*q₂)
end


function factorproject!(V::Factor{T,N}, U::Factor{T,N}, W::Factor{T,N}; rev::Bool=false) where {T<:FloatRC,N}
	n = factorsize(U)
	m = prod(n)
	if factorsize(W) ≠ n
		throw(DimensionMismatch("U and W differ in mode size"))
	end
	if any(factorsize(V) .≠ 1) 
		throw(DimensionMismatch("V should have mode size 1,…,1"))
	end
	p,q = factorranks(U)
	r,s = factorranks(W)
	if rev
		if q ≠ s
			throw(DimensionMismatch("U and W differ in the second rank"))
		end
		if factorranks(V) ≠ (p,r)
			throw(DimensionMismatch("V is inconsistant with U and W in rank"))
		end
		V = reshape(V, p, r)
		U = reshape(U, p, m*q)
		W = reshape(W, r, m*s)
		mul!(V, U, adjoint(W))
		U .-= V*W
		V = reshape(V, p, ones(Int, length(n))..., r)
	else
		if p ≠ r
			throw(DimensionMismatch("U and W differ in the first rank"))
		end
		if factorranks(V) ≠ (s,q)
			throw(DimensionMismatch("V is inconsistant with U and W in rank"))
		end
		V = reshape(V, s, q)
		U = reshape(U, p*m, q)
		W = reshape(W, r*m, s)
		mul!(V, adjoint(W), U)
		U .-= W*V
		V = reshape(V, s, ones(Int, length(n))..., q)
	end
	V
end

"""
    factorqr!(U::Factor{T,N}; rev::Bool=false, factf=(rev ? A -> LinearAlgebra.lq!(A) : A -> LinearAlgebra.qr!(A))) where {T<:FloatRC, N}

Perform a QR or LQ factorization of the tensor `U`, depending on the value of the keyword argument `rev` (reverse). 

# Arguments
- `U::Factor{T, N}`: mutable factor of type `Factor` with elements of type `T` (subtype of `FloatRC`: any real or complex floating point) and with `N` as the number dimensions.
- `rev::Bool=false`: keyword argument that determines the type of factorization. If `false`, performs a QR factorization; if `true`, performs an LQ factorization.
- `factf`: keyword argument that specifies the chosen factorization function. By default, `LinearAlgebra.qr!` and `LinearAlgebra.lq!` are used respectively (depending on `rev`).

# Returns
- tuple `(U, R)`, where:
  - `U`: transformed tensor after applying the QR or LQ factorization. (N-dimensional)
  - `R`: factor tensor obtained by reshaping the factor matrix of the QR or LQ factorization (N-dimensional)

"""

function factorqr!(U::Factor{T,N}; rev::Bool=false, factf=(rev ? A -> LinearAlgebra.lq!(A) : A -> LinearAlgebra.qr!(A))) where {T<:FloatRC,N}
	n = factorsize(U); p,q = factorranks(U); m = ones(Int, length(n))
	if rev
		if p == 0 || q == 0
			R,U = zeros(T, p, m..., 0),zeros(T, 0, n..., q)
		else
			U = reshape(U, p, prod(n)*q)
			R,U = factf(U); R,U = Matrix(R),Matrix(U)
			ρ = size(R, 2)
			R,U = reshape(R, p, m..., ρ),reshape(U, ρ, n..., q)
		end
	else
		if p == 0 || q == 0
			U,R = zeros(T, p, n..., 0),zeros(T, 0, m..., q)
		else
			U = reshape(U, p*prod(n), q)
			U,R = factf(U); U,R = Matrix(U),Matrix(R)
			ρ = size(R, 1)
			U,R = reshape(U, p, n..., ρ),reshape(R, ρ, m..., q)
		end
	end
	U,R
end

factorqr!(U::Factor{T,N}, ::Val{false}; rev::Bool=false) where {T<:FloatRC{<:AbstractFloat},N} = factorqr!(U;  rev=rev)

function factorqr!(U::Factor{T,N}, ::Val{true}; rev::Bool=false, returnS::Bool=false,
	               factf=(A -> LinearAlgebra.qr!(A, LinearAlgebra.ColumnNorm()))) where {T<:FloatRC,N}
	# when returnS==true, a factor S satisfying A ⨝ S = Q if rev==false and S ⨝ A = Q if rev==true is returned
	n = factorsize(U); p,q = factorranks(U); m = ones(Int, length(n))
	if rev
		if p == 0 || q == 0
			R,Q = zeros(T, p, m..., 0),zeros(T, 0, n..., q)
			if returnS
				S = zeros(T, 0, m..., p)
			end
		else
			U = reshape(U, p, prod(n)*q)
			U = permutedims(U) # reallocation
			fact = factf(U)
			π = invperm(fact.p)
			R = permutedims(fact.R[:,π])
			ρ = size(R, 2)
			R = reshape(R, p, m..., ρ)
			Q = permutedims(fact.Q*Matrix{T}(I, ρ, ρ))
			Q = reshape(Q, ρ, n..., q)
			if returnS
				S = inv(fact.R[:,1:ρ])
				(ρ < p) && (S = [S; zeros(T, p-ρ, ρ)])
				S = permutedims(S[π,:])
				S = reshape(S, ρ, m..., p)
			end
		end
	else
		if p == 0 || q == 0
			Q,R = zeros(T, p, n..., 0),zeros(T, 0, m..., q)
			if returnS
				S = zeros(T, q, m..., 0)
			end
		else
			U = reshape(U, p*prod(n), q)
			fact = factf(U)
			π = invperm(fact.p)
			R = fact.R[:,π]
			ρ = size(R, 1)
			R = reshape(R, ρ, m..., q)
			Q = fact.Q*Matrix{T}(I, ρ, ρ)
			Q = reshape(Q, p, n..., ρ)
			if returnS
				S = inv(fact.R[:,1:ρ])
				(ρ < q) && (S = [S; zeros(T, q-ρ, ρ)])
				S = S[π,:]
				S = reshape(S, q, m..., ρ)
			end
		end
	end
	if returnS
		return Q,R,S
	end
	Q,R
end

function factorqradd(Q::Factor{T,N}, R::Union{Factor{T,N},Nothing}, U::Factor{T,N}; rev::Bool=false) where {T<:FloatRC,N}
	# assumes that Q is orthogonal w.r.t the first rank if rev==true and w.r.t the second rank if rev==false
	n = factorsize(Q); m = ones(Int, length(n))
	if factorsize(U) ≠ n
		throw(ArgumentError("Q and U are incompatible in mode size"))
	end
	if rev
		r,q = factorranks(Q)
		s,qq = factorranks(U)
		if isa(R, Nothing)
			R = reshape(Matrix{T}(I, r, r), r, m..., r)
			p = r
		else
			if factorsize(R) ≠ m
				throw(ArgumentError("R should have the same number of mode dimensions as Q and unitary mode sizes"))
			end
			p,rr = factorranks(R)
			if rr ≠ r
				throw(ArgumentError("R and Q are incompatible in rank"))
			end
		end
		if qq ≠ q
			throw(ArgumentError("Q and U are incompatible in the second rank"))
		end
		R = reshape(R, p, r)
		Q = reshape(Q, r, prod(n)*q)
		U = reshape(U, s, prod(n)*q)
		R,Q = lqaddrows(R, Q, U); r = size(R, 2)
		R,Q = reshape(R, p+s, m..., r),reshape(Q, r, n..., q)
	else
		p,r = factorranks(Q)
		pp,s = factorranks(U)
		if isa(R, Nothing)
			R = reshape(Matrix{T}(I, r, r), r, m..., r)
			q = r
		else
			if factorsize(R) ≠ m
				throw(ArgumentError("R should have the same number of mode dimensions as Q and unitary mode sizes"))
			end
			rr,q = factorranks(R)
			if rr ≠ r
				throw(ArgumentError("Q and R are incompatible in rank"))
			end
		end
		if pp ≠ p
			throw(ArgumentError("Q and U are incompatible in the first rank"))
		end
		R = reshape(R, r, q)
		Q = reshape(Q, p*prod(n), r)
		U = reshape(U, p*prod(n), s)
		Q,R = qraddcols(Q, R, U); r = size(R, 1)
		Q,R = reshape(Q, p, n..., r),reshape(R, r, m..., q+s)
	end
	Q,R
end

"""
	factorsvd!(W, m, n; atol=0, rtol=0, rank=0, major="last", rev=false)

produces U and V such that W ≈ U ⋈ V if rev == false and W ≈ V ⋈ U if rev == true
U is orthogonal — with respect to the second or first rank index
                  if rev == false or rev == true respectively
m and n are the mode-size vectors of U and V respectively (at most one of these two arguments may be replaced by ":")
major determines whether the "first" or "last" factor in the product U ⋈ V carries the major bits
rank=0 leads to no rank thresholding
"""
function factorsvd!(W::Factor{T,N},
                    m::Union{FactorSize,Colon},
                    n::Union{FactorSize,Colon};
					soft::S=zero(S),
					hard::S=zero(S),
                    atol::S=zero(S),
                    rtol::S=zero(S),
                    rank::Int=0,
                    major::String="last",
					rev::Bool=false,
					factf=(A -> LinearAlgebra.svd!(A; full=false, alg=LinearAlgebra.QRIteration())) ) where {S<:AbstractFloat,T<:FloatRC{S},N}
	d = factorndims(W)
	k = factorsize(W)
	if isa(m, Colon) && isa(n, Colon)
		throw(ArgumentError("to avoid ambiguity, at lease one of the two mode-size parameters should be specified as a vector"))
	end
	if isa(m, FactorSize)
		if any(m .≤ 0)
			throw(ArgumentError("the elements of the first mode-size parameter, when it is specified, should be positive"))
		end
		if length(m) ≠ d
			throw(DimensionMismatch("the number of entries in the first mode-size vector is inconsistent with the specified factor"))
		end
	else
		m = k.÷n
		if k ≠ m.*n
			throw(DimensionMismatch("not every mode dimension of W is divisible by the corresponding element of the specified mode-size vector"))
		end
	end
	if isa(n, FactorSize)
		if any(n .≤ 0)
			throw(ArgumentError("the elements of the second mode-size vector, when it is specified, should be positive"))
		end
		if length(n) ≠ d
			throw(DimensionMismatch("the number of entries in the second mode-size vector is inconsistent with the specified factor"))
		end
	else
		n = k.÷m
		if k ≠ m.*n
			throw(DimensionMismatch("not every mode dimension of W is divisible by the corresponding element of the specified mode-size vector"))
		end
	end
	if k ≠ m.*n
		throw(DimensionMismatch("the specified mode-size vectors are inconsistent with the specified factor"))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	#
	if soft < 0 || !isfinite(soft)
		throw(ArgumentError("soft, when specified, should be nonnegative and finite"))
	end
	#
	soft² = soft^2
	if !isfinite(soft²)
		throw(ErrorException("overflow encountered while squaring soft, which was passed finite"))
	end
	if soft > 0 && soft² == 0
		throw(ErrorException("underflow encountered while squaring soft, which was passed positive"))
	end
	#
	if hard < 0 || !isfinite(hard)
		throw(ArgumentError("hard, when specified, should be nonnegative and finite"))
	end
	#
	hard² = hard^2
	if !isfinite(hard²)
		throw(ErrorException("overflow encountered while squaring hard, which was passed finite"))
	end
	if hard > 0 && hard² == 0
		throw(ErrorException("underflow encountered while squaring hard, which was passed positive"))
	end
	#
	if atol < 0 || !isfinite(atol)
		throw(ArgumentError("atol, when specified, should be nonnegative and finite"))
	end
	#
	atol² = atol^2
	if !isfinite(atol²)
		throw(ErrorException("overflow encountered while squaring atol, which was passed finite"))
	end
	if atol > 0 && atol² == 0
		throw(ErrorException("underflow encountered while squaring atol, which was passed positive"))
	end
	#
	if rtol < 0 || !isfinite(rtol)
		throw(ArgumentError("rtol, when specified, should be nonnegative and finite"))
	end
	#
	if rank < 0
		throw(ArgumentError("the optional argument rank should be nonnegative"))
	end
	#
	p,q = factorranks(W); prm = collect(1:d)
	k = rev ? [n; m] : [m; n]
	prm = (major == "last") ? [2*prm.-1; 2*prm] : [2*prm; 2*prm.-1]
	W = reshape(W, (p,k[invperm(prm)]...,q))
	W = permutedims(W, vcat(1, prm.+1, 2d+2))
	W = reshape(W, p*prod(k[1:d]), prod(k[d+1:2d])*q)
	μ = norm(W)
	if μ == 0
		ε = zero(S); δ = zero(S); ρ = 0
		σ = Vector{S}()
		U = zeros(T, p, k[1:d]..., ρ)
		V = zeros(T, ρ, k[d+1:2d]..., q)
		rev && ((U,V) = (V,U))
	else
		fact = factf(W)
		U = fact.U
		σ = fact.S
		V = fact.Vt
		μ = norm(σ)
		if !isfinite(μ)
			throw(ErrorException("overflow encountered while computing the norm of the decomposition"))
		end
		rtol = min(rtol, one(S));
		tol = μ*rtol
		if μ > 0 && rtol > 0 && tol == 0
			throw(ErrorException("underflow encountered while computing the absolute accuracy threshold from the relative one"))
		end
		tol² = tol^2
		if μ > 0 && rtol > 0 && tol² == 0
			throw(ErrorException("underflow encountered while computing the squared absolute accuracy threshold from the squared relative one"))
		end
		tol = max(atol, tol)
		σσ,ε,ρ = Auxiliary.threshold(σ, soft, hard, tol, rank)
		δ = ε/μ
		U = U[:,1:ρ]; V = V[1:ρ,:]
		rev ? U = U*Diagonal(σσ) : V = Diagonal(σσ)*V
		U = reshape(U, p, k[1:d]..., ρ)
		V = reshape(V, ρ, k[d+1:2d]..., q)
		rev && ((U,V) = (V,U))
	end
	U,V,ε,δ,μ,ρ,σ
end