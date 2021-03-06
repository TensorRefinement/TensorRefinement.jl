
export FactorSize, Factor, VectorFactor, MatrixFactor
export factorsize, factorranks, factorndims
export factor, factormatrix, factorrankselect, block
export factorvcat, factorhcat, factordcat, factorltcat, factorutcat
export factorranktranspose, factormodetranspose
export factorcontract, factormp, factorkp, factorhp
export factorqr!, factorqradd, factorsvd!


const FactorSize = Vector{Int}
const Factor{T,N} = Array{T,N} where {T<:Number,N}
const VectorFactor{T} = Factor{T,3}
const MatrixFactor{T} = Factor{T,4}

function factorsize(U::Factor{T,N}) where {T<:Number,N}
	if length(U) == 0
		throw(ArgumentError("the factor should not be empty"))
	end
	sz = size(U)
	if length(sz) < 3
		throw(ArgumentError("the factor should have two rank dimensions and at least one mode dimension"))
	end
	return FactorSize(collect(sz[2:end-1]))
end

function factorranks(U::Factor{T,N}) where {T<:Number,N}
	if length(U) == 0
		throw(ArgumentError("the factor should not be empty"))
	end
	sz = size(U)
	if length(sz) < 3
		throw(ArgumentError("the factor should have two rank dimensions and at least one mode dimension"))
	end
	return sz[1],sz[end]
end

function factorndims(U::Factor{T,N}) where {T<:Number,N}
	if N < 3
		throw(ArgumentError("the factor should have two rank dimensions and at least one mode dimension"))
	end
	return N-2
end

function factor(U::Array{T,N}) where {T<:Number,N}
	U = reshape(U, (1,size(U)...,1))
	U = Factor{T,N+2}(U)
	return U
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
	return U
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
	return U
end

function factorrankselect(U::Factor{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	isa(α, Int) && (α = [α])
	isa(β, Int) && (β = [β])
	p,q = factorranks(U)
	α = indvec(α; min=1, max=p)
	β = indvec(β; min=1, max=q)
	if length(α) == 0
		throw(ArgumentError("the index or range for the first rank is empty"))
	end
	if α ⊈ 1:p
		throw(ArgumentError("the index or range for the first rank is incorrect"))
	end
	if length(β) == 0
		throw(ArgumentError("the index or range for the second rank is empty"))
	end
	if β ⊈ 1:q
		throw(ArgumentError("the index or range for the second rank is incorrect"))
	end
	d = N-2
	V = U
	V = selectdim(V, d+2, β)
	V = selectdim(V, 1, α)
	V = copy(V)
	return V
end

function block(U::Factor{T,N}, α::Int, β::Int) where {T<:Number,N}
	p,q = factorranks(U)
	if α ∉ 1:p
		throw(ArgumentError("the first rank index is out of range"))
	end
	if β ∉ 1:q
		throw(ArgumentError("the second rank index is out of range"))
	end
	d = N-2
	V = U
	V = selectdim(V, d+2, β)
	V = selectdim(V, 1, α)
	V = copy(V)
	return V
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
	return cat(U, W...; dims=1)
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
	return cat(U, W...; dims=d+2)
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
	return cat(U, W...; dims=(1,d+2))
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
	return cat(cat(U₁₁, U₂₁; dims=1), cat(U₁₂, U₂₂; dims=1); dims=d+2)
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
	return cat(cat(U₁₁, U₂₁; dims=1), cat(U₁₂, U₂₂; dims=1); dims=d+2)
end

function factorranktranspose(U::Factor{T,N}) where {T<:Number,N}
	d = N-2
	prm = vcat(d+2,(1:d).+1,1)
	U = permutedims(U, prm)
	return U
end

function factormodetranspose(U::Factor{T,N}, π::Union{NTuple{K,Int},Vector{Int}}) where {T<:Number,N,K}
	d = N-2
	if d == 0
		throw(ArgumentError("the factor should have at least one mode dimension"))
	end
	if length(π) ≠ d || !isperm(π)
		throw(ArgumentError("π is not a valid permutation of the mode dimensions of U"))
	end
	isa(π, Vector{Int}) || (π = collect(π))
	prm = vcat(1,π.+1,d+2)
	U = permutedims(U, prm)
	return U
end

factormodetranspose(U::Factor{T,N}) where {T<:Number,N} = factormodetranspose(U, collect(factorndims(U):-1:1))

function factorcontract(U::Factor{T,N}, V::Factor{T,N}; rev::Bool=false, major::String="last") where {T<:Number,N}
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	rev && ((U,V) = (V,U))
	m = factorsize(U); (p,r) = factorranks(U)
	n = factorsize(V); (s,q) = factorranks(V)
	if r ≠ s
		throw(ArgumentError("U and V have inconsistent ranks"))
	end
	d = length(m)
	if d == 0
		throw(ArgumentError("U has no mode dimensions"))
	end
	U = reshape(U,(p*prod(m),r)); V = reshape(V,(r,prod(n)*q))
	W = U*V; W = reshape(W, (p,m...,n...,q))
	prm = collect(1:2d); prm = reshape(prm, (d,2))
	(major == "first") && (prm = reverse(prm; dims=2))
	prm = prm'; prm = prm[:]
	W = permutedims(W, [1,(prm.+1)...,2*d+2])
	k = [m..., n...]; k = k[prm]; k = reshape(k, (2,d)); k = prod(k; dims=1)
	W = reshape(W, (p,k...,q))
	W = Factor{T}(W)
	return W
end

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
	U₁ = reshape(U₁, p₁*q₁*prod(n₁[τ₁]), prod(n₁[σ₁]))
	U₂ = permutedims(U₂, [(σ₂.+1)...,(τ₂.+1)...,1,d₂+2])
	U₂ = reshape(U₂, prod(n₂[σ₂]), prod(n₂[τ₂])*p₂*q₂)
	U = U₁*U₂; n = [n₁[τ₁]..., n₂[τ₂]...]; d = length(τ₁) + length(τ₂)
	if d == 0
		n = [1]; d = 1
	end
	U = reshape(U, (p₁,q₁,n...,p₂,q₂))
	U = permutedims(U, (1,d+3,(3:d+2)...,2,d+4))
	U = reshape(U, (p₁*p₂,n...,q₁*q₂))
	return U
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
# 	U = reshape(U, p, n..., q)
# 	return U
# end

# function factorkp2(U::Factor{T,N}, V::Factor{T,N}) where {T<:Number,N}
# 	d = factorndims(U)
# 	n = factorsize(U).*factorsize(V)
# 	p,q = factorranks(U).*factorranks(V)
# 	W = factormp(U, [], V, [])
# 	prm = collect(1:2*d); prm = reshape(prm, d, 2)
# 	prm = prm'; prm = prm[:]
# 	W = permutedims(W, [1,(prm.+1)...,2*d+2])
# 	W = reshape(W, p, n..., q)
# 	return W
# end

function factorkp(U::Union{Factor{T,N},Tuple{Factor{T,N},Int}}, V::Vararg{Union{Factor{T,N},Tuple{Factor{T,N},Int}},M}) where {T<:Number,N,M}
	V = (U,V...)
	nf = length(V)
	U = Vector{Factor{T,N}}(undef, nf)
	s = Vector{Int}(undef, nf)
	for k ∈ 1:nf
		W = V[k]
		if isa(W, Tuple)
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
	W = permutedims(W, [1,(prm.+1)...,nf*d+2])
	W = reshape(W, p, n..., q)
	return W
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
	W = reshape(W, (p*r,n...,q*s))
	return W
end

function factorqr!(U::Factor{T,N}; rev::Bool=false) where {T<:FloatRC{<:AbstractFloat},N}
	n = factorsize(U); (p,q) = factorranks(U); m = ones(Int, length(n))
	if rev
		U = reshape(U, p, prod(n)*q)
		R,U = lq!(U); R,U = Matrix(R),Matrix(U)
		ρ = size(R, 2)
		R,U = reshape(R, (p,m...,ρ)),reshape(U, (ρ,n...,q))
	else
		U = reshape(U, p*prod(n), q)
		U,R = qr!(U); U,R = Matrix(U),Matrix(R)
		ρ = size(R, 1)
		U,R = reshape(U, (p,n...,ρ)),reshape(R, (ρ,m...,q))
	end
	U = Factor{T,N}(U)
	R = Factor{T,N}(R)
	return U,R
end

factorqr!(U::Factor{T,N}, ::Val{false}; rev::Bool=false) where {T<:FloatRC{<:AbstractFloat},N} = factorqr!(U;  rev=rev)

function factorqr!(U::Factor{T,N}, ::Val{true}; rev::Bool=false) where {T<:FloatRC{<:AbstractFloat},N}
	n = factorsize(U); (p,q) = factorranks(U); m = ones(Int, length(n))
	if rev
		U = reshape(U, p, prod(n)*q)
		U = Matrix(U') # reallocation
		f = qr!(U, Val(true))
		U,R = Matrix(f.Q),Matrix(f.R*adjoint(f.P))
		R,U = Matrix(R'),Matrix(U')
		ρ = size(R, 2)
		R,U = reshape(R, (p,m...,ρ)),reshape(U, (ρ,n...,q))
	else
		U = reshape(U, p*prod(n), q)
		f = qr!(U, Val(true))
		U,R = Matrix(f.Q),Matrix(f.R*adjoint(f.P))
		ρ = size(R, 1)
		U,R = reshape(U, (p,n...,ρ)),reshape(R, (ρ,m...,q))
	end
	U = Factor{T,N}(U)
	R = Factor{T,N}(R)
	return U,R
end

function factorqradd(Q::Factor{T,N}, R::Union{Factor{T,N},Nothing}, U::Factor{T,N}; rev::Bool=false) where {T<:FloatRC{<:AbstractFloat},N}
	# assumes that Q is orthogonal w.r.t the first rank if rev==true and w.r.t the second rank if rev==false
	n = factorsize(Q); m = ones(Int, length(n))
	if factorsize(U) ≠ n
		throw(ArgumentError("Q and U are incompatible in mode size"))
	end
	if rev
		r,q = factorranks(Q); s,qq = factorranks(U)
		if isa(R, Nothing)
			R = reshape(Matrix{T}(I, r, r), (r,m...,r)); p = r
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
		R,Q = reshape(R, (p+s,m...,r)),reshape(Q, (r,n...,q))
	else
		p,r = factorranks(Q); pp,s = factorranks(U)
		if isa(R, Nothing)
			R = reshape(Matrix{S}(I, r, r), (r,m...,r)); q = r
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
		Q,R = reshape(Q, (p,n...,r)),reshape(R, (r,m...,q+s))
	end
	Q,R = Factor{T,N}(Q),Factor{T,N}(R)
	return Q,R
end

"""
	factorsvd!(W, m, n; aTol=0, rTol=0, rank=0, major="last", rev=false)

produces U and V such that W ≈ U ⋈ V if rev == false and W ≈ V ⋈ U if rev == true
U is orthogonal — with respect to the second or first rank index
                  if rev == false or rev == true respectively
m and n are the mode-size vectors of U and V respectively (at most one of the two arguments may be replaced by ":"
major determines whether the "first" or "last" factor in the SKP carries the major bits
rank=0 means no rank thresholding
"""
function factorsvd!(W::Factor{T,N},
                    m::Union{FactorSize,Colon},
                    n::Union{FactorSize,Colon};
                    aTol::S=convert(S, 0),
                    rTol::S=convert(S, 0),
                    rank::Int=0,
                    major::String="last",
					rev::Bool=false)	where {S<:AbstractFloat,T<:FloatRC{S},N}
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
	if aTol < 0 || !isfinite(aTol)
		throw(ArgumentError("aTol, when specified, should be nonnegative and finite"))
	end
	#
	aTol² = aTol.^2
	if !isfinite(aTol²)
		throw(ErrorException("overflow encountered while squaring aTol, which was passed finite"))
	end
	if aTol > 0 && aTol² == 0
		throw(ErrorException("underflow encountered while squaring aTol, which was passed positive"))
	end
	#
	if rTol < 0 || !isfinite(rTol)
		throw(ArgumentError("rTol, when specified, should be nonnegative and finite"))
	end
	#
	if rank < 0
		throw(ArgumentError("the optional argument rank should be nonnegative"))
	end
	ρ = rank
	#
	(p,q) = factorranks(W); prm = collect(1:d)
	k = rev ? [n; m] : [m; n]
	prm = (major == "last") ? [2*prm.-1; 2*prm] : [2*prm; 2*prm.-1]
	W = reshape(W, (p,k[invperm(prm)]...,q))
	W = permutedims(W, vcat(1, prm.+1, 2d+2))
	W = reshape(W, p*prod(k[1:d]), prod(k[d+1:2d])*q)
	U,Σ,V = svd!(W; full=false); V = V';
	σ = copy(Σ); μ = norm(σ)
	if !isfinite(μ)
		throw(ErrorException("overflow encountered while computing the norm of the decomposition"))
	end
	rTol = min(rTol, 1.0);
	τ = μ*rTol
	if μ > 0 && rTol > 0 && τ == 0
		throw(ErrorException("underflow encountered while computing the absolute accuracy threshold from the relative one"))
	end
	τ² = τ^2
	if μ > 0 && rTol > 0 && τ² == 0
		throw(ErrorException("underflow encountered while computing the squared absolute accuracy threshold from the squared relative one"))
	end
	τ² = [aTol²,τ²]; ind = (τ² .> 0)
	τ² = any(ind) ? minimum(τ²[ind]) : 0.0
	if μ == 0
		ε = 0.0; δ = 0.0; ρ = 1
		U = zeros(T, (p,k[1:d]...,1))
		V = zeros(T, (1,k[d+1:2d]...,q))
	else
		ε²,ρ = threshold(σ.^2, τ², ρ); ε = sqrt(ε²); δ = ε/μ
		ρ = max(ρ,1)
		U = U[:,1:ρ]; Σ = Σ[1:ρ]; V = V[1:ρ,:]
		rev ? U = U*Diagonal(Σ) : V = Diagonal(Σ)*V
		U = reshape(U, (p,k[1:d]...,ρ))
		V = reshape(V, (ρ,k[d+1:2d]...,q))
		rev && ((U,V) = (V,U))
	end
	U = Factor{T,N}(U)
	V = Factor{T,N}(V)
	return U,V,ε,δ,μ,ρ,σ
end