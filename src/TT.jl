using LinearAlgebra, Random
using TensorRefinement.Auxiliary

import Base: length, deepcopy, reverse!, ndims, size, ones, zeros
import Base: fill!
import Random: rand!
import LinearAlgebra: rank
import Base: permutedims!
import Base: getindex
import Base: append!, prepend!, push!, pushfirst!, pop!, popfirst!, insert!, deleteat!
import Base: vcat, hcat
import LinearAlgebra: lmul!, rmul!
import Base: *, kron, +
import LinearAlgebra: qr!, svd!

export TT
export length, ndims, size, rank, ranks
export deepcopy, reverse!, permutedims!
export fill!, rand!
export getfirstfactor, getlastfactor, getfactor, setfirstfactor!, setlastfactor!, setfactor!
export rankselect!, rankselect
export getindex
export append!, prepend!, push!, pushfirst!, pop!, popfirst!, insert!, deleteat!
export compose!, compose, composecore!, composecore, composeblock!, composeblock
export vcat, hcat, dcat
export lmul!, rmul!, mul, had, *, kron, ⊗, add, +
export qr!, svd!

struct TT{T<:Number,N}
	factors::Dec{T,N}

	function TT(factors::Dec{T,N}) where {T<:Number,N}
		if N < 2
			throw(ArgumentError("each factor is required to have two rank indices"))
		end
		U = new{T,N}(factors)
		return U
	end

end

const Path = Union{Vector{Int},Int,UnitRange{Int},StepRange{Int,K} where K,NTuple{M,Int} where M,Vector{Any},Nothing,Colon}
const Permutation = Union{NTuple{K,Int},Vector{Int}} where K

function TT(::Type{T}, d::Int, L::Int) where {T<:Number}
	checkndims(d)
	checklength(L)
	factors = Vector{Array{T,d+2}}(undef, L)
	U = TT(factors)
	return U
end

TT(d::Int, L::Int) = TT(Float64, d, L)

function TT(::Type{T}, d::Int) where {T<:Number}
	checkndims(d)
	return TT(Vector{Array{T,d+2}}(undef, 0))
end

TT(d::Int) = TT(Float64, d)

function TT(U::TT{T,N}) where {T<:Number,N}
	return U
end

function TT(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}
	checksize(n[:,:])
	nmat = isa(n, DecSize)
	nlen = size(n, 2)
	rvec = isa(r, DecRank)
	rlen = length(r)
	if any(r .≤ 0)
		throw(ArgumentError("the rank should be a positive integer or a vector of such"))
	end
	if rvec && (rlen < 2)
		throw(ArgumentError("when the rank parameter is specified as a vector, it should contain at least two entries"))
	end
	if rvec && nmat && rlen ≠ nlen+1
		throw(ArgumentError("when the size parameter is specified as a matrix with L columns and the rank parameter is specified as a vector, the latter should contain L+1 entries"))
	end
	if first < 0
		throw(ArgumentError("the first rank, when specified separately, should be positive"))
	end
	if last < 0
		throw(ArgumentError("the last rank, when specified separately, should be positive"))
	end
	if rvec && (first > 0 || last > 0)
		throw(ArgumentError("the first and last ranks are allowed to be specified separately only if the rank parameter is specified as an integer"))
	end
	if len < 0
		throw(ArgumentError("the number of factors, when specified, should be positive"))
	end
	if len > 0
		if nmat && len ≠ nlen
			throw(ArgumentError("when the number of rows in the size matrix is larger than one and the number of factors is specified, the two should be equal"))
		end
		if rvec && len+1 ≠ rlen
			throw(ArgumentError("when the rank parameter is specified as a vector with L+1 entries and the number of factors len is specified, it is required that len=L"))
		end
	end
	L = max(nlen, rlen-1, len)
	if !nmat
		n = repeat(n[:,:], outer=(1,L))
	end
	if !rvec
		r = r*ones(Int, L+1)
		if first > 0
			r[1] = first
		end
		if last > 0
			r[L+1] = last
		end
	end
	checkrank(r; len=L)
	factors = [ Array{T}(undef, r[ℓ], n[:,ℓ]..., r[ℓ+1]) for ℓ ∈ 1:L ]
	U = TT(factors)
	return U
end

TT(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = TT(Float64, n, r; first=first, last=last, len=len)

function TT!(V::Factor{T,N}) where {T<:Number,N}
	factors = [V]
	U = TT(factors)
	return U
end

function TT(V::Factor{T,N}; len::Int=1) where {T<:Number,N}
	if len < 0
		throw(ArgumentError("the number of factors, when specified, should be positive"))
	end
	if len == 0
		return TT(T, N-2)
	end
	if len > 1
		p,q = factorranks(V)
		if p ≠ q
			throw(ArgumentError("the two ranks of V should be equal when the number of factors is specified as larger than one"))
		end
	end
	factors = [ copy(V) for ℓ ∈ 1:len ]
	U = TT(factors)
	return U
end

function length(U::TT{T,N}) where {T<:Number,N}
	L = length(U.factors)
	checklength(L)
	return L
end

function ndims(U::TT{T,N}) where {T<:Number,N}
	d = [ factorndims(V) for V ∈ U.factors ]
	checkndims(d)
	return d[1]
end

function size(U::TT{T,N}) where {T<:Number,N}
	L = length(U)
	d = ndims(U)
	n = [ size(U.factors[ℓ], 1+k) for k ∈ 1:d, ℓ ∈ 1:L ]
	n = n[:,:]
	checksize(n)
	return n
end

function ranks(U::TT{T,N}) where {T<:Number,N}
	L = length(U)
	d = ndims(U)
	p = [ size(U.factors[ℓ], 1) for ℓ ∈ 1:L ]
	q = [ size(U.factors[ℓ], d+2) for ℓ ∈ 1:L ]
	return p,q
end

function rank(U::TT{T,N}) where {T<:Number,N}
	p,q = ranks(U)
	try checkranks(p,q) catch e
		isa(e, DimensionMismatch) && throw(DimensionMismatch("the factors have inconsistent ranks"))
	end
	return [p..., q[end]]
end

deepcopy(U::TT{T,N}) where {T<:Number,N} = TT(deepcopy(U.factors))

function reverse!(W::TT{T,N}) where {T<:Number,N}
	L = length(W)
	reverse!(W.factors)
	for ℓ ∈ 1:L
		W.factors[ℓ] = factorranktranspose(W.factors[ℓ])
	end
	return W
end

function permutedims!(U::TT{T,N}, τ::Permutation) where {T<:Number,N}
	d = N-2
	if d == 0
		throw(ArgumentError("the decomposition should have at least one mode dimension"))
	end
	if length(τ) ≠ d || !isperm(τ)
		throw(ArgumentError("τ is not a valid permutation of the mode dimensions of U"))
	end
	isa(τ, Vector{Int}) || (τ = collect(τ))
	L = length(U)
	for ℓ ∈ 1:L
		U.factors[ℓ] = factormodetranspose(U.fators[ℓ], τ)
	end
	return U
end

function fill!(U::TT{T,N}, v::T) where {T<:Number,N}
	L = length(U)
	for ℓ ∈ 1:L
		fill!(U.factors[ℓ], v)
	end
	return U
end

function rand!(rng::AbstractRNG, U::TT{T,N}) where {T<:Number,N}
	L = length(U)
	for ℓ ∈ 1:L
		rand!(rng, U.factors[ℓ])
	end
	return U
end
rand!(U::TT{T,N}) where {T<:Number,N} = rand!(Random.GLOBAL_RNG, U)

zeros(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = fill!(TT(T, n, r; first=first, last=last, len=len), convert(T, 0))
zeros(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = ttzeros(Float64, n, r; first=first, last=last, len=len)

ones(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = fill!(TT(T, n, r; first=first, last=last, len=len), convert(T, 1))
ones(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = ttzeros(Float64, n, r; first=first, last=last, len=len)

rand(rng::AbstractRNG, ::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = rand!(rng, TT(T, n, r; first=first, last=last, len=len))
rand(rng::AbstractRNG, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = ttrand(rng, Float64, n, r; first=first, last=last, len=len)
rand(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = ttrand(Random.GLOBAL_RNG, T, n, r; first=first, last=last, len=len)
rand(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = ttrand(Random.GLOBAL_RNG, Float64, n, r; first=first, last=last, len=len)

getfirstfactor(U::TT{T,N}) where {T<:Number,N} = U.factors[1]

getlastfactor(U::TT{T,N}) where {T<:Number,N} = U.factors[end]

function getfactor(U::TT{T,N}, ℓ::Int) where {T<:Number,N}
	L = length(U)
	if ℓ ⊈ 1:L
		throw(ArgumentError("ℓ is out of range"))
	end
	return U.factors[ℓ]
end

function setfirstfactor!(U::TT{T,N}, F::Array{T,N}) where {T<:Number,N}
	U.factors[1] = F
	return U
end

function setlastfactor!(U::TT{T,N}, F::Array{T,N}) where {T<:Number,N}
	U.factors[end] = F
	return U
end

function setfactor!(U::TT{T,N}, F::Array{T,N}, ℓ::Int) where {T<:Number,N}
	L = length(U)
	if ℓ ⊈ 1:L
		throw(ArgumentError("ℓ is out of range"))
	end
	U.factors[ℓ] = F
	return U
end

function rankselect!(U::TT{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	# if isa(α, Int) || isa(β, Int)
	# 	throw(ArgumentError("for consistency with Base.selectdim, scalar α and β are not accepted; use α:α or β:β instead of α or β to select a subtensro of the factor whose first or second rank is one"))
	# end
	L = length(U)
	r = rank(U); p,q = r[1],r[L+1]
	α = indvec(α; min=1, max=p)
	β = indvec(β; min=1, max=q)
	if length(α) == 0
		throw(ArgumentError("the range for the first rank is empty"))
	end
	if α ⊈ 1:p
		throw(ArgumentError("the range for the first rank is incorrect"))
	end
	if length(β) == 0
		throw(ArgumentError("the range for the second rank is empty"))
	end
	if β ⊈ 1:q
		throw(ArgumentError("the range for the second rank is incorrect"))
	end
	setlastfactor!(U, factorrankselect(getlastfactor(U), :, β))
	setfirstfactor!(U, factorrankselect(getfirstfactor(U), α, :))
	return U
end

function rankselect(U::TT{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	V = deepcopy(U)
	rankselect!(V, α, β)
	return V
end

getindex(U::TT{T,N}, α::Indices, β::Indices) where {T<:Number,N} = rankselect(U, α, β)

function append!(U::TT{T,N}, V::TT{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if ndims(U) ≠ ndims(V)
		throw(DimensionMismatch("U and V are inconsistent in the number of dimensions"))
	end
	p,q = ranks(U)
	r,s = ranks(V)
	if rankprecheck
		try checkranks(p,q) catch
			throw(ArgumentError("the factors of U have incorrect or inconsistent ranks"))
		end
		try checkranks(r,s) catch
			throw(ArgumentError("the factors of V have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		append!(p, r); append!(q, s);
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return append!(U.factors, V.factors)
end

function prepend!(U::TT{T,N}, V::TT{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if ndims(U) ≠ ndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = ranks(U); r,s = ranks(V)
	if rankprecheck
		try checkranks(p,q) catch
			throw(ArgumentError("the factors of U have incorrect or inconsistent ranks"))
		end
		try checkranks(r,s) catch
			throw(ArgumentError("the factors of V have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		prepend!(p, r); prepend!(q, s);
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return prepend!(U.factors, V.factors)
end

function push!(U::TT{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if ndims(U) ≠ factorndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = ranks(U)
	r,s = factorranks(V)
	if rankprecheck
		try checkranks(p,q) catch
			throw(ArgumentError("the factors of U have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		r,s = factorranks(V); push!(p, r); push!(q, s)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return push!(U.factors, V)
end

function pushfirst!(U::TT{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if ndims(U) ≠ factorndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = ranks(U)
	r,s = factorranks(V)
	if rankprecheck
		try
			checkranks(p,q)
		catch e
			throw(ArgumentError("the factors of U have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		r,s = factorranks(V); pushfirst!(p, r); pushfirst!(q, s)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return pushfirst!(U.factors, V)
end

function pop!(U::TT{T,N}) where {T<:Number,N}
	return pop!(U.factors)
end

function popfirst!(U::TT{T,N}) where {T<:Number,N}
	return popfirst!(U.factors)
end

function insert!(U::TT{T,N}, ℓ::Int, V::Factor{T,N}; path::String="", rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	L = length(U)
	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end
	if ℓ ∉ 1:L
		throw(ArgumentError("ℓ is required to be from 1:L, where L is the number of factors in U"))
	end
	(path == "forward") && (ℓ = ℓ+1)
	(ℓ == 1) && return pushfirst!(U, V; rankprecheck=rankprecheck, rankpostcheck=rankpostcheck)
	(ℓ == L+1) && return push!(U, V; rankprecheck=rankprecheck, rankpostcheck=rankpostcheck)
	p,q = ranks(U)
	if rankprecheck
		try checkranks(p,q) catch
			throw(DimensionMismatch("the factors of U have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		r,s = factorranks(V); insert!(p, ℓ, r); insert!(q, ℓ, s)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U and V are inconsistent for this operation"))
		end
	end
	return insert!(U.factors, ℓ, V)
end

function deleteat!(U::TT{T,N}, Λ::Path; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	L = length(U)
	isa(Λ, Colon) && (Λ = collect(2:L); path = "backward")
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	(length(Λ) == 0) && return W
	if unique(Λ) ≠ Λ
		throw(ArgumentError("the entries of Λ should be unique"))
	end
	if Λ ⊈ 1:L
		throw(ArgumentError("Λ should be an element or a subset of 1:L with unique entries, where L is the number of factors in U"))
	end
	p,q = ranks(U)
	if rankprecheck
		try checkranks(p,q) catch
			throw(DimensionMismatch("the factors of U have incorrect or inconsistent ranks"))
		end
	end
	if rankpostcheck
		deleteat!(p, Λ); deleteat!(q, Λ)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U are inconsistent for this operation"))
		end
	end
	return deleteat!(U.factors, Λ)
end

function compose!(W::TT{T,N}, Λ::Path; path::String="", major::String="last") where {T<:Number,N}
	if path ∉ ("","forward","backward")
		throw(ArgumentError("the value of the keyword argument path should be \"\" (default, accepted only for empty Λ and for Λ=:), \"forward\" or \"backward\""))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	L = length(W)
	rank(W)
	isa(Λ, Colon) && (Λ = collect(2:L); path = "backward")
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	(length(Λ) == 0) && return W
	if path ∉ ("forward","backward")
		throw(ArgumentError("when Λ is neither empty nor a colon, path should be either \"forward\" or \"backward\""))
	end
	(path == "backward") || (Λ .+= 1; path = "backward")
	if unique(Λ) ≠ Λ
		throw(ArgumentError("Λ has duplicate entries"))
	end
	if Λ ⊈ 2:L
		throw(ArgumentError("Λ, when nonempty, should be a colon or a Vector/NTuple/UnitRange of Int, from 1:L-1 for path=\"forward\" and from 2:L for path=\"backward\", where L is the number of factors in W"))
	end
	sort!(Λ; rev=true)
	for ℓ ∈ Λ
		F = getfactor(W, ℓ-1)
		F = factorcontract(F, getfactor(W, ℓ); major=major)
		setfactor!(W, F, ℓ-1)
		deleteat!(W, ℓ; rankprecheck=false, rankpostcheck=false)
	end
	return W
end

compose!(W::TT{T,N}; path::String="", major::String="last") where {T<:Number,N} = compose!(W, :; path=path, major=major)

function compose(W::TT{T,N}, Λ::Path; path::String="", major::String="last") where {T<:Number,N}
	U = deepcopy(W)
	return compose!(U, Λ; path=path, major=major)
end

compose(W::TT{T,N}; path::String="", major::String="last") where {T<:Number,N} = compose(W, :; path=path, major=major)

function composecore(W::TT{T,N}; major::String="last") where {T<:Number,N}
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	L = length(W); rank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	F = copy(getfactor(W, 1))
	for ℓ ∈ 2:L
		F = factorcontract(F, getfactor(W, ℓ); major=major)
	end
	return F
end

function composecore!(U::TT{T,N}; major::String="last") where {T<:Number,N}
	compose!(U; major=major)
	return getfirstfactor(U)
end

function composeblock!(U::TT{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}
	r = rank(U); p,q = r[1],r[end]
	if α ∉ 1:p
		throw(ArgumentError("the first rank index is out of range"))
	end
	if β ∉ 1:q
		throw(ArgumentError("the second rank index is out of range"))
	end
	rankselect!(U, α:α, β:β)
	return compose(U; major=major)
end

function composeblock(U::Dec{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}
	V = deepcopy(U)
	V = composeblock!(V, α, β, major=major)
	return V
end

function vcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M}) where {T<:Number,N,M}
	L = length(U); d = ndims(U); m = size(U); p = rank(U)
	W = (V,W...)
	for V ∈ W
		if length(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if size(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = rank(V)
		if q[L+1] ≠ p[L+1]
			throw(ArgumentError("the decompositions are incompatible in the last rank"))
		end
	end
	Z = TT(T, d, 0)
	for ℓ ∈ 1:L-1
		push!(Z, factordcat(getfactor(U, ℓ), [ getfactor(V, ℓ) for V ∈ W ]...))
	end
	push!(Z, factorvcat(getlastfactor(U), [ getlastfactor(V) for V ∈ W ]...))
	return Z
end

function hcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M}) where {T<:Number,N,M}
	L = length(U); m = size(U); p = rank(U)
	W = (V,W...)
	for V ∈ W
		if length(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if size(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = rank(V)
		if q[1] ≠ p[1]
			throw(ArgumentError("the decompositions are incompatible in the first rank"))
		end
	end
	Z = TT(T, d, 0)
	push!(Z, factorhcat(getfirstfactor(U), [ getfirstfactor(V) for V ∈ W ]...))
	for ℓ ∈ 2:L
		push!(Z, factordcat(getfactor(U, ℓ), [ getfactor(V, ℓ) for V ∈ W ]...))
	end
	return Z
end

function dcat(U::TT{T,N}, V::TT{T,N}, W::Vararg{TT{T,N},M}) where {T<:Number,N,M}
	L = length(U); m = decsize(U)
	W = (V,W...)
	for V ∈ W
		if length(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if size(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
	end
	Z = TT(T, d, 0)
	for ℓ ∈ 1:L
		push!(Z, factordcat(getfactor(U, ℓ), [ getfactor(V, ℓ) for V ∈ W ]...))
	end
	return Z
end

function lmul!(α::T, U::TT{T,N}) where {T<:Number,N}
	F = getlastfactor(U)
	F .= α * F
	return U
end

rmul!(U::TT{T,N}, α::T) where {T<:Number,N} = lmul!(α, U)

function *(α::T, U::TT{T,N}) where {T<:Number,N}
	V = deepcopy(U)
	return lmul!(α, U)
end

*(U::TT{T,N}, α::T) where {T<:Number,N} = *(α, U)

function mul(U₁::TT{T,N₁}, σ₁::Indices, U₂::TT{T,N₂}, σ₂::Indices) where {T<:Number,N₁,N₂}
	n₁ = size(U₁); d₁ = ndims(U₁)
	n₂ = size(U₂); d₂ = ndims(U₂)
	L = length(U₁)
	if length(U₂) ≠ L
		throw(ArgumentError("U₁ and U₂ differ in the number of factors"))
	end
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
	if n₁[σ₁,:] ≠ n₂[σ₂,:]
		throw(ArgumentError("U₁ and U₂ are inconsistent with respect to the specified modes"))
	end
	factors = [ factormp(getfactor(U₁, ℓ), σ₁, getfactor(U₂, ℓ), σ₂) for ℓ ∈ 1:L ]
	return TT(factors)
end

function had(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N}
	L = length(U)
	if length(V) ≠ L
		throw(ArgumentError("U and V differ in the number of factors"))
	end
	if size(U) ≠ size(V)
		throw(ArgumentError("U and V are inconsistent in mode size"))
	end
	factors = [ factorhp(getfactor(U, ℓ), getfactor(V, ℓ)) for ℓ ∈ 1:L ]
	return TT(factors)
end

*(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N} = had(U, V)

function kron(U₁::Dec{T,N}, U₂::Dec{T,N}) where {T<:Number,N}
	L = length(U₁)
	if length(U₂) ≠ L
		throw(ArgumentError("U₁ and U₂ differ in the number of factors"))
	end
	if ndims(U₁) ≠ ndims(U₂)
		throw(ArgumentError("U₁ and U₂ are inconsistent in the number of mode dimensions"))
	end
	factors = [ factorkp(U₁[ℓ], U₂[ℓ]) for ℓ ∈ 1:L ]
	return TT(factors)
end

⊗(U₁::Dec{T,N}, U₂::Dec{T,N}) where {T<:Number,N} = kron(U₁, U₂)

function add(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N}
	m = size(U); L = length(U)
	if length(V) ≠ L
		throw(ArgumentError("U and V differ in the number of factors"))
	end
	(L == 0) && return TT(T, d, 0)
	p = rank(U); q = rank(V)
	if size(V) ≠ m
		throw(ArgumentError("U and V are inconsistent in mode size"))
	end
	if q[1] ≠ p[1]
		throw(ArgumentError("the decompositions are incompatible in the first rank"))
	end
	if q[L+1] ≠ p[L+1]
		throw(ArgumentError("the decompositions are incompatible in the last rank"))
	end
	W = TT(T, d, 0)
	if L == 1
		push!(W, getfirstfactor(U) + getfirstfactor(V))
	elseif L > 1
		push!(W, factorhcat(getfirstfactor(U), getfirstfactor(V)))
		for ℓ ∈ 2:L-1
			push!(W, factordcat(getfactor(U, ℓ), getfactor(V, ℓ)))
		end
		push!(W, factorvcat(getlastfactor(U), getlastfactor(V)))
	end
	return W
end

+(U::TT{T,N}, V::TT{T,N}) where {T<:Number,N} = add(U, V)

function qr!(W::TT{T,N}, Λ::Path; path::String="") where {T<:FloatRC,N}
	L = length(W); rank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if path ∉ ("","forward","backward")
		throw(ArgumentError("path should be either \"\" (default, accepted only when path can be deduced from Λ), \"forward\" or \"backward\""))
	end
	if path == "" && (isa(Λ, Colon) || length(unique(Λ)) == 1)
		throw(ArgumentError("path cannot be deduced from Λ and should be specified as either \"forward\" or \"backward\""))
	end
	isa(Λ, Colon) && (Λ = (path == "forward") ? (1:L) : (L:-1:1))
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	(length(Λ) == 0) && return W
	if unique(Λ) ≠ Λ
		throw(ArgumentError("Λ has duplicate entries"))
	end
	if (minimum(Λ):maximum(Λ)) ⊈ Λ
		throw(ArgumentError("the entries of Λ should form a set of contiguous integers"))
	end
	if Λ ⊈ 1:L
		throw(ArgumentError("Λ is out of range"))
	end
	fw,bw = issorted(Λ, rev=false),issorted(Λ, rev=true)
	if !fw && !bw
		throw(ArgumentError("Λ is not sorted in ascending or descending order, so it is inconsistent with any forward or backward path"))
	end
	@assert !(fw && bw && path == "")
	fw && path == "" && (path = "forward")
	bw && path == "" && (path = "backward")
	if path == "forward" && !fw
		throw(ArgumentError("Λ is not sorted in ascending order, so it is inconsistent with any forward path"))
	end
	if path == "backward" && !bw
		throw(ArgumentError("Λ is not sorted in descending order, so it is inconsistent with any backward path"))
	end
	M = length(Λ)
	for λ ∈ 1:M
		ℓ = Λ[λ]
		Q,R = factorqr!(getfactor(W, ℓ); rev=(path == "backward"))
		setfactor!(W, Q, ℓ)
		if λ < M
			ν = Λ[λ+1]
			F = factorcontract(R, getfactor(W ,ν), rev=(path == "backward"))
			setfactor!(W, F, ν)
		else
			insert!(W, ℓ, R; path=path, rankprecheck=false, rankpostcheck=true)
		end
	end
	return W
end

qr!(W::TT{T,N}; path::String="") where {T<:FloatRC,N} = qr!(W, :; path=path)

function svd!(W::TT{T,N}, Λ::Path, n::Union{Colon,DecSize}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") where {T<:FloatRC,N}
	# the decomposition is assumed to be orthogonal
	L = length(W); rank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if path ∉ ("","forward","backward")
		throw(ArgumentError("path should be either \"\" (default, accepted only when path can be deduced from Λ), \"forward\" or \"backward\""))
	end
	if path == "" && (isa(Λ, Colon) || length(unique(Λ)) == 1)
		throw(ArgumentError("path cannot be deduced from Λ and should be specified as either \"forward\" or \"backward\""))
	end
	isa(Λ, Colon) && (Λ = (path == "forward") ? (1:L) : (L:-1:1))
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	(length(Λ) == 0) && return W
	if (minimum(Λ):maximum(Λ)) ⊈ Λ
		throw(ArgumentError("the entries of Λ should form a set of contiguous integers"))
	end
	if Λ ⊈ 1:L
		throw(ArgumentError("Λ is out of range"))
	end
	fw,bw = issorted(Λ, rev=false),issorted(Λ, rev=true)
	if !fw && !bw
		throw(ArgumentError("Λ is not sorted in ascending or descending order, so it is inconsistent with any forward or backward path"))
	end
	fw && path == "" && (path = "forward")
	bw && path == "" && (path = "backward")
	if path == "forward" && !fw
		throw(ArgumentError("Λ is not sorted in ascending order, so it is inconsistent with any forward path"))
	end
	if path == "backward" && !bw
		throw(ArgumentError("Λ is not sorted in descending order, so it is inconsistent with any backward path"))
	end
	m = size(W)
	isa(n, Colon) && (n = m[:,Λ])
	if size(n,1) ≠ size(m,1)
		throw(ArgumentError("the number of rows in n should be equal to the number of dimensions in each factor of W"))
	end
	if size(n,2) ≠ length(Λ)
		throw(ArgumentError("the number of columns in n should be equal to the number of elements in Λ"))
	end
	for ℓ ∈ unique(Λ)
		λ = (Λ .== ℓ)
		if m[:,ℓ] ≠ prod(n[:,λ]; dims=2)[:]
			throw(ArgumentError("n and Λ are incompatible with the size of the factors of W"))
		end
	end
	K = length(Λ)

	if any(aTol .< 0)
		throw(ArgumentError("aTol should be a nonnegative Float64 or a vector of such"))
	end
	if isa(aTol, Float64)
		aTol = aTol*ones(K)/sqrt(K)
	elseif length(aTol) ≠ K
		throw(ArgumentError("aTol, passed as a vector, has incorrect length"))
	end

	if any(aTolDistr .< 0)
		throw(ArgumentError("aTolDistr should be a nonnegative Float64 or a vector of such"))
	end
	if isa(aTolDistr, Float64)
		aTolDistr = aTolDistr*ones(K)/sqrt(K)
	elseif length(aTolDistr) ≠ K
		throw(ArgumentError("aTolDistr, passed as a vector, has incorrect length"))
	end

	if any(rTol .< 0)
		throw(ArgumentError("rTol should be a nonnegative Float64 or a vector of such"))
	end
	if isa(rTol, Float64)
		rTol = rTol*ones(K)/sqrt(K)
	elseif length(rTol) ≠ K
		throw(ArgumentError("rTol, passed as a vector, has incorrect length"))
	end

	if any(rTolDistr .< 0)
		throw(ArgumentError("rTolDistr should be a nonnegative Float64 or a vector of such"))
	end
	if isa(rTolDistr, Float64)
		rTolDistr = rTolDistr*ones(K)/sqrt(K)
	elseif length(rTolDistr) ≠ K
		throw(ArgumentError("rTolDistr, passed as a vector, has incorrect length"))
	end

	if any(maxrank .< 0)
		throw(ArgumentError("maxrank should be a nonnegative Int or a vector of such"))
	end
	if isa(maxrank, Int)
		maxrank = maxrank*ones(Int, K)
	elseif length(maxrank) ≠ K
		throw(ArgumentError("maxrank, passed as a vector, has incorrect length"))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end

	ε = zeros(Float64, K); δ = zeros(Float64, K)
	σ = Vector{Vector{Float64}}(undef, K)
	ρ = zeros(Int, K)
	μ = 0.0
	aTolAcc = 0.0; rTolAcc = 0.0
	for λ ∈ 1:K
		F = getfactor(W, Λ[λ])
		if λ == 1
			ε₁ = [aTol[λ],aTolDistr[λ]]; ε₁ = ε₁[ε₁ .> 0]; ε₁ = isempty(ε₁) ? 0.0 : minimum(ε₁)
			δ₁ = [rTol[λ],rTolDistr[λ]]; δ₁ = δ₁[δ₁ .> 0]; δ₁ = isempty(δ₁) ? 0.0 : minimum(δ₁)
			U,V,ε[1],δ[1],μ,ρ[1],σ[1] = factorsvd!(F, n[:,λ], :; aTol=ε₁, rTol=δ₁, rank=maxrank[λ], rev=(path == "backward"), major=major)
		else
			(aTolDistr[λ] > 0) && (aTolDistr[λ] = sqrt(aTolDistr[λ]^2+aTolAcc^2); aTolAcc = 0.0)
			(rTolDistr[λ] > 0) && (rTolDistr[λ] = sqrt(rTolDistr[λ]^2+rTolAcc^2); rTolAcc = 0.0)
			ε₁ = [aTol[λ],aTolDistr[λ],μ*rTol[λ],μ*rTolDistr[λ]]; ε₁ = ε₁[ε₁ .> 0]
			ε₁ = isempty(ε₁) ? 0.0 : minimum(ε₁)
			U,V,ε[λ],_,_,ρ[λ],σ[λ] = factorsvd!(F, n[:,λ], :; aTol=ε₁, rank=maxrank[λ], rev=(path == "backward"), major=major)
			δ[λ] = (μ > 0) ? ε[λ]/μ : 0.0
		end
		setfactor!(W, U, Λ[λ])
		if λ < K && Λ[λ+1] ≠ Λ[λ]
			F = getfactor(W, Λ[λ+1])
			F = factorcontract(V, F, rev=(path == "backward"), major=major)
			setfactor!(W, F, Λ[λ+1])
		else
			insert!(W, Λ[λ], V; path=path, rankprecheck=false, rankpostcheck=true)
			(path == "forward") && (Λ[λ:end] .+= 1)
		end
	end
	return W,ε,δ,μ,ρ,σ
end

svd!(W::TT{T,N}, Λ::Path; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") where {T<:FloatRC,N} = svd!(W, Λ, :; path=path, aTol=aTol, aTolDistr=aTolDistr, rTol=rTol, rTolDistr=rTolDistr, maxrank=maxrank, major=major)

svd!(W::TT{T,N}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, maxrank::Int2=0, major::String="last") where {T<:FloatRC,N} = svd!(W, :, :; path=path, aTol=aTol, aTolDistr=aTolDistr, rTol=rTol, rTolDistr=rTolDistr, maxrank=maxrank, major=major)
