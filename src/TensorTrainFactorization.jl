export DecSize, DecRank, Dec, VectorDec, MatrixDec
export checkndims, checklength, checkndims, checksize, checkrank, checkranks
export declength, decndims, decsize, decranks, decrank
export dec, dec!, vector, decrankselect!, decrankselect, factor!, factor, block!, block, decvcat, dechcat, decdcat
export decscale!, decreverse!, decmodetranspose!, decfill!, decrand!
export deczeros, decones, decrand
export decappend!, decprepend!, decpush!, decpushfirst!, decpop!, decpopfirst!, decinsert!, decdeleteat!
export decinsertidentity!
export decskp!, decskp, decmp, deckp, decadd, dechp
export decqr!, decsvd!

const DecSize = Matrix{Int}
const DecRank = Vector{Int}
const Dec{T,N} = Vector{Factor{T,N}} where {T<:Number,N}
const VectorDec{T} = Vector{VectorFactor{T}} where {T<:Number}
const MatrixDec{T} = Vector{MatrixFactor{T}} where {T<:Number}


function checkndims(d::Int)
	if d < 0
		throw(ArgumentError("d should be nonnegative"))
	end
end

function checklength(L::Int)
	if L < 0
		throw(ArgumentError("the number of factors should be nonnegative"))
	end
end

function checkndims(d::Vector{Int})
	L = length(d)
	if L == 0
		throw(ArgumentError("d is empty"))
	end
	if any(d .≤ 0)
		throw(BoundsError("the elements of d should be positive"))
	end
	if length(unique(d)) ≠ 1
		throw(ArgumentError("the values in the dimension vector are not identical"))
	end
end

function checksize(n::DecSize; len::Int=0, dim::Int=0)
	if length(n) == 0
		throw(ArgumentError("the size matrix is empty"))
	end
	if any(n .≤ 0)
		throw(ArgumentError("the elements of the size matrix should be positive"))
	end
	if len < 0
		throw(ArgumentError("the number of factors should be positive"))
	end
	if dim < 0
		throw(ArgumentError("the number of mode dimensions should be positive"))
	end
	if dim > 0 && size(n, 1) ≠ dim
		throw(ArgumentError("the number of rows in n is incorrect"))
	end
	if len > 0 && size(n, 2) ≠ len
		throw(ArgumentError("the number of columns in n is incorrect"))
	end
end

function checkrank(r::DecRank; len::Int=0)
	if length(r) < 2
		throw(ArgumentError("the rank vector should contain at least two elements"))
	end
	if any(r .≤ 0)
		throw(ArgumentError("the elements of the rank vector should be positive"))
	end
	if len > 0 && length(r) ≠ len+1
		throw(ArgumentError("the number of elements in the rank vector is incorrect"))
	end
end

function checkranks(p::Vector{Int}, q::Vector{Int}; len::Int=0)
	if length(p) ≠ length(q)
		throw(ArgumentError("the rank vectors should have the same length"))
	end
	if any([p q] .≤ 0)
		throw(ArgumentError("the elements of p and q should be positive"))
	end
	if p[2:end] ≠ q[1:end-1]
		throw(DimensionMismatch("the ranks are inconsistent"))
	end
	if len > 0 && length(p) ≠ len+1
		throw(ArgumentError("the number of elements in the rank vectors is incorrect"))
	end
end

function declength(U::Dec{T,N}) where {T<:Number,N}
	L = length(U)
	checklength(L)
	return L
end

function decndims(U::Dec{T,N}) where {T<:Number,N}
	# L = declength(U)
	# d = [ ndims(U[ℓ])-2 for ℓ ∈ 1:L ]
	d = [ factorndims(V) for V ∈ U ]
	checkndims(d)
	return d[1]
end

function decsize(U::Dec{T,N}) where {T<:Number,N}
	L = length(U)
	d = decndims(U)
	n = [ size(U[ℓ], 1+k) for k ∈ 1:d, ℓ ∈ 1:L ]
	n = n[:,:]
	checksize(n)
	return n
end

function decranks(U::Dec{T,N}) where {T<:Number,N}
	L = length(U)
	d = decndims(U)
	p = [ size(U[ℓ], 1) for ℓ ∈ 1:L ]
	q = [ size(U[ℓ], d+2) for ℓ ∈ 1:L ]
	return p,q
end

function decrank(U::Dec{T,N}) where {T<:Number,N}
	p,q = decranks(U)
	try checkranks(p,q) catch e
		isa(e, DimensionMismatch) && throw(DimensionMismatch("the factors of the decomposition have inconsistent ranks"))
	end
	return [p..., q[end]]
end

function dec(::Type{T}, d::Int, L::Int) where {T<:Number}
	checkndims(d)
	checklength(L)
	U = Vector{Array{T,d+2}}(undef, L)
	return U
end

dec(d::Int, L::Int) = dec(Float64, d, L)

function dec(::Type{T}, d::Int) where {T<:Number}
	checkndims(d)
	return Vector{Array{T,d+2}}(undef, 0)
end

dec(d::Int) = dec(Float64, d, L)

function dec(U::Dec{T,N}) where {T<:Number,N}
	return U
end

function dec(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number}
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
	U = [ Array{T}(undef, r[ℓ], n[:,ℓ]..., r[ℓ+1]) for ℓ ∈ 1:L ]
	return U
end

dec(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = dec(Float64, n, r; first=first, last=last, len=len)

function dec!(U::Factor{T,N}) where {T<:Number,N}
	return [U]
end

function dec(U::Factor{T,N}; len::Int=1) where {T<:Number,N}
	if len < 0
		throw(ArgumentError("the number of factors, when specified, should be positive"))
	end
	if len == 0
		return dec(T, N-2)
	end
	if len > 1
		p,q = factorranks(U)
		if p ≠ q
			throw(ArgumentError("the two ranks of U should be equal when the number of factors is specified as larger than one"))
		end
	end
	return [ copy(U) for ℓ ∈ 1:len ]
end

function vector(U::Dec{T,N}) where {T<:Number,N}
	return U
end

function decrankselect!(U::Dec{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	# if isa(α, Int) || isa(β, Int)
	# 	throw(ArgumentError("for consistency with Base.selectdim, scalar α and β are not accepted; use α:α or β:β instead of α or β to select a subtensor of the factor whose first or second rank is one"))
	# end
	L = declength(U)
	r = decrank(U); p,q = r[1],r[L+1]
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
	U[L] = factorrankselect(U[L], :, β)
	U[1] = factorrankselect(U[1], α, :)
	return U
end

function decrankselect(U::Dec{T,N}, α::Indices, β::Indices) where {T<:Number,N}
	V = deepcopy(U)
	decrankselect!(V, α, β)
	return V
end

function factor!(U::Dec{T,N}; major::String="last") where {T<:Number,N}
	decskp!(U; major=major); U = U[1]
	return U
end

function factor(U::Dec{T,N}; major::String="last") where {T<:Number,N}
	V = deepcopy(U)
	V = factor!(V; major=major)
	return V
end

function block!(U::Dec{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}
	L = declength(U)
	r = decrank(U); p = r[1]; q = r[L+1]
	if α ∉ 1:p
		throw(ArgumentError("the first rank index is out of range"))
	end
	if β ∉ 1:q
		throw(ArgumentError("the second rank index is out of range"))
	end
	decrankselect!(U, α:α, β:β)
	decskp!(U; major=major); U = U[1]
	U = block(U, 1, 1)
	return U
end

function block(U::Dec{T,N}, α::Int, β::Int; major::String="last") where {T<:Number,N}
	V = deepcopy(U)
	V = block!(V, α, β, major=major)
	return V
end

function decvcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
	L = declength(U); m = decsize(U); p = decrank(U)
	W = (V,W...)
	for V ∈ W
		if declength(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if decsize(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = decrank(V)
		if q[L+1] ≠ p[L+1]
			throw(ArgumentError("the decompositions are incompatible in the last rank"))
		end
	end
	return [ [ factordcat(U[ℓ], [ V[ℓ] for V ∈ W ]...) for ℓ ∈ 1:L-1 ]..., factorvcat(U[L], [ V[L] for V ∈ W ]...) ]
end

function dechcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
	L = declength(U); m = decsize(U); p = decrank(U)
	W = (V,W...)
	for V ∈ W
		if declength(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if decsize(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
		q = decrank(V)
		if q[1] ≠ p[1]
			throw(ArgumentError("the decompositions are incompatible in the first rank"))
		end
	end
	return [ factorhcat(U[1], [ V[1] for V ∈ W ]...), [ factordcat(U[ℓ], [ V[ℓ] for V ∈ W ]...) for ℓ ∈ 2:L ]...]
end

function decdcat(U::Dec{T,N}, V::Dec{T,N}, W::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
	L = declength(U); m = decsize(U)
	W = (V,W...)
	for V ∈ W
		if declength(V) ≠ L
			throw(ArgumentError("the decompositions are incompatible in the number of factors"))
		end
		if decsize(V) ≠ m
			throw(ArgumentError("the decompositions are incompatible in mode size"))
		end
	end
	return [ factordcat(U[ℓ], [ V[ℓ] for V ∈ W ]...) for ℓ ∈ 1:L ]
end

function decscale!(U::Dec{T,N}, α::T) where {T<:Number,N}
	L = declength(U)
	U[L] *= α
	return U
end

function decreverse!(W::Dec{T,N}) where {T<:Number,N}
	L = declength(W)
	reverse!(W)
	for ℓ ∈ 1:L
		W[ℓ] = factorranktranspose(W[ℓ])
	end
	return W
end

function decmodetranspose!(U::Dec{T,N}, τ::Union{NTuple{K,Int},Vector{Int}}) where {T<:Number,N,K}
	d = N-2
	if d == 0
		throw(ArgumentError("the decomposition should have at least one mode dimension"))
	end
	if length(τ) ≠ d || !isperm(τ)
		throw(ArgumentError("τ is not a valid permutation of the mode dimensions of U"))
	end
	isa(τ, Vector{Int}) || (τ = collect(τ))
	L = declength(U)
	for ℓ ∈ 1:L
		U[ℓ] = factormodetranspose(U[ℓ], τ)
	end
	return U
end

decmodetranspose!(U::Dec{T,N}) where {T<:Number,N} = decmodetranspose!(U, collect(decndims(U):-1:1))

function decfill!(U::Dec{T,N}, v::T) where {T<:Number,N}
	L = declength(U)
	for ℓ ∈ 1:L
		fill!(U[ℓ], v)
	end
	return U
end

function decrand!(rng::AbstractRNG, U::Dec{T,N}) where {T<:Number,N}
	L = declength(U)
	for ℓ ∈ 1:L
		rand!(rng, U[ℓ])
	end
	return U
end
decrand!(U::Dec{T,N}) where {T<:Number,N} = decrand!(Random.GLOBAL_RNG, U)

deczeros(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = decfill!(dec(T, n, r; first=first, last=last, len=len), convert(T, 0))
deczeros(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = deczeros(Float64, n, r; first=first, last=last, len=len)

decones(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = decfill!(dec(T, n, r; first=first, last=last, len=len), convert(T, 1))
decones(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = deczeros(Float64, n, r; first=first, last=last, len=len)

decrand(rng::AbstractRNG, ::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = decrand!(rng, dec(T, n, r; first=first, last=last, len=len))
decrand(rng::AbstractRNG, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = decrand(rng, Float64, n, r; first=first, last=last, len=len)
decrand(::Type{T}, n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) where {T<:Number} = decrand(Random.GLOBAL_RNG, T, n, r; first=first, last=last, len=len)
decrand(n::Union{DecSize,FactorSize}, r::Union{Int,DecRank}; first::Int=0, last::Int=0, len::Int=0) = decrand(Random.GLOBAL_RNG, Float64, n, r; first=first, last=last, len=len)

function decappend!(U::Dec{T,N}, V::Dec{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if decndims(U) ≠ decndims(V)
		throw(DimensionMismatch("U and V are inconsistent in the number of dimensions"))
	end
	p,q = decranks(U)
	r,s = decranks(V)
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
	return append!(U, V)
end

function decprepend!(U::Dec{T,N}, V::Dec{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if decndims(U) ≠ decndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = decranks(U); r,s = decranks(V)
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
	return prepend!(U, V)
end

function decpush!(U::Dec{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if decndims(U) ≠ factorndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = decranks(U)
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
	return push!(U, V)
end

function decpushfirst!(U::Dec{T,N}, V::Factor{T,N}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	if decndims(U) ≠ factorndims(V)
		throw(DimensionMismatch("U and V have different numbers of dimensions"))
	end
	p,q = decranks(U)
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
	return pushfirst!(U, V)
end

function decpop!(U::Dec{T,N}) where {T<:Number,N}
	return pop!(U)
end

function decpopfirst!(U::Dec{T,N}) where {T<:Number,N}
	return popfirst!(U)
end

function decinsert!(U::Dec{T,N}, ℓ::Int, V::Factor{T,N}; path::String="", rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	L = declength(U)
	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end
	if ℓ ∉ 1:L
		throw(ArgumentError("ℓ is required to be from 1:L, where L is the number of factors in U"))
	end
	(path == "forward") && (ℓ = ℓ+1)
	(ℓ == 1) && return decpushfirst!(U, V; rankprecheck=rankprecheck, rankpostcheck=rankpostcheck)
	(ℓ == L+1) && return decpush!(U, V; rankprecheck=rankprecheck, rankpostcheck=rankpostcheck)
	p,q = decranks(U)
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
	return insert!(U, ℓ, V)
end

function decdeleteat!(U::Dec{T,N}, Λ::Union{Int,Vector{Int},NTuple{M,Int} where M}; rankprecheck::Bool=true, rankpostcheck::Bool=true) where {T<:Number,N}
	L = declength(U)
	if isa(Λ, Vector{Int}) && unique(Λ) ≠ Λ
		throw(ArgumentError("the entries of Λ should be unique"))
	end
	isa(Λ, Vector{Int}) || (Λ = collect(Λ))
	if Λ ⊈ 1:L
		throw(ArgumentError("Λ should be an element or a subset of 1:L with unique entries, where L is the number of factors in U"))
	end
	p,q = decranks(U)
	if rankprecheck
		try checkranks(p,q) catch
			throw(DimensionMismatch("the factors of U have inconsistent ranks"))
		end
	end
	if rankpostcheck
		deleteat!(p, Λ); deleteat!(q, Λ)
		try checkranks(p,q) catch
			throw(DimensionMismatch("the ranks of U are inconsistent for this operation"))
		end
	end
	return deleteat!(U, Λ)
end

function decinsertidentity!(U::Dec{T,N}, ℓ::Int; path::String="", rankprecheck::Bool=true) where {T<:Number,N}
	L = declength(U)
	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end
	if ℓ ∉ 1:L
		throw(ArgumentError("ℓ is required to be from 1:L, where L is the number of factors in U"))
	end
	p,q = decranks(U)
	d = decndims(U)
	r = (path == "forward") ? q[ℓ] : p[ℓ]
	V = Matrix{T}(I, r, r); V = factor(V, ones(Int, d), [])
	(path == "forward") && (ℓ = ℓ+1)
	(ℓ == 1) && return decpushfirst!(U, V; rankprecheck=rankprecheck, rankpostcheck=false)
	(ℓ == L+1) && return decpush!(U, V; rankprecheck=rankprecheck, rankpostcheck=false)
	p,q = decranks(U)
	if rankprecheck
		try checkranks(p,q) catch
			throw(DimensionMismatch("the factors of U have inconsistent ranks"))
		end
	end
	return insert!(U, ℓ, V)
end

function decskp!(W::Dec{T,N}, Λ::Indices; path::String="", major::String="last") where {T<:Number,N}
	if path ∉ ("","forward","backward")
		throw(ArgumentError("the value of the keyword argument path should be \"\" (default, accepted only for empty Λ and for Λ=:), \"forward\" or \"backward\""))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	L = declength(W); decrank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if isa(Λ, Colon) && path ≠ ""
		throw(ArgumentError("when Λ is a colon, path should be omitted or specfied as \"\" (default)"))
	end
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
		throw(ArgumentError("Λ, when nonempty, should be a colon or an Int/Vector/NTuple/UnitRange of Int, from 1:L-1 for path=\"forward\" and from 2:L for path=\"backward\", where L is the number of factors in W"))
	end
	sort!(Λ; rev=true)
	for ℓ ∈ Λ
		W[ℓ-1] = factorcontract(W[ℓ-1], W[ℓ]; major=major)
		decdeleteat!(W, ℓ; rankprecheck=false, rankpostcheck=false)
	end
	return W
end

decskp!(W::Dec{T,N}; path::String="", major::String="last") where {T<:Number,N} = decskp!(W, :; path=path, major=major)

function decskp(W::Dec{T,N}, Λ::Indices; path::String="", major::String="last") where {T<:Number,N}
	if path ∉ ("","forward","backward")
		throw(ArgumentError("the value of the keyword argument path should be \"\" (default, accepted only for empty Λ and for Λ=:), \"forward\" or \"backward\""))
	end
	if major ∉ ("first","last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	L = declength(W); decrank(W)
	if L == 0
		throw(ArgumentError("the decomposition is empty"))
	end
	if isa(Λ, Colon) && path ≠ ""
		throw(ArgumentError("when Λ is a colon, path should be omitted or specfied as \"\" (default)"))
	end
	isa(Λ, Colon) && (Λ = collect(2:L); path = "backward")
	isa(Λ, Vector{Int}) || (Λ = indvec(Λ))
	if length(Λ) == 0
		throw(ArgumentError("Λ is empty"))
	end
	if (minimum(Λ):maximum(Λ)) ⊈ Λ
		throw(ArgumentError("the entries of Λ should form a set of contiguous integers"))
	end
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
	M = length(Λ)
	sort!(Λ; rev=true)
	U = W[Λ[1]]
	for λ ∈ 2:M
		U = factorcontract(W[Λ[λ]], U; major=major)
	end
	return W
end

decskp(W::Dec{T,N}; path::String="", major::String="last") where {T<:Number,N} = decskp(W, :; path=path, major=major)

function decmp(U₁::Dec{T,N₁}, σ₁::Indices, U₂::Dec{T,N₂}, σ₂::Indices) where {T<:Number,N₁,N₂}
	n₁ = decsize(U₁); d₁ = decndims(U₁); L₁ = declength(U₁)
	n₂ = decsize(U₂); d₂ = decndims(U₂); L₂ = declength(U₂)
	if isa(σ₁, Vector{Any}) && length(σ₁) > 0
		throw(ArgumentError("if σ₁ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	if isa(σ₂, Vector{Any}) && length(σ₂) > 0
		throw(ArgumentError("if σ₂ is passed as a vector, it should be a vector of the type Vector{Int} or an empty vector of the type Vector{Any}"))
	end
	isa(σ₁, Vector{Int}) || (σ₁ = indvec(σ₁; max=d₁))
	isa(σ₂, Vector{Int}) || (σ₂ = indvec(σ₂; max=d₂))
	if L₁ ≠ L₂
		throw(ArgumentError("U₁ and U₂ differ in the number of factors"))
	end
	L = L₁
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
	U = [ factormp(U₁[ℓ], σ₁, U₂[ℓ], σ₂) for ℓ ∈ 1:L ]
	return U
end


# function deckp(U::Dec{T,N}, V::Vararg{Dec{T,N},M}) where {T<:Number,N,M}
# 	n = decsize(U); L = declength(U)
# 	d = 1+length(V)
# 	for k ∈ 2:d
# 		if declength(V[k-1]) ≠ L
# 			throw(ArgumentError("the input decompositions differ in the number of factors"))
# 		end
# 	end
# 	return [ factorkp(U[ℓ], [ V[k-1][ℓ] for k ∈ 2:d ]...) for ℓ ∈ 1:L ]
# end

function deckp(U::Union{Dec{T,N},Tuple{Dec{T,N},Int}}, V::Vararg{Union{Dec{T,N},Tuple{Dec{T,N},Int}},M}) where {T<:Number,N,M}
	V = (U,V...)
	nf = length(V)
	U = Vector{Tuple{Dec{T,N},Int}}(undef, nf)
	for k ∈ 1:nf
		W = V[k]
		if isa(W, Tuple)
			if W[2] < 0
				throw(ArgumentError("all the specified exponents should be nonnegative"))
			end
			U[k] = W[1],W[2]
		else
			U[k] = W,1
		end
	end
	n = decsize(U[1][1]); L = declength(U[1][1])
	for k ∈ 2:nf
		if declength(U[k][1]) ≠ L
			throw(ArgumentError("the input decompositions differ in the number of factors"))
		end
	end
	return [ factorkp([ (U[k][1][ℓ],U[k][2]) for k ∈ 1:nf ]...) for ℓ ∈ 1:L ]
end


function decadd(U::Dec{T,N}, V::Dec{T,N}) where {T<:Number,N}
	m = decsize(U); L = declength(U)
	p = decrank(U); q = decrank(V)
	if declength(V) ≠ L
		throw(ArgumentError("U and V differ in the number of factors"))
	end
	if decsize(V) ≠ m
		throw(ArgumentError("U and V are inconsistent in mode size"))
	end
	if q[1] ≠ p[1]
		throw(ArgumentError("the decompositions are incompatible in the first rank"))
	end
	if q[L+1] ≠ p[L+1]
		throw(ArgumentError("the decompositions are incompatible in the last rank"))
	end
	(L == 1) && return U .+ V
	W = [ factorhcat(U[1], V[1]), [ factordcat(U[ℓ], V[ℓ]) for ℓ ∈ 2:L-1 ]..., factorvcat(U[L], V[L]) ]
	return W
end

function dechp(U::Dec{T,N}, V::Dec{T,N}) where {T<:Number,N}
	m = decsize(U); L = declength(U)
	if declength(V) ≠ L
		throw(ArgumentError("U and V differ in the number of factors"))
	end
	if decsize(V) ≠ m
		throw(ArgumentError("U and V are inconsistent in mode size"))
	end
	W = [ factorhp(U[ℓ], V[ℓ]) for ℓ ∈ 1:L ]
	return W
end

function decqr!(W::Dec{T,N}, Λ::Indices; pivot::Bool=false, path::String="") where {T<:FloatRC,N}
	L = declength(W); decrank(W)
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
		W[ℓ],R = factorqr!(W[ℓ], Val(pivot); rev=(path == "backward"))
		if λ < M
			ν = Λ[λ+1]
			W[ν] = factorcontract(R, W[ν], rev=(path == "backward"))
		else
			decinsert!(W, ℓ, R; path=path, rankprecheck=false, rankpostcheck=true)
		end
	end
	return W
end

decqr!(W::Dec{T,N}; pivot::Bool=false, path::String="") where {T<:FloatRC,N} = decqr!(W, :; pivot=pivot, path=path)


function decsvd!(W::Dec{T,N}, Λ::Indices, n::Union{Colon,DecSize}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, rank::Int2=0, major::String="last") where {T<:FloatRC,N}
	# assumes that the decomposition is orthogonal
	L = declength(W); decrank(W)
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
	m = decsize(W)
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
	#
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

	if any(rank .< 0)
		throw(ArgumentError("rank should be a nonnegative Int or a vector of such"))
	end
	if isa(rank, Int)
		rank = rank*ones(Int, K)
	elseif length(rank) ≠ K
		throw(ArgumentError("rank, passed as a vector, has incorrect length"))
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
		ℓ = Λ[λ]
		if λ == 1
			ε₁ = [aTol[λ],aTolDistr[λ]]; ε₁ = ε₁[ε₁ .> 0]; ε₁ = isempty(ε₁) ? 0.0 : minimum(ε₁)
			δ₁ = [rTol[λ],rTolDistr[λ]]; δ₁ = δ₁[δ₁ .> 0]; δ₁ = isempty(δ₁) ? 0.0 : minimum(δ₁)
			U,V,ε[1],δ[1],μ,ρ[1],σ[1] = factorsvd!(W[ℓ], n[:,λ], :; aTol=ε₁, rTol=δ₁, rank=rank[λ], rev=(path == "backward"), major=major)
		else
			(aTolDistr[λ] > 0) && (aTolDistr[λ] = sqrt(aTolDistr[λ]^2+aTolAcc^2); aTolAcc = 0.0)
			(rTolDistr[λ] > 0) && (rTolDistr[λ] = sqrt(rTolDistr[λ]^2+rTolAcc^2); rTolAcc = 0.0)
			ε₁ = [aTol[λ],aTolDistr[λ],μ*rTol[λ],μ*rTolDistr[λ]]; ε₁ = ε₁[ε₁ .> 0]
			ε₁ = isempty(ε₁) ? 0.0 : minimum(ε₁)
			U,V,ε[λ],_,_,ρ[λ],σ[λ] = factorsvd!(W[ℓ], n[:,λ], :; aTol=ε₁, rank=rank[λ], rev=(path == "backward"), major=major)
			δ[λ] = (μ > 0) ? ε[λ]/μ : 0.0
		end
		W[ℓ] = U
		if λ < K && Λ[λ+1] ≠ ℓ
			ν = Λ[λ+1]
			W[ν] = factorcontract(V, W[ν], rev=(path == "backward"), major=major)
		else
			decinsert!(W, ℓ, V; path=path, rankprecheck=false, rankpostcheck=true)
			(path == "forward") && (Λ[λ:end] .+= 1)
		end
	end
	return W,ε,δ,μ,ρ,σ
end

decsvd!(W::Dec{T,N}, Λ::Indices; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, rank::Int2=0, major::String="last") where {T<:FloatRC,N} = decsvd!(W, Λ, :; path=path, aTol=aTol, aTolDistr=aTolDistr, rTol=rTol, rTolDistr=rTolDistr, rank=rank, major=major)

decsvd!(W::Dec{T,N}; path::String="", aTol::Float2=0.0, aTolDistr::Float2=0.0, rTol::Float2=0.0, rTolDistr::Float2=0.0, rank::Int2=0, major::String="last") where {T<:FloatRC,N} = decsvd!(W, :, :; path=path, aTol=aTol, aTolDistr=aTolDistr, rTol=rTol, rTolDistr=rTolDistr, rank=rank, major=major)
