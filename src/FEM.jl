module FEM

using LinearAlgebra, TensorRefinement.Auxiliary, ..TensorTrain

export extdn, diffextdn, diffdn, intdn, intcholdn, bpxdn, diffext, diffbpxdn, diffextbpxdn, xintdn

function extdn(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if ℓ ∉ 0:L
		throw(ArgumentError("ℓ should be in 0:L"))
	end
	if d ≤ 0
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	U = Matrix{T}([I J'; O J]); U = factor(U, 2, 2)
	W = Matrix{T}([
		1 0
		2 1
		1 2
		0 1
		] / 2 / sqrt(2)); W = factor(W, 2, 1)
	U = factorkp((U,d))
	W = factorkp((W,d))
	P = dec(U; len=ℓ)
	decappend!(P, dec(W; len=L-ℓ))
	decrankselect!(P, 1:1, 1:1)
	(major == "last") && decreverse!(P)
	return P
end

extdn(L::Int, ℓ::Int, d::Int; major::String="last") = extdn(Float64, L, ℓ, d; major=major)


function diffdn(::Type{T}, L::Int, d::Int, k::Int; major::String="last") where {T<:FloatRC}
	if L < 0
		throw(ArgumentError("L should be nonnegative"))
	end
	if d < 0
		throw(ArgumentError("d should be positive"))
	end
	if k ∉ 0:d
		throw(ArgumentError("k should be in 0:d"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	S = Matrix{T}([1 1; 1 -1]); S = factor(S, 1, 1)
	W₀ = Matrix{T}([I+J+J' I-J-J'; I-J+J' I+J-J'] / sqrt(2)); W₀ = factor(W₀, 2, 2)
	W₁ = W₀ * 2
	Z₀ = Matrix{T}([1,0,0,1][:,:] / 2); Z₀ = factor(Z₀, 2, 1)
	Z₁ = Matrix{T}([0,1][:,:]); Z₁ = factor(Z₁, 1, 1)
	S = factorkp((S,d))
	W = factorkp([ (k == i) ? W₁ : W₀ for i ∈ 1:d ]...)
	Z = factorkp([ (k == i) ? Z₁ : Z₀ for i ∈ 1:d ]...)
	M = dec(S)
	decappend!(M, dec(W; len=L))
	decpush!(M, Z)
	decrankselect!(M, 1:1, :)
	decskp!(M, 1, path="forward")
	(major == "last") && decreverse!(M)
	return M
end

diffdn(L::Int, d::Int, k::Int; major::String="last") = diffdn(Float64, L, d, k; major=major)


function intdn(::Type{T}, L::Int, d::Int, kk::Int, k::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if (kk,k) ⊈ 0:d
		throw(ArgumentError("kk and k should be in 0:d"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	I = [1 0; 0 1]
	U = Matrix{T}(I/2); U = factor(U, 2, 2)
	Z = Matrix{T}([2 0; 0 2/3] / 2); Z = factor(Z, 2, 2)

	U = factorkp((U,d))
	Z = [ Z[:,1:((i == kk) ? 1 : 2),1:((i == k) ? 1 : 2),:] for i ∈ 1:d ]
	Z = factorkp(Z...)

	Λ = dec(U; len=L)
	decpush!(Λ, Z)
	(major == "last") && decreverse!(Λ)
	return Λ
end

intdn(L::Int, d::Int, kk::Int, k::Int; major::String="last") = intdn(Float64, L, d, kk, k; major=major)


function xintdn(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end

	I = [1 0; 0 1]
	U = Matrix{T}(I/2); U = factor(U, 2, 2)
	Z₀ = Matrix{T}([1 0; 0 1/3]); Z₀ = factor(Z₀, 2, 2)
	Z₁ = Matrix{T}([0 0; 0 1]); Z₁ = factor(Z₁, 2, 2)

	U = factorkp((U,d))*4
	Z = zeros(T, 1, 2^d, 2^d, 1)
	for k ∈ 1:d
		Z .+= factorkp([ (k == i) ? Z₁ : Z₀ for i ∈ 1:d ]...)*4
	end
	Λ = dec(U; len=L)
	decpush!(Λ, Z)
	(major == "last") && decreverse!(Λ)
	return Λ
end

xintdn(L::Int, d::Int; major::String="last") = xintdn(Float64, L, d; major=major)


function intcholdn(::Type{T}, L::Int, d::Int, kk::Int, k::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if (kk,k) ⊈ 0:d
		throw(ArgumentError("kk and k should be in 0:d"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	I = [1 0; 0 1]
	U = Matrix{T}(I/sqrt(2)); U = factor(U, 2, 2)
	Z = Matrix{T}([1 0; 0 1/sqrt(3)]); Z = factor(Z, 2, 2)

	U = factorkp((U,d))
	Z = [ Z[:,1:((i == kk) ? 1 : 2),1:((i == k) ? 1 : 2),:] for i ∈ 1:d ]
	Z = factorkp(Z...)

	Λ = dec(U; len=L)
	decpush!(Λ, Z)
	(major == "last") && decreverse!(Λ)
	return Λ
end

intcholdn(L::Int, d::Int, kk::Int, k::Int; major::String="last") = intcholdn(Float64, L, d, kk, k; major=major)


function bpxdn(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	U = Matrix{T}([I J'; O J]); U = factor(U, 2, 2)
	W = Matrix{T}([
		1 0
		2 1
		1 2
		0 1
		]); W = factor(W, 2, 1)
	U = factorkp((factormp(U, 2, U, 2),d))./2
	W = factorkp((factormp(W, 2, W, 2)./8,d))

	if L == 1
		C = dec(U + W)
	else
		C = dec(factorhcat(U, U + W))
		decappend!(C, dec(factorutcat(U, U, W); len=L-2))
		decpush!(C, factorvcat(U, W))
	end
	decrankselect!(C, 1:1, 1:1)
	(major == "last") && decreverse!(C)
	return C
end

bpxdn(L::Int, d::Int; major::String="last") = bpxdn(Float64, L, d; major=major)

function diffext(::Type{T}, L::Int, ℓ::Int, d::Int, k::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if ℓ ∉ 0:L
		throw(ArgumentError("ℓ should be in 0:L"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if k ∉ 0:d
		throw(ArgumentError("k should be in 0:d"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	U = Matrix{T}([1 0; 0 1]); U = factor(U, 2, 2)
	if ℓ == L
		Z₀ = Matrix{T}([1 0; 0 1]); Z₀ = factor(Z₀, 2, 2)
		Z₁ = Matrix{T}([1][:,:]); Z₁ = factor(Z₁, 1, 1)

		U  = factorkp((U, d))
		Z  = factorkp([ (k == i) ? Z₁ : Z₀ for i ∈ 1:d ]...)

		P = dec(Z)
		decprepend!(P, dec(U; len=L))
	else
		V₀ = Matrix{T}([ 2 -1 0 1
						 2  1 0 1 ]); V₀ = factor(V₀, 2, 2)
		V₁ = Matrix{T}([1,1][:,:]); V₁ = factor(V₁, 2, 1)
		Y₀ = Matrix{T}([
				 2 0
				 2 0
				-1 1
				 1 1
			] / 2); Y₀ = factor(Y₀, 2, 1)
		Y₁ = Matrix{T}([1,1][:,:]); Y₁ = factor(Y₁, 2, 1)
		Z₀ = Matrix{T}([1,0,0,1][:,:] / 2); Z₀ = factor(Z₀, 2, 1)
		Z₁ = Matrix{T}([1][:,:]); Z₁ = factor(Z₁, 1, 1)

		U  = factorkp((U, d))
		V  = factorkp([ (k == i) ? V₁ : V₀ for i ∈ 1:d ]...)
		Y  = factorkp([ (k == i) ? Y₁ : Y₀ for i ∈ 1:d ]...)
		Z  = factorkp([ (k == i) ? Z₁ : Z₀ for i ∈ 1:d ]...)

		P = dec(Z)
		decprepend!(P, dec(Y; len=L-ℓ-1))
		decpushfirst!(P, V)
		decprepend!(P, dec(U; len=ℓ))
	end
	(major == "last") && decreverse!(P)
	return P
end

diffext(L::Int, ℓ::Int, d::Int, k::Int; major::String="last") = diffext(Float64, L, ℓ, d, k; major=major)

function diffextdn(::Type{T}, L::Int, ℓ::Int, d::Int, k::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if ℓ ∉ 0:L
		throw(ArgumentError("ℓ should be in 0:L"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if k ∉ 0:d
		throw(ArgumentError("k should be in 0:d"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end

	P = diffext(T, L, ℓ, d, k; major="first")
	Q = diffdn(T, ℓ, d, k; major="first")

	I = Matrix{T}(ones(T)[:,:]); I = factor(I, 1, 1)
	decappend!(Q, dec(I; len=L-ℓ))
	Q = decmp(P, 2, Q, 1) 
	(major == "last") && decreverse!(Q)
	return Q
end

diffextdn(L::Int, ℓ::Int, d::Int, k::Int; major::String="last") = diffextdn(Float64, L, ℓ, d, k; major=major)

function diffbpxdn(::Type{T}, L::Int, d::Int, k::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if k ∉ 0:d
		throw(ArgumentError("k should be in 0:d"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]

	A  = Matrix{T}([1 0]); A = factor(A, 1, 1)
	S  = Matrix{T}([1 1; 1 -1]); S = factor(S, 1, 1)
	X  = factorcontract(A, S)
	V₀ = Matrix{T}([I+J+J' I-J-J'; I-J+J' I+J-J'] / sqrt(2)); V₀ = factor(V₀, 2, 2)
	V₁ = V₀ * 2
	R₀ = Matrix{T}([1 0; 0 1]); R₀ = factor(R₀, 1, 1)
	R₁ = Matrix{T}([0; 1][:,:]); R₁ = factor(R₁, 1, 1)
	Y₀ = Matrix{T}([
		 2 0
		 2 0
		-1 1
		 1 1
		 ] / 2); Y₀ = factor(Y₀, 2, 1)
	Y₁ = Y₀[2:2,:,:,2:2] * 2
	Z₀ = Matrix{T}([1,0,0,1][:,:] / 2); Z₀ = factor(Z₀, 2, 1)
	Z₁ = Matrix{T}([1][:,:]); Z₁ = factor(Z₁, 1, 1)
	
	U = Matrix{T}([I J'; O J]); U = factor(U, 2, 2)
	R = Matrix{T}([1 0; 0 1]); R = factor(R, 1, 1)
	W = Matrix{T}([
		1 0
		2 1
		1 2
		0 1
		] / 2 / sqrt(2)); W = factor(W, 2, 1)
	Z = Matrix{T}([1; 0][:,:]); Z = factor(Z, 1, 1)

	X  = factormp(X, 2, A, 2)
	V₀ = factormp(V₀, 2, U, 2)
	V₁ = factormp(V₁, 2, U, 2)
	R₀ = factormp(R₀, 2, R, 2)
	R₁ = factormp(R₁, 2, R, 2)
	Y₀ = factormp(Y₀, 2, W, 2)
	Y₁ = factormp(Y₁, 2, W, 2)
	Z₀ = factormp(Z₀, 2, Z, 2)
	Z₁ = factormp(Z₁, 2, Z, 2)

	X  = factorkp((X, d))
	V  = factorkp([ (k == i) ? V₁ : V₀ for i ∈ 1:d ]...)
	R  = factorkp([ (k == i) ? R₁ : R₀ for i ∈ 1:d ]...)
	Y  = factorkp([ (k == i) ? Y₁ : Y₀ for i ∈ 1:d ]...)
	Z  = factorkp([ (k == i) ? Z₁ : Z₀ for i ∈ 1:d ]...)

	XR = factorcontract(X, R)
	VR = factorcontract(V, R)
	μ = 2*one(T)
	Q = dec(factorhcat(X, XR))
	for ℓ ∈ 1:L-1
		decpush!(Q, factorutcat(V/μ, VR/μ, Y))
	end
	decpush!(Q, factorvcat(VR/μ, Y))
	decpush!(Q, Z)
	decskp!(Q, 1, path="forward")
	(major == "last") && decreverse!(Q)
	return Q
end

diffbpxdn(L::Int, d::Int, k::Int; major::String="last") = diffbpxdn(Float64, L, d, k; major=major)


function diffextbpxdn(::Type{T}, L::Int, ℓ::Int, d::Int, k::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if ℓ ∉ 0:L
		throw(ArgumentError("ℓ should be in 0:L"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if k ∉ 0:d
		throw(ArgumentError("k should be in 0:d"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end

	P = diffext(T, L, ℓ, d, k; major="first")
	Q = diffbpxdn(T, ℓ, d, k; major="first")

	I = Matrix{T}(ones(T)[:,:]); I = factor(I, 1, 1)
	decappend!(Q, dec(I; len=L-ℓ))
	Q = decmp(P, 2, Q, 1) 
	(major == "last") && decreverse!(Q)
	return Q
end

diffextbpxdn(L::Int, ℓ::Int, d::Int, k::Int; major::String="last") = diffextbpxdn(Float64, L, ℓ, d, k; major=major)

end