module FEM

using LinearAlgebra, TensorRefinement.Auxiliary, ..TensorTrain

export extdn, extdd, diffdn, diffdd, dint, dintf, bpxdn, bpxdd, extmix, diffbpxdn, diffbpxdd

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
	A = T[0 1]; A = factor(A, 1, 1)
	U = T[J O; J' I]; U = factor(U, 2, 2)
	V = U
	W = T[2 1; 1 0; 0 1; 1 2] ./ sqrt(8*one(T)); W = factor(W, 2, 1)
	Z = T[0,1][:,:]; Z = factor(Z, 1, 1)
	A = factorkp(A => d)
	U = factorkp(U => d)
	V = factorkp(V => d)
	W = factorkp(W => d)
	Z = factorkp(Z => d)
	P = MatrixDec{T}()
	decpush!(P, A)
	(ℓ > 1) && decappend!(P, dec(U; len=ℓ-1))
	(ℓ > 0) && decpush!(P, V)
	(ℓ < L) && decappend!(P, dec(W; len=L-ℓ))
	decpush!(P, Z)
	decskp!(P, 1; path="forward")
	decskp!(P, L+1; path="backward")
	(major == "last") && decreverse!(P)
	P
end

function extdd(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if ℓ ∉ 1:L
		throw(ArgumentError("ℓ should be in 1:L"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	I1 = [1 0; 0 0]; I2 = [0 0; 0 1]
	A = T[1 0 0]; A = factor(A, 1, 1)
	U = T[I2 I1 J'; O I J'; O O J]; U = factor(U, 2, 2)
	V = U[:,:,:,[3,2]]
	W = T[2 1; 1 0; 0 1; 1 2] ./ sqrt(8*one(T)); W = factor(W, 2, 1)
	Z = T[0,1][:,:]; Z = factor(Z, 1, 1)
	A = factorkp(A => d)
	U = factorkp(U => d)
	V = factorkp(V => d)
	W = factorkp(W => d)
	Z = factorkp(Z => d)
	P = MatrixDec{T}()
	decpush!(P, A)
	(ℓ > 1) && decappend!(P, dec(U; len=ℓ-1)) 
	decpush!(P, V)
	(ℓ < L) && decappend!(P, dec(W; len=L-ℓ))
	decpush!(P, Z)
	decskp!(P, 1; path="forward")
	decskp!(P, L+1; path="backward")
	(major == "last") && decreverse!(P)
	P
end

function diffdn(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 0
		throw(ArgumentError("L should be nonnegative"))
	end
	if ℓ ∉ 0:L
		throw(ArgumentError("ℓ should be in 0:L"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	ρ = √(one(T)*2^d)
	c0 = Int[3,1]
	c1 = Int[0,1]
	wk = (d > 1) ? kron(ntuple(k -> c0, Val(d))...) : c0
	w = wk.*((1/ρ)^(2*ℓ) / (3*one(T))^d)
	for k ∈ 1:d
		wk = (d > 1) ? kron(ntuple(k -> c0, Val(d-k))..., c1, ntuple(k -> c0, Val(k-1))...) : c1
		w .+= wk.*((2/ρ)^(2*ℓ) * 4 / 3^(d-1))
	end
	w .= sqrt.(w)
	C = T[  1//2 -1//2; 1//2 1//2 ]
	B = (d > 1) ? kron(ntuple(k -> C, Val(d))...) : C
	rmul!(B, Diagonal(w))
	###
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	A = T[0 1]; A = factor(A, 1, 1)
	U = T[J O ; J' I ].*sqrt(2*one(T)); U = factor(U, 2, 2)
	V = U
	A = factorkp(A => d)
	(ℓ == 0) && (A = factorcontract(A, B))
	U = factorkp(U => d)
	V = factorkp(V => d)
	V = factorcontract(V, B)
	M = MatrixDec{T}()
	decpush!(M, A)
	(ℓ > 1) && decappend!(M, dec(U; len=ℓ-1))
	(ℓ > 0) && decpush!(M, V)
	for k ∈ ℓ+1:L
		W = reshape(T[1 1 0 0; -1//2 1//2 1//2 1//2], 2, 2, 2)
		W = factorkp(W => d)
		W = factorcontract(Diagonal(1 ./w), W)
		wi = (d > 1) ? kron(ntuple(j -> c0, Val(d))...) : c0
		w = wi.*((1/ρ)^(2*k) / (one(T)*3)^d)
		for i ∈ 1:d
			wi = (d > 1) ? kron(ntuple(j -> c0, Val(d-i))..., c1, ntuple(j -> c0, Val(i-1))...) : c1
			w .+= wi.*((2/ρ)^(2*k) * 4 / (one(T)*3)^(d-1))
		end
		w .= sqrt.(w)
		W = factorcontract(W, Diagonal(w))
		W = reshape(W, 2^d, 2^d, 1, 2^d)
		decpush!(M, W)
	end
	C0 = T[1 0; 0 1]
	C1 = T[0 0; 2 0]
	Z = Array{T,3}(undef, 2^d, 2^d, d+1)
	if d == 1
		Z[:,:,1] = C0
		Z[:,:,2] = C1
	else
		Z[:,:,1] = kron(ntuple(j -> C0, Val(d))...)
		@inbounds for i ∈ 1:d
			Z[:,:,i+1] = kron(ntuple(j -> C0, Val(d-i))..., C1, ntuple(j -> C0, Val(i-1))...)
		end
	end
	Z = reshape(Z, 2^d, 2^d*(d+1))
	lmul!(Diagonal(1 ./w), Z)
	Z = reshape(Z, 2^d, 2^d, d+1)
	Z[:,:,2:d+1] .*= (one(T)*2)^L
	Z = reshape(Z, 2^d, 2^d*(d+1), 1, 1)
	decpush!(M, Z)
	decskp!(M, 1; path="forward")
	(major == "last") && decreverse!(M)
	M
end
diffdn(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC} = diffdn(T, L, L, d; major=major)

function diffdd(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if ℓ ∉ 1:L
		throw(ArgumentError("ℓ should be in 1:L"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	ρ = √(one(T)*2^d)
	c0 = Int[3,1]
	c1 = Int[0,1]
	wk = (d > 1) ? kron(ntuple(k -> c0, Val(d))...) : c0
	w = wk.*((1/ρ)^(2*ℓ) / (3*one(T))^d)
	for k ∈ 1:d
		wk = (d > 1) ? kron(ntuple(k -> c0, Val(d-k))..., c1, ntuple(k -> c0, Val(k-1))...) : c1
		w .+= wk.*((2/ρ)^(2*ℓ) * 4 / 3^(d-1))
	end
	w .= sqrt.(w)
	C = T[  1//2 -1//2; 1//2 1//2 ]
	B = (d > 1) ? kron(ntuple(k -> C, Val(d))...) : C
	rmul!(B, Diagonal(w))
	###
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	I1 = [1 0; 0 0]; I2 = [0 0; 0 1]
	A = T[1 0 0]; A = factor(A, 1, 1)
	U = T[I2 I1 J'; O I J'; O O J].*sqrt(2*one(T)); U = factor(U, 2, 2)
	V = U[:,:,:,[3,2]]
	A = factorkp(A => d)
	U = factorkp(U => d)
	V = factorkp(V => d)
	V = factorcontract(V, B)
	M = MatrixDec{T}()
	decpush!(M, A)
	(ℓ > 1) && decappend!(M, dec(U; len=ℓ-1))
	decpush!(M, V)
	for k ∈ ℓ+1:L
		W = reshape(T[1 1 0 0; -1//2 1//2 1//2 1//2], 2, 2, 2)
		W = factorkp(W => d)
		W = factorcontract(Diagonal(1 ./w), W)
		wi = (d > 1) ? kron(ntuple(j -> c0, Val(d))...) : c0
		w = wi.*((1/ρ)^(2*k) / (one(T)*3)^d)
		for i ∈ 1:d
			wi = (d > 1) ? kron(ntuple(j -> c0, Val(d-i))..., c1, ntuple(j -> c0, Val(i-1))...) : c1
			w .+= wi.*((2/ρ)^(2*k) * 4 / (one(T)*3)^(d-1))
		end
		w .= sqrt.(w)
		W = factorcontract(W, Diagonal(w))
		W = reshape(W, 2^d, 2^d, 1, 2^d)
		decpush!(M, W)
	end
	C0 = T[1 0; 0 1]
	C1 = T[0 0; 2 0]
	Z = Array{T,3}(undef, 2^d, 2^d, d+1)
	if d == 1
		Z[:,:,1] = C0
		Z[:,:,2] = C1
	else
		Z[:,:,1] = kron(ntuple(j -> C0, Val(d))...)
		@inbounds for i ∈ 1:d
			Z[:,:,i+1] = kron(ntuple(j -> C0, Val(d-i))..., C1, ntuple(j -> C0, Val(i-1))...)
		end
	end
	Z = reshape(Z, 2^d, 2^d*(d+1))
	lmul!(Diagonal(1 ./w), Z)
	Z = reshape(Z, 2^d, 2^d, d+1)
	Z[:,:,2:d+1] .*= (one(T)*2)^L
	Z = reshape(Z, 2^d, 2^d*(d+1), 1, 1)
	decpush!(M, Z)
	decskp!(M, 1; path="forward")
	(major == "last") && decreverse!(M)
	M
end
diffdd(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC} = diffdd(T, L, L, d; major=major)

function dint(::Type{T}, L::Int, d::Int, K::AbstractMatrix{T}; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	if size(K) ≠ (d+1,d+1)
		throw(ArgumentError("The coefficient matrix K should be of size (d+1) × (d+1)"))
	end
	U = T[1 0; 0 1] ./ 2; U = factor(U, 2, 2)
	Z = T[1 0; 0 1//3]
	U = factorkp(U => d)
	Z = kron(K, ntuple(j -> Z, Val(d))...)
	Z = factor(Z, 2^d*(d+1), 2^d*(d+1))
	Λ = dec(U; len=L)
	decpush!(Λ, Z)
	(major == "last") && decreverse!(Λ)
	Λ
end

function dintf(::Type{T}, L::Int, d::Int, K::AbstractMatrix{T}; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	if size(K) ≠ (d+1,d+1)
		throw(ArgumentError("The coefficient matrix K should be of size (d+1) × (d+1)"))
	end
	U = T[1 0; 0 1] ./ sqrt(one(T)*2); U = factor(U, 2, 2)
	Z = T[1 0; 0 1/sqrt(one(T)*3)]
	U = factorkp(U => d)
	Z = kron(K, ntuple(j -> Z, Val(d))...)
	Z = factor(Z, 2^d*(d+1), 2^d*(d+1))
	Λ = dec(U; len=L)
	decpush!(Λ, Z)
	(major == "last") && decreverse!(Λ)
	Λ
end

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
	U = T[I J'; O J]; U = factor(U, 2, 2)
	W = T[1 0; 2 1; 1 2; 0 1] ./ sqrt(8*one(T)); W = factor(W, 2, 1)
	U = factorkp(factormp(U, 2, U, 2) => d) ./ 2
	W = factorkp(factormp(W, 2, W, 2) => d)
	if L == 1
		C = dec(U + W)
	else
		C = dec(factorhcat(U, U + W))
		decappend!(C, dec(factorutcat(U, U, W); len=L-2))
		decpush!(C, factorvcat(U, W))
	end
	decrankselect!(C, 1:1, 1:1)
	(major == "last") && decreverse!(C)
	C
end

function bpxdd(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}
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
	I1 = [1 0; 0 0]; I2 = [0 0; 0 1]
	U = T[I2 I1 J'; O I J'; O O J]; U = factor(U, 2, 2)
	V = U[:,:,:,[2,3]]
	W = T[1 0; 2 1; 1 2; 0 1] ./ sqrt(8*one(T)); W = factor(W, 2, 1)
	U = factorkp(factormp(U, 2, U, 2) => d) ./ 2
	V = factorkp(factormp(V, 2, V, 2) => d) ./ 2
	W = factorkp(factormp(W, 2, W, 2) => d)
	if L == 1
		C = dec(V)
	else
		C = dec(factorhcat(U, V))
		decappend!(C, dec(factorutcat(U, V, W); len=L-2))
		decpush!(C, factorvcat(V, W))
	end
	decrankselect!(C, 1:1, 1:1)
	(major == "last") && decreverse!(C)
	C
end

function extmix(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if ℓ ∉ 0:L
		throw(ArgumentError("ℓ should be in 0:L"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	U = T[1 0; 0 1]; U = factor(U, 2, 2)
	U = factorkp(U => d)
	E = MatrixDec{T}()
	decappend!(E, dec(U; len=ℓ))
	if ℓ == L
		Z = Matrix{T}(I, 2^d*(d+1), 2^d*(d+1)); Z = factor(Z, 2^d*(d+1), 2^d*(d+1))
		decpush!(E, Z)
	else
		ρ = √(one(T)*2^d)
		c0 = T[1, 1//3]
		c1 = T[1, 1]
		if d == 1
			wn = [c0 c1]
		else
			wn = Matrix{T}(undef, 2^d, d+1)
			wn[:,1] = kron(ntuple(j -> c0, Val(d))...)
			for i ∈ 1:d
				wn[:,i+1] = kron(ntuple(j -> c0, Val(d-i))..., c1, ntuple(j -> c0, Val(i-1))...)
			end
		end
		wn = reshape(wn, :)

		c0 = Int[3,1]
		c1 = Int[0,1]

		wi = (d > 1) ? kron(ntuple(j -> c0, Val(d))...) : c0
		ww = wi.*((1/ρ)^(2*ℓ) / (one(T)*3)^d)
		for i ∈ 1:d
			wi = (d > 1) ? kron(ntuple(j -> c0, Val(d-i))..., c1, ntuple(j -> c0, Val(i-1))...) : c1
			ww .+= wi.*((2/ρ)^(2*ℓ) * 4 / (one(T)*3)^(d-1))
		end
	
		wi = (d > 1) ? kron(ntuple(j -> c0, Val(d))...) : c0
		w = wi.*((1/ρ)^(2*(ℓ+1)) / (one(T)*3)^d)
		for i ∈ 1:d
			wi = (d > 1) ? kron(ntuple(j -> c0, Val(d-i))..., c1, ntuple(j -> c0, Val(i-1))...) : c1
			w .+= wi.*((2/ρ)^(2*(ℓ+1)) * 4 / (one(T)*3)^(d-1))
		end
		w .= sqrt.(w)

		C0 = T[1 0; 0 1]
		C1 = T[0 2; 0 0]
		W1 = Array{T,3}(undef, 2^d, d+1, 2^d)
		if d == 1
			W1[:,1,:] = C0
			W1[:,2,:] = C1
		else
			W1[:,1,:] = kron(ntuple(j -> C0, Val(d))...)
			@inbounds for i ∈ 1:d
				W1[:,i+1,:] = kron(ntuple(j -> C0, Val(d-i))..., C1, ntuple(j -> C0, Val(i-1))...)
			end
		end

		W1 = reshape(W1, 2^d*(d+1), 2^d)
		rmul!(W1, Diagonal(1 ./ww))
		lmul!(Diagonal(wn), W1)
		W1 ./= ρ^(2*ℓ)
		W1 = reshape(W1, 2^d, d+1, 2^d)
		W1[:,2:d+1,:] .*= (one(T)*2)^ℓ
		W1 = reshape(W1, 2^d*(d+1), 2^d)
		W2 = reshape(T[1 1 0 0; -1//2 1//2 1//2 1//2], 2, 2, 2)
		W2 = factorkp(W2 => d)
		W2 = factorcontract(W2, Diagonal(w))
		W = factorcontract(W1, W2)
		W = reshape(W, 2^d*(d+1), 2^d, 2^d)
		W = permutedims(W, (2,1,3))
		W = reshape(W, 1, 2^d, 2^d*(d+1), 2^d)

		decpush!(E, W)
		for k ∈ ℓ+2:L
			W = reshape(T[1 1 0 0; -1//2 1//2 1//2 1//2], 2, 2, 2)
			W = factorkp(W => d)
			W = factorcontract(Diagonal(1 ./w), W)
			wi = (d > 1) ? kron(ntuple(j -> c0, Val(d))...) : c0
			w = wi.*((1/ρ)^(2*k) / (one(T)*3)^d)
			for i ∈ 1:d
				wi = (d > 1) ? kron(ntuple(j -> c0, Val(d-i))..., c1, ntuple(j -> c0, Val(i-1))...) : c1
				w .+= wi.*((2/ρ)^(2*k) * 4 / (one(T)*3)^(d-1))
			end
			w .= sqrt.(w)
			W = factorcontract(W, Diagonal(w))
			W = reshape(W, 2^d, 2^d, 1, 2^d)
			decpush!(E, W)
		end
		C0 = T[1 0; 0 1]
		C1 = T[0 0; 2 0]
		Z = Array{T,3}(undef, 2^d, 2^d, d+1)
		if d == 1
			Z[:,:,1] = C0
			Z[:,:,2] = C1
		else
			Z[:,:,1] = kron(ntuple(j -> C0, Val(d))...)
			@inbounds for i ∈ 1:d
				Z[:,:,i+1] = kron(ntuple(j -> C0, Val(d-i))..., C1, ntuple(j -> C0, Val(i-1))...)
			end
		end
		Z = reshape(Z, 2^d, 2^d*(d+1))
		lmul!(Diagonal(1 ./w), Z)
		Z = reshape(Z, 2^d, 2^d, d+1)
		Z[:,:,2:d+1] .*= (one(T)*2)^L
		Z = reshape(Z, 2^d, 2^d*(d+1), 1, 1)
		decpush!(E, Z)
	end
	(major == "last") && decreverse!(E)
	E
end

function diffbpxdn(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	ρ = √(one(T)*2^d)
	c0 = Int[3,1]
	c1 = Int[0,1]
	wk = (d > 1) ? kron(ntuple(k -> c0, Val(d))...) : c0
	w = wk./(3*one(T))^d
	for k ∈ 1:d
		wk = (d > 1) ? kron(ntuple(k -> c0, Val(d-k))..., c1, ntuple(k -> c0, Val(k-1))...) : c1
		w .+= wk.*(4*one(T)/3^(d-1))
	end
	w .= sqrt.(w)
	C = T[  1//2 -1//2; 1//2 1//2 ]
	B = (d > 1) ? kron(ntuple(k -> C, Val(d))...) : C
	rmul!(B, Diagonal(w))
	###
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	A = T[0 1]; A = factor(A, 1, 1)
	U = T[J O ; J' I ]; U = factor(U, 2, 2)
	V = U
	WW = T[2 1; 1 0; 0 1; 1 2] ./ sqrt(8*one(T)); WW = factor(WW, 2, 1)
	ZZ = T[0,1][:,:]; ZZ = factor(ZZ, 1, 1)
	A = factorkp(A => d)
	A = factorhcat(factormp(A, 2, A, 2), factormp(factorcontract(A, B), 2, A, 2))
	U = factorkp(U => d); U = factormp(U, 2, U, 2).*(sqrt(2*one(T))^d / 2)
	WW = factorkp(WW => d)
	ZZ = factorkp(ZZ => d)
	V = factorkp(V => d)
	M = MatrixDec{T}()
	decpush!(M, A)
	for k ∈ 1:L
		W = reshape(T[1 1 0 0; -1//2 1//2 1//2 1//2], 2, 2, 2)
		W = factorkp(W => d)
		W = factorcontract(Diagonal(1 ./w), W)
		wi = (d > 1) ? kron(ntuple(j -> c0, Val(d))...) : c0
		w = wi.*((1/ρ)^(2*k) / (one(T)*3)^d)
		for i ∈ 1:d
			wi = (d > 1) ? kron(ntuple(j -> c0, Val(d-i))..., c1, ntuple(j -> c0, Val(i-1))...) : c1
			w .+= wi.*((2/ρ)^(2*k) * 4 / (one(T)*3)^(d-1))
		end
		w .= sqrt.(w)
		C = T[  1//2 -1//2; 1//2 1//2 ]
		B = (d > 1) ? kron(ntuple(k -> C, Val(d))...) : C
		rmul!(B, Diagonal(w))
		VB = factormp(factorcontract(V, B), 2, V, 2).*(sqrt(2*one(T))^d / 2)
		W = factorcontract(W, Diagonal(w))
		W = reshape(W, 2^d, 2^d, 1, 2^d)
		W = factormp(W, 2, WW, 2)
		if k < L
			W = factorutcat(U, VB, W)
		else
			W = factorvcat(VB, W)
		end
		decpush!(M, W)
	end
	C0 = T[1 0; 0 1]
	C1 = T[0 0; 2 0]
	Z = Array{T,3}(undef, 2^d, 2^d, d+1)
	if d == 1
		Z[:,:,1] = C0
		Z[:,:,2] = C1
	else
		Z[:,:,1] = kron(ntuple(j -> C0, Val(d))...)
		@inbounds for i ∈ 1:d
			Z[:,:,i+1] = kron(ntuple(j -> C0, Val(d-i))..., C1, ntuple(j -> C0, Val(i-1))...)
		end
	end
	Z = reshape(Z, 2^d, 2^d*(d+1))
	lmul!(Diagonal(1 ./w), Z)
	Z = reshape(Z, 2^d, 2^d, d+1)
	Z[:,:,2:d+1] .*= (one(T)*2)^L
	Z = reshape(Z, 2^d, 2^d*(d+1), 1, 1)
	decpush!(M, factormp(Z, 2, ZZ, 2))
	decskp!(M, 1; path="forward")
	(major == "last") && decreverse!(M)
	M
end

function diffbpxdd(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	if d < 1
		throw(ArgumentError("d should be positive"))
	end
	if major ∉ ("first", "last")
		throw(ArgumentError("major should be either \"last\" (default) or \"first\""))
	end
	ρ = √(one(T)*2^d)
	c0 = Int[3,1]
	c1 = Int[0,1]
	wk = (d > 1) ? kron(ntuple(k -> c0, Val(d))...) : c0
	w = wk./(3*one(T))^d
	for k ∈ 1:d
		wk = (d > 1) ? kron(ntuple(k -> c0, Val(d-k))..., c1, ntuple(k -> c0, Val(k-1))...) : c1
		w .+= wk.*(4*one(T)/3^(d-1))
	end
	w .= sqrt.(w)
	C = T[  1//2 -1//2; 1//2 1//2 ]
	###
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	I1 = [1 0; 0 0]; I2 = [0 0; 0 1]
	A = T[1 0 0]; A = factor(A, 1, 1)
	U = T[I2 I1 J'; O I J'; O O J]; U = factor(U, 2, 2)
	V = U[:,:,:,[3,2]]
	WW = T[2 1; 1 0; 0 1; 1 2] ./ sqrt(8*one(T)); WW = factor(WW, 2, 1)
	ZZ = T[0,1][:,:]; ZZ = factor(ZZ, 1, 1)
	A = factorkp(A => d)
	A = factorhcat(factormp(A, 2, A, 2), zeros(T, 1, 1, 1, 4^d))
	U = factorkp(U => d); U = factormp(U, 2, U, 2).*(sqrt(2*one(T))^d / 2)
	WW = factorkp(WW => d)
	ZZ = factorkp(ZZ => d)
	V = factorkp(V => d)
	M = MatrixDec{T}()
	decpush!(M, A)
	for k ∈ 1:L
		W = reshape(T[1 1 0 0; -1//2 1//2 1//2 1//2], 2, 2, 2)
		W = factorkp(W => d)
		W = factorcontract(Diagonal(1 ./w), W)
		wi = (d > 1) ? kron(ntuple(j -> c0, Val(d))...) : c0
		w = wi.*((1/ρ)^(2*k) / (one(T)*3)^d)
		for i ∈ 1:d
			wi = (d > 1) ? kron(ntuple(j -> c0, Val(d-i))..., c1, ntuple(j -> c0, Val(i-1))...) : c1
			w .+= wi.*((2/ρ)^(2*k) * 4 / (one(T)*3)^(d-1))
		end
		w .= sqrt.(w)
		C = T[  1//2 -1//2; 1//2 1//2 ]
		B = (d > 1) ? kron(ntuple(k -> C, Val(d))...) : C
		rmul!(B, Diagonal(w))
		VB = factormp(factorcontract(V, B), 2, V, 2).*(sqrt(2*one(T))^d / 2)
		W = factorcontract(W, Diagonal(w))
		W = reshape(W, 2^d, 2^d, 1, 2^d)
		W = factormp(W, 2, WW, 2)
		if k < L
			W = factorutcat(U, VB, W)
		else
			W = factorvcat(VB, W)
		end
		decpush!(M, W)
	end
	C0 = T[1 0; 0 1]
	C1 = T[0 0; 2 0]
	Z = Array{T,3}(undef, 2^d, 2^d, d+1)
	if d == 1
		Z[:,:,1] = C0
		Z[:,:,2] = C1
	else
		Z[:,:,1] = kron(ntuple(j -> C0, Val(d))...)
		@inbounds for i ∈ 1:d
			Z[:,:,i+1] = kron(ntuple(j -> C0, Val(d-i))..., C1, ntuple(j -> C0, Val(i-1))...)
		end
	end
	Z = reshape(Z, 2^d, 2^d*(d+1))
	lmul!(Diagonal(1 ./w), Z)
	Z = reshape(Z, 2^d, 2^d, d+1)
	Z[:,:,2:d+1] .*= (one(T)*2)^L
	Z = reshape(Z, 2^d, 2^d*(d+1), 1, 1)
	decpush!(M, factormp(Z, 2, ZZ, 2))
	decskp!(M, 1; path="forward")
	(major == "last") && decreverse!(M)
	M
end













checkdim(d) = throw(ArgumentError("Dimension parameter d should be a positive integer"))
checkdim(d::Int) = d > 0 || throw(ArgumentError("Dimension parameter d should be a positive integer"))

implementationrequired(::Type{S}) where {S} = throw(ErrorException("The implementation of method "*string(StackTraces.stacktrace()[2].func)*" for type "*string(S)*" is currently missing"))


##################
### Refinement ###

abstract type RefinementBasis{d} end #  for now, this is always of order one

abstract type Refinement{d,B<:RefinementBasis{d}} end


### Specification of required methods
cube_nodal_to_normedQ(::S) where {S<:Refinement} = implementationrequired(S)
cube_nodal_evaluate(::S, ::NTuple{d,Vector{T}}) where {T<:Number,d,S<:Refinement{d}} = implementationrequired(S)
## some of the following four previously lead to ambiguity??? Currently that does not occur...
cube_normedQ_to_normedF(::S, ::T, ::T) where {T<:AbstractFloat,S<:Refinement} = implementationrequired(S)
### TO BE FIXED # cube_normedQ_to_normedF(::S, ::T, ::Matrix{T}) where {T<:AbstractFloat,S<:Refinement} = implementationrequired(S)
### NOT NEEDED? # cube_normedQ_to_normedF(::S, ::T, ::T) where {T<:AbstractFloat,d,B<:RefinementBasis,S<:Refinement{d,B}} = implementationrequired(S)
### TO BE FIXED # cube_normedQ_to_normedF(::S, ::T, ::Matrix{T}) where {T<:AbstractFloat,d,B<:RefinementBasis,S<:Refinement{d,B}} = implementationrequired(S)


### Compositions
cube_nodal_to_normedF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,S<:Refinement} = cube_normedQ_to_normedF(re, κ₀, κ₁)*cube_nodal_to_normedQ(re)
### TO BE FIXED # cube_nodal_to_normedF(re::S, κ₀::T, K₁::Matrix{T}) where {T<:AbstractFloat,S<:Refinement} = cube_normedQ_to_normedF(re, κ₀, K₁)*cube_nodal_to_normedQ(re)


### Extensions

function cube_nodal_refinement_factor(re::S, ::Val{n}) where {d,S<:Refinement{d},n}
	T = Rational{Int}
	m = ntuple(k -> n, Val(d))
	p = ntuple(k -> 2, Val(d))
	t = T[ i//n for i ∈ 0:n ]
	V = cube_nodal_evaluate(re, ntuple(k -> t, Val(d)))
	V = reshape(V, (m.+1)..., 2^d)
	W = zeros(T, 2^d, m..., p...)
	for α ∈ 1:2^d, i ∈ CartesianIndices(m), β ∈ CartesianIndices(p)
		γ = Tuple(β).+Tuple(i).-1
		W[α,i,β] = V[γ...,α]
	end
	reshape(W, 2^d, n^d, 2^d)
end


#########################################
### Refinement with orthogonalization ###

abstract type OrthMethod end

struct OrthRefinement{d,B<:RefinementBasis{d},O<:OrthMethod} <: Refinement{d,B} end
OrthRefinement(::B,::M) where {d,B<:RefinementBasis{d},M<:OrthMethod} = OrthRefinement{d,B,M}()

cube_normedF_to_orth(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,S<:OrthRefinement} = adjoint(cube_orth_to_normedF(re, κ₀, κ₁))

### Specification of required methods
cube_auxQ_to_nodal(::S) where {S<:OrthRefinement} = implementationrequired(S)
cube_nodal_to_auxQ(::S) where {S<:OrthRefinement} = implementationrequired(S)
cube_auxF_to_auxQ(::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,S<:OrthRefinement} = implementationrequired(S)
cube_auxQ_to_auxF(::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,S<:OrthRefinement} = implementationrequired(S)


### Compositions
cube_auxF_to_nodal(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,S<:OrthRefinement} = cube_auxQ_to_nodal(re)*cube_auxF_to_auxQ(re, κ₀, κ₁)
cube_nodal_to_auxF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,S<:OrthRefinement} = cube_auxQ_to_auxF(re, κ₀, κ₁)*cube_nodal_to_auxQ(re)
cube_auxQ_to_normedQ(re::S) where {S<:OrthRefinement} = cube_nodal_to_normedQ(re)*cube_auxQ_to_nodal(re)
cube_auxF_to_normedF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,S<:OrthRefinement} = cube_normedQ_to_normedF(re, κ₀, κ₁)*cube_auxQ_to_normedQ(re)*cube_auxF_to_auxQ(re, κ₀, κ₁)

### Extensions

cube_auxQ_refinement_factor(re::S, nn::Val{n}) where {S<:OrthRefinement,n} = factorcontract(factorcontract(permutedims(cube_auxQ_to_nodal(re)), cube_nodal_refinement_factor(re, nn)), permutedims(cube_nodal_to_auxQ(re)))

cube_auxF_refinement_factor(re::S, κ₀::T, κ₁::T, nn::Val{n}) where {T<:AbstractFloat,d,S<:OrthRefinement{d},n} = factorcontract(factorcontract(permutedims(cube_auxF_to_auxQ(re, κ₀, κ₁)), Array{T}(cube_auxQ_refinement_factor(re, nn))), permutedims(cube_auxQ_to_auxF(re, κ₀/(√(n*one(T)))^d, κ₁*n/(√(n*one(T)))^d)))


### Explicit orthogonalization
struct OrthExplicit <: OrthMethod end

cube_auxF_to_orth_to_normedF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthExplicit}} = cube_auxF_to_normedF(re, κ₀, κ₁),I

cube_orth_to_normedF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthExplicit}} = cube_auxF_to_normedF(re, κ₀, κ₁)

cube_auxF_to_orth(::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthExplicit}} = I

cube_orth_to_auxF(::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthExplicit}} = I

cube_auxF_to_orth_to_auxF(::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthExplicit}} = I,I

cube_normedF_to_auxF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthExplicit}} = cube_normedF_to_orth(re, κ₀, κ₁)


### Orthogonalization by pivoted QR
struct OrthPQR <: OrthMethod end

function cube_auxF_to_orth_to_normedF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthPQR}}
	W = cube_auxF_to_normedF(re, κ₀, κ₁)
	n = size(W, 2)
	fact = qr!(W, Val(true))
	Q = fact.Q*Matrix{T}(I, n, n)
	R = fact.R[:,invperm(fact.p)]
	Q,R
end

function cube_orth_to_normedF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthPQR}}
	W = cube_auxF_to_normedF(re, κ₀, κ₁)
	n = size(W, 2)
	fact = qr!(W, Val(true))
	Q = fact.Q*Matrix{T}(I, n, n)
	Q
end

function cube_auxF_to_orth(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthPQR}}
	W = cube_auxF_to_normedF(re, κ₀, κ₁)
	fact = qr!(W, Val(true))
	R = fact.R[:,invperm(fact.p)]
	R
end

function cube_orth_to_auxF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthPQR}}
	W = cube_auxF_to_normedF(re, κ₀, κ₁)
	n = size(W, 2)
	fact = qr!(W, Val(true))
	R = UpperTriangular(triu!(fact.R))
	Rinv = R\Matrix{T}(I, n, n)
	Rinv = Rinv[invperm(fact.p),:]
	Rinv
end

function cube_auxF_to_orth_to_auxF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthPQR}}
	W = cube_auxF_to_normedF(re, κ₀, κ₁)
	n = size(W, 2)
	fact = qr!(W, Val(true))
	R = UpperTriangular(triu!(fact.R))
	Rinv = R\Matrix{T}(I, n, n)
	R = R[:,invperm(fact.p)]
	Rinv = Rinv[invperm(fact.p),:]
	Rinv,R # orth_to_auxF, auxF_to_orth
end

function cube_normedF_to_auxF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,<:RefinementBasis{d},OrthPQR}}
	W = cube_auxF_to_normedF(re, κ₀, κ₁)
	n = size(W, 2)
	fact = qr!(W, Val(true))
	Q = fact.Q*Matrix{T}(I, n, n)
	R = UpperTriangular(triu!(fact.R))
	(R\Q')[invperm(fact.p),:]
end



function cube_normedF_refine_dec(re::S, ::Type{C}, κ₀::T, κ₁::T, ℓ::Int, L::Int) where {d,S<:OrthRefinement{d},T<:AbstractFloat,C<:FloatRC{T}}
	𝟙 = one(T)
	if ℓ < 0
		throw(ArgumentError("ℓ should be nonnegative"))
	end
	if L < ℓ
		throw(ArgumentError("L should be at least ℓ"))
	end
	U = MatrixDec{C}(undef, L+2)
	V = ones(C, 1, 1);
	U[1] = factor(V, 1, 1)
	VV = Matrix{C}(I, 2^d, 2^d);
	V = factor(VV, 2^d, 2^d)
	for j ∈ 1:ℓ
		U[j+1] = V
	end
	ρ = √((2^d)𝟙)

	FF = VectorFactor{T}(cube_auxQ_refinement_factor(re, Val(2)))
	F = factormodereshape(FF, [2^d,1])
	local YY,HH
	for j ∈ L+1:-1:ℓ+1
		μ₀ = κ₀*(1/ρ)^(j-1)
		μ₁ = κ₁*(2/ρ)^(j-1)
		X,Y = cube_auxF_to_orth_to_auxF(re, μ₀, μ₁)	# X is orth → auxF, Y is auxF → orth
		Z = cube_auxF_to_auxQ(re, μ₀, μ₁)
		H = cube_auxQ_to_auxF(re, μ₀, μ₁)
		if j == L+1
			V = cube_normedQ_to_normedF(re, μ₀, μ₁)*cube_auxQ_to_normedQ(re)*(Z*X) # Z*X is orth → auxQ
			U[j+1] = reshape(permutedims(V), 2^d, :, 1, 1)
		else
			U[j+1] = factorcontract(permutedims(Z*X), factorcontract(F, permutedims(YY*HH))) # Z*X is orth → auxQ, YY*HH is auxQ → orth
			if j == ℓ+1
				QQ = cube_orth_to_normedF(re, μ₀, μ₁)
				Q = reshape(QQ, 1, 1, :, 2^d)
				U[j+1] = factorcontract(Q, U[j+1])
			end
		end
		YY,HH = Y,H
	end
	U
end



### Convenience functions for the default values of parameters
cube_auxF_to_nodal(re::S, ::Type{T}) where {T<:AbstractFloat,S<:OrthRefinement} = cube_auxF_to_nodal(re, one(T), one(T))
cube_nodal_to_auxF(re::S, ::Type{T}) where {T<:AbstractFloat,S<:OrthRefinement} = cube_nodal_to_auxF(re, one(T), one(T))
cube_auxF_to_normedF(re::S, ::Type{T}) where {T<:AbstractFloat,S<:OrthRefinement} = cube_auxF_to_normedF(re, one(T), one(T))
cube_auxF_to_auxQ(re::S, ::Type{T}) where {T<:AbstractFloat,S<:OrthRefinement} = cube_auxF_to_auxQ(re, one(T), one(T))
cube_auxQ_to_auxF(re::S, ::Type{T}) where {T<:AbstractFloat,S<:OrthRefinement} = cube_auxQ_to_auxF(re, one(T), one(T))
cube_normedF_refine_dec(re::S, ::Type{C}, ℓ::Int, L::Int) where {d,S<:OrthRefinement{d},T<:AbstractFloat,C<:FloatRC{T}} = cube_normedF_refine_dec(re, C, one(T), one(T), ℓ, L)


function cube_basisfactors_dn(::Val{d}, L::Int) where {d}
	checkdim(d)
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	UU = [J O; J' I]; UU = factor(UU, 2, 2)
	VV = [0 1]; VV = factor(VV, 1, 1)
	U = MatrixDec{Int}()
	decpush!(U, VV)
	for ℓ ∈ 1:L
		decpush!(U, UU)
	end
	QQ = [1 0; 0 1]
	Q = [ QQ for ℓ ∈ 0:L ]
	P = [0,1]
	ntuple(k -> U, Val(d)),ntuple(k -> Q, Val(d)),ntuple(k -> P, Val(d))
end

function cube_basisfactors_nn(::Val{d}, L::Int) where {d}
	checkdim(d)
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]
	UU = [I O O; O I J; O O J']; UU = factor(UU, 2, 2)
	VV = [1 0 1 0 0 1]; VV = factor(VV, 1, 2)
	U = MatrixDec{Int}()
	decpush!(U, VV)
	for ℓ ∈ 1:L
		decpush!(U, UU)
	end
	QQ = [1 0; 0 0; 0 1]
	Q = [ QQ for ℓ ∈ 0:L ]
	P = [1,0]
	ntuple(k -> U, Val(d)),ntuple(k -> Q, Val(d)),ntuple(k -> P, Val(d))
end

function cube_basisfactors_dd(::Val{d}, L::Int) where {d}
	checkdim(d)
	if L < 1
		throw(ArgumentError("L should be positive"))
	end
	I = [1 0; 0 1]; J = [0 1; 0 0]; O = [0 0; 0 0]; I1 = [1 0; 0 0]; I2 = [0 0; 0 1];
	UU = [J O O; J' I O; J' I1 I2]; UU = factor(UU, 2, 2)
	VV = [0 0 1]; VV = factor(VV, 1, 1)
	U = MatrixDec{Int}()
	decpush!(U, VV)
	for ℓ ∈ 1:L
		decpush!(U, UU)
	end
	QQ = [1 0; 0 1; 0 0]
	Q = [ QQ for ℓ ∈ 0:L ]
	P = [0,1]
	ntuple(k -> U, Val(d)),ntuple(k -> Q, Val(d)),ntuple(k -> P, Val(d))
end

function cube_mixed_basisfactors(re::S, κ₀::T, κ₁::T, U::NTuple{d,MatrixDec{Int}}, Q::NTuple{d,Vector{Matrix{Int}}}) where {d,S<:OrthRefinement{d},T<:AbstractFloat}
	L = declength(U[1])-1
	if L < 0
		throw(ArgumentError("each U[k] should be a decomposition of length L+1, where L is the (nonegative) number of levels"))
	end
	for k ∈ 1:d
		if declength(U[k]) ≠ L+1 || declength(Q[k]) ≠ L+1
			throw(ArgumentError("all elements of U and Q should be decompositions of the same length"))
		end
	end
	m = zeros(Int, d, L+1)
	r = zeros(Int, d, L+2)
	for k ∈ 1:d
		sz = decsize(U[k])
		m[k:k,:] = sz[1:1,:]
		r[k:k,:] = decrank(U[k])
	end
	if any(m[:,2:L+1] .≠ 2)
		throw(ArgumentError("each mode dimension, except the first, of each element of U should be equal to 2"))
	end
	for k ∈ 1:d, ℓ ∈ 0:L
		if size(Q[k][ℓ+1], 1) ≠ r[k,ℓ+2]
			throw(ArgumentError("U and Q have inconsitent ranks"))
		end
	end
	Umix = ntuple(k -> MatrixDec{T}(undef, L+1), Val(d))
	Qmix = ntuple(k -> Vector{Matrix{T}}(undef, L+1), Val(d))
	Cmix = Vector{Matrix{T}}(undef, L+1)
	Wmix = VectorDec{T}(undef, L+1)

	ρ = √(one(T)*2)
	F = VectorFactor{T}(cube_auxQ_refinement_factor(re, Val(2)))
	G = cube_nodal_to_auxQ(re)
	local YY,HH
	for ℓ ∈ L+1:-1:1
		μ₀ = κ₀*(1/ρ^d)^(ℓ-1)
		μ₁ = κ₁*(2/ρ^d)^(ℓ-1)
		X,Y = cube_auxF_to_orth_to_auxF(re, μ₀, μ₁)	# X is orth → auxF, Y is auxF → orth
		Z = cube_auxF_to_auxQ(re, μ₀, μ₁)
		H = cube_auxQ_to_auxF(re, μ₀, μ₁)
		Cmix[ℓ] = Matrix{T}(transpose(Y*H*G))
		Cmix[ℓ] .*= ρ^(d*ℓ) / (2*one(T))^(ℓ-1)
		if ℓ == L+1
			V = cube_normedQ_to_normedF(re, μ₀, μ₁)*cube_auxQ_to_normedQ(re)*(Z*X) # Z*X is orth → auxQ
			Wmix[L+1] = reshape(permutedims(V), 2^d, :, 1)
		else
			Wmix[ℓ] = factorcontract(permutedims(Z*X), factorcontract(F, permutedims(YY*HH))) # Z*X is orth → auxQ, YY*HH is auxQ → orth
		end
		for k ∈ 1:d
			Qmix[k][ℓ] = Matrix{T}(Q[k][ℓ])
			Qmix[k][ℓ] ./= ρ
			Umix[k][ℓ] = MatrixFactor{T}(U[k][ℓ])
		end
		YY,HH = Y,H
	end
	Umix,Qmix,Cmix,Wmix
end

function cube_nodal_to_normedF!(re::S, κ₀::T, κ₁::T, U::VectorDec{C}) where {d,S<:Refinement{d},T<:AbstractFloat,C<:FloatRC{T}}
	L = declength(U)-2
	if L < 0
		throw(ArgumentError("the input decompositions should have at least two factors"))
	end
	n,r = decsize(U),decrank(U)
	if r[1] ≠ 1 || r[L+3] ≠ 1
		throw(ArgumentError("both terminal ranks of U should be unit"))
	end
	if n[L+2] ≠ 2^d
		throw(ArgumentError("the last factor of U should have mode size 2^d, where d is the dimension parameter"))
	end
	if any(n[2:L+1] .≠ 2^d)
		throw(ArgumentError("all factors of U except the first should have mode size 2^d, where d is the dimension parameter"))
	end
	ρ = √(one(T)*2^d)
	μ₀ = κ₀*(1/ρ)^L
	μ₁ = κ₁*(2/ρ)^L
	# W = cube_nodal_to_normedF(re, μ₀, μ₁)
	W = Matrix{C}(cube_nodal_to_normedF(re, μ₀, μ₁)) # TODO the conversion should not be necessary, but a segfault may occur without it in subsequent computations…
	W = factor(W, size(W)...)
	U[L+2] = factormp(W, 2, U[L+2], 1)
	U
end








include("FEMQ1.jl")








end
