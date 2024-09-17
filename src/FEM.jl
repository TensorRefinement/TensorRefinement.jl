module FEM

using LinearAlgebra, TensorRefinement.Auxiliary, ..TensorTrain

export extdn, extdd, diffdn, diffdd, dint, dintf, bpxdn, bpxdd, extmix, diffbpxdn, diffbpxdd

"""
    extdn(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}

Generate a matrix decomposition which specifies ... based on the parameters provided.

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `ℓ::Int`: Index of the current factor.
- `d::Int`: ...
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `MatrixDec{T}`: Matrix decomposition `P` representing ...

# Throws
- `ArgumentError`: If `L` is not positive.
- `ArgumentError`: If `ℓ` is not in `0:L`.
- `ArgumentError`: If `d` is not positive.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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

"""
    extdd(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}

Generate a matrix decomposition specifiying... 

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `ℓ::Int`: Index of the current factor.
- `d::Int`: Size of the ...
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `MatrixDec{T}`: Matrix decomposition `P` representing ...

# Throws
- `ArgumentError`: If `L` is not positive.
- `ArgumentError`: If `ℓ` is not in `1:L`.
- `ArgumentError`: If `d` is not positive.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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

"""
    diffdn(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}

Construct a matrix decomposition that represents ...

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `ℓ::Int`: Index of the current factor.
- `d::Int`: Size of the ...
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `MatrixDec{T}`: Matrix decomposition `M` representing ...

# Throws
- `ArgumentError`: If `L` is negative.
- `ArgumentError`: If `ℓ` is not in `0:L`.
- `ArgumentError`: If `d` is not positive.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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
	w = wk.*((1/ρ)^(2*ℓ) / 3^d)
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

"""
    diffdn(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}

Sentence summarizing the function.

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `d::Int`: Size of the ...
- `major::String="last"`: Specifies the major axis ("first" or "last") for the operation. The default is `"last"`.

# Returns
Returns the result ....

# Throws
- `ArgumentError`: If `L` is negative.
- `ArgumentError`: If `d` is not positive.
- `ArgumentError`: If `major` is not `"first"` or `"last"`.
"""
diffdn(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC} = diffdn(T, L, L, d; major=major)

"""
    diffdd(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}

Construct a matrix decomposition representing ... based on the parameters provided.

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `ℓ::Int`: Index of the current factor. Should be in the range `1:L`.
- `d::Int`: Size of the...
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `MatrixDec{T}`: Matrix decomposition `M` representing ...

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `ℓ` is not in `1:L`.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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
	w = wk.*((1/ρ)^(2*ℓ) / 3^d)
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

"""
    diffdd(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}

Summary Sentence for function.

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `d::Int`: Size of the...
- `major::String="last"`: Specifies the major axis ("first" or "last") for the operation. The default is `"last"`.

# Returns
Returns the result...

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
diffdd(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC} = diffdd(T, L, L, d; major=major)

"""
    dint(::Type{T}, L::Int, d::Int, K::AbstractMatrix{T}; major::String="last") where {T<:FloatRC}

Construct a decomposition representing integration ... with a given coefficient matrix `K`.

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `d::Int`: Size of the ...
- `K::AbstractMatrix{T}`: Coefficient matrix, should be of size `(d+1, d+1)`.
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `Dec{T,N}`: Decomposition `Λ` representing the integration.

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `K` is not of size `(d+1, d+1)`.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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

"""
    dintf(::Type{T}, L::Int, d::Int, K::AbstractMatrix{T}; major::String="last") where {T<:FloatRC}

Construct a decomposition representing integration ... with a given coefficient matrix `K`.

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `d::Int`: Size of the ...
- `K::AbstractMatrix{T}`: Coefficient matrix, should be of size `(d+1, d+1)`.
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `Dec{T,N}`: Decomposition `Λ` representing the integration...

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `K` is not of size `(d+1, d+1)`.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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

"""
    bpxdn(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}

Construct a decomposition representing ...

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `d::Int`: Size of the...
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `Dec{T,N}`: Decomposition `C` representing ...

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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

"""
    bpxdd(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}

Construct a decomposition representing ...

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `d::Int`: Size of the ... 
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `Dec{T,N}`: Decomposition `C` representing ...

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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

"""
    extmix(::Type{T}, L::Int, ℓ::Int, d::Int; major::String="last") where {T<:FloatRC}

Construct a decomposition representing ... for a given set of parameters.

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `ℓ::Int`: Current factor index (must be between 0 and L).
- `d::Int`: Size of the...
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `MatrixDec{T}`: Matrix decomposition `E` representing ...

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `ℓ` is not in the range `0:L`.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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

"""
    diffbpxdn(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}

Generate a decomposition representing ... using ...

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `d::Int`: Size of the ...
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `MatrixDec{T}`: Matrix decomposition `M` representing ...

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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
	w = wk./3^d
	for k ∈ 1:d
		wk = (d > 1) ? kron(ntuple(k -> c0, Val(d-k))..., c1, ntuple(k -> c0, Val(k-1))...) : c1
		w .+= wk.*(4/3^(d-1))
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

"""
    diffbpxdd(::Type{T}, L::Int, d::Int; major::String="last") where {T<:FloatRC}

Generate decomposition representing ... using ...

# Arguments
- `::Type{T}`: Numeric type (subtype of `FloatRC`).
- `L::Int`: Total number of factors in the decomposition.
- `d::Int`: Size of the ...
- `major::String`: Specifies the major dimension of the decomposition, either `"last"` (default) or `"first"`.

# Returns
- `MatrixDec{T}`: Matrix decomposition `M` representing ...

# Throws
- `ArgumentError`: If `L` is less than 1.
- `ArgumentError`: If `d` is less than 1.
- `ArgumentError`: If `major` is neither `"first"` nor `"last"`.
"""
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
	w = wk./3^d
	for k ∈ 1:d
		wk = (d > 1) ? kron(ntuple(k -> c0, Val(d-k))..., c1, ntuple(k -> c0, Val(k-1))...) : c1
		w .+= wk.*(4/3^(d-1))
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


end