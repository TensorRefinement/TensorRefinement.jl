using LinearAlgebra, TensorRefinement
using Test

function factorqr!_test(::Type{T}, p::Int, q::Int, n::Vector{Int}, rtol::Real) where {T<:FloatRC{<:AbstractFloat}}
	U = rand(T, p, n..., q)
	atol = rtol*norm(U)
	@testset "without pivoting" begin
		Q,R = factorqr!(copy(U)); V = factorcontract(Q, R)
		@test norm(V-U) ≈ 0 atol=atol
		Q,R = factorqr!(copy(U); rev=false); V = factorcontract(Q, R)
		@test norm(V-U) ≈ 0 atol=atol
		Q,R = factorqr!(copy(U); rev=true); V = factorcontract(R, Q)
		@test norm(V-U) ≈ 0 atol=atol
	end
	@testset "without pivoting, specified explicitly" begin
		Q,R = factorqr!(copy(U), Val(false)); V = factorcontract(Q, R)
		@test norm(V-U) ≈ 0 atol=atol
		Q,R = factorqr!(copy(U), Val(false); rev=false); V = factorcontract(Q, R)
		@test norm(V-U) ≈ 0 atol=atol
		Q,R = factorqr!(copy(U), Val(false); rev=true); V = factorcontract(R, Q)
		@test norm(V-U) ≈ 0 atol=atol
	end
	@testset "with pivoting, specified explicitly" begin
		Q,R = factorqr!(copy(U), Val(true)); V = factorcontract(Q, R)
		@test norm(V-U) ≈ 0 atol=atol
		Q,R = factorqr!(copy(U), Val(true); rev=false); V = factorcontract(Q, R)
		@test norm(V-U) ≈ 0 atol=atol
		Q,R = factorqr!(copy(U), Val(true); rev=true); V = factorcontract(R, Q)
		@test norm(V-U) ≈ 0 atol=atol
	end
end

@testset "Factor orthogonalization test" begin
	p,q = 7,19
	n = [4,5,6,7]
	@testset "Underlying type: $T" for T ∈ [Float64, Float32]
		rtol = sqrt(eps(T))
		@testset "$T" for i ∈ 1:3
			factorqr!_test(T, p, q, n, rtol)
		end
		@testset "Complex{$T}" for i ∈ 1:3
			factorqr!_test(Complex{T}, p, q, n, rtol)
		end
    end
end
