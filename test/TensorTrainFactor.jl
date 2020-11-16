using LinearAlgebra, TensorRefinement
using Test

function factorqr!_test(::Type{T}, p::Int, q::Int, n::Vector{Int}, atol::Real) where {T<:FloatRC{<:AbstractFloat}}
	U = rand(T, p, n..., q)
	# without pivoting
	Q,R = factorqr!(copy(U)); V = factorcontract(Q, R)
	@test norm(V-U) ≈ 0 atol=atol
	Q,R = factorqr!(copy(U); rev=false); V = factorcontract(Q, R)
	@test norm(V-U) ≈ 0 atol=atol
	Q,R = factorqr!(copy(U); rev=true); V = factorcontract(R, Q)
	@test norm(V-U) ≈ 0 atol=atol
	# without pivoting, specified explicitly
	Q,R = factorqr!(copy(U), Val(false)); V = factorcontract(Q, R)
	@test norm(V-U) ≈ 0 atol=atol
	Q,R = factorqr!(copy(U), Val(false); rev=false); V = factorcontract(Q, R)
	@test norm(V-U) ≈ 0 atol=atol
	Q,R = factorqr!(copy(U), Val(false); rev=true); V = factorcontract(R, Q)
	@test norm(V-U) ≈ 0 atol=atol
	# with pivoting
	Q,R = factorqr!(copy(U), Val(true)); V = factorcontract(Q, R)
	@test norm(V-U) ≈ 0 atol=atol
	Q,R = factorqr!(copy(U), Val(true); rev=false); V = factorcontract(Q, R)
	@test norm(V-U) ≈ 0 atol=atol
	Q,R = factorqr!(copy(U), Val(true); rev=true); V = factorcontract(R, Q)
	@test norm(V-U) ≈ 0 atol=atol
end

@testset "Factor orthogonalization test" begin
	p,q = 7,19
	n = [4,5,6,7]
	@testset "Underlying type: $T" for T ∈ [Float64, Float32]
		atol = sqrt(eps(T))
		@testset "$T" for i ∈ 1:3
			factorqr!_test(T, p, q, n, atol)
		end
		@testset "Complex{$T}" for i ∈ 1:3
			factorqr!_test(Complex{T}, p, q, n, atol)
		end
    end
end
