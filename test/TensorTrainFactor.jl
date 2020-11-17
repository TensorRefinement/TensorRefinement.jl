using Random, LinearAlgebra, TensorRefinement, Test

Random.seed!(1365)

@testset "Factor orthogonalization" begin
	n = [4,5,6,7]; p,q = 7,19
	d = length(n)
	@testset "Type: $S" for T ∈ (Float64, Float32), S ∈ (T, Complex{T})
		tol = 100*eps(T)
		U = rand(S, p, n..., q); nrm = norm(U)
		@testset "pivot=$pivot, rev=$rev" for pivot ∈ (false, true), rev ∈ (false, true)
			Q,R = factorqr!(copy(U), Val(pivot); rev=rev)
			r,s = factorranks(Q)
			rev || (r = s)
			if rev
				@test factorranks(R) == (p,r)
				@test factorranks(Q) == (r,q)
				@test factorsize(R) == ones(Int, d)
				@test factorsize(Q) == n
			else
				@test factorranks(Q) == (p,r)
				@test factorranks(R) == (r,q)
				@test factorsize(R) == ones(Int, d)
				@test factorsize(Q) == n
			end
			V = factorcontract(Q, R; rev=rev)
			@test norm(V-U)/nrm ≈ 0 atol=tol
			if rev
				Q = reshape(Q, r, prod(n)*q)
				E = Q*Q'-I
				@test norm(E)/r ≈ 0 atol=tol
			else
				Q = reshape(Q, p*prod(n), r)
				E = Q'*Q-I
				@test norm(E)/sqrt(r) ≈ 0 atol=tol
			end
		end
    end
end
