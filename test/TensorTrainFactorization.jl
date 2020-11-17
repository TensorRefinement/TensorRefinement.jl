using Random, LinearAlgebra, TensorRefinement, Test

Random.seed!(1365)

@time @testset "Factorization orthogonalization" begin
	n = [2,1,3]; r = [5,4,3,7,8]
	d = length(n)
	L = length(r)-1
	@testset "Type: $S" for T ∈ (Float64, Float32), S ∈ (T, Complex{T})
		tol = 100*eps(T)
		U = decrand(n, r); Uf = factor(U); nrm = norm(Uf)
		@testset "pivot=$pivot, path=$path" for pivot ∈ (false, true), path ∈ ("forward", "backward")
			V = decqr!(deepcopy(U); pivot=pivot, path=path)
			Vf = factor(V)
			@test norm(Vf-Uf)/nrm ≈ 0 atol=tol
			(path == "backward") && decreverse!(V)
			p = decrank(V)
			for ℓ ∈ 1:L
				Q = reshape(V[ℓ], p[ℓ]*prod(n), p[ℓ+1])
				E = Q'*Q-I
				@test norm(E)/p[ℓ+1]^2 ≈ 0 atol=tol
			end
		end
    end
end