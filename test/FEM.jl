using LinearAlgebra, TensorRefinement, Test

function refextdn(L::Int, ℓ::Int, d::Int)
	I = diagm(0 => ones(2^ℓ))
	if ℓ > 0
		S = diagm(-1 => ones(2^ℓ-1))
	else
		S = [0.0]
	end
	η = collect(1:2^(L-ℓ))/2^(L-ℓ)
	ζ = collect(2^(L-ℓ)-1:-1:0)/2^(L-ℓ)
	P = kron(I, η) + kron(S, ζ)
	P .*= 2.0^((ℓ-L)/2)
	PP = P
	for k ∈ 2:d
		PP = kron(P, PP)
	end
	return PP
end

function refdiffdn(ℓ::Int, d::Int, K::Int)
	I = diagm(0 => ones(2^ℓ))
	if ℓ > 0
		S = diagm(-1 => ones(2^ℓ-1))
	else
		S = [0.0]
	end
	M0 = 2.0^(ℓ/2-1) * ( kron(I+S, [1; 0]) + kron(I-S, [0; 1]) )
	M1 = 2.0^(ℓ/2-1+(ℓ+1)) * (I-S)
	M = (K == 1) ? M1 : M0
	for κ ∈ 2:d
		M = kron((κ == K) ? M1 : M0, M)
	end
	return M
end

function refintdn(ℓ::Int, d::Int, K1::Int, K2::Int)
	Λ00 = kron(diagm(0 => ones(2^ℓ))/2^ℓ, diagm(0 => [1,1/3]))
	Λ11 = kron(diagm(0 => ones(2^ℓ))/2^ℓ, [1.0])
	Λ01 = kron(diagm(0 => ones(2^ℓ))/2^ℓ, [1.0; 0.0])
	Λ10 = kron(diagm(0 => ones(2^ℓ))/2^ℓ, [1.0 0.0])
	κ = 1
	if κ == K1 && κ == K2
		Λ = Λ11
	elseif κ == K1 && κ ≠ K2
		Λ = Λ10
	elseif κ ≠ K1 && κ == K2
		Λ = Λ01
	else
		Λ = Λ00
	end
	for κ ∈ 2:d
		if κ == K1 && κ == K2
			Λ = kron(Λ11, Λ)
		elseif κ == K1 && κ ≠ K2
			Λ = kron(Λ10, Λ)
		elseif κ ≠ K1 && κ == K2
			Λ = kron(Λ01, Λ)
		else
			Λ = kron(Λ00, Λ)
		end
	end
	return Λ
end

function refbpxdn(L::Int, d::Int)
	C = zeros(2^(d*L), 2^(d*L))
	for ℓ ∈ 0:L
		P = refextdn(L, ℓ, d)
		C = C + 2.0^(-ℓ) * P*P'
	end
	return C
end


@testset "extdn" begin
	@testset "d = $d, ℓ = $ℓ, L = $L" for (d,L,ℓ) ∈ ((1,7,0),(1,7,4),(1,7,7),(2,5,0),(2,5,3),(2,5,5),(3,4,0),(3,4,3),(3,4,4))
		P = FEM.extdn(L, ℓ, d; major="first")
		P = [ [ reshape(P[k], size(P[k], 1), 2*ones(Int, 2*d)..., size(P[k], 4)) for k ∈ 1:ℓ ]...,
			  [ reshape(P[k], size(P[k], 1), 2*ones(Int, d)..., ones(Int, d)..., size(P[k], 4)) for k ∈ ℓ+1:L ]... ]
		Pf = block(P, 1, 1; major="first")
		Pf = reshape(Pf, 2^(d*L), 2^(d*ℓ))
		P0f = refextdn(L, ℓ, d)
		@test Pf ≈ P0f rtol=1e-14
	end
end

@testset "diffdn" begin
	@testset "d = $d, ℓ = $ℓ" for (d,ℓ) ∈ ((1,7),(2,5),(3,4))
		@testset "K = $K" for K ∈ 0:d
			n = 2*ones(Int, d)
			if K ≠ 0
				n[K] = 1
			end
			M = FEM.diffdn(ℓ, d, K; major="first")
			M = [ [ reshape(M[k], size(M[k], 1), 2*ones(Int, 2*d)..., size(M[k], 4)) for k ∈ 1:ℓ ]...,
				  reshape(M[ℓ+1], size(M[ℓ+1], 1), n..., ones(Int, d)..., size(M[ℓ+1], 4)) ]
			Mf = block(M, 1, 1; major="first")
			Mf = reshape(Mf, 2^(d*ℓ)*prod(n), 2^(d*ℓ))
			M0f = refdiffdn(ℓ, d, K)
			@test Mf ≈ M0f rtol=1e-14
		end
	end
end

@testset "intdn" begin
	@testset "d = $d, ℓ = $ℓ" for (d,ℓ) ∈ ((1,7),(2,4),(3,3))
		@testset "K1 = $K1, K2 = $K2" for K1 ∈ 0:d, K2 ∈ 0:d
			n1 = 2*ones(Int, d)
			n2 = 2*ones(Int, d)
			if K1 ≠ 0
				n1[K1] = 1
			end
			if K2 ≠ 0
				n2[K2] = 1
			end
			Λ = FEM.intdn(ℓ, d, K1, K2; major="first")
			Λ = [ [ reshape(Λ[k], size(Λ[k], 1), 2*ones(Int, 2*d)..., size(Λ[k], 4)) for k ∈ 1:ℓ ]...,
				  reshape(Λ[ℓ+1], size(Λ[ℓ+1], 1), n1..., n2..., size(Λ[ℓ+1], 4)) ]
			Λf = block(Λ, 1, 1; major="first")
			Λf = reshape(Λf, 2^(d*ℓ)*prod(n1), 2^(d*ℓ)*prod(n2))
			Λ0f = refintdn(ℓ, d, K1, K2)
			@test Λf ≈ Λ0f rtol=1e-14
		end
	end
end

@testset "bpxdn" begin
	@testset "d = $d, ℓ = $ℓ" for (d,ℓ) ∈ ((1,7),(2,5),(3,4))
		C = FEM.bpxdn(ℓ, d; major="first")
		C = [ reshape(C[k], size(C[k], 1), 2*ones(Int, 2*d)..., size(C[k], 4)) for k ∈ 1:ℓ ]
		Cf = block(C, 1, 1; major="first")
		Cf = reshape(Cf, 2^(d*ℓ), 2^(d*ℓ))
		C0f = refbpxdn(ℓ, d)
		@test Cf ≈ C0f rtol=1e-14
	end
end

@testset "diffextdn" begin
	@testset "d = $d, ℓ = $ℓ, L = $L" for (d,ℓ,L) ∈ ((1,0,7),(1,1,7),(1,3,7),(1,7,7),(2,3,5),(3,2,4))
		@testset "K = $K" for K ∈ 0:d
			n = 2*ones(Int, d)
			if K ≠ 0
				n[K] = 1
			end
			MP = FEM.diffextdn(L, ℓ, d, K; major="first")
			MP = [ [ reshape(MP[k], size(MP[k], 1), 2*ones(Int, 2*d)..., size(MP[k], 4)) for k ∈ 1:ℓ ]...,
				   [ reshape(MP[k], size(MP[k], 1), 2*ones(Int, d)..., ones(Int, d)..., size(MP[k], 4)) for k ∈ ℓ+1:L ]...,
				  reshape(MP[L+1], size(MP[L+1], 1), n..., ones(Int, d)..., size(MP[L+1], 4)) ]
			MPf = block(MP, 1, 1; major="first")
			MPf = reshape(MPf, 2^(d*L)*prod(n), 2^(d*ℓ))
			MP0f = refdiffdn(L, d, K)*refextdn(L, ℓ, d)
			@test MPf ≈ MP0f rtol=1e-14
		end
	end
end

@testset "diffbpxdn" begin
	@testset "d = $d, ℓ = $ℓ" for (d,ℓ) ∈ ((1,1),(1,7),(2,1),(2,4))
		@testset "K = $K" for K ∈ 0:d
			n = 2*ones(Int, d)
			if K ≠ 0
				n[K] = 1
			end
			Q = FEM.diffbpxdn(ℓ, d, K; major="first")
			Q = [ [ reshape(Q[k], size(Q[k], 1), 2*ones(Int, 2*d)..., size(Q[k], 4)) for k ∈ 1:ℓ ]...,
				  reshape(Q[ℓ+1], size(Q[ℓ+1], 1), n..., ones(Int, d)..., size(Q[ℓ+1], 4)) ]
			Qf = block(Q, 1, 1; major="first")
			Qf = reshape(Qf, 2^(d*ℓ)*prod(n), 2^(d*ℓ))
			M0f = refdiffdn(ℓ, d, K)
			C0f = refbpxdn(ℓ, d)

			Q0f = M0f*C0f
			@test Qf ≈ Q0f rtol=1e-14
		end
	end
end

@testset "stiffness and mass matrices" begin
	@testset "d = 1, ℓ = $ℓ" for ℓ ∈ (1,8,9)
		d = 1
		M0 = FEM.diffdn(ℓ, d, 0; major="first")
		Λ0 = FEM.intdn(ℓ, d, 0, 0; major="first")
		M = decmp(M0, 1, decmp(Λ0, 2, M0, 1), 1)
		decskp!(M, ℓ+1; path="backward")
		Mf = block(M, 1, 1; major="first")

		M1 = FEM.diffdn(ℓ, d, 1; major="first")
		Λ1 = FEM.intdn(ℓ, d, 1, 1; major="first")
		S = decmp(M1, 1, decmp(Λ1, 2, M1, 1), 1)
		decskp!(S, ℓ+1; path="backward")
		Sf = block(S, 1, 1; major="first")

		n = 2^ℓ
		Sf0 = 2^(2*ℓ) * diagm(1 => -ones(Float64, n-1), -1 => -ones(Float64, n-1), 0 => [2*ones(Float64, n-1); 1])
		Mf0 = diagm(1 => 1/6*ones(Float64, n-1), -1 => 1/6*ones(Float64, n-1), 0 => 1/3*[2*ones(Float64, n-1); 1])
		@testset "mass matrix" begin
			@test Mf ≈ Mf0 rtol=1e-14
		end
		@testset "stiffness matrix" begin
			@test Sf ≈ Sf0 rtol=1e-14
		end
	end
	@testset "d = 2, ℓ = $ℓ" for ℓ ∈ (1,4,5)
		d = 2
		M0 = FEM.diffdn(ℓ, d, 0; major="first")
		Λ0 = FEM.intdn(ℓ, d, 0, 0; major="first")
		M = decmp(M0, 1, decmp(Λ0, 2, M0, 1), 1)
		decskp!(M, ℓ+1; path="backward")
		M = [ reshape(M[k], size(M[k], 1), 2*ones(Int, 2*d)..., size(M[k], 4)) for k ∈ 1:ℓ ]
		Mf = block(M, 1, 1; major="first")
		Mf = reshape(Mf, 2^(d*ℓ), 2^(d*ℓ))

		M1 = FEM.diffdn(ℓ, d, 1; major="first")
		Λ1 = FEM.intdn(ℓ, d, 1, 1; major="first")
		M2 = FEM.diffdn(ℓ, d, 2; major="first")
		Λ2 = FEM.intdn(ℓ, d, 2, 2; major="first")
		S1 = decmp(M1, 1, decmp(Λ1, 2, M1, 1), 1)
		decskp!(S1, ℓ+1; path="backward")
		S2 = decmp(M2, 1, decmp(Λ2, 2, M2, 1), 1)
		decskp!(S2, ℓ+1; path="backward")
		S = decadd(S1, S2)

		S1 = [ reshape(S1[k], size(S1[k], 1), 2*ones(Int, 2*d)..., size(S1[k], 4)) for k ∈ 1:ℓ ]
		S2 = [ reshape(S2[k], size(S2[k], 1), 2*ones(Int, 2*d)..., size(S2[k], 4)) for k ∈ 1:ℓ ]
		S  = [ reshape(S[k], size(S[k], 1), 2*ones(Int, 2*d)..., size(S[k], 4)) for k ∈ 1:ℓ ]
		
		S1f = block(S1, 1, 1; major="first")
		S1f = reshape(S1f, 2^(d*ℓ), 2^(d*ℓ))
		S2f = block(S2, 1, 1; major="first")
		S2f = reshape(S2f, 2^(d*ℓ), 2^(d*ℓ))

		Sf = block(S, 1, 1; major="first")
		Sf = reshape(Sf, 2^(d*ℓ), 2^(d*ℓ))

		n = 2^ℓ
		Sf0 = 2^(2*ℓ) * diagm(1 => -ones(Float64, n-1), -1 => -ones(Float64, n-1), 0 => [2*ones(Float64, n-1); 1])
		Mf0 = diagm(1 => 1/6*ones(Float64, n-1), -1 => 1/6*ones(Float64, n-1), 0 => 1/3*[2*ones(Float64, n-1); 1])
		S1f0 = kron(Mf0, Sf0)
		S2f0 = kron(Sf0, Mf0)
		Mf0 = kron(Mf0, Mf0)
		Sf0 = S1f0 + S2f0
		@testset "mass matrix" begin
			@test Mf ≈ Mf0 rtol=1e-14
		end
		@testset "stiffness matrix" begin
			@test S1f ≈ S1f0 rtol=1e-14
			@test S2f ≈ S2f0 rtol=1e-14
			@test Sf ≈ Sf0 rtol=1e-14
		end
	end
end

@testset "preconditioned stiffness and mass matrices" begin
	@testset "d = 1, ℓ = $ℓ" for ℓ ∈ (1,8,9)
		d = 1
		M0 = FEM.diffbpxdn(ℓ, d, 0; major="first")
		Λ0 = FEM.intdn(ℓ, d, 0, 0; major="first")
		M = decmp(M0, 1, decmp(Λ0, 2, M0, 1), 1)
		decskp!(M, ℓ+1; path="backward")
		Mf = block(M, 1, 1; major="first")

		M1 = FEM.diffbpxdn(ℓ, d, 1; major="first")
		Λ1 = FEM.intdn(ℓ, d, 1, 1; major="first")
		S = decmp(M1, 1, decmp(Λ1, 2, M1, 1), 1)
		decskp!(S, ℓ+1; path="backward")
		Sf = block(S, 1, 1; major="first")

		n = 2^ℓ
		C0f = refbpxdn(ℓ, d)
		Sf0 = 2^(2*ℓ) * diagm(1 => -ones(Float64, n-1), -1 => -ones(Float64, n-1), 0 => [2*ones(Float64, n-1); 1])
		Mf0 = diagm(1 => 1/6*ones(Float64, n-1), -1 => 1/6*ones(Float64, n-1), 0 => 1/3*[2*ones(Float64, n-1); 1])
		Mf0 = C0f*Mf0*C0f
		Sf0 = C0f*Sf0*C0f
		@testset "mass matrix" begin
			@test Mf ≈ Mf0 rtol=1e-14
		end
		@testset "stiffness matrix" begin
			@test Sf ≈ Sf0 rtol=1e-14
		end
	end
	@testset "d = 2, ℓ = $ℓ" for ℓ ∈ (1,4,5)
		d = 2
		M0 = FEM.diffbpxdn(ℓ, d, 0; major="first")
		Λ0 = FEM.intdn(ℓ, d, 0, 0; major="first")
		M = decmp(M0, 1, decmp(Λ0, 2, M0, 1), 1)
		decskp!(M, ℓ+1; path="backward")
		M = [ reshape(M[k], size(M[k], 1), 2*ones(Int, 2*d)..., size(M[k], 4)) for k ∈ 1:ℓ ]
		Mf = block(M, 1, 1; major="first")
		Mf = reshape(Mf, 2^(d*ℓ), 2^(d*ℓ))

		M1 = FEM.diffbpxdn(ℓ, d, 1; major="first")
		Λ1 = FEM.intdn(ℓ, d, 1, 1; major="first")
		M2 = FEM.diffbpxdn(ℓ, d, 2; major="first")
		Λ2 = FEM.intdn(ℓ, d, 2, 2; major="first")
		S1 = decmp(M1, 1, decmp(Λ1, 2, M1, 1), 1)
		decskp!(S1, ℓ+1; path="backward")
		S2 = decmp(M2, 1, decmp(Λ2, 2, M2, 1), 1)
		decskp!(S2, ℓ+1; path="backward")
		S = decadd(S1, S2)

		S1 = [ reshape(S1[k], size(S1[k], 1), 2*ones(Int, 2*d)..., size(S1[k], 4)) for k ∈ 1:ℓ ]
		S2 = [ reshape(S2[k], size(S2[k], 1), 2*ones(Int, 2*d)..., size(S2[k], 4)) for k ∈ 1:ℓ ]
		S  = [ reshape(S[k], size(S[k], 1), 2*ones(Int, 2*d)..., size(S[k], 4)) for k ∈ 1:ℓ ]
		
		S1f = block(S1, 1, 1; major="first")
		S1f = reshape(S1f, 2^(d*ℓ), 2^(d*ℓ))
		S2f = block(S2, 1, 1; major="first")
		S2f = reshape(S2f, 2^(d*ℓ), 2^(d*ℓ))

		Sf = block(S, 1, 1; major="first")
		Sf = reshape(Sf, 2^(d*ℓ), 2^(d*ℓ))

		n = 2^ℓ
		C0f = refbpxdn(ℓ, d)
		Sf0 = 2^(2*ℓ) * diagm(1 => -ones(Float64, n-1), -1 => -ones(Float64, n-1), 0 => [2*ones(Float64, n-1); 1])
		Mf0 = diagm(1 => 1/6*ones(Float64, n-1), -1 => 1/6*ones(Float64, n-1), 0 => 1/3*[2*ones(Float64, n-1); 1])
		S1f0 = C0f*kron(Mf0, Sf0)*C0f
		S2f0 = C0f*kron(Sf0, Mf0)*C0f
		Mf0 = C0f*kron(Mf0, Mf0)*C0f
		Sf0 = S1f0 + S2f0
		@testset "mass matrix" begin
			@test Mf ≈ Mf0 rtol=1e-14
		end
		@testset "stiffness matrix" begin
			@test S1f ≈ S1f0 rtol=1e-14
			@test S2f ≈ S2f0 rtol=1e-14
			@test Sf ≈ Sf0 rtol=1e-14
		end
	end
end