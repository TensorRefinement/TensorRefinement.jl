using LinearAlgebra, TensorRefinement, Test

refmassdn(L::Int) = diagm(0 => [ones(2^L-1).*4; 2], 1 => ones(2^L-1), -1 => ones(2^L-1))./6
refmassdd(L::Int) = diagm(0 => [ones(2^L-1).*4; 0], 1 => ones(2^L-2), -1 => ones(2^L-2))./6
refstiffdn(L::Int) = diagm(0 => [ones(2^L-1).*2; 1], 1 => -ones(2^L-1), -1 => -ones(2^L-1)).*4^L
refstiffdd(L::Int) = diagm(0 => [ones(2^L-1).*2; 0], 1 => -ones(2^L-2), -1 => -ones(2^L-2)).*4^L

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

function refextdd(L::Int, ℓ::Int, d::Int)
	I = diagm(0 => [ones(2^ℓ-1); 0])
	S = diagm(-1 => ones(2^ℓ-1))
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
	M1 = 2.0^(ℓ/2-1+(ℓ+1)) * kron(I-S, [1; 0])
	M = (K == 1) ? M1 : M0
	for κ ∈ 2:d
		M = kron((κ == K) ? M1 : M0, M)
	end
	return M
end

function refdiffdd(ℓ::Int, d::Int, K::Int)
	I = diagm(0 => ones(2^ℓ))
	I[2^ℓ,2^ℓ] = 0
	S = diagm(-1 => ones(2^ℓ-1))
	M0 = 2.0^(ℓ/2-1) * ( kron(I+S, [1; 0]) + kron(I-S, [0; 1]) )
	M1 = 2.0^(ℓ/2-1+(ℓ+1)) * kron(I-S, [1; 0])
	M = (K == 1) ? M1 : M0
	for κ ∈ 2:d
		M = kron((κ == K) ? M1 : M0, M)
	end
	return M
end

function refdint(ℓ::Int, d::Int, K1::Int, K2::Int)
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

function refbpxdd(L::Int, d::Int)
	C = zeros(2^(d*L), 2^(d*L))
	for ℓ ∈ 1:L
		P = refextdd(L, ℓ, d)
		C = C + 2.0^(-ℓ) * P*P'
	end
	return C
end

###
T = Float64
###

@testset "extdn" begin
	@testset "d = $d, ℓ = $ℓ, L = $L" for (d,L,ℓ) ∈ ((1,7,0),(1,7,4),(1,7,7),(2,5,0),(2,5,3),(2,5,5),(3,4,0),(3,4,3),(3,4,4))
		P = FEM.extdn(T, L, ℓ, d; major="first")
		P = [ [ reshape(P[k], size(P[k], 1), 2*ones(Int, 2*d)..., size(P[k], 4)) for k ∈ 1:ℓ ]...,
			  [ reshape(P[k], size(P[k], 1), 2*ones(Int, d)..., ones(Int, d)..., size(P[k], 4)) for k ∈ ℓ+1:L ]... ]
		Pf = block(P, 1, 1; major="first")
		Pf = reshape(Pf, 2^(d*L), 2^(d*ℓ))
		P0f = refextdn(L, ℓ, d)
		@test Pf ≈ P0f rtol=1e-14
	end
end

@testset "extdd" begin
	@testset "d = $d, ℓ = $ℓ, L = $L" for (d,L,ℓ) ∈ ((1,7,1),(1,7,4),(1,7,7),(2,5,1),(2,5,3),(2,5,5),(3,4,1),(3,4,3),(3,4,4))
		P = FEM.extdd(T, L, ℓ, d; major="first")
		P = [ [ reshape(P[k], size(P[k], 1), 2*ones(Int, 2*d)..., size(P[k], 4)) for k ∈ 1:ℓ ]...,
			  [ reshape(P[k], size(P[k], 1), 2*ones(Int, d)..., ones(Int, d)..., size(P[k], 4)) for k ∈ ℓ+1:L ]... ]
		Pf = block(P, 1, 1; major="first")
		Pf = reshape(Pf, 2^(d*L), 2^(d*ℓ))
		P0f = refextdd(L, ℓ, d)
		@test Pf ≈ P0f rtol=1e-14
	end
end

@testset "diffdn ℓ = L" begin
	@testset "d = $d, ℓ = $ℓ" for (d,ℓ) ∈ ((1,0),(1,1),(1,7),(2,0),(2,3),(2,5),(3,2))
		M = diffdn(T, ℓ, d; major="first")
		M = decmodereshape(M, [2*ones(Int, d, ℓ) 2*ones(Int, d, 1); 2*ones(Int, d, ℓ) ones(Int, d, 1); ones(Int, 1, ℓ) d+1])
		Mf = block(M, 1, 1; major="first")
		Mf = reshape(Mf, 2^(d*(ℓ+1)), 2^(d*ℓ), d+1)
		@testset "k = $k" for k ∈ 0:d
			M0f = refdiffdn(ℓ, d, k)
			@test Mf[:,:,k+1] ≈ M0f rtol=1e-14
		end
	end
end

@testset "diffdn ℓ ≤ L" begin
	@testset "d = $d, ℓ = $ℓ, L = $L" for (d,ℓ,L) ∈ ((1,0,0),(1,0,7),(1,0,1),(1,1,7),(1,3,7),(1,7,7),(2,0,0),(2,3,5),(3,1,3),(3,2,3))
		M = diffdn(T, L, ℓ, d; major="first")
		M = decmodereshape(M, [2*ones(Int, d, ℓ) 2*ones(Int, d, L-ℓ) 2*ones(Int, d, 1); 2*ones(Int, d, ℓ) ones(Int, d, L-ℓ) ones(Int, d, 1); ones(Int, 1, ℓ) ones(Int, 1, L-ℓ) d+1])
		Mf = block(M, 1, 1; major="first")
		Mf = reshape(Mf, 2^(d*(L+1)), 2^(d*ℓ), d+1)
		@testset "k = $k" for k ∈ 0:d
			M0f = refdiffdn(L, d, k)*refextdn(L, ℓ, d)
			@test Mf[:,:,k+1] ≈ M0f rtol=1e-14
		end
	end
end

@testset "diffdd" begin
	@testset "d = $d, ℓ = $ℓ" for (d,ℓ) ∈ ((1,1),(1,7),(2,1),(2,5),(2,3),(3,2))
		M = diffdd(T, ℓ, d; major="first")
		M = decmodereshape(M, [2*ones(Int, d, ℓ) 2*ones(Int, d, 1); 2*ones(Int, d, ℓ) ones(Int, d, 1); ones(Int, 1, ℓ) d+1])
		Mf = block(M, 1, 1; major="first")
		Mf = reshape(Mf, 2^(d*(ℓ+1)), 2^(d*ℓ), d+1)
		@testset "k = $k" for k ∈ 0:d
			M0f = refdiffdd(ℓ, d, k)
			@test Mf[:,:,k+1] ≈ M0f rtol=1e-14
		end
	end
end

@testset "diffdd ℓ ≤ L" begin
	@testset "d = $d, ℓ = $ℓ, L = $L" for (d,ℓ,L) ∈ ((1,1,1),(1,1,7),(1,1,2),(1,3,7),(1,7,7),(2,1,1),(2,3,5),(3,1,3),(3,2,3))
		M = diffdd(T, L, ℓ, d; major="first")
		M = decmodereshape(M, [2*ones(Int, d, ℓ) 2*ones(Int, d, L-ℓ) 2*ones(Int, d, 1); 2*ones(Int, d, ℓ) ones(Int, d, L-ℓ) ones(Int, d, 1); ones(Int, 1, ℓ) ones(Int, 1, L-ℓ) d+1])
		Mf = block(M, 1, 1; major="first")
		Mf = reshape(Mf, 2^(d*(L+1)), 2^(d*ℓ), d+1)
		@testset "k = $k" for k ∈ 0:d
			M0f = refdiffdd(L, d, k)*refextdd(L, ℓ, d)
			@test Mf[:,:,k+1] ≈ M0f rtol=1e-14
		end
	end
end

@testset "dint (d = 1) on mass and stiffness matrices" begin
	d = 1
	@testset "dn" begin
		@testset "L = $L" for L ∈ (1,2,4,7)
			M = diffdn(T, L, d; major="first")
			Mf = block(M, 1, 1; major="first")
			@testset "stiffness matrix" begin
				K = Diagonal([0; ones(T, d)])
				Λ = dint(T, L, d, K; major="first")
				A = decmp(M, 1, decmp(Λ, 2, M, 1), 1)
				Af = block(A, 1, 1; major="first")
				Aref = refstiffdn(L)
				@test maximum(abs.(Af-Aref)[:])./4^L < 1e-14
			end
			@testset "mass matrix" begin
				K = Diagonal([1; zeros(T, d)])
				Λ = dint(T, L, d, K; major="first")
				A = decmp(M, 1, decmp(Λ, 2, M, 1), 1)
				Af = block(A, 1, 1; major="first")
				Aref = refmassdn(L)
				@test maximum(abs.(Af-Aref)[:]).*6 < 1e-14
			end
		end
	end
	@testset "dd" begin
		@testset "L = $L" for L ∈ (1,2,4,7)
			M = diffdd(T, L, d; major="first")
			Mf = block(M, 1, 1; major="first")
			@testset "stiffness matrix" begin
				K = Diagonal([0; ones(T, d)])
				Λ = dint(T, L, d, K; major="first")
				A = decmp(M, 1, decmp(Λ, 2, M, 1), 1)
				Af = block(A, 1, 1; major="first")
				Aref = refstiffdd(L)
				@test maximum(abs.(Af-Aref)[:])./4^L < 1e-14
			end
			@testset "mass matrix" begin
				K = Diagonal([1; zeros(T, d)])
				Λ = dint(T, L, d, K; major="first")
				A = decmp(M, 1, decmp(Λ, 2, M, 1), 1)
				Af = block(A, 1, 1; major="first")
				Aref = refmassdd(L)
				@test maximum(abs.(Af-Aref)[:]).*6 < 1e-14
			end
		end
	end
end

@testset "dint (d > 1) on the assembly of operator discretizations" begin
	@testset "dn" begin
		@testset "d = $d, L = $L" for (d,L) ∈ ((2,3),(2,4),(3,2))
			c = collect(0:d)./d .+ 1
			K = Diagonal(c)
			Λ = dint(T, L, d, K; major="first")
			M = diffdn(T, L, d; major="first")
			A = decmp(M, 1, decmp(Λ, 2, M, 1), 1)
			A = decmodereshape(A, [2*ones(Int, 2*d, L) ones(Int, 2*d, 1)])
			Af = block(A, 1, 1; major="first")
			Af = reshape(Af, 2^(d*L), 2^(d*L))
			Mref,Sref = refmassdn(L),refstiffdn(L)
			Aref = kron(ntuple(k -> Mref, Val(d))...).*c[1]
			for k ∈ 1:d
				Aref += kron(ntuple(k -> Mref, Val(d-k))..., Sref, ntuple(k -> Mref, Val(k-1))...).*c[k+1]
			end
			@test maximum(abs.(Aref-Af)[:]) / 4^L * 6^(d-1) < 1e-12
		end
	end
	@testset "dd" begin
		@testset "d = $d, L = $L" for (d,L) ∈ ((2,3),(2,4),(3,2))
			c = collect(0:d)./d .+ 1
			K = Diagonal(c)
			Λ = dint(T, L, d, K; major="first")
			M = diffdd(T, L, d; major="first")
			A = decmp(M, 1, decmp(Λ, 2, M, 1), 1)
			A = decmodereshape(A, [2*ones(Int, 2*d, L) ones(Int, 2*d, 1)])
			Af = block(A, 1, 1; major="first")
			Af = reshape(Af, 2^(d*L), 2^(d*L))
			Mref,Sref = refmassdd(L),refstiffdd(L)
			Aref = kron(ntuple(k -> Mref, Val(d))...).*c[1]
			for k ∈ 1:d
				Aref += kron(ntuple(k -> Mref, Val(d-k))..., Sref, ntuple(k -> Mref, Val(k-1))...).*c[k+1]
			end
			@test maximum(abs.(Aref-Af)[:]) / 4^L * 6^(d-1) < 1e-12
		end
	end
end


@testset "dintf" begin
	@testset "d = $d, L = $L" for (d,L) ∈ ((1,2),(1,7),(2,3),(2,4),(3,2))
		c = collect(0:d)./d .+ 1
		K = Diagonal(c)
		Λ = dint(T, L, d, K'*K; major="first")
		Λf = block(Λ, 1, 1)
		ΛΛ = dintf(T, L, d, K; major="first")
		ΛΛf = block(ΛΛ, 1, 1)
		@test norm(ΛΛf'*ΛΛf-Λf)*2^(d*L) < 1e-12
	end
end

@testset "bpxdn" begin
	@testset "d = $d, L = $L" for (d,L) ∈ ((1,7),(2,3),(3,2))
		C = FEM.bpxdn(T, L, d; major="first")
		C = decmodereshape(C, 2*ones(Int, 2*d, L))
		Cf = block(C, 1, 1; major="first")
		Cf = reshape(Cf, 2^(d*L), 2^(d*L))
		C0f = refbpxdn(L, d)
		@test Cf ≈ C0f rtol=1e-14
	end
end

@testset "bpxdd" begin
	@testset "d = $d, L = $L" for (d,L) ∈ ((1,7),(2,3),(3,2))
		C = FEM.bpxdd(T, L, d; major="first")
		C = decmodereshape(C, 2*ones(Int, 2*d, L))
		Cf = block(C, 1, 1; major="first")
		Cf = reshape(Cf, 2^(d*L), 2^(d*L))
		C0f = refbpxdd(L, d)
		@test Cf ≈ C0f rtol=1e-14
	end
end


@testset "extmix" begin
	@testset "d = $d, ℓ = $ℓ, L = $L" for (d,ℓ,L) ∈ ((1,1,1),(1,1,7),(1,1,2),(1,3,7),(1,7,7),(2,1,1),(2,3,5),(3,1,3),(3,2,3))
		E = extmix(T, L, ℓ, d; major="first")
		M = diffdn(T, ℓ, ℓ, d; major="first")
		decappend!(M, dec(ones(T, 1, 1, 1, 1); len=L-ℓ))
		ME = decmp(E, 2, M, 1)
		MEf = block(ME, 1, 1; major="first")
		ME0 = diffdn(T, L, ℓ, d; major="first")
		ME0f = block(ME0, 1, 1; major="first")
		@test MEf ≈ ME0f rtol=1e-14
	end
end


@testset "diffbpxdn" begin
	@testset "d = $d, L = $L" for (d,L) ∈ ((1,1),(1,7),(2,1),(2,3),(3,2))
		Q = diffbpxdn(T, L, d; major="first")
		M = diffdn(T, L, d; major="first")
		C = bpxdn(T, L, d; major="first")
		Qf = block(Q, 1, 1; major="first")
		Mf = block(M, 1, 1; major="first")
		Cf = block(C, 1, 1; major="first")
		@test Qf ≈ Mf*Cf rtol=1e-14
	end
end

@testset "diffbpxdd" begin
	@testset "d = $d, L = $L" for (d,L) ∈ ((1,1),(1,7),(2,1),(2,3),(3,2))
		Q = diffbpxdd(T, L, d; major="first")
		M = diffdd(T, L, d; major="first")
		C = bpxdd(T, L, d; major="first")
		Qf = block(Q, 1, 1; major="first")
		Mf = block(M, 1, 1; major="first")
		Cf = block(C, 1, 1; major="first")
		@test Qf ≈ Mf*Cf rtol=1e-14
	end
end


@testset "preconditioned operator" begin
	d = 2
	L = 2
	@testset "dn d = $d, L = $L" begin
		c = [1.0, 2.0, 3.0]
		K = Diagonal(c)
		Λ = dint(T, L, d, K; major="first")
		M = diffbpxdn(T, L, d; major="first")
		A = decmp(M, 1, decmp(Λ, 2, M, 1), 1)
		A = decmodereshape(A, [2*ones(Int, 2*d, L) ones(Int, 2*d, 1)])
		Af = block(A, 1, 1; major="first")
		Af = reshape(Af, 2^(d*L), 2^(d*L))
		Mref,Sref = refmassdn(L),refstiffdn(L)
		Cref = refbpxdn(L, d)
		Aref = kron(Mref, Mref).*c[1] + kron(Mref, Sref).*c[2] + kron(Sref, Mref).*c[3]
		Aref = Cref'*Aref*Cref
		@test Af ≈ Aref rtol=1e-14
	end
	@testset "dd d = $d, L = $L" begin
		c = [1.0, 2.0, 3.0]
		K = Diagonal(c)
		Λ = dint(T, L, d, K; major="first")
		M = diffbpxdd(T, L, d; major="first")
		A = decmp(M, 1, decmp(Λ, 2, M, 1), 1)
		A = decmodereshape(A, [2*ones(Int, 2*d, L) ones(Int, 2*d, 1)])
		Af = block(A, 1, 1; major="first")
		Af = reshape(Af, 2^(d*L), 2^(d*L))
		Mref,Sref = refmassdd(L),refstiffdd(L)
		Cref = refbpxdd(L, d)
		Aref = kron(Mref, Mref).*c[1] + kron(Mref, Sref).*c[2] + kron(Sref, Mref).*c[3]
		Aref = Cref'*Aref*Cref
		@test Af ≈ Aref rtol=1e-14
	end
end
