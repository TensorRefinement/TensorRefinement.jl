
module SwitchingEdgeTensorTrain

using TensorRefinement.Auxiliary, ..TensorTrain, LinearAlgebra, Printf


export VectorFrameDec
export framedeclength, framedecsize, framedecrank, framedecrearrank
export framedecaxpby
export framedecqr, framedecsvd, framedecproj, matveclaplacelikeproject, framedecevalorth, framedecproject, framedeceval

export VectorFrameDec1
export framedecqr1, framedecsvd1



const VectorFrameDec{T} = Tuple{VectorDec{T},Vector{VectorFactor{T}}} where T<:Number
VectorFrameDec(Z::VectorDec{T}, R::Vector{VectorFactor{T}}) where T<:Number = (Z,R)

framedeclength(F::VectorFrameDec{T}) where T<:Number = declength(F[1])+1
framedecsize(F::VectorFrameDec{T}) where T<:Number = [factorsize(F[2][ℓ])[1] for k ∈ 1:1, ℓ ∈ 1:length(F[2])]
framedecrank(F::VectorFrameDec{T}) where T<:Number = decrank(F[1])
framedecrearrank(F::VectorFrameDec{T}) where T<:Number = [ factorranks(F[2][ℓ])[2] for ℓ ∈ 1:length(F[2]) ]

function framedecconsistency(F::VectorFrameDec{T}; requiresizeconsistency::Bool=true) where T<:Number
	U,V = F
	@assert declength(U) + 1 == length(V)
	r = framedecrank(F)
	for ℓ ∈ 1:framedeclength(F)
		@assert factorranks(V[ℓ])[1] == r[ℓ]
	end
	if requiresizeconsistency
		for ℓ ∈ 1:framedeclength(F)-1
			@assert factorsize(F[1][ℓ]) == factorsize(F[2][ℓ])
		end
	end
	true
end



### this is for a nonstandard frame dec (L+1 factors with modes, L+1 factors without modes)
const VectorFrameDec1{T} = Tuple{VectorDec{T},Vector{Matrix{T}}} where T<:Number
VectorFrameDec1(Z::VectorDec{T}, R::Vector{Matrix{T}}) where T<:Number = (Z,R)

framedeclength(F::VectorFrameDec1{T}) where T<:Number = declength(F[1])
framedecsize(F::VectorFrameDec1{T}) where T<:Number = decsize(F[1])
framedecrank(F::VectorFrameDec1{T}) where T<:Number = decrank(F[1])
framedecrearrank(F::VectorFrameDec1{T}) where T<:Number = [ factorranks(F[2][ℓ])[2] for k ∈ 1:1, ℓ ∈ 1:length(F[2]) ]

function framedecconsistency(F::VectorFrameDec1{T}) where T<:Number
	U,V = F
	@assert declength(U) == length(V)
	r = framedecrank(F)
	for ℓ ∈ 1:framedeclength(F)
		@assert factorranks(V[ℓ])[1] == r[ℓ+1]
	end
	true
end



function framedecaxpby(α::T, U::VectorFrameDec{T}, β::T, V::VectorFrameDec{T}) where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	framedecconsistency(U)
	framedecconsistency(V)
	@assert framedecsize(U) == framedecsize(V)
	@assert framedecrearrank(U) == framedecrearrank(V)

	L = framedeclength(U)

	W = dechcat(U[1], V[1])
	Z = [ α*U[2][1]+β*V[2][1], [ factorvcat(α*U[2][ℓ+1], β*V[2][ℓ+1]) for ℓ ∈ 1:L-1 ]... ]

	VectorFrameDec(W, Z)
end


function framedecqr(F::VectorFrameDec{T}, Ω::TΩ; path::String="forward", requiresizeconsistency::Bool=true) where {Treal<:AbstractFloat,T<:FloatRC{Treal},TΩ<:Union{Nothing,Vector{Matrix{T}}}}

	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end

	framedecconsistency(F; requiresizeconsistency=requiresizeconsistency)
	Z,S = F
	L = framedeclength(F)-1
	@assert isa(Ω, Nothing) || (length(Ω) == L)

	Q = Vector{Factor{T,3}}(undef, L)
	R = Vector{Factor{T,3}}(undef, L+1)

	if path == "forward"
		R[1] = copy(S[1])
		Q[1] = copy(Z[1])
		for ℓ ∈ 1:L
			Q[ℓ],RR = factorqr!(Q[ℓ], Val(true))
			RR = factormodereshape(RR, Int[])
			isa(Ω, Nothing) || (Ω[ℓ] = RR)
			R[ℓ+1] = factorcontract(RR, S[ℓ+1])
			(ℓ < L) && (Q[ℓ+1] = factorcontract(RR, Z[ℓ+1]))
		end
	elseif path == "backward"
		for ℓ ∈ L+1:-1:2
			if ℓ == L+1
				Φ = copy(S[ℓ])
			else
				Φ = S[ℓ]
			end
			pΦ,qΦ = factorranks(Φ)
			nΦ = factorsize(Φ)[1]
			ΦΦ = reshape(Φ, pΦ, nΦ*qΦ)
			G = ΦΦ
			if ℓ ≤ L
				Ψ = Q[ℓ]
				pΨ,qΨ = factorranks(Ψ)
				nΨ = factorsize(Ψ)[1]
				ΨΨ = reshape(Ψ, pΨ, nΨ*qΨ)
				G = factorhcat(G, ΨΨ)
			end
			G,ΩΩ = factorqr!(G, Val(true); rev=true)
			ΩΩ = factormodereshape(ΩΩ, Int[])
			isa(Ω, Nothing) || (Ω[ℓ-1] = ΩΩ)
			r = factorranks(G)[1]
			# RR = factorrankselect(G, :, 1:rΦ)
			RR = G[:,1:nΦ*qΦ] ### TODO replace with the above once the function has been fixed
			R[ℓ] = reshape(RR, r, nΦ, qΦ)
			Q[ℓ-1] = factorcontract(Z[ℓ-1], ΩΩ)
			if ℓ ≤ L
				# QQ = factorrankselect(G, :, nΦ*qΦ+1:qΦ*qΦ+nΨ*qΨ)
				QQ = G[:,nΦ*qΦ+1:end] ### TODO replace with the above once the function has been fixed
				Q[ℓ] = reshape(QQ, r, nΨ, qΨ)
			end
			# F = factorhcat(S[ℓ], factorcontract(Z[ℓ], RR))
			# F,RR = factorqr!(F, Val(true); rev=true)
			# R[ℓ] = F[:,:,1:q[ℓ]]
			# Q[ℓ] = F[:,:,q[ℓ]+1:end]
		end
		R[1] = copy(S[1])
	end
	VectorFrameDec(Q, R)
end

framedecqr(F::VectorFrameDec{T}; path::String="forward", requiresizeconsistency::Bool=true) where {Treal<:AbstractFloat,T<:FloatRC{Treal}} = framedecqr(F, nothing; path=path, requiresizeconsistency=requiresizeconsistency)

### this is for a nonstandard frame dec (L+1 factors with modes, L+1 factors without modes)
### (!) no extensive accuracy or orthogonalty tests have been performed
function framedecqr1(F::VectorFrameDec1{T}; path::String="forward") where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end

	framedecconsistency(F)
	q = framedecrearrank(F)
	Z,S = F

	Q = Vector{Factor{T,3}}(undef, L+1)
	R = Vector{Matrix{T}}(undef, L+1)

	if path == "forward"
		Q[1] = copy(Z[1])
		for ℓ ∈ 1:L+1
			Q[ℓ],RR = factorqr!(Q[ℓ], Val(true))
			RR = factormodereshape(RR, Int[])
			R[ℓ] = factorcontract(RR, S[ℓ])
			if ℓ ≤ L
				Q[ℓ+1] = factorcontract(RR, Z[ℓ+1])
			end
		end
	elseif path == "backward"
		# G = copy(S[L+1])
		# G,RR = factorqr!(G, Val(true); rev=true)
		# R[L+1] = G
		# for ℓ ∈ L:-1:2
		# 	SS = S[ℓ]
		# 	pS,qS = factorranks(SS)
		# 	SM = reshape(SS, pS, :)
		# 	qSM = factorranks(SM)[2]
		# 	#
		# 	ZZ = factorcontract(Z[ℓ], RR)
		# 	pZ,qZ = factorranks(ZZ)
		# 	ZM = reshape(ZZ, pZ, :)
		# 	qZM = factorranks(ZM)[2]
		# 	#
		# 	G = factorhcat(SM, ZM)
		# 	G,RR = factorqr!(G, Val(true); rev=true)
		# 	RM = factorrankselect(G, :, 1:qSM)
		# 	QM = factorrankselect(G, :, qSM+1:qSM+qZM)
		# 	r = factorranks(G)[1]
		# 	R[ℓ] = reshape(RM, r, :, qS)
		# 	Q[ℓ] = reshape(QM, r, :, qZ)
		# 	# F = factorhcat(S[ℓ], factorcontract(Z[ℓ], RR))
		# 	# F,RR = factorqr!(F, Val(true); rev=true)
		# 	# R[ℓ] = F[:,:,1:q[ℓ]]
		# 	# Q[ℓ] = F[:,:,q[ℓ]+1:end]
		# end
		# R[1] = copy(S[1])
		# Q[1] = factorcontract(Z[1], RR)

		for ℓ ∈ L+1:-1:1
			F = S[ℓ]
			if ℓ ≤ L
				F = factorhcat(F, RR)
			end
			r = factorranks(F)[2]
			if ℓ > 1
				F = factorcontract(Z[ℓ], F)
			end
			F,RR = factorqr!(F, Val(true); rev=true)
			RR = factormodereshape(RR, Int[])
			if ℓ > 1
				Q[ℓ] = F
				F = Matrix{T}(I, r, r)
			else
				Q[ℓ] = factorcontract(Z[ℓ], RR)
			end
			R[ℓ] = F[:,1:q[ℓ]]
			if ℓ ≤ L
				Q[ℓ+1] = factorcontract(F[:,q[ℓ]+1:end], Q[ℓ+1])
			end
		end
	end
	VectorFrameDec1(Q, R)
end


function framedecsvd(F::VectorFrameDec{T},
	              τsoft::Vector{Treal}, τhard::Vector{Treal}, τ::Vector{Treal}, ρ::Vector{Int};
				  path::String="backward",
                  toldistr::Bool=false, verbose::Bool=false, requiresizeconsistency::Bool=true) where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end

	framedecconsistency(F; requiresizeconsistency=requiresizeconsistency)
	Z,S = F
	L = framedeclength(F)-1

	@assert length(τsoft) == L
	@assert length(τhard) == L
	@assert length(τ) == L
	@assert length(ρ) == L
	ε = copy(τ)

	Q = Vector{VectorFactor{T}}(undef, L)
	R = Vector{VectorFactor{T}}(undef, L+1)
	Ω = Vector{Array{T,2}}(undef, L+1)
	σ = Vector{Vector{Float64}}(undef, L)

	if path == "forward"
		R[1] = copy(S[1])
		Q[1] = copy(Z[1])
		for ℓ ∈ 1:L
			Q[ℓ],ΩΩ,η,_,_,ρ[ℓ],σ[ℓ] = factorsvd!(Q[ℓ], :, [1]; hard=τ[ℓ], atol=ε[ℓ], rank=ρ[ℓ])
			if toldistr && ℓ < L && η < ε[ℓ]
				ε[ℓ+1] = sqrt(ε[ℓ+1]^2+ε[ℓ]^2-η^2)
			end
			ε[ℓ] = η
			R[ℓ+1] = factorcontract(ΩΩ, S[ℓ+1])
			(ℓ < L) && (Q[ℓ+1] = factorcontract(ΩΩ, Z[ℓ+1]))
			Ω[ℓ] = factormodereshape(ΩΩ, [])
		end
	elseif path == "backward"
		for ℓ ∈ L+1:-1:2
			if ℓ == L+1
				Φ = copy(S[ℓ])
			else
				Φ = S[ℓ]
			end
			pΦ,qΦ = factorranks(Φ)
			nΦ = factorsize(Φ)[1]
			ΦΦ = reshape(Φ, pΦ, nΦ*qΦ)
			G = ΦΦ
			if ℓ ≤ L
				Ψ = Q[ℓ]
				pΨ,qΨ = factorranks(Ψ)
				nΨ = factorsize(Ψ)[1]
				ΨΨ = reshape(Ψ, pΨ, nΨ*qΨ)
				G = factorhcat(G, ΨΨ)
			end
			G,ΩΩ,η,_,_,ρ[ℓ-1],σ[ℓ-1] = factorsvd!(G, :, Int[]; soft=τsoft[ℓ-1], hard=τhard[ℓ-1], atol=ε[ℓ-1], rank=ρ[ℓ-1], rev=true)
			if toldistr && ℓ > 2 && η < ε[ℓ-1]
				ε[ℓ-2] = sqrt(ε[ℓ-2]^2+ε[ℓ-1]^2-η^2)
			end
			ε[ℓ-1] = η
			Ω[ℓ-1] = ΩΩ
			r = factorranks(G)[1]
			# RR = factorrankselect(G, :, 1:rΦ)
			RR = G[:,1:nΦ*qΦ] ### TODO replace with the above
			R[ℓ] = reshape(RR, r, nΦ, qΦ)
			Q[ℓ-1] = factorcontract(Z[ℓ-1], ΩΩ)
			if ℓ ≤ L
				# QQ = factorrankselect(G, :, nΦ*qΦ+1:qΦ*qΦ+nΨ*qΨ)
				QQ = G[:,nΦ*qΦ+1:end] ### TODO replace with the above
				Q[ℓ] = reshape(QQ, r, nΨ, qΨ)
			end
		end
		R[1] = copy(S[1])
	end
	VectorFrameDec(Q, R),Ω,ε,ρ,σ
end



function framedecsvd1(F::VectorFrameDec1{T},
	              τsoft::Vector{Treal}, τhard::Vector{Treal}, τ::Vector{Treal}, ρ::Vector{Int};
				  path::String="backward",
                  toldistr::Bool=false, verbose::Bool=false) where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	framedecconsistency(F)
	Q,R = F

	L = declength(Q)-1
	q = framedecrearrank(F)

	@assert length(τsoft) == L+1
	@assert length(τhard) == L+1
	@assert length(τ) == L+1
	@assert length(ρ) == L+1
	ε = copy(τ)

	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end

	Z = Vector{VectorFactor{T}}(undef, L+1)
	S = Vector{Matrix{T}}(undef, L+1)
	Ω = Vector{Matrix{T}}(undef, L+1) ### check the length
	σ = Vector{Vector{Float64}}(undef, L+1)

	if path == "forward"
		Z[1] = copy(Q[1])
		for ℓ ∈ 1:L+1
			Z[ℓ],ΩΩ,η,_,_,ρ[ℓ],σ[ℓ] = factorsvd!(Z[ℓ], :, [1]; hard=τ[ℓ], atol=ε[ℓ], rank=ρ[ℓ])
			ΩΩ = factormodereshape(ΩΩ, Int[])			
			S[ℓ] = factorcontract(ΩΩ, R[ℓ])
			if ℓ ≤ L
				Z[ℓ+1] = factorcontract(ΩΩ, Q[ℓ+1])
			end
			if toldistr && ℓ < L+1 && η < ε[ℓ]
				ε[ℓ+1] = sqrt(ε[ℓ+1]^2+ε[ℓ]^2-η^2)
			end
			ε[ℓ] = η
			Ω[ℓ] = ΩΩ
		end
	elseif path == "backward"
		local ΩΩ
		for ℓ ∈ L+1:-1:1
			F = R[ℓ]
			if ℓ ≤ L
				F = factorhcat(F, ΩΩ)
				F = factormodereshape(F, [1])
			end
			r = factorranks(F)[2]
			if ℓ > 1
				F = factorcontract(Q[ℓ], F)
			end
			F,ΩΩ,η,_,_,ρ[ℓ],σ[ℓ] = factorsvd!(F, :, [1]; soft=τsoft[ℓ], hard=τhard[ℓ], atol=ε[ℓ], rank=ρ[ℓ], rev=true)
			if toldistr && ℓ > 1 && η < ε[ℓ]
				ε[ℓ-1] = sqrt(ε[ℓ-1]^2+ε[ℓ]^2-η^2)
			end
			ε[ℓ] = η
			ΩΩ = factormodereshape(ΩΩ, Int[])
			Ω[ℓ] = ΩΩ
			if ℓ > 1
				Z[ℓ] = F
				F = Matrix{T}(I, r, r)
			else
				F = factormodereshape(F, Int[])
				Z[ℓ] = factorcontract(Q[ℓ], ΩΩ)
			end
			S[ℓ] = F[:,1:q[ℓ]]
			if ℓ ≤ L
				Z[ℓ+1] = factorcontract(F[:,q[ℓ]+1:end], Z[ℓ+1])
			end
		end
	end
	VectorFrameDec1(Z, S),Ω,ε,ρ,σ
end



function matveclaplacelikeproject(W::VectorDec{T},
                           V::MatrixDec{T}, Q::Vector{Factor{T,2}}, Y::VectorDec{T};
                           verbose::Bool=false, maxlvl::Int=declength(W)-2, maxnrm::Treal=zero(Treal)) where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	L = declength(W)-2
	@assert declength(V) == L+1
	@assert declength(Q) == L+1
	@assert declength(Y) == L+1
	
	rV = decrank(V)
	rY = decrank(Y); pushfirst!(rY, 1)
	@assert rV[1] == 1
	@assert rY[L+3] == 1
	for ℓ ∈ 1:L+1
		@assert factorranks(Q[ℓ]) == (rV[ℓ+1],rY[ℓ+1])
	end

	rW = decrank(W)
	@assert rW[1] == 1
	@assert rW[L+3] == 1

	@assert decsize(V)[1,2:L+1] == decsize(Y)[1,1:L]
	n = decsize(V)[1,:]; push!(n, decsize(Y)[1,L+1])
	@assert decsize(W)[1,:] == n

	@assert maxlvl ∈ 0:L
	@assert maxnrm ≥ 0
	Lmax = maxlvl
	Smax² = maxnrm^2

	C = Vector{Array{T,2}}(undef, L+1)
	Z = Vector{Array{T,3}}(undef, Lmax)
	R = Vector{Array{T,3}}(undef, Lmax+1)
	μ = Vector{Treal}(undef, Lmax+1)

	CC = ones(T, rY[L+3]*rW[L+3], 1)
	for ℓ ∈ L+1:-1:1
		CC = factorcontract(factormp(W[ℓ+1], 1, conj(Y[ℓ]), 1), CC)
		C[ℓ] = reshape(CC, rW[ℓ+1], rY[ℓ+1])
	end

	S = zero(Treal)
	RR = ones(T, 1, 1)
	for ℓ ∈ 1:Lmax+1
		F = factorcontract(RR, factormp(W[ℓ], 1, conj(V[ℓ]), 1))
		QC = C[ℓ]*adjoint(Q[ℓ])
		QC = reshape(QC, rW[ℓ+1]*rV[ℓ+1], 1)
		R[ℓ] = factorcontract(F, QC)
		μ[ℓ] = norm(R[ℓ])
		S += μ[ℓ]^2
		if Smax² > 0 && S ≥ Smax²
			Lmax = ℓ-1
			Z = Z[1:Lmax]
			R = R[1:Lmax+1]
			μ = μ[1:Lmax+1]
			break
		end
		# (ℓ ≤ Lmax) && (A[ℓ],RR = factorqr!(F, Val(true)))
		if ℓ ≤ Lmax
			Z[ℓ],RR = factorqr!(F, Val(true))
		end
	end

	return VectorFrameDec(Z, R),μ,Lmax
end




function framedecevalorth(F::VectorFrameDec{T},
	                      U::NTuple{d,MatrixDec{T}}, Q::NTuple{d,Vector{Matrix{T}}}, C::Vector{Matrix{T}}, W::VectorDec{T},
						  τsoft::Matrix{Treal}, τhard::Matrix{Treal}, τ::Matrix{Treal}, ρ::Matrix{Int};
						  toldistr::Bool=false, verbose::Bool=false) where {d,Treal<:AbstractFloat,T<:FloatRC{Treal}}
	### we assume here that the norm of UQ does not exceed one
	### we also assume that W is ortogonal


	framedecconsistency(F)
	Lmax = framedeclength(F)-1

	@assert all(framedecrearrank(F) .== 1)
	r = framedecrank(F)
	@assert r[1] == 1

	L = declength(U[1])-1

	if L < 0
		throw(ArgumentError("all U[k] and Q[k] should be of the same positive length"))
	end
	@assert L ≥ Lmax
	for k ∈ 1:d
		if declength(U[k]) ≠ L+1 || length(Q[k]) ≠ L+1
			throw(ArgumentError("all U[k] and Q[k] should be of the same positive length"))
		end
	end

	@assert size(τsoft) == (d+1,Lmax)
	@assert size(τhard) == (d+1,Lmax)
	@assert size(τ) == (d+1,Lmax)
	@assert size(ρ) == (d+1,Lmax)
	ε = copy(τ)
	ρ = copy(ρ)

	mU = zeros(Int, d, L+1)
	nU = zeros(Int, d, L+1)
	rU = zeros(Int, d, L+2)
	pQ = zeros(Int, d, L+1)
	qQ = zeros(Int, d, L+1)
	for k ∈ 1:d
		sz = decsize(U[k])
		mU[k:k,:] = sz[1:1,:]
		nU[k:k,:] = sz[2:2,:]
		rU[k:k,:] = decrank(U[k])
		for ℓ ∈ 0:L
			pQ[k,ℓ+1],qQ[k,ℓ+1] = size(Q[k][ℓ+1])
		end
	end
	@assert all(rU[:,1] .== 1)
	@assert length(C) == L+1
	@assert declength(W) == L+1
	
	rW = decrank(W)
	@assert rW[L+2] == 1
	for ℓ ∈ 1:L+1
		@assert factorranks(C[ℓ]) == (prod(qQ[:,ℓ]),rW[ℓ])
		@assert all(pQ[:,ℓ] .== rU[:,ℓ+1])
	end

	mW = decsize(W)
	@assert mW[:,1:L] == prod(mU[:,2:L+1]; dims=1)
	mF = framedecsize(F)
	@assert mF[:,1:L+1] == prod(mU; dims=1)


	Z,S = F
	for ℓ ∈ 1:Lmax
		@assert factorsize(Z[ℓ])[1] == prod(nU[:,ℓ])
		@assert factorsize(S[ℓ])[1] == prod(nU[:,ℓ])
	end
	@assert factorsize(S[Lmax+1])[1] == prod(nU[:,Lmax+1])


	Φ = VectorDec{T}(undef, Lmax)
	Ψ = Vector{VectorFactor{T}}(undef, Lmax+1)
	Θ = Vector{VectorFactor{T}}(undef, Lmax)
	μ = Vector{Treal}(undef, Lmax+1)

	X = [ Matrix{T}(transpose(reshape(W[ℓ], rW[ℓ], mW[ℓ]*rW[ℓ+1]))) for ℓ ∈ 1:Lmax ]
	pΘ = [ mW[ℓ]*rW[ℓ+1]-rW[ℓ] for ℓ ∈ 1:Lmax ]
	Y = [ LinearAlgebra.qr(X[ℓ]).Q*(Matrix{T}(I, mW[ℓ]*rW[ℓ+1], mW[ℓ]*rW[ℓ+1])[:,rW[ℓ]+1:rW[ℓ]+pΘ[ℓ]]) for ℓ ∈ 1:Lmax ]
	Θ = [ reshape(Matrix{T}(transpose(Y[ℓ])), pΘ[ℓ], mW[ℓ], rW[ℓ+1]) for ℓ ∈ 1:Lmax ]

	U2 = deckp(U...)
	Q2 = [ (d == 1) ? Q[1][ℓ] : kron(ntuple(k -> Q[d+1-k][ℓ], Val(d))...) for ℓ ∈ 1:L+1 ]

	local Δ
	for ℓ ∈ Lmax:-1:0
		if verbose
			@printf("framedecevalorth projection (←) ℓ = %4d out of %4d\n", ℓ, Lmax)
			flush(stdout)
		end
		US = factormp(S[ℓ+1], 1, U2[ℓ+1], 2)
		QC = Q2[ℓ+1]*C[ℓ+1]
		US1 = reshape(US, r[ℓ+1]*prod(rU[:,ℓ+1])*prod(mU[:,ℓ+1]), prod(rU[:,ℓ+2]))
		VS1 = US1*QC ### TODO smarter?
		VS = reshape(VS1, r[ℓ+1]*prod(rU[:,ℓ+1]), prod(mU[:,ℓ+1]), rW[ℓ+1])
		if ℓ < Lmax
			UZ = factormp(Z[ℓ+1], 1, U2[ℓ+1], 2)
			VS += factorcontract(UZ, Δ)
		end
		VS = reshape(VS, r[ℓ+1]*prod(rU[:,ℓ+1]), prod(mU[:,ℓ+1])*rW[ℓ+1])
		if ℓ == 0
			VS = reshape(VS, r[ℓ+1], prod(rU[:,ℓ+1])*prod(mU[:,ℓ+1]), rW[ℓ+1])
		else
			Δ,VS = VS*conj(X[ℓ]),VS*conj(Y[ℓ])
			VS = reshape(VS, r[ℓ+1], prod(rU[:,ℓ+1]), pΘ[ℓ])
		end
		Ψ[ℓ+1] = VS
	end

	Φ = deepcopy(Z[1:Lmax])
	mU = mU[:,1:Lmax]
	nU = nU[:,1:Lmax]


	F = VectorFrameDec(Φ, Ψ)
	δ = 0
	for k ∈ 1:d
		Φ,Ψ = F
		r = decrank(Φ)
		sz = [nU[k:k,:]; prod(nU[k+1:d,:]; dims=1).*prod(mU[1:k-1,:]; dims=1)]
		ΦΦ = decmodereshape(Φ, sz)
		ΦΦ = decmp(ΦΦ, 1, U[k][1:Lmax], 2)
		sz = prod(nU[k+1:d,:]; dims=1).*prod(mU[1:k,:]; dims=1)
		Φ = decmodereshape(ΦΦ, sz)
		for ℓ ∈ 0:Lmax
			if ℓ == 0
				Ψ[ℓ+1] = reshape(Ψ[ℓ+1], r[ℓ+1]*rU[k,ℓ+1], prod(rU[k+1:d,ℓ+1])*prod(mU[:,ℓ+1]), rW[ℓ+1])
			else
				Ψ[ℓ+1] = reshape(Ψ[ℓ+1], r[ℓ+1]*rU[k,ℓ+1], prod(rU[k+1:d,ℓ+1]), pΘ[ℓ])
			end
		end
		F = VectorFrameDec(Φ, Ψ)
		if verbose
			@printf("framedecevalorth orthogonalization (→) k = %2d out of %2d\n", k, d)
			flush(stdout)
		end
		F = framedecqr(F; path="forward", requiresizeconsistency=false)
		nrmε0 = norm(ε[k,:])
		if toldistr && nrmε0 > 0
			ε[k,:] .*= 1+δ/nrmε0
			nrmε0 += δ
			δ = 0
		end
		if verbose
			@printf("framedecevalorth approximation (←) k = %2d out of %2d\n", k, d)
			flush(stdout)
		end
		F,_,ε[k,:],ρ[k,:],_ = framedecsvd(F, τsoft[k,:], τhard[k,:], ε[k,:], ρ[k,:]; toldistr=toldistr, path="backward", requiresizeconsistency=false)
		nrmε1 = norm(ε[k,:])
		if nrmε1 < nrmε0
			δ += nrmε0-nrmε1
		end
	end

	Φ,Ψ = F
	r = decrank(Φ)
	for ℓ ∈ 1:Lmax
		Ψ[ℓ+1] = factorcontract(reshape(Ψ[ℓ+1], r[ℓ+1], 1, pΘ[ℓ]), Θ[ℓ])
	end


	if verbose
		@printf("framedecevalorth assembly\n")
		flush(stdout)
	end
	Ξ = VectorDec{T}(undef, L+2)
	for ℓ ∈ 0:Lmax
		if ℓ == 0
			Ξ[ℓ+1] = factorhcat(Ψ[ℓ+1], Φ[ℓ+1])
		elseif ℓ == Lmax
			Ξ[ℓ+1] = factorvcat(W[ℓ], Ψ[ℓ+1])
		else
			Ξ[ℓ+1] = factorltcat(W[ℓ], Ψ[ℓ+1], Φ[ℓ+1])
		end
	end
	for ℓ ∈ Lmax+1:L+1
		Ξ[ℓ+1] = copy(W[ℓ])
	end
	if verbose
		@printf("framedecevalorth approximation (→)\n")
		flush(stdout)
	end
	Ξ,ε[d+1,:],_,_,ρ[d+1,:],_ = decsvd!(Ξ, 1:Lmax; aTolDistr=ε[d+1,:], rank=ρ[d+1,:], path="forward")
	decskp!(Ξ, Lmax+1; path="backward")
	Φ,Ψ,Ξ,ε,ρ
end


function framedecproject(X::VectorDec{T},
                         U::NTuple{d,MatrixDec{T}}, Q::NTuple{d,Vector{Matrix{T}}}, C::Vector{Matrix{T}}, W::VectorDec{T},
						 τsoft::Matrix{Treal}, τhard::Matrix{Treal}, τ::Matrix{Treal}, ρ::Matrix{Int};
						 path::String="backward", toldistr::Bool=false,
                         verbose::Bool=false, maxlvl::Int=declength(X)-2) where {d,Treal<:AbstractFloat,T<:FloatRC{Treal}}
	### we assume here that the norm of UQ does not exceed one
	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end
	pathorth = (path == "backward") ? "forward" : "backward"
	
	L = declength(X)-2
	if L < 0
		throw(ArgumentError("X should be a decomposition of length L+2, where L is the (nonegative) number of levels"))
	end
	for k ∈ 1:d
		if declength(U[k]) ≠ L+1 || declength(Q[k]) ≠ L+1
			throw(ArgumentError("all U[k] and Q[k] should be of length L+1"))
		end
	end

	@assert size(τsoft) == (d,L)
	@assert size(τhard) == (d,L)
	@assert size(τ) == (d,L)
	@assert size(ρ) == (d,L)
	ε = copy(τ)

	mU = zeros(Int, d, L+1)
	nU = zeros(Int, d, L+1)
	rU = zeros(Int, d, L+2)
	pQ = zeros(Int, d, L+1)
	qQ = zeros(Int, d, L+1)
	for k ∈ 1:d
		sz = decsize(U[k])
		mU[k:k,:] = sz[1:1,:]
		nU[k:k,:] = sz[2:2,:]
		rU[k:k,:] = decrank(U[k])
		for ℓ ∈ 0:L
			pQ[k,ℓ+1],qQ[k,ℓ+1] = size(Q[k][ℓ+1])
		end
	end
	@assert all(rU[:,1] .== 1)
	@assert length(C) == L+1
	@assert declength(W) == L+1
	
	rW = decrank(W)
	@assert all(rU[:,1] .== 1)
	@assert rW[L+2] == 1
	for ℓ ∈ 1:L+1
		@assert factorranks(C[ℓ]) == (prod(qQ[:,ℓ]),rW[ℓ])
		@assert all(pQ[:,ℓ] .== rU[:,ℓ+1])
	end

	rX = decrank(X)
	@assert rX[1] == 1
	@assert rX[L+3] == 1

	mW = decsize(W)
	@assert mW[:,1:L] == prod(mU[:,2:L+1]; dims=1)
	mX = decsize(X)
	@assert mX[:,1:L+1] == prod(mU; dims=1)
	@assert mX[L+2] == mW[L+1]

	@assert maxlvl ∈ 0:L
	Lmax = maxlvl

	Z = X[1:Lmax]
	R = Vector{Array{T,3}}(undef, Lmax+1)
	R = Vector{VectorFactor{T}}(undef, L+1)
	F = VectorFrameDec(Z, R)

	U2 = deckp(U...)
	Q2 = [ (d == 1) ? Q[1][ℓ] : kron(ntuple(k -> Q[d+1-k][ℓ], Val(d))...) for ℓ ∈ 1:L+1 ]

	G = ones(T, rX[L+3]*rW[L+2])
	for ℓ ∈ L+1:-1:1
		WX = factormp(X[ℓ+1], 1, conj(W[ℓ]), 1)
		G = factormodereshape(WX, Int[])*G
		if ℓ ≤ Lmax+1
			UX = factormp(X[ℓ], 1, conj(U2[ℓ]), 1)
			QC = Q2[ℓ]*C[ℓ]
			UX1 = reshape(UX, rX[ℓ]*prod(rU[:,ℓ])*prod(nU[:,ℓ])*rX[ℓ+1], prod(rU[:,ℓ+1]))
			VX1 = UX1*conj(QC) ### TODO smarter?
			VX = reshape(VX1, rX[ℓ]*prod(rU[:,ℓ])*prod(nU[:,ℓ]), rX[ℓ+1]*rW[ℓ])
			VX = VX*G
			VX = reshape(VX, rX[ℓ], prod(rU[:,ℓ])*prod(nU[:,ℓ]), 1)
			R[ℓ] = VX
		end
	end
	δ = zero(T)
	for k ∈ 1:d
		Z,R = F
		rZ = decrank(Z)
		sz = [mU[k:k,1:Lmax]; prod(mU[k+1:d,1:Lmax]; dims=1).*prod(nU[1:k-1,1:Lmax]; dims=1)]
		ZZ = decmodereshape(Z, sz)
		ZZ = decmp(ZZ, 1, conj(U[k][1:Lmax]), 1)
		sz = prod(mU[k+1:d,1:Lmax]; dims=1).*prod(nU[1:k,1:Lmax]; dims=1)
		Z = decmodereshape(ZZ, sz)
		for ℓ ∈ 1:Lmax+1
			R[ℓ] = reshape(R[ℓ], rZ[ℓ]*rU[k,ℓ], prod(rU[k+1:d,ℓ])*prod(nU[:,ℓ]), 1)
		end
		F = VectorFrameDec(Z, R)
		F = framedecqr(F; path=pathorth, requiresizeconsistency=false)
		nrmε0 = norm(ε[k,:])
		if toldistr && nrmε0 > 0
			ε[k,:] .*= 1+δ/nrmε0
			nrmε0 += δ
			δ = 0
		end
		F,_,ε[k,:],_ = framedecsvd(F, τsoft[k,:], τhard[k,:], ε[k,:], ρ[k,:]; toldistr=toldistr, path=path, requiresizeconsistency=false)
		nrmε1 = norm(ε[k,:])
		if nrmε1 < nrmε0
			δ += nrmε0-nrmε1
		end
	end
	F,ε
end





function framedeceval(F::VectorFrameDec{T}, U::MatrixDec{T}, V::Vector{MatrixFactor{T}}, W::VectorDec{T}) where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	framedecconsistency(F)
	Lmax = framedeclength(F)-1

	@assert all(framedecrearrank(F) .== 1)
	r = framedecrank(F)

	Q,R = F

	L = declength(U)
	@assert L ≥ Lmax
	@assert declength(V) == L+1
	@assert declength(W) == L+1

	rU = decrank(U)
	rW = decrank(W); pushfirst!(rW, 1)
	@assert rU[1] == 1
	@assert rW[L+3] == 1
	for ℓ ∈ 1:L+1
		@assert factorranks(V[ℓ]) == (rU[ℓ],rW[ℓ+1])
	end

	m = zeros(Int, L+2)
	k = zeros(Int, L+1)
	for ℓ ∈ 1:L+1
		sz = factorsize(V[ℓ])
		m[ℓ],k[ℓ] = sz[1],sz[2]
	end
	m[L+2] = decsize(W)[1,L+1]

	@assert decsize(U) == [m[1:L]'; k[1:L]']
	@assert decsize(W) == m[2:L+2]'

	for ℓ ∈ 1:Lmax
		@assert factorsize(Q[ℓ]) == [k[ℓ]]
	end
	for ℓ ∈ 1:Lmax+1
		@assert factorsize(R[ℓ]) == [k[ℓ]]
	end


	Φ = Vector{Array{T,3}}(undef, L+2)

	for ℓ ∈ 1:Lmax
		D = factormp(V[ℓ], 2, R[ℓ], 1)
		C = factormp(U[ℓ], 2, Q[ℓ], 1)
		if ℓ == 1
			Φ[ℓ] = factorhcat(D, C)
		else
			Φ[ℓ] = factorltcat(W[ℓ-1], D, C)
		end
	end
	D = factormp(V[Lmax+1], 2, R[Lmax+1], 1)
	Φ[Lmax+1] = factorvcat(W[Lmax], D)
	for ℓ ∈ Lmax+1:L+1
		Φ[ℓ+1] = copy(W[ℓ])
	end

	Φ
end






function framedeceval(F::VectorFrameDec{T},
	U::MatrixDec{T}, V::Vector{MatrixFactor{T}}, W::VectorDec{T},
	τsoft::Vector{Treal}, τhard::Vector{Treal}, τ::Vector{Treal}, ρ::Vector{Int};
	path::String="backward",
	toldistr::Bool=false, verbose::Bool=false, minlvl::Int=0) where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	Q,R = F

	Lmax = declength(Q)
	@assert declength(R) == Lmax+1

	Lmin = minlvl
	@assert Lmin ∈ 0:Lmax
	@assert Lmin == 0 # TODO?

	@assert length(τsoft) == Lmax-Lmin
	@assert length(τhard) == Lmax-Lmin
	@assert length(τ) == Lmax-Lmin
	@assert length(ρ) == Lmax-Lmin

	r = decrank(Q)
	for ℓ ∈ 1:Lmax+1
		@assert factorranks(R[ℓ]) == (r[ℓ],1)
	end

	L = declength(U)
	@assert L ≥ Lmax
	@assert declength(V) == L+1
	@assert declength(W) == L+1

	rU = decrank(U)
	rW = decrank(W); pushfirst!(rW, 1)
	@assert rU[1] == 1
	@assert rW[L+3] == 1
	for ℓ ∈ 1:L+1
		@assert factorranks(V[ℓ]) == (rU[ℓ],rW[ℓ+1])
	end

	m = zeros(Int, L+2)
	k = zeros(Int, L+1)
	for ℓ ∈ 1:L+1
		sz = factorsize(V[ℓ])
		m[ℓ],k[ℓ] = sz[1],sz[2]
	end
	m[L+2] = decsize(W)[1,L+1]

	@assert decsize(U) == [m[1:L]'; k[1:L]']
	@assert decsize(W) == m[2:L+2]'

	for ℓ ∈ 1:Lmax
		@assert factorsize(Q[ℓ]) == [k[ℓ]]
	end
	for ℓ ∈ 1:Lmax+1
		@assert factorsize(R[ℓ]) == [k[ℓ]]
	end

	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end

	Φ = framedeceval(F, U, V, W)
	decqr!(Φ, 1:Lmax+1; path="forward")
	decskp!(Φ, Lmax+2; path="backward")
	σ = Vector{Vector{Float64}}(undef, Lmax)
	ε = copy(τ)
	if path=="backward"
		verbose && @printf("NB: W is assumed to be orthogonal from the right\n")
		verbose && @printf("NB: the orthogonalization of F from the left may improve the stability of this computation\n")

		# Ω = ones(T, 0, 0)
		local Ω
		for ℓ ∈ Lmax:-1:1
		# if ℓ == Lmax+1
		# 	Φ[ℓ+1],Ω = factorqr!(Φ[ℓ+1], Val(true); rev=true)
		# else
			Φ[ℓ+1],Ω,η,_,_,ρ[ℓ],σ[ℓ] = factorsvd!(Φ[ℓ+1], :, [1]; soft=τsoft[ℓ], hard=τhard[ℓ], atol=ε[ℓ], rank=ρ[ℓ], rev=true)
			if toldistr && ℓ > 1 && η < ε[ℓ]
				ε[ℓ-1] = sqrt(ε[ℓ-1]^2+ε[ℓ]^2-η^2)
			end
			ε[ℓ] = η
		# end
			Φ[ℓ] = factorcontract(Φ[ℓ], Ω)
		end
	elseif path=="forward"
		throw(ArgumentError("there is no implementation for path=\"forward\""))
	end
	μ = nothing
	Φ,ε,μ,ρ,σ
end



function framedeceval(F::VectorFrameDec{T}, U::MatrixDec{T}, P::Vector{Factor{T,2}}, W::VectorDec{T}) where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	framedecconsistency(F)
	Lmax = framedeclength(F)-1

	@assert all(framedecrearrank(F) .== 1)
	r = framedecrank(F)

	Q,R = F

	L = declength(U)-1
	@assert L ≥ Lmax
	@assert declength(P) == L+1
	@assert declength(W) == L+1

	rU = decrank(U)
	rW = decrank(W); pushfirst!(rW, 1)
	@assert rU[1] == 1
	@assert rW[L+3] == 1
	for ℓ ∈ 1:L+1
		@assert factorranks(P[ℓ]) == (rU[ℓ+1],rW[ℓ+1])
	end

	@assert decsize(U)[1,2:L+1] == decsize(W)[1,1:L]
	m = decsize(U)[1,:]; push!(m, decsize(W)[1,L+1])
	k = decsize(U)[2,:]
	for ℓ ∈ 1:Lmax
		@assert factorsize(Q[ℓ]) == [k[ℓ]]
		@assert factorsize(R[ℓ]) == [k[ℓ]]
	end
	@assert factorsize(R[Lmax+1]) == [k[Lmax+1]]

	Φ = Vector{Array{T,3}}(undef, L+2)

	for ℓ ∈ 1:Lmax
		UP = factorcontract(U[ℓ], P[ℓ])
		D = factormp(UP, 2, R[ℓ], 1)
		C = factormp(U[ℓ], 2, Q[ℓ], 1)
		if ℓ == 1
			Φ[ℓ] = factorhcat(D, C)
		else
			Φ[ℓ] = factorltcat(W[ℓ-1], D, C)
		end
	end
	UP = factorcontract(U[Lmax+1], P[Lmax+1])
	D = factormp(UP, 2, R[Lmax+1], 1)
	Φ[Lmax+1] = factorvcat(W[Lmax], D)
	for ℓ ∈ Lmax+1:L+1
		Φ[ℓ+1] = copy(W[ℓ])
	end

	Φ
end






function framedeceval(F::VectorFrameDec{T},
	U::MatrixDec{T}, P::Vector{Factor{T,2}}, W::VectorDec{T},
	τsoft::Vector{Treal}, τhard::Vector{Treal}, τ::Vector{Treal}, ρ::Vector{Int};
	path::String="backward",
	toldistr::Bool=false, verbose::Bool=false, minlvl::Int=0) where {Treal<:AbstractFloat,T<:FloatRC{Treal}}

	Q,R = F

	Lmax = declength(Q)
	@assert declength(R) == Lmax+1

	Lmin = minlvl
	@assert Lmin ∈ 0:Lmax
	@assert Lmin == 0 # TODO?

	@assert length(τsoft) == Lmax-Lmin
	@assert length(τhard) == Lmax-Lmin
	@assert length(τ) == Lmax-Lmin
	@assert length(ρ) == Lmax-Lmin

	r = decrank(Q)
	for ℓ ∈ 1:Lmax+1
		@assert factorranks(R[ℓ]) == (r[ℓ],1)
	end

	L = declength(U)-1
	@assert L ≥ Lmax
	@assert declength(P) == L+1
	@assert declength(W) == L+1

	rU = decrank(U)
	rW = decrank(W); pushfirst!(rW, 1)
	@assert rU[1] == 1
	@assert rW[L+3] == 1
	for ℓ ∈ 1:L+1
		@assert factorranks(P[ℓ]) == (rU[ℓ+1],rW[ℓ+1])
	end

	@assert decsize(U)[1,2:L+1] == decsize(W)[1,1:L]
	m = decsize(U)[1,:]; push!(m, decsize(W)[1,L+1])
	k = decsize(U)[2,:]
	for ℓ ∈ 1:Lmax
		@assert factorsize(Q[ℓ]) == [k[ℓ]]
		@assert factorsize(R[ℓ]) == [k[ℓ]]
	end
	@assert factorsize(R[Lmax+1]) == [k[Lmax+1]]

	if path ∉ ("forward","backward")
		throw(ArgumentError("path should be either \"forward\" or \"backward\""))
	end

	Φ = framedeceval(F, U, P, W)
	decqr!(Φ, 1:Lmax+1; path="forward")
	decskp!(Φ, Lmax+2; path="backward")
	σ = Vector{Vector{Float64}}(undef, Lmax)
	ε = copy(τ)
	if path=="backward"
		verbose && @printf("NB: W is assumed to be orthogonal from the right\n")
		verbose && @printf("NB: the orthogonalization of F from the left may improve the stability of this computation\n")

		# Ω = ones(T, 0, 0)
		local Ω
		for ℓ ∈ Lmax:-1:1
		# if ℓ == Lmax+1
		# 	Φ[ℓ+1],Ω = factorqr!(Φ[ℓ+1], Val(true); rev=true)
		# else
			Φ[ℓ+1],Ω,η,_,_,ρ[ℓ],σ[ℓ] = factorsvd!(Φ[ℓ+1], :, [1]; soft=τsoft[ℓ], hard=τhard[ℓ], atol=ε[ℓ], rank=ρ[ℓ], rev=true)
			if toldistr && ℓ > 1 && η < ε[ℓ]
				ε[ℓ-1] = sqrt(ε[ℓ-1]^2+ε[ℓ]^2-η^2)
			end
			ε[ℓ] = η
		# end
			Φ[ℓ] = factorcontract(Φ[ℓ], Ω)
		end
	elseif path=="forward"
		throw(ArgumentError("there is no implementation for path=\"forward\""))
	end
	μ = nothing
	Φ,ε,μ,ρ,σ
end










end
