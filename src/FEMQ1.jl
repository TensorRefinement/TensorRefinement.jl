
struct RefinementBasisQ1{d} <: RefinementBasis{d}
	function RefinementBasisQ1{d}() where {d}
		checkdim(d)
		new{d}()
	end
end
RefinementBasisQ1(d::Int) = RefinementBasisQ1{d}()


function cube_nodal_evaluate(::S, t::NTuple{d,Vector{T}}) where {d,S<:Refinement{d,RefinementBasisQ1{d}},T<:Number}
	### each entry of every element of t should be from [0,1]
	n = ntuple(k -> length(t[k]), Val(d))
	p = ntuple(k -> 2, Val(d))
	W = ones(T, n..., p...)
	for j ∈ CartesianIndices(p), i ∈ CartesianIndices(n)
		@inbounds for k ∈ 1:d
			W[i,j] *= (j[k] == 1) ? 1-t[k][i[k]] : t[k][i[k]]
		end
	end
	W = reshape(W, prod(n), 2^d)
end

function cube_nodal_to_normedQ(::S) where {d,S<:Refinement{d,RefinementBasisQ1{d}}}
	T = Rational{Int}
	C0 = T[ 1 1; -1 1].//2
	C1 = T[-1 1;  0 0]
	W = Array{T,3}(undef, 2^d, d+1, 2^d) # DoF × derivative × vertex
	if d == 1
		W[:,1,:] = C0
		W[:,2,:] = C1
	else
		W[:,1,:] = kron(ntuple(k -> C0, Val(d))...)
		@inbounds for k ∈ 1:d
			W[:,k+1,:] = kron(ntuple(k -> C0, Val(d-k))..., C1, ntuple(k -> C0, Val(k-1))...)
		end
	end
	reshape(W, 2^d*(d+1), 2^d)
end

function _cube_normed_weight(::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:Refinement{d,RefinementBasisQ1{d}}}
	c0 = T[1, 1/(sqrt(one(T)*3))]
	c1 = T[1, 1]
	if d == 1
		w = [c0 c1]
	else
		w = Matrix{T}(undef, 2^d, d+1)
		w[:,1] = kron(ntuple(k -> c0, Val(d))...)
		@inbounds for k ∈ 1:d
			w[:,k+1] = kron(ntuple(k -> c0, Val(d-k))..., c1, ntuple(k -> c0, Val(k-1))...)
		end
	end
	w[:,1] .*= κ₀
	w[:,2:d+1] .*= κ₁
	w = reshape(w, :)
	w
end

function cube_normedQ_to_normedF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:Refinement{d,RefinementBasisQ1{d}}}
	w = _cube_normed_weight(re, κ₀, κ₁)
	Diagonal(w)
end


function cube_normedF_to_normedQ(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:Refinement{d,RefinementBasisQ1{d}}}
	w = _cube_normed_weight(re, κ₀, κ₁)
	w .= 1 ./w
	Diagonal(w)
end




#############################
### Any orthogonalization ###

function cube_auxQ_to_nodal(::S) where {d,S<:OrthRefinement{d,RefinementBasisQ1{d}}}
	C = Rational{Int}[ 1 -1
	                   1  1 ]
	(d > 1) ? kron(ntuple(k -> C, Val(d))...) : C
end

function cube_nodal_to_auxQ(::S) where {d,S<:OrthRefinement{d,RefinementBasisQ1{d}}}
	C = Rational{Int}[  1 1
	                   -1 1 ].//2
	(d > 1) ? kron(ntuple(k -> C, Val(d))...) : C
end

function _cube_aux_weight(::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,RefinementBasisQ1{d}}}
	c0 = Int[3,1]
	c1 = Int[0,1]
	wk = (d > 1) ? kron(ntuple(k -> c0, Val(d))...) : c0
	w = wk.*(κ₀^2 / 3^d)
	for k ∈ 1:d
		wk = (d > 1) ? kron(ntuple(k -> c0, Val(d-k))..., c1, ntuple(k -> c0, Val(k-1))...) : c1
		w .+= wk.*(κ₁^2 * 4 / 3^(d-1))
	end
	w .= sqrt.(w)
	w
end

function cube_auxF_to_auxQ(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,RefinementBasisQ1{d}}}
	if κ₀ == 0
		throw(ArgumentError("κ₀ should be nonzero"))
	end
	if κ₁ == 0
		throw(ArgumentError("κ₁ should be nonzero"))
	end
	w = _cube_aux_weight(re, κ₀, κ₁)
	w .= 1 ./w
	Diagonal(w)
end

function cube_auxQ_to_auxF(re::S, κ₀::T, κ₁::T) where {T<:AbstractFloat,d,S<:OrthRefinement{d,RefinementBasisQ1{d}}}
	if κ₀ == 0
		throw(ArgumentError("κ₀ should be nonzero"))
	end
	if κ₁ == 0
		throw(ArgumentError("κ₁ should be nonzero"))
	end
	w = _cube_aux_weight(re, κ₀, κ₁)
	Diagonal(w)
end

