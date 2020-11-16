__precompile__()
module TTPy

using PyCall
using ..Aux, ..TensorTrain

import ..TensorTrain.dec

export pydec, dec, pyamen

const tt = PyNULL()
const ttamen = PyNULL()

function __init__()
	copy!(tt, pyimport("tt"))
	copy!(ttamen, pyimport("tt.amen"))
end

pydec(U::Dec{T,3}) where {T<:FloatRC} = tt.vector.from_list(U)
pydec(U::Dec{T,4}) where {T<:FloatRC} = tt.matrix.from_list(U)

dec(U::PyObject) = dec(tt.vector.to_list(U))

pyamen(A::Dec{T,4}, b::Dec{T,3}, x::Dec{T,3}, ɛ::Real; kickrank::Int=4, nswp::Int=20, local_prec::String="n", local_iters::Int=2, local_restart::Int=40, trunc_norm::Int=1, max_full_size::Int=50, verb::Int=1) where {T<:FloatRC} = dec(ttamen.amen_solve(pydec(A), pydec(b), pydec(x), ɛ, kickrank, nswp, local_prec, local_iters, local_restart, trunc_norm, max_full_size, verb))

end
