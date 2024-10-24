using StaticArrays: SVector, SMatrix, MVector, MMatrix
using LinearAlgebra: Diagonal, I

###########

"""
    Transform{Dim}

Supertype for mathematical functions mapping the `Dim`-dimensional hypercube 
`[0,1]^Dim` into the `Dim`-dimensional coordinate space.
"""
abstract type Transform{Dim} <: Function end

"""
    (f::Transform{Dim})(x::Real...)

Apply the transform `f` to a point `x` of the `Dim`-dimensional coordinate space.
"""
function (f::Transform{Dim})(x::Real...) where {Dim}
    if length(x) != Dim
        error("dimension mismatch")
    end

    return f(SVector{Dim}(x))
end

"""
    (f::Transform{Dim})(x::AbstractVector{<:Real})

Apply the transform `f` to a point `x` of the `Dim`-dimensional coordinate space.
"""
function (f::Transform{Dim})(x::AbstractVector{<:Real}) where {Dim}
    if length(x) != Dim
        error("dimension mismatch")
    end

    return f(SVector{Dim}(x))
end

"""
    jacobian(f::Transform{Dim}, x::AbstractVector{<:Real}, δ::Real=1e-10)

Numerically compute the Jacobian of a transform `f` at a point `x` of the 
`Dim`-dimensional coordinate space, using central difference approximation with
step size `δ`.
"""
function jacobian(f::Transform{Dim}, x::AbstractVector{<:Real}, δ::Real=1e-10) where {Dim}
    if length(x) != Dim
        error("dimension mismatch")
    end

    xp = MVector{Dim,Float64}(x)
    xm = MVector{Dim,Float64}(x)
    jac = zero(MMatrix{Dim,Dim,Float64})
    for indc in 1:Dim
        if x[indc] - RefIntervalBeg < eps(Float64)
            xp[indc] += δ
            jac[:, indc] = (f(xp) - f(xm)) / (δ)
            xp[indc] -= δ
        elseif RefIntervalEnd - x[indc] < eps(Float64)
            xm[indc] -= δ
            jac[:, indc] = (f(xp) - f(xm)) / (δ)
            xm[indc] += δ
        else
            xp[indc] += δ
            xm[indc] -= δ
            jac[:, indc] = (f(xp) - f(xm)) / (2 * δ)
            xp[indc] -= δ
            xm[indc] += δ
        end
    end
    return jac
end

###########

"""
    CompwiseTransform{Dim} <: Transform{Dim}

`Dim`-dimensional transform represented as `Dim` real-valued functions on the `Dim`-dimensional hypercube:
f(x) = (f1(x), ..., fDim(x)).
"""
struct CompwiseTransform{Dim} <: Transform{Dim}
    components::SVector{Dim, Function}
end

function CompwiseTransform(components::AbstractVector{Function})
    n = length(components)
    return CompwiseTransform{n}(SVector{n}(components))
end

CompwiseTransform(components::Function...) = CompwiseTransform(SVector(components...))

function (f::CompwiseTransform{Dim})(x::SVector{Dim,<:Real}) where {Dim}
    return SVector((g(x...) for g in f.components)...)
end

###########

"""
    CompositeTransform{Dim, T1<:Transform{Dim}, T2<:Transform{Dim}} <: Transform{Dim}

Composition of two `Dim`-dimensional transforms `f1 ∘ f2` with `f1::T1` and `f2::T2`.
"""
struct CompositeTransform{Dim, T1<:Transform{Dim}, T2<:Transform{Dim}} <: Transform{Dim}
    outer::T1
    inner::T2

    function CompositeTransform{Dim}(outer::T1, inner::T2) where {Dim, T1<:Transform{Dim}, T2<:Transform{Dim}}
        return new{Dim,T1,T2}(outer, inner)
    end
end

function (f::CompositeTransform{Dim})(x::SVector{Dim,<:Real}) where {Dim}
    return f.outer(f.inner(x))
end

function jacobian(f::CompositeTransform, x::AbstractVector{<:Real}, δ::Real=1e-10)
    return jacobian(f.outer, f.inner(x), δ) * jacobian(f.inner, x, δ)
end

Base.:(∘)(f::Transform) = f
Base.:(∘)(f₂::Transform{Dim}, f₁::Transform{Dim}) where {Dim} = CompositeTransform{Dim}(f₂,f₁)
Base.:(∘)(f₃::Transform{Dim}, f₂::Transform{Dim}, f₁::Transform{Dim}...) where {Dim} = ∘(f₃, f₂ ∘ f₁)

###########

"""
    IdentityTransform{Dim} <: Transform{Dim}

`Dim`-dimensional identity transform.
"""
struct IdentityTransform{Dim} <: Transform{Dim} end

function (f::IdentityTransform{Dim})(x::SVector{Dim,<:Real}) where {Dim}
    return x
end

jacobian(::IdentityTransform, ::AbstractVector{<:Real}) = I

Base.:(∘)(f::Transform{Dim}, ::IdentityTransform{Dim}) where {Dim} = f
Base.:(∘)(::IdentityTransform{Dim}, f::Transform{Dim}) where {Dim} = f

###########

"""
    AbstractAffineTransform{Dim} <: Transform{Dim}

Supertype for `Dim`-dimensional affine transforms such that `f(x) = linear(f) * x + offset(f)`.
"""
abstract type AbstractAffineTransform{Dim} <: Transform{Dim} end
"""
    offset(f::AbstractAffineTransform{Dim})

Return the `Dim`-dimensional offset vector of the affine transform `f`.
"""
function offset end
"""
    linear(f::AbstractAffineTransform{Dim})

Return the `Dim`×`Dim` matrix of the affine transform `f`.
"""
function linear end

function (f::AbstractAffineTransform{Dim})(x::SVector{Dim,<:Real}) where {Dim}
    return offset(f) .+ linear(f) * x
end

"""
    jacobian(f::AbstractAffineTransform, ::AbstractVector{<:Real})

Compute the Jacobian of an affine transform `f` as its matrix.
"""
jacobian(f::AbstractAffineTransform, ::AbstractVector{<:Real}) = linear(f)

function Base.:(∘)(f₂::AbstractAffineTransform{Dim}, f₁::AbstractAffineTransform{Dim}) where {Dim}
    dx = f₂(offset(f₁))
    A = linear(f₂) * linear(f₁)
    return AffineTransform(dx, A)
end

###########

"""
    AffineTransform{Dim,T} <: AbstractAffineTransform{Dim}

`Dim`-dimensional affine transform with explicitly specified `Dim`×`Dim` matrix
and `Dim`-dimensional offset vector with elements of type `T`.
"""
struct AffineTransform{Dim,T} <: AbstractAffineTransform{Dim}
    dx::SVector{Dim,T}
    A::SMatrix{Dim,Dim,T}

    AffineTransform(dx::SVector{Dim,T}, A::SMatrix{Dim,Dim,T}) where {Dim, T<:Real} = new{Dim,T}(dx, A)
end

function AffineTransform(dx::AbstractVector, A::AbstractMatrix)
    T = promote_type(eltype(dx), eltype(A))
    n = length(dx)
    ((size(A,1) != n) || (size(A,2) != n)) && error("dimension mismatch")
    return AffineTransform(SVector{n,T}(dx), SMatrix{n,n,T}(A))
end

offset(f::AffineTransform) = f.dx
linear(f::AffineTransform) = f.A

##########

"""
    ShiftTransform{Dim,T} <: AbstractAffineTransform{Dim}

`Dim`-dimensional affine transform with the identity `Dim`×`Dim` matrix and `Dim`-dimensional offset vector with elements of type `T`.
"""
struct ShiftTransform{Dim,T} <: AbstractAffineTransform{Dim}
    dx::SVector{Dim,T}

    ShiftTransform(dx::SVector{Dim,T}) where {Dim, T<:Real} = new{Dim,T}(dx)
end

function ShiftTransform(dx::AbstractVector)
    T = eltype(dx)
    n = length(dx)
    return ShiftTransform(SVector{n,T}(dx))
end

function ShiftTransform(dx::Real...)
    return ShiftTransform(SVector(dx))
end

offset(f::ShiftTransform) = f.dx
linear(f::ShiftTransform) = I 

###########

"""
    ScaleTransform{Dim,T} <: AbstractAffineTransform{Dim}

`Dim`-dimensional affine transform with a diagonal `Dim`×`Dim` matrix with elements of type `T` and zero offset vector.
"""
struct ScaleTransform{Dim,T} <: AbstractAffineTransform{Dim}
    scalars::SVector{Dim,T}
    Λ::Diagonal{T,SVector{Dim,T}}

    ScaleTransform(scalars::SVector{Dim,T}) where {Dim, T<:Real} = new{Dim,T}(scalars, Diagonal(scalars))
end

function ScaleTransform(scalars::AbstractVector)
    T = eltype(scalars)
    n = length(scalars)
    return ScaleTransform(SVector{n,T}(scalars))
end

function ScaleTransform(scalars::Real...)
    return ScaleTransform(SVector(scalars))
end

offset(f::ScaleTransform) = zero(f.scalars)
linear(f::ScaleTransform) = f.Λ

Base.:(∘)(f₂::ScaleTransform{Dim}, f₁::ScaleTransform{Dim}) where {Dim} = ScaleTransform(f₂.scalars .* f₁.scalars)

###########

"""
    RotateTransform2D{T} <: AbstractAffineTransform{2}

Two-dimensional rotation transform specified by the rotation angle (in radians). The 2×2 rotation matrix has elements of type `T`.
"""
struct RotateTransform2D{T} <: AbstractAffineTransform{2}
    angle_rad::T
    rotation::SMatrix{2,2,T}

    function RotateTransform2D(angle_rad::Real)
        c = cos(angle_rad)
        s = sin(angle_rad)
        T = typeof(c)
        rotation = SMatrix{2,2}(c, s, -s, c)
        return new{T}(T(angle_rad), rotation)
    end
end

offset(f::RotateTransform2D) = SVector(0,0)
linear(f::RotateTransform2D) = f.rotation

Base.:(∘)(f₂::RotateTransform2D, f₁::RotateTransform2D) = RotateTransform2D(f₂.angle_rad + f₁.angle_rad)

##########

const NodeVals{Dim,T} = Array{SVector{Dim,T}, Dim}
const AbsNodeVals{Dim,T} = AbstractArray{SVector{Dim,T}, Dim}
const NodeJacVals{Dim,T} = Array{SMatrix{Dim,Dim,T}, Dim}
_mutable_nodeval(::Type{SVector{Dim,T}}) where {Dim,T} = MVector{Dim,T}
_mutable_nodeval(::Type{SMatrix{Dim,Dim,T}}) where {Dim,T} = MMatrix{Dim,Dim,T}

"""
    InterPolyTransform{Dim,AP,T} <: Transform{Dim}

`Dim`-dimensional transform represented as a vector-valued Lagrange polynomial interpolant on the product
of `Dim` Chebyshev grids on [0,1]. Along the kth dimension, the nodes of the grid are `xj = (1 + cos(jπ/nk)) / 2` for j = 0,...,nk.

The values of the interpolant on the grid are of type `SVector{Dim,T}` and stored as an object of type `AP`.

The vector-valued Lagrange polynomial interpolant is represneted in the barycentric format [Berrut JP, Trefethen LN. Barycentric Lagrange interpolation. SIAM Review. 2004;46(3):501-17].
"""
struct InterPolyTransform{Dim,AP,T} <: Transform{Dim}
    values::AP
    jac_values::NodeJacVals{Dim,Float64}

    function InterPolyTransform{Dim,AP,T}(values::AP) where {Dim,T,AP<:AbsNodeVals{Dim,T}}
        n = size(values)
        jvals = Vector{NodeVals{Dim,Float64}}(undef, Dim)
        for ix in 1:Dim
            D = _cheb_differentiation_matrix(n[ix]-1)
            perm = vcat(ix, setdiff(1:Dim,ix))
            unfolding = reshape(permutedims(values, perm), n[ix], :)
            jvals[ix] = permutedims(reshape(D * unfolding, n[perm]...), invperm(perm))
        end
        jac_values = NodeJacVals{Dim,Float64}(undef, n...)
        for ci in CartesianIndices(jac_values)
            jac_values[ci] = reduce(hcat, SVector{Dim}([jvals[i][ci] for i in 1:Dim]))
        end
        return new{Dim,AP,T}(values, jac_values)
    end
end

function InterPolyTransform(values::AbstractArray{P,Dim}) where {Dim,P<:SVector{Dim,<:Real}}
    return InterPolyTransform{Dim,typeof(values),eltype(P)}(values)
end

function (f::InterPolyTransform{Dim})(x::SVector{Dim,<:Real}) where {Dim}
    return _cheb_eval(f.values, x)
end

"""
    jacobian(f::InterPolyTransform{Dim}, x::AbstractVector{<:Real})

Compute the Jacobian of a polynomial interpolant transform `f` at a point `x` by interpolating the values of its Jacobian on the Chebyshev grid.
"""
function jacobian(f::InterPolyTransform{Dim}, x::AbstractVector{<:Real}) where {Dim}
    return _cheb_eval(f.jac_values, x)
end

"""
    interpolate(f::Transform{Dim}, degrees::Int...)

Interpolate the `Dim`-dimensional transform `f` on the product of Chebyshev grids. Along the kth dimension, the polynomial
interpolation is of degree `degrees[k]`.
"""
function interpolate(f::Transform{Dim}, degrees::Int...) where {Dim}
    Dim != length(degrees) && error("bad length")
    vals = NodeVals{Dim,Float64}(undef, (degrees .+ 1)...)
    for ci in CartesianIndices(vals)
        vals[ci] = f((_cheb_node(ci[i]-1, degrees[i]) for i in 1:Dim)...)
    end
    return InterPolyTransform(vals)
end

_cheb_node(i,deg) = (RefIntervalEnd + RefIntervalBeg)/2 + (RefIntervalEnd - RefIntervalBeg) / 2 * cos((deg-i) * π / deg)
_cheb_weight(i,deg) = (-1)^i * (i == 0 || i == deg ? 0.5 : 1.0)
_cheb_rational_term(i,deg,x) = _cheb_weight(i,deg) / (x - _cheb_node(i,deg))

function _cheb_eval(vals::AbstractArray, x::AbstractVector)
    degrees = size(vals) .- 1
    isempty(degrees) && error("0-dimensional array")
    dim = length(degrees)

    exact_nodes = [findfirst(j -> abs(_cheb_node(j,degrees[i]) - x[i]) < eps(Float64), 0:degrees[i]) for i in 1:dim]
    slices = (s == nothing ? Colon() : s for s in exact_nodes)
    subarr = findall(s -> s == nothing, exact_nodes)
    return _cheb_eval_offgrid(view(vals, slices...), x[subarr])
end

function _cheb_eval_offgrid(vals::AbstractArray, x::AbstractVector)
    degrees = size(vals) .- 1
    isempty(degrees) && return vals[1]
    dim = length(degrees)

    ratsums = [sum(j -> _cheb_rational_term(j,degrees[i],x[i]), 0:degrees[i]) for i in 1:dim]
    numer = zero(_mutable_nodeval(eltype(vals)))
    for ci in CartesianIndices(vals)
        numer += vals[ci] * prod(i -> _cheb_rational_term(ci[i]-1,degrees[i],x[i]), 1:dim)
    end
    return numer / prod(ratsums)
end

function _cheb_differentiation_matrix(degree::Int)
    D = zeros(Float64, degree+1, degree+1)
    for i in 1:degree+1
        for j in 1:degree+1
            j == i && continue
            D[i,j] = (_cheb_weight(j-1,degree) / _cheb_weight(i-1,degree)) / (_cheb_node(i-1,degree) - _cheb_node(j-1,degree))
        end
        D[i,i] = -sum(view(D, i, :))
    end
    return D
end

###########

struct AnnularSectorTransform{T1,T2} <: Transform{2}
    radii::SVector{2,T1}
    angles::SVector{2,T2}
end

function AnnularSectorTransform(radii::SVector{2,T1}, angles::SVector{2,T2}) where {T1<:Real, T2<:Real}
    return AnnularSectorTransform{T1,T2}(radii, angles)
end
function AnnularSectorTransform(inner_radius::Real, outer_radius::Real, inner_angle::Real, outer_angle::Real)
    return AnnularSectorTransform(SVector(inner_radius, outer_radius), SVector(inner_angle, outer_angle))
end

function (f::AnnularSectorTransform{T1,T2})(x::SVector{2,<:Real}) where {T1<:Real, T2 <:Real}
    r = f.radii[1] * (1-x[1]) + f.radii[2] * x[1]
    ang = f.angles[1] * (1-x[1]) + f.angles[2] * x[1]
    c = cos(ang * (x[2]-0.5))
    s = sin(ang * (x[2]-0.5))
    return SVector(r*c, r*s)
end
