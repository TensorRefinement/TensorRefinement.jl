########################################
# Abstraction
########################################

abstract type AbstractSignedPermutation{N} end

dimension(::AbstractSignedPermutation{N}) where {N} = N

function (Π::AbstractSignedPermutation{N})(i::Int)::Tuple{Int,Bool} where {N}
    if (i < 1) || (i > N)
        throw(ArgumentError("Index $i is out of bounds [1,$N]."))
    end
    return _apply(Π, i)
end

function Base.:(==)(Π₁::AbstractSignedPermutation{N₁}, Π₂::AbstractSignedPermutation{N₂}) where {N₁,N₂}
    N₁ != N₂ && return false
    return all(i -> Π₁(i) == Π₂(i), 1:N₁)
end

Base.:(∘)(Π::AbstractSignedPermutation) = Π
function Base.:(∘)(Π₁::AbstractSignedPermutation, Π₂::AbstractSignedPermutation, Π₃::AbstractSignedPermutation...)
    return ∘(Π₁, Π₂ ∘ Π₃)
end

########################################
# Concrete implementation
########################################

struct IdentityPermutation{N} <: AbstractSignedPermutation{N}
    signs::SVector{N,Bool}
end

function IdentityPermutation(signs)
    N = length(signs)
    return IdentityPermutation{N}(SVector{N}(signs))
end

function IdentityPermutation(signs::Bool...)
    return IdentityPermutation(SVector(signs))
end

_apply(Π::IdentityPermutation, i::Int) = i, Π.signs[i]
inverse(Π::IdentityPermutation) = Π

##########

struct FlipPermutation{N} <: AbstractSignedPermutation{N}
    signs::SVector{N,Bool}
end

function FlipPermutation(signs)
    N = length(signs)
    return FlipPermutation{N}(SVector{N}(signs))
end

function FlipPermutation(signs::Bool...)
    return FlipPermutation(SVector(signs))
end

_apply(Π::FlipPermutation, i::Int) = dimension(Π)-i+1, Π.signs[i]
inverse(Π::FlipPermutation) = FlipPermutation(reverse(Π.signs))

##########

struct GeneralPermutation{N} <: AbstractSignedPermutation{N}
    order::SVector{N,Int}
    signs::SVector{N,Bool}

    function GeneralPermutation{N}(order, signs) where {N}
        sorted = sort(order)
        if any(i -> sorted[i] != i, 1:N)
            throw(ArgumentError("Not a permutation of [1,$N]."))
        end
        return new(order, signs)
    end
end

function GeneralPermutation(order, signs)
    if length(order) != length(signs)
       throw(DimensionMismatch("Length of order $(length(order)) not equal to dimension of signs $(length(signs))."))
    end
    N = length(order)
    return GeneralPermutation{N}(SVector{N}(order), SVector{N}(signs))
end

_apply(Π::GeneralPermutation, i::Int) = Π.order[i], Π.signs[i]

##########

function are_inverse(Π₁::AbstractSignedPermutation{N}, Π₂::AbstractSignedPermutation{N}) where {N}
    for i = 1:N
        j₂, σ₂ = Π₂(i)
        j₁, σ₁ = Π₁(j₂)
        ((j₁ != i) || (σ₁ != σ₂)) && return false
    end
    return true
end

function inverse(Π::AbstractSignedPermutation{N}) where {N}
    order = [Π(i)[1] for i in 1:N]
    signs = [Π(i)[2] for i in 1:N]
    inverted_order = (findfirst(isequal(i), order) for i in 1:N)
    return GeneralPermutation{N}(SVector{N}(inverted_order), SVector{N}(signs))
end

function Base.:(∘)(Π₁::AbstractSignedPermutation{N}, Π₂::AbstractSignedPermutation{N}) where {N}
    order = Vector{Int}(undef,N)
    signs = Vector{Bool}(undef,N)
    for i in 1:N
        j₂, σ₂ = Π₂(i)
        j₁, σ₁ = Π₁(j₂)
        order[i] = j₁
        signs[i] = σ₁ ? σ₂ : !σ₂   
    end
    return GeneralPermutation(order, signs)
end

function Base.:(∘)(Π₁::IdentityPermutation{N}, Π₂::IdentityPermutation{N}) where {N}
    signs = Vector{Bool}(undef,N)
    for i in 1:N
        signs[i] = Π₁.signs[i] ? Π₂.signs[i] : !Π₂.signs[i]
    end
    return IdentityPermutation(signs)
end

function Base.:(∘)(Π₁::IdentityPermutation{N}, Π₂::FlipPermutation{N}) where {N}
    signs = Vector{Bool}(undef,N)
    for i in 1:N
        signs[i] = Π₁.signs[N-i+1] ? Π₂.signs[i] : !Π₂.signs[i]
    end
    return FlipPermutation(signs)
end

function Base.:(∘)(Π₁::FlipPermutation{N}, Π₂::IdentityPermutation{N}) where {N}
    signs = Vector{Bool}(undef,N)
    for i in 1:N
        signs[i] = Π₁.signs[i] ? Π₂.signs[i] : !Π₂.signs[i]
    end
    return FlipPermutation(signs)
end

function Base.:(∘)(Π₁::FlipPermutation{N}, Π₂::FlipPermutation{N}) where {N}
    signs = Vector{Bool}(undef,N)
    for i in 1:N
        signs[i] = Π₁.signs[N-i+1] ? Π₂.signs[i] : !Π₂.signs[i]
    end
    return IdentityPermutation(signs)
end

##########

const SignedPermutation1D = IdentityPermutation{1}
const SignedPermutation2D = Union{IdentityPermutation{2}, FlipPermutation{2}}
