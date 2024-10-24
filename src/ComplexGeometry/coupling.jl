########################################
# Abstraction
########################################

abstract type AbstractCouplingEdge{D,K,Act,Cpl} <: AbstractEdge{D,K} end
const AbstractCouplingFace{D,K,Act,Cpl} = AbstractCouplingEdge{D,1,Act,Cpl}

is_coupling(::AbstractEdge) = false
is_coupling(::AbstractCouplingEdge) = true

abstract type Activity end
struct Active <: Activity end
struct Inactive <: Activity end

abstract type Coupling end
struct Interior <: Coupling end
struct Periodic <: Coupling end

activity(::AbstractCouplingEdge{D,K,Act,Cpl}) where {D,K,Act<:Activity,Cpl} = Act
coupling(::AbstractCouplingEdge{D,K,Act,Cpl}) where {D,K,Act,Cpl<:Coupling} = Cpl

is_active(e::AbstractCouplingEdge) = (activity(e) == Active)

# Interface contract for AbstractCouplingEdge:
#   neighbor
#   orientation
function neighbor end
function orientation end

########################################
# Concrete implementation: interface
########################################

struct CouplingInterface{D,K,Perm}
    nbr_subdomain::Int
    nbr_edge::Edge{D,K}
    Π::Perm

    function CouplingInterface{D,K,Perm}(ids,e,Π) where {D,K,Perm<:AbstractSignedPermutation}
        if D-K != dimension(Π)
            throw(DimensionMismatch("Permutation's dimension $(dimension(Π)) needs to be $(D-K)."))
        end
        return new(ids,e,Π)
    end
end

function CouplingInterface(ids::Int, e::Edge{D,K}, Π::Perm) where {D,K,Perm}
    return CouplingInterface{D,K,Perm}(ids,e,Π)
end

########################################
# Concrete implementation: edge, face
########################################

struct CouplingEdge{D,K,Act,Cpl,Perm} <: AbstractCouplingEdge{D,K,Act,Cpl}
    self::Edge{D,K}
    interface::CouplingInterface{D,K,Perm}

    function CouplingEdge{D,K,Act,Cpl,Perm}(self,interface) where {D,K,Act<:Activity,Cpl<:Coupling,Perm}
        return new(self, interface)
    end
end

function CouplingEdge{Act,Cpl}(edge::Edge{D,K}, interface::CouplingInterface{D,K,Perm}) where {D,K,Act,Cpl,Perm}
    return CouplingEdge{D,K,Act,Cpl,Perm}(edge,interface)
end

function CouplingEdge{Act,Cpl}(edge, nbr_subdomain, nbr_edge, Π) where {Act,Cpl}
    interface = CouplingInterface(nbr_subdomain, nbr_edge, Π)
    return CouplingEdge{Act,Cpl}(edge, interface)
end

const CouplingFace{D,Act,Cpl,Perm} = CouplingEdge{D,1,Act,Cpl,Perm}

function CouplingFace{D,Act,Cpl}(
    ax::Int, lr::Int,
    nbr_subdomain::Int, nbr_ax::Int, nbr_lr::Int, Π::Perm
) where {D,Act,Cpl,Perm}
    face = Edge{D}( (ax,lr) )
    nbr_face = Edge{D}( (nbr_ax, nbr_lr) )
    return CouplingEdge{Act,Cpl}(face, nbr_subdomain, nbr_face, Π)
end

fixed_axes(e::CouplingEdge) = fixed_axes(e.self)
free_axes(e::CouplingEdge) = free_axes(e.self)
Base.getindex(e::CouplingEdge, ax) = e.self[ax]
neighbor(e::CouplingEdge) = (e.interface.nbr_subdomain, e.interface.nbr_edge)
orientation(e::CouplingEdge) = e.interface.Π
