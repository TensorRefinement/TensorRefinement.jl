########################################
# Abstraction
########################################

abstract type AbstractBoundaryFace{D,BC} <: AbstractFace{D} end

abstract type BoundaryCondition end
struct Dirichlet <: BoundaryCondition end
struct Neumann <: BoundaryCondition end
struct Robin <: BoundaryCondition end

boundary_condition(::AbstractBoundaryFace{D,BC}) where {D,BC<:BoundaryCondition} = BC

is_dirichlet(::AbstractEdge) = false
is_dirichlet(f::AbstractBoundaryFace) = (boundary_condition(f) == Dirichlet)

########################################
# Concrete implementation
########################################

struct BoundaryFace{D,BC} <: AbstractBoundaryFace{D,BC} 
    self::Edge{D,1}

    function BoundaryFace{D,BC}(face) where {D,BC<:BoundaryCondition}
        return new(face)
    end
end

function BoundaryFace{D,BC}(ax::Int, lr::Int) where {D,BC}
    face = Edge{D}( (ax,lr) )
    return BoundaryFace{D,BC}(face)
end

fixed_axes(f::BoundaryFace) = fixed_axes(f.self)
free_axes(f::BoundaryFace) = free_axes(f.self)
Base.getindex(f::BoundaryFace, ax) = f.self[ax]
