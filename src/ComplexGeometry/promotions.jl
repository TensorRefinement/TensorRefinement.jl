const AbstractFaceBVP{D} = Union{AbstractCouplingFace{D}, AbstractBoundaryFace{D}}
Base.promote_rule(::Type{<:AbstractCouplingFace{D}}, ::Type{<:AbstractBoundaryFace{D}}) where {D} = AbstractFaceBVP{D}

const FaceBVP{D} = Union{CouplingFace{D}, BoundaryFace{D}}
Base.promote_rule(::Type{<:CouplingFace{D}}, ::Type{<:BoundaryFace{D}}) where {D} = FaceBVP{D}
