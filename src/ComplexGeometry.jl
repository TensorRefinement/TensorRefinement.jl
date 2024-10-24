module ComplexGeometry

# from "transforms.jl"
export Transform, AbstractAffineTransform
export CompwiseTransform, CompositeTransform, AffineTransform, IdentityTransform, ShiftTransform, 
       ScaleTransform, RotateTransform2D, InterPolyTransform, AnnularSectorTransform
export jacobian, offset, linear, interpolate

# from "signed_permutations.jl"
export AbstractSignedPermutation
export IdentityPermutation, FlipPermutation, GeneralPermutation
export SignedPermutation1D, SignedPermutation2D
export dimension, inverse, are_inverse

# from "edges.jl"
export AbstractEdge, AbstractFace
export Edge
export dimension, codimension, fixed_axes, free_axes, coordinates, adjacent_edges, map_and_permute

# from "coupling.jl"
export AbstractCouplingEdge, AbstractCouplingFace
export Activity, Active, Inactive
export Coupling, Interior, Periodic
export CouplingInterface, CouplingEdge, CouplingFace
export is_coupling, activity, coupling, is_active, neighbor, orientation

# from "boundary.jl"
export AbstractBoundaryFace
export BoundaryCondition, Dirichlet, Neumann, Robin
export BoundaryFace
export boundary_condition, is_dirichlet

# from "promotions.jl"
export AbstractFaceBVP
export FaceBVP

# from "subdomains.jl"
export AbstractSubdomain, AbstractFaceBVP
export Subdomain
export dimension, parametrization, faces, inactive_edges

# from "partitions.jl"
export AbstractPartition
export DomainPartition

###########

const RefIntervalBeg = 0
const RefIntervalEnd = 1

###########

include("ComplexGeometry/transforms.jl")
include("ComplexGeometry/signed_permutations.jl")
include("ComplexGeometry/edges.jl")
include("ComplexGeometry/coupling.jl")
include("ComplexGeometry/boundary.jl")
include("ComplexGeometry/promotions.jl")
include("ComplexGeometry/subdomains.jl")
include("ComplexGeometry/partitions.jl")

end #module ComplexGeometry
