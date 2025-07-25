module TensorRefinement

using Reexport

include("Auxiliary.jl")
@reexport using .Auxiliary

include("TensorTrain.jl")
@reexport using .TensorTrain

include("SwitchingEdgeTensorTrain.jl")
@reexport using .SwitchingEdgeTensorTrain

include("Exponential.jl")
@reexport using .Exponential

include("Legendre.jl")
@reexport using .Legendre

include("Chebyshev.jl")
@reexport using .Chebyshev

include("FEM.jl")
@reexport using .FEM

include("ComplexGeometry.jl")
@reexport using .ComplexGeometry

end
