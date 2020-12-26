module TensorRefinement

using Reexport

include("Auxiliary.jl")
@reexport using .Auxiliary

include("TensorTrain.jl")
@reexport using .TensorTrain

include("Exponential.jl")
@reexport using .Exponential

include("FEM.jl")
@reexport using .FEM

end
