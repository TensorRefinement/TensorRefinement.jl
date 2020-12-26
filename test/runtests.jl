using SafeTestsets

@safetestset "TensorTrainFactor tests" begin include("TensorTrainFactor.jl") end
@safetestset "TensorTrainFactorization tests" begin include("TensorTrainFactorization.jl") end
@safetestset "FEM tests" begin include("FEM.jl") end
