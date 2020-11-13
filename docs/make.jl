using TensorRefinement
using Documenter

makedocs(;
    modules=[TensorRefinement],
    authors="TensorRefinement Contributors",
    repo="https://github.com/TensorRefinement/TensorRefinement.jl/blob/{commit}{path}#L{line}",
    sitename="TensorRefinement.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://TensorRefinement.github.io/TensorRefinement.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/TensorRefinement/TensorRefinement.jl",
)
