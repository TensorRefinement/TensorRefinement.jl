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
        size_threshold = 500000  # Increased size threshold to avoid file size error
    ),
    pages=[
        "Home" => "index.md",
    ],
    checkdocs=:none,
)

deploydocs(;
    repo="github.com/TensorRefinement/TensorRefinement.jl",
)
