using Documenter
using SimpleNorm

makedocs(
    sitename = "SimpleNorm.jl",
    authors = "SciML",
    modules = [SimpleNorm],
    clean = true,
    format = Documenter.HTML(;
        canonical = "https://docs.sciml.ai/SimpleNorm/stable/",
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/SciML/SimpleNorm.jl.git",
    push_preview = true,
)
