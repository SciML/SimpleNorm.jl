using SimpleNorm
using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    @testset "QA" begin
        include("qa/qa.jl")
    end
end

if GROUP == "All" || GROUP == "Core"
    @safetestset "Explicit Imports" begin
        include("explicit_imports_test.jl")
    end

    @safetestset "SimpleNorm.jl" begin
        include("norm_tests.jl")
    end
end
