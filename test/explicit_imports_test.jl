using Test
using SimpleNorm
using ExplicitImports

@testset "ExplicitImports" begin
    @testset "check_no_implicit_imports" begin
        @test check_no_implicit_imports(SimpleNorm) === nothing
    end

    @testset "check_no_stale_explicit_imports" begin
        @test check_no_stale_explicit_imports(SimpleNorm) === nothing
    end
end
