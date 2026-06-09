using SimpleNorm
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(SimpleNorm)
end

@testset "JET" begin
    JET.test_package(SimpleNorm; target_defined_modules = true)
end
