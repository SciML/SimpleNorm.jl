using SafeTestsets

@safetestset "Aqua" begin
    using SimpleNorm
    using Aqua
    using Test
    Aqua.test_all(SimpleNorm)
end

@safetestset "JET" begin
    using SimpleNorm
    using JET
    using Test
    JET.test_package(SimpleNorm; target_defined_modules = true)
end
