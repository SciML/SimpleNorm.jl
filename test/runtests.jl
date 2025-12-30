using SimpleNorm
using Test

@testset "Explicit Imports" begin
    include("explicit_imports_test.jl")
end

@testset "SimpleNorm.jl" begin
    @testset "Vector norms" begin
        # Test vectors
        v1 = [3.0, 4.0]
        v2 = [1.0, -2.0, 3.0, -4.0]
        v3 = Float64[]
        v4 = [1.0 + 2.0im, 3.0 - 4.0im]

        # 2-norm (default)
        @test norm(v1) ≈ 5.0
        @test norm(v1, 2) ≈ 5.0
        @test norm(v2, 2) ≈ sqrt(1 + 4 + 9 + 16)
        @test norm(v3) == 0.0

        # 1-norm
        @test norm(v1, 1) ≈ 7.0
        @test norm(v2, 1) ≈ 10.0
        @test norm(v3, 1) == 0.0

        # Infinity norm
        @test norm(v1, Inf) ≈ 4.0
        @test norm(v2, Inf) ≈ 4.0
        @test norm(v3, Inf) == 0.0

        # Negative infinity norm
        @test norm(v1, -Inf) ≈ 3.0
        @test norm(v2, -Inf) ≈ 1.0

        # 0-norm (counting non-zeros)
        @test norm(v1, 0) == 2.0
        @test norm(v2, 0) == 4.0
        @test norm([1.0, 0.0, 3.0, 0.0, 5.0], 0) == 3.0

        # p-norm for various p
        @test norm(v1, 3) ≈ (3^3 + 4^3)^(1/3)
        @test norm(v2, 4) ≈ (1 + 16 + 81 + 256)^(1/4)

        # Complex vectors
        @test norm(v4, 1) ≈ sqrt(5) + 5
        @test norm(v4, 2) ≈ sqrt(5 + 25)
        @test norm(v4, Inf) ≈ 5.0
    end

    @testset "Matrix norms" begin
        A = [1.0 2.0 3.0;
             4.0 5.0 6.0]

        B = [1.0 -2.0;
             3.0 4.0;
             -5.0 6.0]

        # 1-norm (maximum column sum)
        @test norm(A, 1) ≈ 9.0  # max(1+4, 2+5, 3+6)
        @test norm(B, 1) ≈ 12.0  # max(1+3+5, 2+4+6)

        # Infinity norm (maximum row sum)
        @test norm(A, Inf) ≈ 15.0  # max(1+2+3, 4+5+6)
        @test norm(B, Inf) ≈ 11.0  # max(1+2, 3+4, 5+6)

        # Frobenius norm
        @test norm(A, "fro") ≈ sqrt(1 + 4 + 9 + 16 + 25 + 36)
        @test norm(A, :fro) ≈ sqrt(1 + 4 + 9 + 16 + 25 + 36)
        @test norm(B, "fro") ≈ sqrt(1 + 4 + 9 + 16 + 25 + 36)

        # Empty matrix
        @test norm(zeros(0, 0), 1) == 0.0
        @test norm(zeros(0, 5), 1) == 0.0
        @test norm(zeros(5, 0), Inf) == 0.0
    end

    @testset "Scalar norms" begin
        @test norm(5.0) == 5.0
        @test norm(-3.0) == 3.0
        @test norm(3.0 + 4.0im) == 5.0
        @test norm(0.0) == 0.0

        # All p-norms of a scalar should return absolute value
        @test norm(5.0, 1) == 5.0
        @test norm(-3.0, 2) == 3.0
        @test norm(5.0, Inf) == 5.0
    end

    @testset "Special cases" begin
        # Infinity and NaN handling
        v_inf = [1.0, Inf, 3.0]
        v_nan = [1.0, NaN, 3.0]

        @test norm(v_inf, 1) == Inf
        @test norm(v_inf, 2) == Inf
        @test norm(v_inf, Inf) == Inf
        @test isnan(norm(v_nan, 1))
        @test isnan(norm(v_nan, 2))
        @test isnan(norm(v_nan, Inf))

        # Large values (overflow prevention)
        large_val = 1e308
        v_large = [large_val, large_val]
        @test norm(v_large, 2) ≈ sqrt(2) * large_val
        @test !isinf(norm(v_large, 2))

        # Small values (underflow prevention)
        small_val = 1e-308
        v_small = [small_val, small_val]
        @test norm(v_small, 2) ≈ sqrt(2) * small_val
        @test norm(v_small, 2) > 0
    end

    @testset "Error handling" begin
        v = [1.0, 2.0, 3.0]

        # Invalid p values
        @test_throws ArgumentError norm(v, -2)
        @test_throws ArgumentError norm(v, -0.5)

        # Matrix 2-norm not implemented
        A = [1.0 2.0; 3.0 4.0]
        @test_throws ErrorException norm(A, 2)

        # Invalid matrix norm
        @test_throws ArgumentError norm(A, 3)
    end

    @testset "Type stability" begin
        # Test that output types are predictable
        v_int = [1, 2, 3]
        v_float = [1.0, 2.0, 3.0]
        v_complex = [1.0 + 0.0im, 2.0 + 0.0im]

        @test typeof(norm(v_int)) <: AbstractFloat
        @test typeof(norm(v_float)) <: AbstractFloat
        @test typeof(norm(v_complex)) <: AbstractFloat

        # Different element types
        @test norm(Int8[3, 4]) ≈ 5.0
        @test norm(Float32[3, 4]) ≈ 5.0
        @test norm(BigFloat[3, 4]) ≈ 5.0
    end
end
