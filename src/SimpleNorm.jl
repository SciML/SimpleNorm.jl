module SimpleNorm

export norm

"""
    norm(x[, p]) -> Real
    norm(A::AbstractMatrix, p::Union{AbstractString, Symbol}) -> Real

Compute a scalar norm using pure-Julia implementations that do not depend on
`LinearAlgebra`, BLAS, or LAPACK. `norm(x)` is equivalent to `norm(x, 2)`.

# Arguments

  - `x::Union{Number, AbstractArray}`: A number or array of numeric values.
  - `A::AbstractMatrix`: A matrix for matrix-specific norms.
  - `p::Real`: The requested numeric norm order. It defaults to `2` when omitted.
    Matrices also accept `"fro"` or `:fro` for the Frobenius norm.

# Returns

The absolute value for numbers, or a real floating-point scalar for array
inputs. Empty arrays return the floating-point zero corresponding to their
element type.

# Supported Values

For vectors and general arrays, including custom `AbstractArray` implementations:

  - `p = 1`: sum of absolute values.
  - `p = 2`: Euclidean norm.
  - `p = Inf`: maximum absolute value.
  - `p = -Inf`: minimum absolute value.
  - `p = 0`: count of non-zero elements.
  - `p > 0`: scaled ``(sum(abs(x_i)^p))^(1/p)`` computation.

For matrices:

  - `p = 1`: maximum absolute column sum.
  - `p = Inf`: maximum absolute row sum.
  - `p = "fro"` or `p = :fro`: Frobenius norm.

For numbers, all supported `p` values return `abs(x)`.

# Array Interface

`norm` consumes the public `AbstractArray` interface; custom arrays must support
their usual public Base operations, including iteration, `isempty`, `eltype`,
and scalar indexing. Matrix inputs must additionally provide axes and
two-dimensional scalar indexing `A[i, j]` for values from those axes. No
SimpleNorm-specific subtype or method extension is required: define the normal
`AbstractArray` interface and call `norm` on the resulting value.

# Throws

Throws `ArgumentError` for unsupported negative vector orders, unsupported matrix
orders, and unknown matrix norm strings or symbols. Matrix spectral norms
(`p = 2`) are intentionally not implemented because they require singular value
decomposition.

# Notes

The Euclidean, general positive-order, and Frobenius paths use scaling algorithms
to reduce overflow and underflow in floating-point computations.

# Examples

```jldoctest
julia> using SimpleNorm

julia> SimpleNorm.norm([3.0, 4.0])
5.0

julia> SimpleNorm.norm([3.0, 4.0], 1)
7.0

julia> A = [1 2 3; 4 5 6];

julia> SimpleNorm.norm(A, :fro)
9.539392014169456
```
"""
norm(x) = norm(x, 2)

# Dispatch for vectors and general arrays with numeric p
function norm(x::AbstractArray{T}, p::Real) where {T}
    # For matrices, handle matrix-specific norms
    if ndims(x) == 2
        return _matrix_norm(x, p)
    end

    # Vector and general array norms
    if isempty(x)
        return float(abs(zero(T)))
    end

    if p == 2
        return norm2(x)
    elseif p == 1
        return norm1(x)
    elseif p == Inf
        return normInf(x)
    elseif p == -Inf
        return normMinusInf(x)
    elseif p == 0
        return norm0(x)
    elseif p > 0
        return normp(x, p)
    else
        throw(ArgumentError("p-norm is not defined for p < 0 (except p = -Inf)"))
    end
end

# Dispatch for matrix string/symbol norms
function norm(A::AbstractMatrix, p::Union{AbstractString, Symbol})
    if p in ("fro", :fro)
        return normFrobenius(A)
    else
        throw(ArgumentError("Unknown matrix norm: $p"))
    end
end

# Internal function for matrix norms with numeric p
function _matrix_norm(A::AbstractMatrix, p::Real)
    if p == 1
        return norm1_matrix(A)
    elseif p == Inf
        return normInf_matrix(A)
    elseif p == 2
        error("Spectral norm (2-norm) for matrices requires singular value decomposition, which is not implemented in SimpleNorm.jl to avoid dependencies.")
    else
        throw(ArgumentError("Matrix norm not supported for p = $p"))
    end
end

# Specialized implementations for common norms

function norm1(x)
    s = float(abs(zero(eltype(x))))
    for xi in x
        s += abs(xi)
    end
    return s
end

function norm2(x)
    # Use a scaling algorithm to avoid overflow/underflow
    # Similar to BLAS dnrm2 but in pure Julia
    T = float(real(eltype(x)))

    # Find the maximum absolute value for scaling
    scale = zero(T)
    for xi in x
        scale = max(scale, abs(xi))
    end

    if scale == zero(T)
        return zero(T)
    elseif isinf(scale)
        return T(Inf)
    end

    # Compute scaled sum of squares
    sumsq = zero(T)
    for xi in x
        sumsq += abs2(xi / scale)
    end

    return scale * sqrt(sumsq)
end

function normInf(x)
    if isempty(x)
        return float(abs(zero(eltype(x))))
    end

    m = abs(first(x))
    for xi in x
        m = max(m, abs(xi))
    end
    return float(m)
end

function normMinusInf(x)
    if isempty(x)
        return float(abs(zero(eltype(x))))
    end

    m = abs(first(x))
    for xi in x
        m = min(m, abs(xi))
    end
    return float(m)
end

function norm0(x)
    # Count non-zero elements
    count = 0
    for xi in x
        if xi != zero(xi)
            count += 1
        end
    end
    return float(count)
end

function normp(x, p::Real)
    T = float(real(eltype(x)))

    if p == 1
        return norm1(x)
    elseif p == 2
        return norm2(x)
    elseif isinf(p)
        return normInf(x)
    end

    # General p-norm with scaling to avoid overflow
    scale = zero(T)
    for xi in x
        scale = max(scale, abs(xi))
    end

    if scale == zero(T)
        return zero(T)
    elseif isinf(scale)
        return T(Inf)
    end

    # Compute scaled p-norm
    sump = zero(T)
    for xi in x
        sump += abs(xi / scale)^p
    end

    return scale * sump^(1 / p)
end

# Matrix norm implementations

function norm1_matrix(A::AbstractMatrix)
    if isempty(A)
        return float(abs(zero(eltype(A))))
    end

    # Maximum absolute column sum
    maxsum = zero(float(real(eltype(A))))
    for j in axes(A, 2)
        colsum = zero(float(real(eltype(A))))
        for i in axes(A, 1)
            colsum += abs(A[i, j])
        end
        maxsum = max(maxsum, colsum)
    end
    return maxsum
end

function normInf_matrix(A::AbstractMatrix)
    if isempty(A)
        return float(abs(zero(eltype(A))))
    end

    # Maximum absolute row sum
    maxsum = zero(float(real(eltype(A))))
    for i in axes(A, 1)
        rowsum = zero(float(real(eltype(A))))
        for j in axes(A, 2)
            rowsum += abs(A[i, j])
        end
        maxsum = max(maxsum, rowsum)
    end
    return maxsum
end

function normFrobenius(A::AbstractMatrix)
    # Frobenius norm is just the 2-norm of A viewed as a vector
    return norm2(A)
end

# Type conversion for norm of numbers
norm(x::Number, p::Real = 2) = abs(x)

end # module
