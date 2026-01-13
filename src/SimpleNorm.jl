module SimpleNorm

export norm

"""
    norm(x, p::Real=2)

Compute the p-norm of a vector or array.

For vectors and arrays, the p-norm is defined as:
- `p = 1`: sum of absolute values
- `p = 2`: Euclidean norm (default)
- `p = Inf`: maximum absolute value
- `p = -Inf`: minimum absolute value
- `p = 0`: count of non-zero elements
- `p > 0`: (Σ|xᵢ|^p)^(1/p)

For matrices, special norms are supported:
- `p = 1`: maximum absolute column sum
- `p = Inf`: maximum absolute row sum
- `p = "fro"` or `p = :fro`: Frobenius norm
- `p = 2`: spectral norm (not implemented, as it requires SVD)

This implementation does not depend on BLAS or LAPACK and uses scaling
algorithms to prevent overflow and underflow in floating-point computations.

# Examples

```julia
# Vector norms
v = [3.0, 4.0]
norm(v)       # 5.0
norm(v, 1)    # 7.0
norm(v, Inf)  # 4.0

# Matrix norms
A = [1 2 3; 4 5 6]
norm(A, 1)     # 9.0
norm(A, Inf)   # 15.0
norm(A, :fro)  # 9.539392014169456
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
    m, n = size(A)
    if m == 0 || n == 0
        return float(abs(zero(eltype(A))))
    end

    # Maximum absolute column sum
    maxsum = zero(float(real(eltype(A))))
    for j in 1:n
        colsum = zero(float(real(eltype(A))))
        for i in 1:m
            colsum += abs(A[i, j])
        end
        maxsum = max(maxsum, colsum)
    end
    return maxsum
end

function normInf_matrix(A::AbstractMatrix)
    m, n = size(A)
    if m == 0 || n == 0
        return float(abs(zero(eltype(A))))
    end

    # Maximum absolute row sum
    maxsum = zero(float(real(eltype(A))))
    for i in 1:m
        rowsum = zero(float(real(eltype(A))))
        for j in 1:n
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
