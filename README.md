# SimpleNorm.jl

A lightweight implementation of norm functions without LinearAlgebra.jl, BLAS, or LAPACK dependencies.

## Features

This package provides pure Julia implementations of common norm functions:

### Vector norms
- `norm(x)` or `norm(x, 2)` - Euclidean norm (default)
- `norm(x, 1)` - 1-norm (sum of absolute values)
- `norm(x, Inf)` - Infinity norm (maximum absolute value)
- `norm(x, -Inf)` - Minimum absolute value
- `norm(x, p)` - p-norm for any `p ≥ 1`
- `norm(x, 0)` - Count of non-zero elements

### Matrix norms
- `norm(A, 1)` - Maximum absolute column sum
- `norm(A, Inf)` - Maximum absolute row sum
- `norm(A, "fro")` or `norm(A, :fro)` - Frobenius norm

Note: The spectral norm (`norm(A, 2)`) is not implemented as it requires singular value decomposition.

## Installation

```julia
using Pkg
Pkg.add("SimpleNorm")
```

## Usage

```julia
using SimpleNorm

# Vector norms
v = [3.0, 4.0]
norm(v)       # 5.0 (Euclidean norm)
norm(v, 1)    # 7.0 (1-norm)
norm(v, Inf)  # 4.0 (infinity norm)

# Matrix norms
A = [1 2 3; 4 5 6]
norm(A, 1)    # 9.0 (max column sum)
norm(A, Inf)  # 15.0 (max row sum)
norm(A, "fro") # Frobenius norm
```

## Why SimpleNorm.jl?

This package is useful when you need norm computations but want to avoid:
- Binary dependencies (BLAS/LAPACK)
- Heavy LinearAlgebra.jl dependency
- Deployment issues in constrained environments

All implementations are in pure Julia with careful attention to numerical stability, including overflow/underflow prevention in the 2-norm calculation.
