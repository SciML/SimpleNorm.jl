# SimpleNorm.jl

SimpleNorm.jl is a lightweight implementation of norm functions without
`LinearAlgebra.jl`, BLAS, or LAPACK dependencies. It provides pure-Julia
`norm` implementations that are useful in dependency-constrained or
precompilation-sensitive contexts.

## Installation

```julia
using Pkg
Pkg.add("SimpleNorm")
```

## Usage

```julia
using SimpleNorm

norm([3.0, 4.0])        # 5.0  (Euclidean / 2-norm, the default)
norm([3.0, 4.0], 1)     # 7.0  (1-norm)
norm([3.0, 4.0], Inf)   # 4.0  (infinity norm)
```

## Supported norms

### Vector norms

- `norm(x)` or `norm(x, 2)` — Euclidean norm (default)
- `norm(x, 1)` — 1-norm (sum of absolute values)
- `norm(x, Inf)` — infinity norm (maximum absolute value)
- `norm(x, -Inf)` — minimum absolute value
- `norm(x, p)` — p-norm for any `p ≥ 1`
- `norm(x, 0)` — count of non-zero elements

### Matrix norms

- `norm(A, 1)` — maximum absolute column sum
- `norm(A, Inf)` — maximum absolute row sum
- `norm(A, "fro")` or `norm(A, :fro)` — Frobenius norm

!!! note
    The spectral norm (`norm(A, 2)`) is not implemented, as it requires a
    singular value decomposition.

## API

```@docs
SimpleNorm.norm
```
