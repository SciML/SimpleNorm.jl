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

```jldoctest
julia> using SimpleNorm

julia> norm([3.0, 4.0])
5.0

julia> norm([3.0, 4.0], 1)
7.0

julia> norm([3.0, 4.0], Inf)
4.0
```

## Array contract

`norm` accepts ordinary arrays and custom `AbstractArray` implementations. A
custom array must implement the public Base array operations used by its shape:
iteration, `isempty`, `eltype`, and scalar indexing; matrices must additionally
implement `size` and two-dimensional scalar indexing. There is no
SimpleNorm-specific abstract type to subtype or method to extend. Define the
normal Base array interface, then call `norm`.

## Supported norms

### Vector norms

- `norm(x)` or `norm(x, 2)` — Euclidean norm (default)
- `norm(x, 1)` — 1-norm (sum of absolute values)
- `norm(x, Inf)` — infinity norm (maximum absolute value)
- `norm(x, -Inf)` — minimum absolute value
- `norm(x, p)` — scaled positive-order norm for any `p > 0`
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
