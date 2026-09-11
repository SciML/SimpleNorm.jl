using SimpleNorm, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

x = rand(rng, 10_000)
xm = rand(rng, 300, 300)

# =============================================================================
# Vector norms
# =============================================================================

SUITE["vector"] = BenchmarkGroup()

SUITE["vector"]["norm2"] = @benchmarkable norm($x)
SUITE["vector"]["norm1"] = @benchmarkable norm($x, 1)
SUITE["vector"]["normInf"] = @benchmarkable norm($x, Inf)
SUITE["vector"]["norm3"] = @benchmarkable norm($x, 3)

# =============================================================================
# Matrix norms
# =============================================================================

SUITE["matrix"] = BenchmarkGroup()

SUITE["matrix"]["norm1"] = @benchmarkable norm($xm, 1)
SUITE["matrix"]["normInf"] = @benchmarkable norm($xm, Inf)
