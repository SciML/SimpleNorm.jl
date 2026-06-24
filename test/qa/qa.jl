using SimpleNorm
using Aqua
using JET
using ExplicitImports
using SciMLTesting

run_qa(
    SimpleNorm;
    Aqua = Aqua,
    JET = JET,
    jet = true,
    ExplicitImports = ExplicitImports,
    explicit_imports = true,
)
