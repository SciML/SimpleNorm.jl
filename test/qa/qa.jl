using SciMLTesting, SimpleNorm, Test
using JET

run_qa(SimpleNorm; explicit_imports = true, api_docs_kwargs = (; rendered = true))
