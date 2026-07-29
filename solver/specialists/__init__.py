"""
Importing this package registers every category's verification check with
solver.verification. main.py (or any test) should `import solver.specialists`
once before calling verify_candidate/SolverEngine.solve for the checks to
be active — plain imports below are intentional, not unused.
"""

from solver.specialists import code, cryptic, data, documents, optimization  # noqa: F401
