# Project Instructions

- Run ABACUS runtime tests, MPI tests, and any test where process visibility matters outside the sandbox using an escalated command. This avoids sandbox interference with MPI launch, process inspection, and runtime behavior.
- Set `OMP_NUM_THREADS=1` when running ABACUS tests unless a test explicitly requires a different OpenMP thread count.

# Coding Guidelines

- Follow the local coding guidelines in `docs/developers_guide/coding_guidelines.md`.
- Key points: avoid new dependencies on `GlobalV`, `GlobalC`, or `PARAM`; avoid hidden control-flow state in class members; keep header dependencies light; avoid heavy `.hpp` dependencies; be careful around EXX-related dependency chains; do not add new default arguments to existing functions; and add short, focused tests for important behavior.
