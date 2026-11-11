# Architecture

`ranking_metrics` is a small, dependency-free Python library for evaluating ranked retrieval systems. It implements the core metrics used in TREC-style evaluation and provides statistical significance tests. The design prioritizes:

- **Zero runtime dependencies**: Only the Python standard library is used. This simplifies deployment in CI environments.
- **Trec_eval compatibility**: Code follows trec_eval conventions for metric definitions and run ordering. Tests include acceptance values computed to match reference implementations.
- **Explicit error handling**: All input errors raise a `RankingError` with a machine-readable code, and messages are truncated to avoid leaking sensitive data.
- **Resource limits**: Parsing functions enforce a maximum number of lines (default 5 million) to prevent excessive memory consumption.

## Key Decisions

- **Linear gain for nDCG**: The specification requires linear relevance scaling (not 2^rel - 1) to match trec_eval defaults.
- **Student t CDF via incomplete beta**: Implementing the CDF without scipy keeps the library self-contained. The continued fraction method (Lentz) is used with a relative tolerance of 1e-14 and a maximum of 300 iterations.
- **Randomization test enumeration**: For n ≤ 20, all 2^n sign assignments are enumerated exactly; otherwise Monte Carlo with seed for reproducibility.
- **CLI separate from library**: The `main` function uses `argparse` to provide an easy interface but remains within the package as a callable for testing.
