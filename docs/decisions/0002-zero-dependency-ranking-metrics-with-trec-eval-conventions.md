# Zero-dependency ranking metrics with trec_eval conventions

Status: accepted (generated, pending human review)

## Context

We need a Python library for evaluating retrieval runs with nDCG@k, MAP, MRR, P@k, plus significance tests, that is easy to use in CI without installing heavy dependencies.

## Decision

Implement all functionality using only the Python standard library, with scipy as a dev-only oracle for testing. Adopt trec_eval conventions exactly: linear gain for nDCG, discount log2(rank+1), tie-breaking by docid descending, and handling of missing queries.

## Consequences

The package has no runtime dependencies, simplifying deployment. However, the Student t CDF and incomplete beta functions must be implemented from scratch, which is done with a continued fraction method. The randomization test enumerates all sign assignments for n ≤ 20 and uses Monte Carlo otherwise.
