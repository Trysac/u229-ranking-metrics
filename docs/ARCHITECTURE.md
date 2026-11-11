# Architecture

## Purpose

`trysac-ranking-metrics` is a small, dependency-free Python library: Rank evaluation over TREC qrels and run files: nDCG@k, MAP, MRR and P@k with trec_eval conventions, plus a paired t-test and an exact or seeded randomization test

## Shape

```text
caller code --> ranking_metrics (public API in __init__.py)
                    |
                    +-- pure functions, no I/O, no global state
```

- **Public API**: the names listed in `__all__` of `src/ranking_metrics/__init__.py`. Anything else is
  internal and may change without a major version.
- **Errors**: invalid input raises built-in exceptions (`ValueError`, `TypeError`) with a clear
  message; the library never prints or exits.
- **Side effects**: none. If a feature needs I/O, isolate it in its own module behind a small
  interface and record the choice in an ADR.

## Quality contract

`scripts/check.py` is the single definition of the gates; `fabrica.json` tells the factory and CI
how to run it. See [TESTING.md](TESTING.md).

## Decisions

Architecture decisions are recorded in [decisions/](decisions/) as numbered ADRs.
