# Testing

## Running

```sh
pytest
```

The test module is `tests/test_ranking_metrics.py`. Tests import the public package as `ranking_metrics` and assert return values and `RankingError.code` values.

## Policy

- Tests are deterministic and use local string fixtures; no network access is required.
- Tests exercise the public API and assert observable results, including exact metric values, t-test results, and error codes.
- A development comparison checks `paired_t_test` p-values against `scipy.stats.ttest_rel` on 50 seeded random vectors within `1e-9`; scipy is a dev-only dependency and is not required at runtime.

## Tiers

The repository contains one local pytest suite. No separate download or conformance tier is present in the checked-in files.

## Layout

- `tests/test_ranking_metrics.py` mirrors the public package.
- The recorded test functions cover: `test_evaluate_acceptance_k4`, `test_evaluate_acceptance_k2`, `test_tie_breaking_docid_desc`, `test_paired_t_test_acceptance`, `test_paired_t_test_zero_variance_zero_mean`, and `test_paired_t_test_mismatch`.
