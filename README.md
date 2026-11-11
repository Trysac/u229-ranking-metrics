# ranking-metrics

Ranking metrics for TREC qrels and run files. The library parses the standard four-field qrels format and six-field run format, computes nDCG@k, MAP, MRR and P@k with documented trec_eval conventions, and provides a paired t-test and a randomization test for comparing two systems. It uses only the Python standard library at runtime.

## Features

- Parses TREC qrels lines `qid iter docid rel` and run lines `qid Q0 docid rank score tag`.
- Computes nDCG@k, MAP, MRR and P@k per query and their means.
- Orders each query's run by descending float score, with ties broken by descending docid; the run rank column is ignored.
- Skips qrels queries with no judged document at or above `relevance_level`; with `missing_as_zero=True`, qrels queries absent from the run are evaluated as empty rankings.
- Provides a paired two-sided Student t-test and a two-sided randomization test.
- Enforces a configurable maximum number of input lines in both parsers.
- Provides the `python -m ranking_metrics` CLI with `eval` and `compare` subcommands.
- Attaches a `NullHandler` to the package logger and has no runtime dependencies beyond the standard library.

## Installation

```bash
pip install trysac-ranking-metrics
```

The import package is `ranking_metrics`.

## Usage

```python
from ranking_metrics import parse_qrels, parse_run, evaluate

qrels = parse_qrels("q1 0 d1 3\nq1 0 d2 2\nq1 0 d3 0\nq1 0 d4 1\n")
run = parse_run("q1 Q0 d3 1 0.9 tag\nq1 Q0 d1 2 0.8 tag\nq1 Q0 d4 3 0.7 tag\nq1 Q0 d2 4 0.6 tag\n")
res = evaluate(qrels, run, k=4)
print(res.per_query["q1"])
```

CLI examples:

```bash
python -m ranking_metrics eval qrels.txt run.txt --k 10
python -m ranking_metrics compare qrels.txt run_a.txt run_b.txt --metric ndcg@10 --k 10 --trials 10000 --seed 0
```

## API

- `parse_qrels(text, *, max_lines=5_000_000) -> dict[str, dict[str, int]]`
- `parse_run(text, *, max_lines=5_000_000) -> dict[str, list[str]]`
- `evaluate(qrels, run, *, k=10, relevance_level=1, missing_as_zero=False) -> EvalResult`
  - `EvalResult.per_query: dict[str, dict[str, float]]`
  - `EvalResult.mean: dict[str, float]`
- `paired_t_test(a: Mapping[str, float], b: Mapping[str, float]) -> TTest`
  - `TTest.t: float`, `TTest.df: int`, `TTest.p: float`
- `randomization_test(a, b, *, trials=10_000, seed=0) -> float`
- `main(argv: Sequence[str] | None = None) -> int`

`RankingError` has a `code` attribute with one of: `parse_error`, `duplicate_entry`, `input_too_large`, `query_mismatch`, `too_few_queries`, `invalid_options`. Parser error messages quote at most 200 characters of the offending line. Parsers reject duplicate `(qid, docid)` pairs and inputs whose line count exceeds `max_lines`. `evaluate` rejects `k <= 0` or `relevance_level <= 0`. `paired_t_test` and `randomization_test` require the same query id set and at least two queries. `randomization_test` enumerates all `2**n` sign assignments exactly when `n <= 20`; otherwise it uses `trials` random flips plus the observed assignment, seeded by `seed`.

## Conventions

- nDCG@k uses linear gain `rel` and discount `log2(rank + 1)`, with ideal DCG from the query's judged relevance values sorted descending and cut at `k`. Negative relevance values contribute zero gain.
- MAP uses binary relevance (`rel >= relevance_level`) over the full ranking and divides by the number of relevant judged documents.
- MRR is `1 / rank` of the first relevant document, or 0 if none.
- P@k divides relevant hits in the top `k` by `k`; `k` larger than the ranking is allowed.
- Run ordering sorts by descending score, with ties by descending docid.

## Why not pytrec_eval or ranx

`pytrec_eval` and `ranx` are established alternatives. This package is for cases where a zero-runtime-dependency install is required; it states and tests the tie-breaking and gain conventions it implements, and it includes the significance tests in the same small package.

## Testing

Run `pytest`. The test module is `tests/test_ranking_metrics.py`; it covers the acceptance values for `evaluate`, docid tie-breaking, paired t-test results, zero-variance t-test behavior, and query-set mismatch errors. A development test compares `paired_t_test` p-values with `scipy.stats.ttest_rel` on seeded vectors; scipy is a dev-only dependency.

See [docs/SECURITY.md](docs/SECURITY.md) for the threat model.
