# Project record

## Objective

Rank evaluation over TREC qrels and run files: nDCG@k, MAP, MRR and P@k with trec_eval conventions, plus a paired t-test and an exact or seeded randomization test.

## Scope

- In scope: parsing TREC qrels and run text; nDCG@k with linear gain and `log2(rank + 1)` discount; MAP with binary relevance; MRR; P@k; per-query results and means; paired two-sided Student t-test; two-sided randomization test with exact enumeration for `n <= 20` and seeded Monte Carlo otherwise; `RankingError` with documented codes; the `eval` and `compare` CLI subcommands.
- Out of scope: other trec_eval measures, graded MAP, bootstrap confidence intervals, multiple-comparison correction, and runtime dependencies beyond the Python standard library.

## Timeline

| Date | Event |
|---|---|
| 2026 | Repository created from the factory utility template. |

## Effort metrics

Measured values only; a value that has not been measured stays "pending measurement".

| Metric | Value |
|---|---|
| Tokens (input / output) | 1430669 / 95687 |
| Hours (human) | pending measurement |
| Cost (USD) | 0.1243 |

## Factory run u229 (measured from API usage)

| Date (UTC) | Phase | Model | Batch | Input | Output | Cache write | Cache read | Cost (USD) |
|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | generation | deepseek-v4-pro | yes | 1970 | 24389 | 0 | 3328 | 0.0497 |
| 2026-11-11 | fix-1 | deepseek-flash | no | 20992 | 19426 | 0 | 488960 | 0.0163 |
| 2026-11-11 | fix-2 | deepseek-flash | no | 12669 | 5318 | 0 | 91392 | 0.0054 |
| 2026-11-11 | fix-3 | deepseek-flash | no | 16441 | 4348 | 0 | 61568 | 0.0053 |
| 2026-11-11 | audit | deepseek-flash | no | 27658 | 31017 | 0 | 454016 | 0.0241 |
| 2026-11-11 | fix-4 | deepseek-flash | no | 14460 | 3438 | 0 | 141824 | 0.0047 |
| 2026-11-11 | docs | deepseek-flash | no | 95135 | 7751 | 0 | 256 | 0.0189 |

Total: $0.1243. Prices: deepseek line, as measured by its adapter. Providers serve the current version of each model id, so the date tells which version answered; versions, launch stages and price sources per line are in the factory's model register (docs/MODELS.md).
