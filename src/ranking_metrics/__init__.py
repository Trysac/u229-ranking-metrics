from __future__ import annotations

import logging
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "EvalResult",
    "RankingError",
    "TTest",
    "evaluate",
    "main",
    "paired_t_test",
    "parse_qrels",
    "parse_run",
    "randomization_test",
]

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


class RankingError(Exception):
    """Base exception for ranking_metrics errors."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class EvalResult:
    per_query: dict[str, dict[str, float]]
    mean: dict[str, float]


@dataclass(frozen=True)
class TTest:
    t: float
    df: int
    p: float


def _excerpt(line: str, max_len: int = 200) -> str:
    if len(line) <= max_len:
        return line
    return line[: max_len - 3] + "..."


def parse_qrels(text: str, *, max_lines: int = 5_000_000) -> dict[str, dict[str, int]]:
    """Parse TREC qrels format into {query_id: {doc_id: relevance}}.

    Each non-empty line must contain exactly 4 whitespace-separated fields:
    qid, iter (ignored), docid, rel (integer).

    Args:
        text: qrels text.
        max_lines: maximum allowed number of lines (including blank/comment lines).

    Returns:
        Mapping from query id to a mapping of document id to relevance score.

    Raises:
        RankingError: with code 'parse_error', 'duplicate_entry', or 'input_too_large'.
    """
    lines = text.splitlines()
    if len(lines) > max_lines:
        raise RankingError("input_too_large", f"input exceeds {max_lines} lines")
    result: dict[str, dict[str, int]] = {}
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 4:
            raise RankingError(
                "parse_error",
                f"line {line_no}: expected 4 fields, got {len(parts)}: {_excerpt(stripped)}",
            )
        qid, _, docid, rel_str = parts
        try:
            rel = int(rel_str)
        except ValueError:
            raise RankingError(
                "parse_error",
                f"line {line_no}: relevance score is not an integer: {_excerpt(stripped)}",
            ) from None
        if qid in result and docid in result[qid]:
            raise RankingError(
                "duplicate_entry",
                f"line {line_no}: duplicate (qid, docid) pair: {_excerpt(stripped)}",
            )
        result.setdefault(qid, {})[docid] = rel
    return result


def parse_run(text: str, *, max_lines: int = 5_000_000) -> dict[str, list[str]]:
    """Parse a TREC run file into {query_id: [doc_ids ordered by score desc, tie docid desc]}.

    Each non-empty line must contain exactly 6 whitespace-separated fields:
    qid, Q0, docid, rank, score, tag. The Q0, rank, and tag are ignored for ordering;
    result lists are sorted by descending score, with ties broken by descending docid
    (as trec_eval does).

    Args:
        text: run text.
        max_lines: maximum allowed number of lines.

    Returns:
        Mapping from query id to an ordered list of document ids.

    Raises:
        RankingError: with code 'parse_error', 'duplicate_entry', or 'input_too_large'.
    """
    lines = text.splitlines()
    if len(lines) > max_lines:
        raise RankingError("input_too_large", f"input exceeds {max_lines} lines")
    tmp: dict[str, list[tuple[float, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 6:
            raise RankingError(
                "parse_error",
                f"line {line_no}: expected 6 fields, got {len(parts)}: {_excerpt(stripped)}",
            )
        qid, _q0, docid, _rank, score_str, _tag = parts
        try:
            score = float(score_str)
        except ValueError:
            raise RankingError(
                "parse_error",
                f"line {line_no}: score is not a float: {_excerpt(stripped)}",
            ) from None
        if (qid, docid) in seen:
            raise RankingError(
                "duplicate_entry",
                f"line {line_no}: duplicate (qid, docid) pair: {_excerpt(stripped)}",
            )
        seen.add((qid, docid))
        tmp.setdefault(qid, []).append((score, docid))
    result: dict[str, list[str]] = {}
    for qid, docs in tmp.items():
        # sort by score descending, then docid descending
        docs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        result[qid] = [docid for _, docid in docs]
    return result


def _dcg(relevance_values: Sequence[float], k: int) -> float:
    """Compute DCG@k for a sequence of relevance values (already in ranking order)."""
    dcg = 0.0
    for i, rel in enumerate(relevance_values[:k], start=1):
        dcg += rel / math.log2(i + 1)
    return dcg


def _idcg(relevance_values: Sequence[float], k: int) -> float:
    """Compute ideal DCG@k from a list of relevance values (sorted descending)."""
    sorted_rels = sorted(relevance_values, reverse=True)
    return _dcg(sorted_rels, k)


def _average_precision(
    run_docs: Sequence[str], rels: Mapping[str, int], relevance_level: int
) -> float:
    """Compute Average Precision over the entire run."""
    relevant_docs = [doc for doc, rel in rels.items() if rel >= relevance_level]
    total_relevant = len(relevant_docs)
    if total_relevant == 0:
        return 0.0
    num_rel_so_far = 0
    ap = 0.0
    for i, doc in enumerate(run_docs, start=1):
        if doc in rels and rels[doc] >= relevance_level:
            num_rel_so_far += 1
            ap += num_rel_so_far / i
    return ap / total_relevant


def _reciprocal_rank(
    run_docs: Sequence[str], rels: Mapping[str, int], relevance_level: int
) -> float:
    """Reciprocal rank of first relevant document, 0 if none."""
    for i, doc in enumerate(run_docs, start=1):
        if doc in rels and rels[doc] >= relevance_level:
            return 1.0 / i
    return 0.0


def _precision_at_k(
    run_docs: Sequence[str], rels: Mapping[str, int], relevance_level: int, k: int
) -> float:
    """Precision at k."""
    if k <= 0:
        return 0.0
    hits = sum(1 for doc in run_docs[:k] if doc in rels and rels[doc] >= relevance_level)
    return hits / k


def evaluate(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    *,
    k: int = 10,
    relevance_level: int = 1,
    missing_as_zero: bool = False,
) -> EvalResult:
    """Evaluate a run against qrels.

    Computes nDCG@k, MAP, MRR, and P@k for each query, following trec_eval conventions:
    - nDCG uses linear gain (rel) and discount log2(rank + 1), ideal DCG from sorted
      relevance values, cut at k.
    - MAP uses binary relevance (rel >= relevance_level) over the full ranking.
    - MRR is reciprocal rank of first relevant document.
    - P@k is proportion of relevant in top k.
    Queries with no relevant judged documents are skipped.
    If missing_as_zero is False (default), only queries present in both qrels and run are evaluated.
    If True, queries in qrels but absent from run are treated as having an empty ranking.

    Args:
        qrels: Result of parse_qrels.
        run: Result of parse_run.
        k: Cutoff for nDCG and P.
        relevance_level: Minimum relevance to be considered relevant (default 1).
        missing_as_zero: Include queries absent from run as zeros.

    Returns:
        EvalResult with per-query metrics and their means.

    Raises:
        RankingError: with code 'invalid_options' if k <= 0 or relevance_level <= 0.
    """
    if k <= 0:
        raise RankingError("invalid_options", "k must be positive")
    if relevance_level <= 0:
        raise RankingError("invalid_options", "relevance_level must be positive")

    # Determine queries to evaluate: those in qrels with at least one relevant judged doc.
    qids_to_eval = []
    for qid, rels in qrels.items():
        if any(rel >= relevance_level for rel in rels.values()):
            qids_to_eval.append(qid)

    per_query: dict[str, dict[str, float]] = {}
    for qid in qids_to_eval:
        rels = qrels[qid]
        run_docs = run.get(qid)
        if not missing_as_zero and run_docs is None:
            continue
        if run_docs is None:
            run_docs = []

        # nDCG@k
        dcg = _dcg([max(0, rels.get(doc, 0)) for doc in run_docs], k)
        ideal_rels = [max(0, rel) for rel in rels.values()]
        idcg = _idcg(ideal_rels, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # MAP
        ap = _average_precision(run_docs, rels, relevance_level)

        # MRR
        mrr = _reciprocal_rank(run_docs, rels, relevance_level)

        # P@k
        p_at_k = _precision_at_k(run_docs, rels, relevance_level, k)

        per_query[qid] = {
            f"ndcg@{k}": ndcg,
            "map": ap,
            "mrr": mrr,
            f"p@{k}": p_at_k,
        }

    if not per_query:
        return EvalResult(per_query={}, mean={})

    metric_names = list(next(iter(per_query.values())).keys())
    mean: dict[str, float] = {}
    for metric in metric_names:
        values = [per_query[qid][metric] for qid in per_query]
        mean[metric] = sum(values) / len(values)
    return EvalResult(per_query=per_query, mean=mean)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta function (Lentz's algorithm)."""
    MAX_ITER = 300
    EPS = 1e-14
    FPMIN = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < EPS:
            break
    else:
        raise RuntimeError("Continued fraction failed to converge")
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: int) -> float:
    """Cumulative distribution function of Student's t distribution."""
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    if t >= 0:
        return 1.0 - 0.5 * ib
    else:
        return 0.5 * ib


def paired_t_test(a: Mapping[str, float], b: Mapping[str, float]) -> TTest:
    """Paired two-sided Student's t-test on common query ids.

    Args:
        a: Per-query metric values for system A.
        b: Per-query metric values for system B.

    Returns:
        TTest with t-statistic, degrees of freedom, and two-sided p-value.

    Raises:
        RankingError: 'query_mismatch' if id sets differ, 'too_few_queries' if fewer than 2 ids.
    """
    if set(a.keys()) != set(b.keys()):
        raise RankingError("query_mismatch", "query id sets do not match")
    ids = sorted(a.keys())
    n = len(ids)
    if n < 2:
        raise RankingError("too_few_queries", "need at least 2 queries for paired t-test")
    diffs = [a[qid] - b[qid] for qid in ids]
    mean_diff = sum(diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    if variance == 0.0:
        if abs(mean_diff) < 1e-12:
            return TTest(t=0.0, df=n - 1, p=1.0)
        else:
            t = float("inf") if mean_diff > 0 else -float("inf")
            return TTest(t=t, df=n - 1, p=0.0)
    t = mean_diff / math.sqrt(variance / n)
    p = 2.0 * (1.0 - _student_t_cdf(abs(t), n - 1))
    return TTest(t=t, df=n - 1, p=p)


def randomization_test(
    a: Mapping[str, float],
    b: Mapping[str, float],
    *,
    trials: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided randomization test on per-query differences.

    The null hypothesis is that the two systems are equivalent, so differences
    a[i] - b[i] can be randomly sign-flipped. The statistic is |sum(differences)|.

    If n <= 20, enumerate all 2**n sign assignments exactly. Otherwise use `trials`
    random sign flips plus the observed assignment.

    Args:
        a: Per-query metric values for system A.
        b: Per-query metric values for system B.
        trials: Number of random permutations for Monte Carlo (used only when n > 20).
        seed: Seed for random.Random.

    Returns:
        Two-sided p-value.

    Raises:
        RankingError: 'query_mismatch' if id sets differ, 'too_few_queries' if fewer than 2 ids.
    """
    if set(a.keys()) != set(b.keys()):
        raise RankingError("query_mismatch", "query id sets do not match")
    ids = sorted(a.keys())
    diffs = [a[qid] - b[qid] for qid in ids]
    n = len(diffs)
    if n < 2:
        raise RankingError("too_few_queries", "need at least 2 queries for randomization test")
    observed = abs(sum(diffs))
    tol = 1e-12
    count = 0
    if n <= 20:
        total_assignments: int = 2**n
        for bits in range(total_assignments):
            s = 0.0
            for i, d in enumerate(diffs):
                if (bits >> i) & 1:
                    s += d
                else:
                    s -= d
            if abs(s) >= observed - tol:
                count += 1
        return count / total_assignments
    else:
        rng = random.Random(seed)  # noqa: S311
        total_assignments = trials + 1  # observed assignment is always included
        count_random = 0
        for _ in range(trials):
            s = 0.0
            for d in diffs:
                if rng.random() < 0.5:
                    s += d
                else:
                    s -= d
            if abs(s) >= observed - tol:
                count_random += 1
        # observed assignment corresponds to all + signs; its |sum| equals observed, so it counts.
        return (count_random + 1) / total_assignments


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line interface.

    Two subcommands:
      eval <qrels> <run> [--k K]
      compare <qrels> <run_a> <run_b> --metric METRIC [--k K] [--trials N] [--seed S]

    Returns:
        0 on success, 2 on any error (usage, I/O, parse).
    """
    import argparse

    parser = argparse.ArgumentParser(prog="ranking-metrics")
    subparsers = parser.add_subparsers(dest="command")

    # eval
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("qrels")
    eval_parser.add_argument("run")
    eval_parser.add_argument("--k", type=int, default=10)

    # compare
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("qrels")
    compare_parser.add_argument("run_a")
    compare_parser.add_argument("run_b")
    compare_parser.add_argument("--metric", required=True)
    compare_parser.add_argument("--k", type=int, default=10)
    compare_parser.add_argument("--trials", type=int, default=10000)
    compare_parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    try:
        if args.command == "eval":
            with open(args.qrels) as f:
                qrels_text = f.read()
            with open(args.run) as f:
                run_text = f.read()
            qrels = parse_qrels(qrels_text)
            run = parse_run(run_text)
            res = evaluate(qrels, run, k=args.k)
            if not res.per_query:
                # no queries evaluated, print nothing
                return 0
            metric_names = sorted(next(iter(res.per_query.values())).keys())
            for metric in metric_names:
                print(f"{metric}\t{res.mean[metric]:.4f}")
            return 0

        elif args.command == "compare":
            with open(args.qrels) as f:
                qrels_text = f.read()
            with open(args.run_a) as f:
                run_a_text = f.read()
            with open(args.run_b) as f:
                run_b_text = f.read()
            qrels = parse_qrels(qrels_text)
            run_a = parse_run(run_a_text)
            run_b = parse_run(run_b_text)
            eval_a = evaluate(qrels, run_a, k=args.k)
            eval_b = evaluate(qrels, run_b, k=args.k)
            if args.metric not in eval_a.mean or args.metric not in eval_b.mean:
                raise RankingError("invalid_options", f"unknown metric: {args.metric}")
            mean_a = eval_a.mean[args.metric]
            mean_b = eval_b.mean[args.metric]
            per_query_a = {qid: metrics[args.metric] for qid, metrics in eval_a.per_query.items()}
            per_query_b = {qid: metrics[args.metric] for qid, metrics in eval_b.per_query.items()}
            ttest = paired_t_test(per_query_a, per_query_b)
            p_rand = randomization_test(
                per_query_a, per_query_b, trials=args.trials, seed=args.seed
            )
            print(f"mean_a\t{mean_a:.4f}")
            print(f"mean_b\t{mean_b:.4f}")
            print(f"t\t{ttest.t:.4f}")
            print(f"p_t\t{ttest.p:.4f}")
            print(f"p_rand\t{p_rand:.4f}")
            return 0
        else:
            return 2
    except RankingError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
