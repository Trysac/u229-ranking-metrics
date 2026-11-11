from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import ranking_metrics.__main__ as package_main
from ranking_metrics import (
    RankingError,
    evaluate,
    main,
    paired_t_test,
    parse_qrels,
    parse_run,
    randomization_test,
)

QRELS_TEXT = """\
q1 0 d1 3
q1 0 d2 2
q1 0 d3 0
q1 0 d4 1
"""
RUN_TEXT = """\
q1 Q0 d3 1 0.9 tag
q1 Q0 d1 2 0.8 tag
q1 Q0 d4 3 0.7 tag
q1 Q0 d2 4 0.6 tag
"""


def test_evaluate_acceptance_k4():
    qrels = parse_qrels(QRELS_TEXT)
    run = parse_run(RUN_TEXT)
    res = evaluate(qrels, run, k=4)
    q1_metrics = res.per_query["q1"]
    assert q1_metrics["ndcg@4"] == pytest.approx(0.6833763936083916, rel=1e-12, abs=1e-12)
    assert q1_metrics["map"] == pytest.approx(0.6388888888888888, rel=1e-12, abs=1e-12)
    assert q1_metrics["mrr"] == 0.5
    assert q1_metrics["p@4"] == pytest.approx(0.75)


def test_evaluate_acceptance_k2():
    qrels = parse_qrels(QRELS_TEXT)
    run = parse_run(RUN_TEXT)
    res = evaluate(qrels, run, k=2)
    q1_metrics = res.per_query["q1"]
    assert q1_metrics["ndcg@2"] == pytest.approx(0.44412286644879784, rel=1e-12, abs=1e-12)
    assert q1_metrics["p@2"] == pytest.approx(0.5)
    # map and mrr should be unchanged (not cut at k)
    assert q1_metrics["map"] == pytest.approx(0.6388888888888888, rel=1e-12, abs=1e-12)
    assert q1_metrics["mrr"] == 0.5


def test_tie_breaking_docid_desc():
    run_text = "\n".join(
        [
            "q1 Q0 da 1 0.5 tag",
            "q1 Q0 db 2 0.5 tag",  # same score, rank column ignored
        ]
    )
    run = parse_run(run_text)
    assert run["q1"] == ["db", "da"]


def test_paired_t_test_acceptance():
    a = {"q1": 0.5, "q2": 0.6, "q3": 0.7, "q4": 0.8}
    b = {"q1": 0.4, "q2": 0.6, "q3": 0.5, "q4": 0.6}
    result = paired_t_test(a, b)
    assert result.t == pytest.approx(2.6111648393354674, rel=1e-9, abs=1e-9)
    assert result.df == 3
    assert result.p == pytest.approx(0.0796049808179063, rel=1e-9, abs=1e-9)


def test_paired_t_test_zero_variance_zero_mean():
    a = {"q1": 0.1, "q2": 0.1, "q3": 0.1, "q4": 0.1}
    b = {"q1": 0.1, "q2": 0.1, "q3": 0.1, "q4": 0.1}
    result = paired_t_test(a, b)
    assert result.t == 0.0
    assert result.df == 3
    assert result.p == 1.0


def test_paired_t_test_mismatch():
    a = {"q1": 0.5}
    b = {"q2": 0.5}
    with pytest.raises(RankingError) as exc_info:
        paired_t_test(a, b)
    assert exc_info.value.code == "query_mismatch"


def test_paired_t_test_too_few():
    a = {"q1": 0.5}
    b = {"q1": 0.5}
    with pytest.raises(RankingError) as exc_info:
        paired_t_test(a, b)
    assert exc_info.value.code == "too_few_queries"


def test_randomization_test_exact_enumeration():
    a = {"q1": 0.1, "q2": 0.2, "q3": 0.3, "q4": 0.4}
    b = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}
    p = randomization_test(a, b)
    assert p == 0.125


def test_randomization_test_identical():
    a = {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5}
    b = {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5}
    p = randomization_test(a, b)
    assert p == 1.0


def test_randomization_test_monte_carlo_seed_repeatable():
    # 25 queries -> n > 20, use Monte Carlo
    qids = [f"q{i}" for i in range(25)]
    a = {qid: (i % 5) * 0.01 for i, qid in enumerate(qids)}
    b = {qid: 0.0 for qid in qids}
    p1 = randomization_test(a, b, trials=10000, seed=42)
    p2 = randomization_test(a, b, trials=10000, seed=42)
    assert p1 == p2
    # p should be between 0 and 1
    assert 0.0 <= p1 <= 1.0


def test_randomization_test_mismatch():
    a = {"q1": 0.5}
    b = {"q2": 0.5}
    with pytest.raises(RankingError) as exc_info:
        randomization_test(a, b)
    assert exc_info.value.code == "query_mismatch"


def test_randomization_test_too_few():
    a = {"q1": 0.5}
    b = {"q1": 0.5}
    with pytest.raises(RankingError) as exc_info:
        randomization_test(a, b)
    assert exc_info.value.code == "too_few_queries"


def test_parse_run_wrong_fields():
    run_text = "q1 Q0 d1 1 0.5\n"  # only 5 fields
    with pytest.raises(RankingError) as exc_info:
        parse_run(run_text)
    assert exc_info.value.code == "parse_error"
    # message should include line number
    assert "line 1" in str(exc_info.value)


def test_parse_run_duplicate():
    run_text = "\n".join(
        [
            "q1 Q0 d1 1 0.5 tag",
            "q1 Q0 d1 2 0.6 tag",
        ]
    )
    with pytest.raises(RankingError) as exc_info:
        parse_run(run_text)
    assert exc_info.value.code == "duplicate_entry"


def test_parse_qrels_duplicate():
    qrels_text = "\n".join(
        [
            "q1 0 d1 3",
            "q1 0 d1 2",
        ]
    )
    with pytest.raises(RankingError) as exc_info:
        parse_qrels(qrels_text)
    assert exc_info.value.code == "duplicate_entry"


def test_parse_qrels_wrong_fields():
    qrels_text = "q1 0 d1\n"  # 3 fields
    with pytest.raises(RankingError) as exc_info:
        parse_qrels(qrels_text)
    assert exc_info.value.code == "parse_error"
    assert "line 1" in str(exc_info.value)


def test_input_too_large():
    text = "line1\nline2\nline3\nline4\n"
    with pytest.raises(RankingError) as exc_info:
        parse_qrels(text, max_lines=3)
    assert exc_info.value.code == "input_too_large"


def test_evaluate_edge_case_no_relevant_retrieved():
    qrels = parse_qrels("q1 0 d1 1\nq1 0 d2 1\n")
    run = parse_run("q1 Q0 d3 1 0.9 tag\nq1 Q0 d4 2 0.8 tag\n")
    res = evaluate(qrels, run, k=10)
    assert res.per_query["q1"]["ndcg@10"] == 0.0
    assert res.per_query["q1"]["map"] == 0.0
    assert res.per_query["q1"]["mrr"] == 0.0
    assert res.per_query["q1"]["p@10"] == 0.0


def test_evaluate_k_larger_than_ranking():
    qrels = parse_qrels("q1 0 d1 2\nq1 0 d2 1\n")
    run = parse_run("q1 Q0 d1 1 0.9 tag\nq1 Q0 d2 2 0.8 tag\n")
    res = evaluate(qrels, run, k=10)
    # ndcg@10 should be computed over all 2 docs; ideal DCG over 2 docs
    dcg = 2 / math.log2(2) + 1 / math.log2(3)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)  # best order
    assert res.per_query["q1"]["ndcg@10"] == pytest.approx(dcg / idcg, rel=1e-12)
    # p@10 = 2/10 = 0.2
    assert res.per_query["q1"]["p@10"] == pytest.approx(0.2)


def test_evaluate_relevance_level_2():
    qrels = parse_qrels("q1 0 d1 3\nq1 0 d2 2\nq1 0 d3 1\nq1 0 d4 0\n")
    run = parse_run(
        "q1 Q0 d3 1 0.9 tag\nq1 Q0 d1 2 0.8 tag\nq1 Q0 d2 3 0.7 tag\nq1 Q0 d4 4 0.6 tag\n"
    )
    # relevance_level=2: relevant docs are d1 (3) and d2 (2)
    res = evaluate(qrels, run, k=4, relevance_level=2)
    q1 = res.per_query["q1"]
    # ndcg@4 uses raw judged relevance gains (relevance_level does not change nDCG gains):
    # dcg = 1 at pos1 (d3 rel1) + 3 at pos2 (d1) + 2 at pos3 (d2) + 0 at pos4 (d4)
    dcg = 1 / math.log2(2) + 3 / math.log2(3) + 2 / math.log2(4) + 0 / math.log2(5)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4) + 0 / math.log2(5)
    assert q1["ndcg@4"] == pytest.approx(dcg / idcg, rel=1e-12)
    # map: relevant docs d1 and d2, positions 2 and 3 -> AP = (1/2 + 2/3)/2
    ap = (0.5 + 0.6666666666666666) / 2
    assert q1["map"] == pytest.approx(ap, rel=1e-12)
    # mrr: first relevant is d1 at rank2 -> 0.5
    assert q1["mrr"] == 0.5
    # p@4: hits=2 -> 0.5
    assert q1["p@4"] == pytest.approx(0.5)


def test_evaluate_negative_relevance_as_zero():
    qrels = parse_qrels("q1 0 d1 -1\nq1 0 d2 1\n")
    run = parse_run("q1 Q0 d1 1 0.9 tag\nq1 Q0 d2 2 0.8 tag\n")
    res = evaluate(qrels, run, k=2)
    q1 = res.per_query["q1"]
    # dcg: d1 rel 0 (negative -> 0), d2 rel 1: 0/log2(2) + 1/log2(3) = 0.6309
    dcg = 0.0 + 1.0 / math.log2(3)
    # ideal: rels sorted [1, 0] -> idcg = 1/log2(2) + 0/log2(3) = 1.0
    idcg = 1.0
    assert q1["ndcg@2"] == pytest.approx(dcg / idcg, rel=1e-12)
    # map: relevant docs with rel >=1: only d2, so total_relevant=1, ap = 0 (d2 is at position 2, precision = 1/2 = 0.5? Wait AP formula: sum precision at relevant docs / total_relevant. Only d2 relevant at pos2 -> precision at 2 = 1/2 = 0.5, total_relevant=1, so AP=0.5.
    assert q1["map"] == pytest.approx(0.5, rel=1e-12)
    # mrr: first relevant d2 at rank2 -> 0.5
    assert q1["mrr"] == 0.5
    # p@2: hits=1 -> 0.5
    assert q1["p@2"] == pytest.approx(0.5)


def test_evaluate_missing_as_zero():
    qrels = parse_qrels("q1 0 d1 1\nq2 0 d1 1\n")
    run = parse_run("q1 Q0 d1 1 0.9 tag\n")
    # without missing_as_zero: q2 not in run, skipped, only q1 evaluated
    res_no = evaluate(qrels, run, k=10)
    assert set(res_no.per_query.keys()) == {"q1"}
    # with missing_as_zero=True: q2 included with empty ranking => all metrics zero
    res_yes = evaluate(qrels, run, k=10, missing_as_zero=True)
    assert set(res_yes.per_query.keys()) == {"q1", "q2"}
    assert res_yes.per_query["q2"]["ndcg@10"] == 0.0
    assert res_yes.per_query["q2"]["map"] == 0.0
    assert res_yes.per_query["q2"]["mrr"] == 0.0
    assert res_yes.per_query["q2"]["p@10"] == 0.0


def test_evaluate_no_relevant_queries_skipped():
    qrels = parse_qrels("q1 0 d1 0\nq1 0 d2 0\n")
    run = parse_run("q1 Q0 d1 1 0.9 tag\n")
    res = evaluate(qrels, run, k=10)
    assert res.per_query == {}
    assert res.mean == {}


def test_evaluate_invalid_options():
    with pytest.raises(RankingError) as exc_info:
        evaluate({}, {}, k=0)
    assert exc_info.value.code == "invalid_options"
    with pytest.raises(RankingError) as exc_info:
        evaluate({}, {}, relevance_level=0)
    assert exc_info.value.code == "invalid_options"


def test_main_compare(tmp_path, capsys):
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("q1 0 d1 3\nq1 0 d2 2\nq1 0 d3 1\nq2 0 d1 2\nq2 0 d2 1\n")
    run_a_file = tmp_path / "a.txt"
    run_a_file.write_text(
        "q1 Q0 d1 1 0.9 tag\nq1 Q0 d2 2 0.8 tag\nq1 Q0 d3 3 0.7 tag\nq2 Q0 d1 1 0.9 tag\nq2 Q0 d2 2 0.8 tag\n"
    )
    run_b_file = tmp_path / "b.txt"
    run_b_file.write_text(
        "q1 Q0 d3 1 0.9 tag\nq1 Q0 d2 2 0.8 tag\nq1 Q0 d1 3 0.7 tag\nq2 Q0 d2 1 0.9 tag\nq2 Q0 d1 2 0.8 tag\n"
    )
    argv = [
        "compare",
        str(qrels_file),
        str(run_a_file),
        str(run_b_file),
        "--metric",
        "ndcg@10",
        "--k",
        "10",
        "--trials",
        "1000",
        "--seed",
        "0",
    ]
    ret = main(argv)
    captured = capsys.readouterr()
    out_lines = captured.out.strip().splitlines()
    assert ret == 0
    assert len(out_lines) == 5
    # each line should have two tab-separated parts with 4 decimal places
    for line in out_lines:
        key, val = line.split("\t")
        assert float(val) == pytest.approx(float(f"{float(val):.4f}"))
        assert key in ["mean_a", "mean_b", "t", "p_t", "p_rand"]
    # check the first line is mean_a
    assert out_lines[0].startswith("mean_a\t")


def test_main_compare_unknown_metric(tmp_path, capsys):
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("q1 0 d1 1\n")
    run_a_file = tmp_path / "a.txt"
    run_a_file.write_text("q1 Q0 d1 1 0.9 tag\n")
    run_b_file = tmp_path / "b.txt"
    run_b_file.write_text("q1 Q0 d1 1 0.8 tag\n")
    argv = [
        "compare",
        str(qrels_file),
        str(run_a_file),
        str(run_b_file),
        "--metric",
        "unknown",
    ]
    ret = main(argv)
    assert ret == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.err


def _acceptance_line_pattern(template: str) -> re.Pattern[str]:
    """Turn an acceptance line template such as 'mean_a<4 decimals>' into a regex."""
    text = template.replace("\\t", "\t")
    prefix, _, suffix = text.partition("<4 decimals>")
    return re.compile("^" + re.escape(prefix) + r"-?\d+\.\d{4}" + re.escape(suffix) + "$")


def test_main_compare_acceptance_line_formats(tmp_path, capsys):
    # The acceptance criterion requires the compare output to be exactly these lines.
    expected_templates = (
        r"mean_a\t<4 decimals>",
        r"mean_b\t<4 decimals>",
        r"t\t<4 decimals>",
        r"p_t\t<4 decimals>",
        r"p_rand\t<4 decimals>",
    )
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("q1 0 d1 3\nq1 0 d2 2\nq1 0 d3 1\nq2 0 d1 2\nq2 0 d2 1\n")
    run_a_file = tmp_path / "a.txt"
    run_a_file.write_text(
        "q1 Q0 d1 1 0.9 tag\nq1 Q0 d2 2 0.8 tag\nq1 Q0 d3 3 0.7 tag\n"
        "q2 Q0 d1 1 0.9 tag\nq2 Q0 d2 2 0.8 tag\n"
    )
    run_b_file = tmp_path / "b.txt"
    run_b_file.write_text(
        "q1 Q0 d3 1 0.9 tag\nq1 Q0 d2 2 0.8 tag\nq1 Q0 d1 3 0.7 tag\n"
        "q2 Q0 d2 1 0.9 tag\nq2 Q0 d1 2 0.8 tag\n"
    )
    argv = [
        "compare",
        str(qrels_file),
        str(run_a_file),
        str(run_b_file),
        "--metric",
        "ndcg@10",
        "--k",
        "10",
        "--trials",
        "1000",
        "--seed",
        "0",
    ]
    assert main(argv) == 0
    out_lines = capsys.readouterr().out.strip("\n").splitlines()
    assert len(out_lines) == len(expected_templates)
    for template, line in zip(expected_templates, out_lines, strict=True):
        assert _acceptance_line_pattern(template).match(line), (template, line)


def test_module_entry_point_exposes_main():
    # `python -m ranking_metrics` resolves to the package __main__ module.
    assert package_main.main is main


def test_module_entry_point_runs_as_subprocess(tmp_path):
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("q1 0 d1 1\n")
    run_file = tmp_path / "run.txt"
    run_file.write_text("q1 Q0 d1 1 0.9 tag\n")
    src_dir = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src_dir), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(src_dir)]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ranking_metrics",
            "eval",
            str(qrels_file),
            str(run_file),
            "--k",
            "10",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ndcg@10\t1.0000" in proc.stdout


def test_scipy_oracle_paired_t_test():
    import random

    try:
        from scipy import stats
    except ImportError:
        pytest.skip("scipy not installed")

    rng = random.Random(12345)  # noqa: S311
    for _ in range(50):
        n = rng.randint(2, 30)
        a = {f"q{i}": rng.gauss(0.5, 0.1) for i in range(n)}
        b = {f"q{i}": rng.gauss(0.5, 0.1) for i in range(n)}
        # occasionally make differences nonzero
        if rng.random() < 0.5:
            for qid in a:
                b[qid] += rng.gauss(0, 0.05)
        our = paired_t_test(a, b)
        scipy_res = stats.ttest_rel([a[qid] for qid in sorted(a)], [b[qid] for qid in sorted(b)])
        assert our.t == pytest.approx(scipy_res.statistic, rel=1e-9, abs=1e-9)
        assert our.df == scipy_res.df
        assert our.p == pytest.approx(scipy_res.pvalue, rel=1e-9, abs=1e-9)
