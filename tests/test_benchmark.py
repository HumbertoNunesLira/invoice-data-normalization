from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.benchmark import (
    build_result,
    compute_per_original_stats,
    constancy_for_group,
    load_rows,
    main,
    run,
)
from benchmark.weighted_distance import (
    constancy_with_weighted_distance,
    normalized_weighted_distance,
    run_benchmark_with_weighted_distance,
    weighted_edit_distance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "benchmark" / "fixtures" / "normalized_sample.csv"


def test_constancy_coca_example() -> None:
    norms = [
        "cocacola 2l",
        "cocacola 2l",
        "coca cola 2l",
        "coca cola 2l",
    ]
    c, sz, dom, counter = constancy_for_group(norms)
    assert sz == 4
    assert dom == 2
    assert c == pytest.approx(0.5)
    assert counter["cocacola 2l"] == 2
    assert counter["coca cola 2l"] == 2


def test_fixture_coca_and_agua_constancy() -> None:
    rows = load_rows(FIXTURE)
    per = compute_per_original_stats(rows)
    coca_key = "coca cola 2 l"
    agua_key = "agua mineral 500ml"
    assert per[coca_key][0] == pytest.approx(0.5)
    assert per[agua_key][0] == pytest.approx(1.0)


def test_single_row_group_is_perfect() -> None:
    rows = [{"original": "x", "attacked": "y", "normalized": "z"}]
    per = compute_per_original_stats(rows)
    assert per["x"][0] == pytest.approx(1.0)


def test_best_worst_lexicographic_tiebreak() -> None:
    rows = [
        {"original": "zebra", "attacked": "a", "normalized": "n1"},
        {"original": "zebra", "attacked": "b", "normalized": "n2"},
        {"original": "apple", "attacked": "c", "normalized": "n3"},
        {"original": "apple", "attacked": "d", "normalized": "n4"},
    ]
    result = build_result(rows)
    best = result["best_grouping"]
    worst = result["worst_grouping"]
    assert isinstance(best, dict) and isinstance(worst, dict)
    assert best["constancy"] == worst["constancy"] == pytest.approx(0.5)
    assert best["original"] == "apple"
    assert worst["original"] == "zebra"


def test_build_result_fixture_histogram_and_ordering() -> None:
    rows = load_rows(FIXTURE)
    result = build_result(rows)
    m = result["metrics"]
    assert m["total_rows"] == 7
    assert m["num_originals"] == 2
    assert m["mean_constancy"] == pytest.approx((0.5 + 1.0) / 2)
    assert m["weighted_constancy"] == pytest.approx(5 / 7)
    best = result["best_grouping"]
    worst = result["worst_grouping"]
    assert isinstance(best, dict) and isinstance(worst, dict)
    assert best["constancy"] >= worst["constancy"]
    assert best["original"] == "agua mineral 500ml"
    assert worst["original"] == "coca cola 2 l"
    assert "normalized_histogram" in best
    assert isinstance(best["normalized_histogram"], list)


def test_empty_rows_result() -> None:
    result = build_result([])
    assert result["metrics"]["total_rows"] == 0
    assert result["best_grouping"] is None


def test_main_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    code = main(["--input", str(FIXTURE), "--out", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "metrics" in data and "best_grouping" in data
    assert data["worst_grouping"]["constancy"] == pytest.approx(0.5)


def test_main_missing_input(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    out = tmp_path / "results.json"
    assert main(["--input", str(missing), "--out", str(out)]) == 1


def test_run_integration(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    run(FIXTURE, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metrics"]["min_constancy"] == pytest.approx(0.5)


def test_cli_module_smoke(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    proc = subprocess.run(
        [sys.executable, "-m", "benchmark.benchmark", "--input", str(FIXTURE), "--out", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "wrote" in proc.stdout.lower() or str(out.resolve()) in proc.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["metrics"]["num_originals"] == 2


def test_load_rows_rejects_missing_column(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("original,normalized\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="attacked"):
        load_rows(p)


def test_weighted_edit_distance_penalizes_digits() -> None:
    assert weighted_edit_distance("1", "2") == pytest.approx(100.0)
    assert normalized_weighted_distance("coca", "koka") == pytest.approx(0.5)


def test_weighted_distance_benchmark_groups_letters_but_not_digits(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "original,attacked,normalized\n"
        "x,a,coca cola 2 l\n"
        "x,b,coca cola 2 l\n"
        "x,c,coca cola 1 l\n",
        encoding="utf-8",
    )

    out = tmp_path / "out.json"
    result = run_benchmark_with_weighted_distance(csv_path, out, distance_threshold=1.0)

    assert result["metrics"]["num_originals"] == 1
    assert result["best_grouping"]["constancy_weighted_distance"] == pytest.approx(2 / 3)
    assert result["best_grouping"]["group_size"] == 3
