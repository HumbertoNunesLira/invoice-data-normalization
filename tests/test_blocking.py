from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.blocking import (
    Measure,
    constancy_with_blocking,
    extract_measures,
    is_blocked,
    measures_are_compatible,
    run_benchmark_with_blocking,
    similarity,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "benchmark" / "fixtures" / "normalized_sample.csv"


class TestExtractMeasures:
    def test_volume_liters(self) -> None:
        m = extract_measures("coca cola 2l")
        assert len(m) == 1
        assert m[0].value_base == pytest.approx(2000.0)
        assert m[0].dimension == "volume"

    def test_volume_ml(self) -> None:
        m = extract_measures("agua mineral 500ml")
        assert len(m) == 1
        assert m[0].value_base == pytest.approx(500.0)
        assert m[0].dimension == "volume"

    def test_volume_decimal_liters(self) -> None:
        m = extract_measures("coca cola 1,5l")
        assert len(m) == 1
        assert m[0].value_base == pytest.approx(1500.0)

    def test_mass_kg(self) -> None:
        m = extract_measures("arroz 5kg")
        assert len(m) == 1
        assert m[0].value_base == pytest.approx(5000.0)
        assert m[0].dimension == "mass"

    def test_mass_grams(self) -> None:
        m = extract_measures("cafe 500g")
        assert len(m) == 1
        assert m[0].value_base == pytest.approx(500.0)
        assert m[0].dimension == "mass"

    def test_mass_kilos_text(self) -> None:
        m = extract_measures("arroz 5 kilos")
        assert len(m) == 1
        assert m[0].value_base == pytest.approx(5000.0)

    def test_no_measures(self) -> None:
        assert extract_measures("televisão samsung 55pol") == []

    def test_empty_string(self) -> None:
        assert extract_measures("") == []

    def test_multiple_measures_same_dimension(self) -> None:
        m = extract_measures("kit 500ml + 1l")
        assert len(m) == 2
        values = sorted([x.value_base for x in m])
        assert values == pytest.approx([500.0, 1000.0])


class TestBlocking:
    def test_same_volume_different_value_blocked(self) -> None:
        assert is_blocked("coca cola 1l", "coca cola 2l") is True

    def test_equivalent_volume_not_blocked(self) -> None:
        assert is_blocked("coca cola 2l", "coca cola 2000ml") is False

    def test_equivalent_mass_not_blocked(self) -> None:
        assert is_blocked("arroz 5kg", "arroz 5000g") is False

    def test_different_mass_blocked(self) -> None:
        assert is_blocked("arroz 5kg", "arroz 10kg") is True

    def test_different_dimensions_blocked(self) -> None:
        assert is_blocked("coca cola 1l", "coca cola 1kg") is True

    def test_one_has_measure_other_not_blocked(self) -> None:
        assert is_blocked("coca cola 1l", "televisão samsung") is True

    def test_neither_has_measure_not_blocked(self) -> None:
        assert is_blocked("televisão samsung", "televisão lg") is False

    def test_identical_not_blocked(self) -> None:
        assert is_blocked("coca cola 500ml", "coca cola 500ml") is False


class TestSimilarity:
    def test_identical_returns_one(self) -> None:
        assert similarity("coca cola 2l", "coca cola 2l") == pytest.approx(1.0)

    def test_blocked_returns_zero(self) -> None:
        assert similarity("coca cola 1l", "coca cola 2l") == pytest.approx(0.0)

    def test_similar_strings_same_unit(self) -> None:
        sim = similarity("coca cola 2l", "cocacola 2l")
        assert sim > 0.8

    def test_equivalent_units_similar(self) -> None:
        sim = similarity("coca cola 2l", "coca cola 2000ml")
        assert sim > 0.7

    def test_empty_returns_zero(self) -> None:
        assert similarity("", "coca cola 2l") == pytest.approx(0.0)
        assert similarity("coca cola 2l", "") == pytest.approx(0.0)

    def test_threshold_filters(self) -> None:
        sim = similarity("abc", "xyz", threshold=0.9)
        assert sim == pytest.approx(0.0)


class TestConstancyWithBlocking:
    def test_all_equivalent_perfect_constancy(self) -> None:
        group = ["coca cola 2l", "cocacola 2l", "coca cola 2000ml", "coca-cola 2l"]
        c, sz, dom, clusters = constancy_with_blocking(group, similarity_threshold=0.7)
        assert c == pytest.approx(1.0)
        assert sz == 4
        assert dom == 4
        assert len(clusters) == 1

    def test_blocked_pair_splits_cluster(self) -> None:
        group = ["coca cola 1l", "coca cola 2l", "coca cola 1000ml"]
        c, sz, dom, clusters = constancy_with_blocking(group, similarity_threshold=0.7)
        assert sz == 3
        assert dom == 2
        assert c == pytest.approx(2 / 3)
        assert len(clusters) == 2

    def test_empty_group(self) -> None:
        c, sz, dom, clusters = constancy_with_blocking([], similarity_threshold=0.85)
        assert c == pytest.approx(0.0)
        assert sz == 0

    def test_single_element(self) -> None:
        c, sz, dom, clusters = constancy_with_blocking(["x"], similarity_threshold=0.85)
        assert c == pytest.approx(1.0)
        assert dom == 1


class TestRunBenchmarkWithBlocking:
    def test_fixture_improves_over_exact(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        result = run_benchmark_with_blocking(FIXTURE, out, similarity_threshold=0.7)
        m = result["metrics"]
        assert m["mean_constancy_blocking"] >= m["mean_constancy_exact"]
        assert m["delta_blocking_vs_exact"] >= 0

    def test_output_file_written(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        run_benchmark_with_blocking(FIXTURE, out, similarity_threshold=0.85)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "metrics" in data
        assert "best_grouping" in data
        assert "worst_grouping" in data

    def test_best_worse_ordering(self, tmp_path: Path) -> None:
        out = tmp_path / "results.json"
        result = run_benchmark_with_blocking(FIXTURE, out, similarity_threshold=0.85)
        best = result["best_grouping"]
        worst = result["worst_grouping"]
        assert best["constancy_blocking"] >= worst["constancy_blocking"]
