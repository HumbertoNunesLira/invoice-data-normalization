from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import NamedTuple


class Measure(NamedTuple):
    value_base: float
    dimension: str
    raw: str


_PATTERNS: list[tuple[re.Pattern, str, float, str]] = [
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*l(?:itro)?s?\b", re.I), "volume_l", 1000.0, "volume"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*ml\b", re.I), "volume_ml", 1.0, "volume"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*k(?:g|ilo|ilos|ilograma|ilogramas)\b", re.I), "mass_kg", 1000.0, "mass"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*g(?:r|rama|ramas)?\b", re.I), "mass_g", 1.0, "mass"),
]


def extract_measures(text: str) -> list[Measure]:
    """Extract all volume/mass measures from text, normalized to base units (ml/g)."""
    if not text:
        return []

    measures: list[Measure] = []
    used_spans: list[tuple[int, int]] = []

    for pattern, _label, factor, dimension in _PATTERNS:
        for m in pattern.finditer(text):
            span = m.span()
            if any(s <= span[0] < e or s < span[1] <= e for s, e in used_spans):
                continue

            raw_num = m.group(1).replace(",", ".")
            try:
                value = float(raw_num) * factor
            except ValueError:
                continue

            measures.append(Measure(value_base=value, dimension=dimension, raw=m.group(0)))
            used_spans.append(span)

    return measures


def measures_are_compatible(measures_a: list[Measure], measures_b: list[Measure]) -> bool:
    """Check if measures from two products are compatible (same dimension and value)."""
    if not measures_a and not measures_b:
        return True

    if not measures_a or not measures_b:
        return False

    dims_a: dict[str, list[float]] = defaultdict(list)
    dims_b: dict[str, list[float]] = defaultdict(list)

    for m in measures_a:
        dims_a[m.dimension].append(m.value_base)
    for m in measures_b:
        dims_b[m.dimension].append(m.value_base)

    if set(dims_a.keys()) != set(dims_b.keys()):
        return False

    for dim in dims_a:
        vals_a = sorted(dims_a[dim])
        vals_b = sorted(dims_b[dim])
        if len(vals_a) != len(vals_b):
            return False
        for va, vb in zip(vals_a, vals_b):
            if abs(va - vb) > max(va, vb) * 0.001:
                return False

    return True


def is_blocked(text_a: str, text_b: str) -> bool:
    """Return True if the pair should be blocked (similarity forced to 0)."""
    measures_a = extract_measures(text_a)
    measures_b = extract_measures(text_b)
    return not measures_are_compatible(measures_a, measures_b)


def similarity(text_a: str, text_b: str, threshold: float = 0.0) -> float:
    """Compute similarity with blocking. Returns 0.0 if units are incompatible."""
    if not text_a or not text_b:
        return 0.0

    a = text_a.strip().lower()
    b = text_b.strip().lower()

    if a == b:
        return 1.0

    if is_blocked(a, b):
        return 0.0

    sim = SequenceMatcher(None, a, b).ratio()
    return sim if sim >= threshold else 0.0


def constancy_with_blocking(
    normalized_values: list[str],
    similarity_threshold: float = 0.85,
) -> tuple[float, int, int, list[list[int]]]:
    """Compute constancy using blocking + string similarity with Union-Find clustering."""
    n = len(normalized_values)
    if n == 0:
        return 0.0, 0, 0, []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity(normalized_values[i], normalized_values[j], threshold=similarity_threshold)
            if sim >= similarity_threshold:
                union(i, j)

    clusters_map: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters_map[find(i)].append(i)

    clusters = list(clusters_map.values())
    dominant_count = max(len(c) for c in clusters)
    constancy = dominant_count / n

    return constancy, n, dominant_count, clusters


def run_benchmark_with_blocking(
    input_path: Path,
    out_path: Path,
    similarity_threshold: float = 0.85,
) -> dict[str, object]:
    """Run constancy benchmark using blocking + similarity. Compatible with benchmark.py format."""
    from benchmark.benchmark import load_rows

    rows = load_rows(input_path)
    if not rows:
        result: dict[str, object] = {
            "metrics": {
                "mean_constancy_blocking": None,
                "mean_constancy_exact": None,
                "num_originals": 0,
                "total_rows": 0,
                "similarity_threshold": similarity_threshold,
                "total_blocked_pairs": 0,
            },
            "best_grouping": None,
            "worst_grouping": None,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["original"].strip()].append(row["normalized"].strip())

    per_original: dict[str, dict] = {}
    total_blocked_pairs = 0

    for original, norms in groups.items():
        constancy_b, group_size, dominant_count, clusters = constancy_with_blocking(
            norms, similarity_threshold
        )

        counter = Counter(norms)
        exact_dominant = max(counter.values())
        constancy_exact = exact_dominant / group_size

        blocked = 0
        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                if is_blocked(norms[i], norms[j]):
                    blocked += 1
        total_blocked_pairs += blocked

        per_original[original] = {
            "constancy_blocking": constancy_b,
            "constancy_exact": constancy_exact,
            "group_size": group_size,
            "dominant_count_blocking": dominant_count,
            "dominant_count_exact": exact_dominant,
            "num_clusters": len(clusters),
            "blocked_pairs": blocked,
            "clusters": clusters,
            "normalized_values": norms,
        }

    constancies_b = [v["constancy_blocking"] for v in per_original.values()]
    constancies_e = [v["constancy_exact"] for v in per_original.values()]

    mean_b = sum(constancies_b) / len(constancies_b)
    mean_e = sum(constancies_e) / len(constancies_e)

    total_rows = sum(v["group_size"] for v in per_original.values())
    weighted_b = sum(v["dominant_count_blocking"] for v in per_original.values()) / total_rows

    sorted_origs = sorted(per_original.keys())
    best_orig = min(sorted_origs, key=lambda o: (-per_original[o]["constancy_blocking"], o))
    worst_orig = min(sorted_origs, key=lambda o: (per_original[o]["constancy_blocking"], o))

    def _pack_grouping(orig: str) -> dict:
        info = per_original[orig]
        largest_cluster = max(info["clusters"], key=len) if info["clusters"] else []
        dominant_norm = info["normalized_values"][largest_cluster[0]] if largest_cluster else ""

        return {
            "original": orig,
            "constancy_blocking": info["constancy_blocking"],
            "constancy_exact": info["constancy_exact"],
            "dominant_count_blocking": info["dominant_count_blocking"],
            "dominant_count_exact": info["dominant_count_exact"],
            "group_size": info["group_size"],
            "num_clusters": info["num_clusters"],
            "blocked_pairs": info["blocked_pairs"],
            "normalized_histogram": [
                {"normalized": n, "count": c}
                for n, c in Counter(info["normalized_values"]).most_common()
            ],
        }

    result = {
        "metrics": {
            "mean_constancy_blocking": round(mean_b, 6),
            "mean_constancy_exact": round(mean_e, 6),
            "delta_blocking_vs_exact": round(mean_b - mean_e, 6),
            "weighted_constancy_blocking": round(weighted_b, 6),
            "min_constancy_blocking": round(min(constancies_b), 6),
            "max_constancy_blocking": round(max(constancies_b), 6),
            "num_originals": len(per_original),
            "total_rows": total_rows,
            "total_blocked_pairs": total_blocked_pairs,
            "similarity_threshold": similarity_threshold,
        },
        "best_grouping": _pack_grouping(best_orig),
        "worst_grouping": _pack_grouping(worst_orig),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Constancy benchmark with unit-based blocking.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/blocking_results.json"))
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1

    result = run_benchmark_with_blocking(args.input.resolve(), args.out.resolve(), args.threshold)

    print(f"wrote {args.out.resolve()}")
    print()
    m = result["metrics"]
    print(f"  {'Metric':<30s} {'Value':>10s}")
    print(f"  {'-' * 30} {'-' * 10}")
    print(f"  {'mean_constancy (blocking)':<30s} {m['mean_constancy_blocking']:>10.4f}")
    print(f"  {'mean_constancy (exact)':<30s} {m['mean_constancy_exact']:>10.4f}")
    print(f"  {'delta blocking vs exact':<30s} {m['delta_blocking_vs_exact']:>+10.4f}")
    print(f"  {'weighted_constancy (blocking)':<30s} {m['weighted_constancy_blocking']:>10.4f}")
    print(f"  {'total_blocked_pairs':<30s} {m['total_blocked_pairs']:>10d}")
    print(f"  {'similarity_threshold':<30s} {m['similarity_threshold']:>10.2f}")
    print(f"  {'num_originals':<30s} {m['num_originals']:>10d}")
    print(f"  {'total_rows':<30s} {m['total_rows']:>10d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
