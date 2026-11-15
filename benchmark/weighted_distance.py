from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

from benchmark.benchmark import load_rows


class DistanceConfig(NamedTuple):
    letter_cost: float = 1.0
    digit_cost: float = 100.0
    symbol_cost: float = 5.0
    space_cost: float = 0.5


def _substitution_cost(a: str, b: str, config: DistanceConfig) -> float:
    if a == b:
        return 0.0
    if a.isdigit() or b.isdigit():
        return config.digit_cost
    if a.isalpha() and b.isalpha():
        return config.letter_cost
    if a.isspace() or b.isspace():
        return config.space_cost
    return config.symbol_cost


def _indel_cost(ch: str, config: DistanceConfig) -> float:
    if ch.isdigit():
        return config.digit_cost
    if ch.isalpha():
        return config.letter_cost
    if ch.isspace():
        return config.space_cost
    return config.symbol_cost


def weighted_edit_distance(s1: str, s2: str, config: DistanceConfig | None = None) -> float:
    """Compute a weighted edit distance where digits are penalized heavily."""
    if config is None:
        config = DistanceConfig()

    a = s1.strip().lower()
    b = s2.strip().lower()
    n, m = len(a), len(b)

    if n == 0:
        return sum(_indel_cost(ch, config) for ch in b)
    if m == 0:
        return sum(_indel_cost(ch, config) for ch in a)

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _indel_cost(a[i - 1], config)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _indel_cost(b[j - 1], config)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = dp[i - 1][j - 1] + _substitution_cost(a[i - 1], b[j - 1], config)
            cost_del = dp[i - 1][j] + _indel_cost(a[i - 1], config)
            cost_ins = dp[i][j - 1] + _indel_cost(b[j - 1], config)
            dp[i][j] = min(cost_sub, cost_del, cost_ins)

    return dp[n][m]


def normalized_weighted_distance(s1: str, s2: str, config: DistanceConfig | None = None) -> float:
    distance = weighted_edit_distance(s1, s2, config)
    denom = max(len(s1.strip()), len(s2.strip()), 1)
    return distance / denom


def constancy_with_weighted_distance(
    normalized_values: list[str],
    distance_threshold: float = 1.0,
) -> tuple[float, int, int, list[list[int]]]:
    """Cluster normalized values by weighted distance and compute constancy."""
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
            if normalized_weighted_distance(normalized_values[i], normalized_values[j]) <= distance_threshold:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    cluster_list = list(clusters.values())
    dominant_count = max(len(c) for c in cluster_list)
    constancy = dominant_count / n
    return constancy, n, dominant_count, cluster_list


def run_benchmark_with_weighted_distance(
    input_path: Path,
    out_path: Path,
    distance_threshold: float = 1.0,
) -> dict[str, object]:
    rows = load_rows(input_path)
    if not rows:
        result: dict[str, object] = {
            "metrics": {
                "mean_constancy_weighted_distance": None,
                "mean_constancy_exact": None,
                "num_originals": 0,
                "total_rows": 0,
                "distance_threshold": distance_threshold,
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
    for original, norms in groups.items():
        constancy_wd, group_size, dominant_count, clusters = constancy_with_weighted_distance(
            norms, distance_threshold=distance_threshold
        )
        counter = Counter(norms)
        exact_dominant = max(counter.values())
        constancy_exact = exact_dominant / group_size
        per_original[original] = {
            "constancy_weighted_distance": constancy_wd,
            "constancy_exact": constancy_exact,
            "group_size": group_size,
            "dominant_count_weighted_distance": dominant_count,
            "dominant_count_exact": exact_dominant,
            "num_clusters": len(clusters),
            "clusters": clusters,
            "normalized_values": norms,
        }

    constancies_wd = [v["constancy_weighted_distance"] for v in per_original.values()]
    constancies_e = [v["constancy_exact"] for v in per_original.values()]
    total_rows = sum(v["group_size"] for v in per_original.values())
    weighted_wd = sum(v["dominant_count_weighted_distance"] for v in per_original.values()) / total_rows
    sorted_origs = sorted(per_original.keys())
    best_orig = min(sorted_origs, key=lambda o: (-per_original[o]["constancy_weighted_distance"], o))
    worst_orig = min(sorted_origs, key=lambda o: (per_original[o]["constancy_weighted_distance"], o))

    def _pack_grouping(orig: str) -> dict[str, object]:
        info = per_original[orig]
        largest_cluster = max(info["clusters"], key=len) if info["clusters"] else []
        dominant_norm = info["normalized_values"][largest_cluster[0]] if largest_cluster else ""
        return {
            "original": orig,
            "constancy_weighted_distance": info["constancy_weighted_distance"],
            "constancy_exact": info["constancy_exact"],
            "dominant_count_weighted_distance": info["dominant_count_weighted_distance"],
            "dominant_count_exact": info["dominant_count_exact"],
            "group_size": info["group_size"],
            "num_clusters": info["num_clusters"],
            "normalized_histogram": [
                {"normalized": n, "count": c}
                for n, c in Counter(info["normalized_values"]).most_common()
            ],
        }

    result = {
        "metrics": {
            "mean_constancy_weighted_distance": round(sum(constancies_wd) / len(constancies_wd), 6),
            "mean_constancy_exact": round(sum(constancies_e) / len(constancies_e), 6),
            "delta_weighted_vs_exact": round(sum(constancies_wd) / len(constancies_wd) - sum(constancies_e) / len(constancies_e), 6),
            "weighted_constancy_weighted_distance": round(weighted_wd, 6),
            "min_constancy_weighted_distance": round(min(constancies_wd), 6),
            "max_constancy_weighted_distance": round(max(constancies_wd), 6),
            "num_originals": len(per_original),
            "total_rows": total_rows,
            "distance_threshold": distance_threshold,
        },
        "best_grouping": _pack_grouping(best_orig),
        "worst_grouping": _pack_grouping(worst_orig),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Constancy benchmark using weighted string distance.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/weighted_string_distance.json"))
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 1

    result = run_benchmark_with_weighted_distance(args.input.resolve(), args.out.resolve(), args.threshold)
    print(f"wrote {args.out.resolve()}")
    print()
    m = result["metrics"]
    print(f"  {'Metric':<35s} {'Value':>10s}")
    print(f"  {'-' * 35} {'-' * 10}")
    print(f"  {'mean_constancy (weighted)':<35s} {m['mean_constancy_weighted_distance']:>10.4f}")
    print(f"  {'mean_constancy (exact)':<35s} {m['mean_constancy_exact']:>10.4f}")
    print(f"  {'delta weighted vs exact':<35s} {m['delta_weighted_vs_exact']:>+10.4f}")
    print(f"  {'weighted_constancy (weighted)':<35s} {m['weighted_constancy_weighted_distance']:>10.4f}")
    print(f"  {'distance_threshold':<35s} {m['distance_threshold']:>10.2f}")
    print(f"  {'num_originals':<35s} {m['num_originals']:>10d}")
    print(f"  {'total_rows':<35s} {m['total_rows']:>10d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
