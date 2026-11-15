from __future__ import annotations

import argparse
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import TypedDict

import pandas as pd


RANDOM_SEED = 42
POOL_SIZE = 30_000
RANDOM_PAIRS = 300
CANDIDATE_PAIRS_PER_CLASS = 5_000
STRATIFIED_PAIRS = 900
MATCH_THRESHOLD = 0.70


class Pair(TypedDict):
    title_a: str
    title_b: str
    candidate_label: str
    similarity_score: float


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def make_pair(a: str, b: str, label: str, sim: float) -> Pair:
    return Pair(title_a=a, title_b=b, candidate_label=label, similarity_score=round(sim, 4))


def build_category_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group pool titles by category for match-biased sampling."""
    if "category" not in df.columns:
        return {}
    return {
        cat: group["title"].tolist()
        for cat, group in df.dropna(subset=["title"]).groupby("category")
    }


def _same_category_pairs(
    category_groups: dict[str, list[str]],
    rng: random.Random,
) -> list[tuple[str, str]]:
    """Generate all within-category pairs, shuffled. Capped per category to avoid O(n²) blowup."""
    MAX_PAIRS_PER_CATEGORY = 500
    pairs: list[tuple[str, str]] = []
    for titles in category_groups.values():
        if len(titles) < 2:
            continue
        cat_pairs = [
            (titles[i], titles[j])
            for i in range(len(titles))
            for j in range(i + 1, len(titles))
        ]
        if len(cat_pairs) > MAX_PAIRS_PER_CATEGORY:
            cat_pairs = rng.sample(cat_pairs, MAX_PAIRS_PER_CATEGORY)
        pairs.extend(cat_pairs)
    rng.shuffle(pairs)
    return pairs


def build_random_pairs(titles: list[str], n: int, rng: random.Random) -> list[Pair]:
    """Sample n fully random pairs with no similarity criterion."""
    if len(titles) < 2:
        raise ValueError(f"Need at least 2 titles to build pairs, got {len(titles)}")

    pairs: list[Pair] = []
    max_attempts = n * 10
    attempts = 0

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(titles, 2)
        if a != b:
            pairs.append(make_pair(a, b, "random", string_similarity(a, b)))

    return pairs


def build_candidate_pairs(
    titles: list[str],
    n_match: int,
    n_nonmatch: int,
    threshold: float,
    rng: random.Random,
    category_groups: dict[str, list[str]] | None = None,
    max_attempts_factor: int = 20,
) -> tuple[list[Pair], list[Pair]]:
    """
    Build candidate match and non-match pairs using string similarity.

    Same-category pairs are prioritized for match candidates since titles
    within the same category are more likely to share >= threshold similarity.
    """
    matches: list[Pair] = []
    nonmatches: list[Pair] = []

    same_cat_pairs = _same_category_pairs(category_groups or {}, rng)
    same_cat_idx = 0

    max_attempts = (n_match + n_nonmatch) * max_attempts_factor
    attempts = 0

    while (len(matches) < n_match or len(nonmatches) < n_nonmatch) and attempts < max_attempts:
        attempts += 1

        if len(matches) < n_match and same_cat_idx < len(same_cat_pairs):
            a, b = same_cat_pairs[same_cat_idx]
            same_cat_idx += 1
        else:
            a, b = rng.sample(titles, 2)

        if a == b:
            continue

        sim = string_similarity(a, b)

        if sim >= threshold and len(matches) < n_match:
            matches.append(make_pair(a, b, "match", sim))
        elif sim < threshold and len(nonmatches) < n_nonmatch:
            nonmatches.append(make_pair(a, b, "non_match", sim))

    return matches, nonmatches


def stratified_sample(
    matches: list[Pair],
    nonmatches: list[Pair],
    total: int,
    rng: random.Random,
) -> list[Pair]:
    """Draw `total` pairs preserving the observed match/non-match ratio."""
    n_candidates = len(matches) + len(nonmatches)
    match_ratio = len(matches) / n_candidates if n_candidates > 0 else 0.5

    n_match = min(round(total * match_ratio), len(matches))
    n_nonmatch = min(total - n_match, len(nonmatches))

    drawn = rng.sample(matches, n_match) + rng.sample(nonmatches, n_nonmatch)
    rng.shuffle(drawn)
    return drawn


def load_pool(path: Path, pool_size: int, language: str, seed: int) -> pd.DataFrame:
    """Load and filter the dataset, then draw a random pool."""
    df = pd.read_csv(path)
    print(f"  Total rows: {len(df):,}")

    if language != "all" and "language" in df.columns:
        df = df[df["language"] == language]
        print(f"  After language filter ({language}): {len(df):,}")

    return df.sample(n=min(pool_size, len(df)), random_state=seed)


def _print_summary(df_out: pd.DataFrame, out_path: Path) -> None:
    total_target = RANDOM_PAIRS + STRATIFIED_PAIRS
    print(f"\n{'=' * 55}")
    print(f"  Output: {out_path}")
    print(f"  Total pairs: {len(df_out):,}  (target: {total_target})")
    print(f"\n  Distribution:")
    print(df_out["candidate_label"].value_counts().to_string())
    print(f"\n  Similarity stats:")
    print(df_out["similarity_score"].describe().round(4).to_string())
    print(f"{'=' * 55}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build stratified pair dataset for labeling.")
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/pairs_to_label.csv"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    parser.add_argument(
        "--language",
        type=str,
        default="portuguese",
        choices=["portuguese", "spanish", "all"],
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)

    print(f"Loading {args.input} ...")
    df_pool = load_pool(args.input, args.pool_size, args.language, args.seed)
    pool = df_pool["title"].dropna().tolist()
    print(f"  Pool size: {len(pool):,}")

    category_groups = build_category_groups(df_pool)
    if category_groups:
        print(f"  Categories in pool: {len(category_groups):,}")

    print(f"\nBuilding {RANDOM_PAIRS} fully random pairs ...")
    random_pairs = build_random_pairs(pool, RANDOM_PAIRS, rng)
    print(f"  Done: {len(random_pairs)} pairs")

    print(f"\nBuilding {CANDIDATE_PAIRS_PER_CLASS:,} candidate pairs per class "
          f"(threshold={args.threshold}) ...")
    matches, nonmatches = build_candidate_pairs(
        pool,
        n_match=CANDIDATE_PAIRS_PER_CLASS,
        n_nonmatch=CANDIDATE_PAIRS_PER_CLASS,
        threshold=args.threshold,
        rng=rng,
        category_groups=category_groups,
    )
    print(f"  Match candidates:     {len(matches):,}")
    print(f"  Non-match candidates: {len(nonmatches):,}")

    n_candidates = len(matches) + len(nonmatches)
    match_ratio = len(matches) / n_candidates if n_candidates > 0 else 0.0
    print(f"  Observed match ratio: {match_ratio:.1%} / {1 - match_ratio:.1%}")

    print(f"\nDrawing {STRATIFIED_PAIRS} stratified pairs ...")
    stratified = stratified_sample(matches, nonmatches, STRATIFIED_PAIRS, rng)
    n_match_drawn = sum(1 for p in stratified if p["candidate_label"] == "match")
    print(f"  Match: {n_match_drawn}  |  Non-match: {len(stratified) - n_match_drawn}")

    all_pairs = random_pairs + stratified
    rng.shuffle(all_pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(all_pairs)
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")

    _print_summary(df_out, args.out)


if __name__ == "__main__":
    main()
