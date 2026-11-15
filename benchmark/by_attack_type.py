"""
Análise de consistência por tipo de ataque.

Unindo o arquivo normalized.csv (original, atacado, normalizado) com o arquivo attack.csv
(id, original, tipo_variacao, atacado) na coluna `attacked`,
e então, para cada `tipo_variacao`, são relatadas as proporções de variantes cujo valor `normalizado`
é igual ao valor `normalizado` dominante dentro do seu grupo `original`
(taxa de correspondência de modo).

Uma taxa de correspondência de modo mais baixa significa que a classe de ataque desestabiliza o normalizador
com mais frequência.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path


def load_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _dominant(values: Iterable[str]) -> str:
    """Most frequent value; lex-smallest on ties (matches benchmark.py)."""
    c = Counter(values)
    if not c:
        return ""
    max_count = max(c.values())
    return min(n for n, k in c.items() if k == max_count)


def compute(
    normalized_rows: list[Mapping[str, str]],
    attack_rows: list[Mapping[str, str]],
) -> dict[str, object]:
    attack_by_text: dict[str, Mapping[str, str]] = {
        r["attacked"]: r for r in attack_rows if r.get("attacked")
    }

    groups: dict[str, list[str]] = defaultdict(list)
    for r in normalized_rows:
        groups[r["original"]].append(r["normalized"])

    mode_per_original = {orig: _dominant(norms) for orig, norms in groups.items()}

    type_hits: dict[str, list[int]] = defaultdict(list)
    matched = 0
    skipped = 0
    for r in normalized_rows:
        meta = attack_by_text.get(r["attacked"])
        if not meta:
            skipped += 1
            continue
        tipo = meta.get("tipo_variacao") or "Desconhecido"
        is_match = int(r["normalized"] == mode_per_original[r["original"]])
        type_hits[tipo].append(is_match)
        matched += 1

    rows = []
    for tipo, hits in type_hits.items():
        rows.append({
            "tipo_variacao": tipo,
            "n": len(hits),
            "mode_match_rate": sum(hits) / len(hits) if hits else 0.0,
        })
    rows.sort(key=lambda r: (r["mode_match_rate"], r["tipo_variacao"]))

    return {
        "rows_joined": matched,
        "rows_unmatched": skipped,
        "n_originals": len(groups),
        "by_attack_type": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--normalized", type=Path, required=True, help="Path to normalized.csv")
    p.add_argument("--attack", type=Path, required=True, help="Path to attack.csv")
    p.add_argument("--out", type=Path, default=Path("output/by_attack_type.json"))
    args = p.parse_args(argv)

    if not args.normalized.is_file():
        print(f"error: normalized csv not found: {args.normalized}", file=sys.stderr)
        return 1
    if not args.attack.is_file():
        print(f"error: attack csv not found: {args.attack}", file=sys.stderr)
        return 1

    norm = load_csv_dicts(args.normalized)
    atk = load_csv_dicts(args.attack)

    for required in ("original", "attacked", "normalized"):
        if norm and required not in norm[0]:
            print(f"error: normalized csv missing column {required!r}", file=sys.stderr)
            return 1
    for required in ("attacked", "tipo_variacao"):
        if atk and required not in atk[0]:
            print(f"error: attack csv missing column {required!r}", file=sys.stderr)
            return 1

    result = compute(norm, atk)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {args.out.resolve()}")
    print(f"  joined {result['rows_joined']} rows ({result['rows_unmatched']} unmatched)"
          f" across {result['n_originals']} originals")
    print()
    print(f"  {'tipo_variacao':<24s} {'n':>5s}  {'mode_match_rate':>15s}")
    print(f"  {'-' * 24} {'-' * 5}  {'-' * 15}")
    for row in result["by_attack_type"]:
        print(f"  {row['tipo_variacao']:<24s} {row['n']:>5d}  {row['mode_match_rate']:>15.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
