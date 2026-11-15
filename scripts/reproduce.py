from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_single(backend_name: str, run_id: int, output_base: Path) -> dict:
    from inference_harness import get_backend
    from inference_harness.runner import build_consistency_csv, build_pairwise_csv
    from benchmark.pairwise_f1 import run as run_pairwise_f1
    from benchmark.benchmark import run as run_constancy

    out_dir = output_base / backend_name / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s | run %d] starting", backend_name, run_id)

    backend = get_backend(backend_name)

    normalized_csv = out_dir / "normalized.csv"
    build_consistency_csv(
        backend,
        attack_csv=PROJECT_ROOT / "data" / "attack.csv",
        out_csv=normalized_csv,
    )

    pairs_csv = out_dir / "pairs_normalized.csv"
    build_pairwise_csv(
        backend,
        pairs_csv=PROJECT_ROOT / "benchmark" / "labeled_pairs" / "pairs_raw.csv",
        out_csv=pairs_csv,
    )

    pairwise_results = run_pairwise_f1(
        input_path=str(pairs_csv),
        out_path=str(out_dir / "pairwise_results.json"),
    )

    constancy_results = run_constancy(
        input_path=normalized_csv,
        out_path=out_dir / "constancy_results.json",
    )

    metrics = pairwise_results["metrics"]
    summary = {
        "backend": backend_name,
        "run_id": run_id,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "constancy": constancy_results["metrics"]["mean_constancy"],
    }

    logger.info("[%s | run %d] F1=%.4f  P=%.4f  R=%.4f  C=%.4f",
                backend_name, run_id,
                summary["f1"], summary["precision"],
                summary["recall"], summary["constancy"])

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["gemini"],
                        choices=["gemini", "deepseek", "gpt"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    args = parser.parse_args(argv)

    results = []
    for name in args.backends:
        for rid in range(1, args.runs + 1):
            try:
                results.append(run_single(name, rid, args.output_dir))
            except Exception as e:
                logger.error("[%s | run %d] FAILED: %s", name, rid, e)
                results.append({"backend": name, "run_id": rid, "error": str(e)})

    summary_path = args.output_dir / "reproduction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    logger.info("summary -> %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
