import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def predict(df: pd.DataFrame) -> pd.DataFrame:
   
    def _match(row) -> int:
        a = str(row["normalized_a"]).strip()
        b = str(row["normalized_b"]).strip()
        if not a or not b or a == "nan" or b == "nan":
            return 0
        return int(a == b)

    df = df.copy()
    df["predicted"] = df.apply(_match, axis=1)
    return df

def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["label"].tolist()
    y_pred = df["predicted"].tolist()

    metrics = {
        "precision":            round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":               round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                   round(f1_score(y_true, y_pred, zero_division=0), 4),
        "n_pairs":              len(df),
        "n_positive":           int(df["label"].sum()),
        "n_negative":           int((df["label"] == 0).sum()),
        "n_predicted_positive": int(df["predicted"].sum()),
    }

    false_negatives = df[
        (df["label"] == 1) & (df["predicted"] == 0)
    ][["title_a", "title_b", "normalized_a", "normalized_b"]].head(5).to_dict("records")

    false_positives = df[
        (df["label"] == 0) & (df["predicted"] == 1)
    ][["title_a", "title_b", "normalized_a", "normalized_b"]].head(5).to_dict("records")

    return {
        "metrics": metrics,
        "error_examples": {
            "false_negatives": false_negatives,  # mesmo produto, sistema separou
            "false_positives": false_positives,  # produtos diferentes, sistema uniu
        },
    }

def print_report(results: dict) -> None:
    m = results["metrics"]
    print()
    print("=" * 65)
    print("  PAIRWISE F1-SCORE — AVALIAÇÃO DO PIPELINE")
    print("=" * 65)
    print(f"  Pares totais         : {m['n_pairs']}")
    print(f"  Positivos (label=1)  : {m['n_positive']}")
    print(f"  Negativos (label=0)  : {m['n_negative']}")
    print(f"  Preditos positivos   : {m['n_predicted_positive']}")
    print("-" * 65)
    print(f"  Precision            : {m['precision']:.4f}")
    print(f"  Recall               : {m['recall']:.4f}")
    print(f"  F1-Score             : {m['f1']:.4f}")
    print("=" * 65)

    fn = results["error_examples"]["false_negatives"]
    fp = results["error_examples"]["false_positives"]

    if fn:
        print(f"\n  Falsos Negativos (mesmo produto, sistema separou) — top {len(fn)}:")
        for ex in fn:
            print(f"    A: {ex['title_a']}")
            print(f"    B: {ex['title_b']}")
            print(f"    norm_A: {ex['normalized_a']}")
            print(f"    norm_B: {ex['normalized_b']}")
            print()

    if fp:
        print(f"\n  Falsos Positivos (produtos diferentes, sistema uniu) — top {len(fp)}:")
        for ex in fp:
            print(f"    A: {ex['title_a']}")
            print(f"    B: {ex['title_b']}")
            print(f"    norm_A: {ex['normalized_a']}")
            print(f"    norm_B: {ex['normalized_b']}")
            print()

def run(
    input_path: str = "output/pairs_normalized.csv",
    out_path: str = "output/pairwise_results.json",
) -> dict:
    
    df = pd.read_csv(input_path)
    logger.info("CSV carregado: %d pares", len(df))

    required = {"title_a", "title_b", "label", "normalized_a", "normalized_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing}")

    assert df["label"].isin([0, 1]).all(), "Coluna 'label' deve conter apenas 0 ou 1."

    df = predict(df)
    results = compute_metrics(df)
    print_report(results)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Resultados salvos em: %s", out_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise F1-Score para normalização de títulos")
    parser.add_argument("--input", default="output/pairs_normalized.csv", help="pairs_normalized.csv")
    parser.add_argument("--out",   default="output/pairwise_results.json", help="JSON de saída")
    args = parser.parse_args()

    run(input_path=args.input, out_path=args.out)