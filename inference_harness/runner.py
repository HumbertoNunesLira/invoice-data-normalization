"""Runner — produz os CSVs que o benchmark já consome, a partir de `normalize()`.

Dois artefatos, ambos via a MESMA interface de normalização:

  1. `output/normalized.csv`  (original, attacked, normalized, display_name)
     → alimenta TODO o benchmark.ipynb (consistência, by_attack_type,
       weighted_distance, blocking e o pairwise F1 inline).

  2. `output/pairs_normalized.csv`  (title_a, title_b, label, normalized_a, normalized_b)
     → alimenta o módulo standalone benchmark/pairwise_f1.py.

A pré-normalização e o filtro de distância de edição são model-independent e
aplicados aqui UMA vez, de modo que todo backend veja o mesmo conjunto de
entrada — a única coisa que muda entre rodadas é o modelo.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .backends import Backend
from .core import normalize_many
from .normalization import (
    EDIT_DISTANCE_THRESHOLD,
    normalized_weighted_distance,
    pre_normalize_title,
)

logger = logging.getLogger(__name__)


def build_consistency_csv(
    backend: Backend,
    attack_csv: str | Path = "data/attack.csv",
    out_csv: str | Path = "output/normalized.csv",
    apply_filter: bool = True,
    distance_threshold: float = EDIT_DISTANCE_THRESHOLD,
) -> pd.DataFrame:
    """Gera `original, attacked, normalized, display_name` a partir do attack.csv.

    `attack_csv` precisa das colunas `original` e `attacked` (o attack.csv padrão
    tem `id, original, tipo_variacao, attacked`). O filtro pré-LLM compara as
    versões pré-normalizadas de original × attacked e descarta pares divergentes
    demais (mesma lógica do classifier_api). A coluna `attacked` no CSV de saída
    preserva o texto atacado ORIGINAL (não pré-normalizado).
    """
    df = pd.read_csv(attack_csv)
    for col in ("original", "attacked"):
        if col not in df.columns:
            raise ValueError(f"{attack_csv}: coluna obrigatória ausente: {col!r}")

    n_total = len(df)
    if apply_filter:
        orig_pre = df["original"].astype(str).apply(pre_normalize_title)
        att_pre = df["attacked"].astype(str).apply(pre_normalize_title)
        keep = [
            normalized_weighted_distance(o, a) <= distance_threshold
            for o, a in zip(orig_pre, att_pre)
        ]
        df = df[keep].reset_index(drop=True)
        logger.info(
            "Filtro de distância: %d/%d pares mantidos (threshold=%.2f)",
            len(df), n_total, distance_threshold,
        )

    results = normalize_many(df["attacked"].astype(str).tolist(), backend=backend)

    out = pd.DataFrame({
        "original": df["original"].values,
        "attacked": df["attacked"].values,
        "normalized": [r["match_key"] for r in results],
        "display_name": [r["canonical"] for r in results],
    })
    _write_csv(out, out_csv)
    logger.info("Exportado: %s (%d linhas)", out_csv, len(out))
    return out


def build_pairwise_csv(
    backend: Backend,
    pairs_csv: str | Path = "benchmark/labeled_pairs/pairs_raw.csv",
    out_csv: str | Path = "output/pairs_normalized.csv",
) -> pd.DataFrame:
    """Gera `title_a, title_b, label, normalized_a, normalized_b` para o pairwise_f1.

    `pairs_csv` precisa de `title_a, title_b, label` (pairs_raw.csv tem esses +
    colunas extras de mineração). Normaliza cada lado com o mesmo `normalize()`.
    """
    df = pd.read_csv(pairs_csv)
    for col in ("title_a", "title_b", "label"):
        if col not in df.columns:
            raise ValueError(f"{pairs_csv}: coluna obrigatória ausente: {col!r}")

    res_a = normalize_many(df["title_a"].astype(str).tolist(), backend=backend)
    res_b = normalize_many(df["title_b"].astype(str).tolist(), backend=backend)

    out = pd.DataFrame({
        "title_a": df["title_a"].values,
        "title_b": df["title_b"].values,
        "label": df["label"].values,
        "normalized_a": [r["match_key"] for r in res_a],
        "normalized_b": [r["match_key"] for r in res_b],
    })
    _write_csv(out, out_csv)
    logger.info("Exportado: %s (%d pares)", out_csv, len(out))
    return out


def _write_csv(df: pd.DataFrame, out_csv: str | Path) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
