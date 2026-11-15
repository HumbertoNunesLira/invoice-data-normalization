#!/usr/bin/env python3
"""Roda a extração do TAIL sobre os títulos do Golden Set do AI-PAVE-Br.

Usa o pipeline canônico (``inference_harness.normalize_many``): pré-normalização
simbólica → LLM → limpeza determinística. Só os títulos entram, como na Seção 3.1
do artigo.

Escreve JSONL incremental — o script é retomável: relê o que já foi feito e
continua de onde parou.

    python revisao/pave_extract.py --limit 100     # piloto
    python revisao/pave_extract.py                 # completo
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")

from inference_harness import get_backend, normalize_many  # noqa: E402

PAVE = Path(__file__).resolve().parent / "pave"


def carregar_feitos(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    feitos = set()
    with path.open(encoding="utf-8") as fh:
        for linha in fh:
            try:
                feitos.add(json.loads(linha)["item_uid"])
            except Exception:
                continue  # linha truncada por interrupção; será refeita
    return feitos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = tudo")
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview",
                    help="modelo do artigo (Seção 3.3)")
    ap.add_argument("--batch-size", type=int, default=5, help="BATCH_SIZE=5, como no artigo")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--chunk", type=int, default=100, help="títulos por flush no JSONL")
    ap.add_argument("--out", default=str(PAVE / "tail_pred.jsonl"))
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with (PAVE / "itens.csv").open(encoding="utf-8") as fh:
        itens = list(csv.DictReader(fh))
    if a.limit:
        itens = itens[: a.limit]

    feitos = carregar_feitos(out)
    pend = [i for i in itens if i["item_uid"] not in feitos]
    print(f"total={len(itens)}  já feitos={len(feitos)}  pendentes={len(pend)}")
    if not pend:
        print("nada a fazer.")
        return

    backend = get_backend(
        "gemini",
        model_name=a.model,
        temperature=0.0,
        batch_size=a.batch_size,
        sleep_between_batches=a.sleep,
    )

    t0 = time.time()
    feito = 0
    with out.open("a", encoding="utf-8") as fh:
        for ini in range(0, len(pend), a.chunk):
            lote = pend[ini: ini + a.chunk]
            res = normalize_many([i["title"] for i in lote], backend=backend)
            for item, r in zip(lote, res):
                fh.write(json.dumps({
                    "item_uid": item["item_uid"],
                    "categoria": item["categoria"],
                    "title": item["title"],
                    "cleaned": r["cleaned"],
                    "match_key": r["match_key"],
                }, ensure_ascii=False) + "\n")
            fh.flush()
            feito += len(lote)
            dt = time.time() - t0
            vazios = sum(1 for r in res if not r["cleaned"])
            print(f"  {feito}/{len(pend)}  {dt/60:.1f} min  "
                  f"({feito/max(dt,1e-9):.1f} títulos/s)  vazios_no_lote={vazios}",
                  flush=True)

    print(f"OK — {feito} títulos em {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
