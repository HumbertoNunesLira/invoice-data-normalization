#!/usr/bin/env python3
"""Monta o conjunto de avaliação AI-PAVE-Br a partir dos CSVs publicados.

Saída: um CSV longo, uma linha por (item, atributo gold), com o valor gold e as
predições dos dois sistemas deles. As predições do TAIL entram depois
(``pave_extract.py``), casadas por ``item_uid``.

Estrutura dos CSVs deles: a linha 0 traz rótulos de grupo mesclados, a linha 1 os
nomes reais das colunas. ``title`` e ``description`` são nomeados só na linha 0.
Três blocos: ``*_ol`` (empresa terceira), ``*_llm`` (AI-PAVE-Br) e nomes sem
sufixo (gold).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import pandas as pd

META_ANOT = {"analista", "data", "revisor", "observacao", "observacoes", "status"}
META_COLS = {"id", "product_type", "extracted_attributes"}


def build(pave_dir: str) -> tuple[list[dict], list[dict]]:
    longo: list[dict] = []
    itens: list[dict] = []
    for p in sorted(glob.glob(os.path.join(pave_dir, "*.csv"))):
        cat = os.path.basename(p)[:-4]
        row0 = pd.read_csv(p, nrows=0).columns.tolist()
        df = pd.read_csv(p, header=1, low_memory=False)
        cols = df.columns.tolist()

        i_title = row0.index("title") if "title" in row0 else None
        i_desc = row0.index("description") if "description" in row0 else None
        if i_title is None:
            raise ValueError(f"{cat}: coluna 'title' ausente na linha 0")

        gold_cols = [c for c in cols
                     if not c.endswith("_ol") and not c.endswith("_llm")
                     and c not in META_COLS and not str(c).startswith("Unnamed")
                     and c.strip().lower() not in META_ANOT]

        for i, r in df.iterrows():
            uid = f"{cat}||{i}"
            titulo = r.iloc[i_title]
            if pd.isna(titulo) or not str(titulo).strip():
                continue
            itens.append(dict(
                item_uid=uid, categoria=cat,
                product_type=r["product_type"] if "product_type" in cols else cat,
                title=str(titulo).strip(),
                description=("" if i_desc is None or pd.isna(r.iloc[i_desc])
                             else str(r.iloc[i_desc])[:512]),
            ))
            for g in gold_cols:
                longo.append(dict(
                    item_uid=uid, categoria=cat, atributo=g,
                    gold=r[g],
                    pred_ol=r.get(f"{g}_ol"),
                    pred_llm=r.get(f"{g}_llm"),
                ))
    return itens, longo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pave-dir", required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "pave"))
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    itens, longo = build(a.pave_dir)

    with (out / "itens.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(itens[0].keys())); w.writeheader(); w.writerows(itens)
    with (out / "gold_longo.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(longo[0].keys())); w.writeheader(); w.writerows(longo)

    print(f"itens.csv      : {len(itens)} itens em {len({i['categoria'] for i in itens})} categorias")
    print(f"gold_longo.csv : {len(longo)} pares (item, atributo)")
    print(f"com description: {sum(1 for i in itens if i['description'])}")
