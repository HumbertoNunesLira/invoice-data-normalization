#!/usr/bin/env python3
"""Quantos valores numéricos extraídos aparecem só na descrição (não no título)?

Indício de que a descrição fez parte da entrada de um sistema: se ele extrai um
número que não está no título mas está na descrição, provavelmente leu a descrição.
Não é prova (o modelo pode ter inferido o valor por conhecimento prévio).

Para cada valor não vazio de cada sistema (e do gold), extrai os números
(``\\d+(?:[.,]\\d+)?``, separador de milhar removido, vírgula decimal -> ponto) e
classifica cada número em: aparece no título / só na descrição / em nenhum dos dois.
Faz isso com a descrição cortada em 512 caracteres (como em ``pave/itens.csv``) e
com a descrição completa dos CSVs originais.

Uso:  python revisao/numeros_descricao.py --pave-dir Dataset_in_CSVs
Saída: revisao/T7_numeros_so_na_descricao.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import pandas as pd

from pave_score import GOLD2TAIL, PAVE, eh_nulo

csv.field_size_limit(10 ** 8)
OUT = Path(__file__).resolve().parent
NUM = re.compile(r"\d+(?:\.\d+)?")


def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"(\d)\.(?=\d{3}\b)", r"\1", s)     # 38.000 -> 38000
    return s.replace(",", ".")                     # 8,5 -> 8.5


def _tem(numero: str, texto: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(numero)}(?![\d])", texto) is not None


def descricoes_completas(pave_dir: str) -> dict[str, str]:
    """item_uid -> descrição completa (mesma indexação de pave_dataset.build)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(pave_dir, "*.csv"))):
        cat = os.path.basename(p)[:-4]
        row0 = pd.read_csv(p, nrows=0).columns.tolist()
        df = pd.read_csv(p, header=1, low_memory=False)
        i_t = row0.index("title")
        i_d = row0.index("description") if "description" in row0 else None
        for i, r in df.iterrows():
            t = r.iloc[i_t]
            if pd.isna(t) or not str(t).strip():
                continue
            out[f"{cat}||{i}"] = "" if i_d is None or pd.isna(r.iloc[i_d]) else str(r.iloc[i_d])
    return out


def contar(linhas, coluna, titulo, descricao) -> dict:
    tot = no_titulo = so_desc = nenhum = 0
    for r in linhas:
        v = r[coluna]
        if eh_nulo(v):
            continue
        T, D = _norm(titulo[r["item_uid"]]), _norm(descricao(r["item_uid"]))
        for n in NUM.findall(_norm(v)):
            tot += 1
            if _tem(n, T):
                no_titulo += 1
            elif _tem(n, D):
                so_desc += 1
            else:
                nenhum += 1
    return dict(n_valores_numericos=tot, pct_no_titulo=round(100 * no_titulo / tot, 1),
                pct_so_na_descricao=round(100 * so_desc / tot, 1), pct_em_nenhum=round(100 * nenhum / tot, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pave-dir", required=True, help="pasta Dataset_in_CSVs do AI-PAVE-Br")
    a = ap.parse_args()

    longo = list(csv.DictReader((PAVE / "gold_longo.csv").open(encoding="utf-8")))
    itens = {r["item_uid"]: r for r in csv.DictReader((PAVE / "itens.csv").open(encoding="utf-8"))}
    titulo = {u: r["title"] for u, r in itens.items()}
    completa = descricoes_completas(a.pave_dir)
    d512 = lambda u: itens[u]["description"]
    dful = lambda u: completa.get(u, "")
    longas = sum(1 for v in completa.values() if len(v) > 512)
    print(f"{sum(1 for v in completa.values() if v)} descrições preenchidas; {longas} com mais de 512 caracteres")

    escopos = {
        "todos os atributos gold": [r for r in longo if r["item_uid"] in itens],
        "atributos mapeados ao TAIL": [r for r in longo if r["item_uid"] in itens and r["atributo"] in GOLD2TAIL],
    }
    sistemas = {"AI-PAVE-Br (pred_llm)": "pred_llm", "Baseline empresa terceira (pred_ol)": "pred_ol", "Gold": "gold"}
    linhas = []
    for esc, rows in escopos.items():
        for nome, col in sistemas.items():
            for rot, dfn in [("512 caracteres", d512), ("completa", dful)]:
                m = contar(rows, col, titulo, dfn)
                linhas.append(dict(escopo=esc, fonte=nome, descricao=rot, **m))
                print(f"{esc:28s} {nome:36s} {rot:15s} só na descrição: {m['pct_so_na_descricao']:5.1f}%  "
                      f"no título: {m['pct_no_titulo']:5.1f}%  em nenhum: {m['pct_em_nenhum']:5.1f}%  (n={m['n_valores_numericos']})")
    with (OUT / "T7_numeros_so_na_descricao.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    print("-> revisao/T7_numeros_so_na_descricao.csv")


if __name__ == "__main__":
    main()
