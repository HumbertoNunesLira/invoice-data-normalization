#!/usr/bin/env python3
"""Gera os números da resposta ao revisor (ticket de comparação).

Uso, da raiz do repo:

    python revisao/gerar_numeros.py                       # só a tabela de significância
    python revisao/gerar_numeros.py --pave-dir CAMINHO    # + cobertura de schema do AI-PAVE-Br

Os CSVs do AI-PAVE-Br não são versionados aqui (CC BY-NC-SA). Baixe de
https://github.com/ai-luizalabs/AI-PAVE-Br (pasta Dataset_in_CSVs) e aponte
--pave-dir para ela.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DADOS = REPO / "main_supersaurus" / "dados"
OUT = REPO / "revisao"

MODELOS = {
    "Gemini 3.1 Flash-Lite": "pairs_normalized.csv",
    "GPT-5.4-mini": "pairs_normalized_openai_gpt5.4mini.csv",
    "DeepSeek V4 Flash": "pairs_normalized_deepseek.csv",
}


# ── significância ────────────────────────────────────────────────────
def _f1(tp, fp, fn):
    tp, fp, fn = np.asarray(tp, float), np.asarray(fp, float), np.asarray(fn, float)
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)


def _clusters(title_a, title_b, label):
    """cluster_id por par: union-find sobre label==1 (mesma definição da Seção 3.5.1)."""
    par: dict[str, str] = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b, y in zip(title_a, title_b, label):
        if int(y) == 1:
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb

    ids, seen = [], {}
    for i, (a, b) in enumerate(zip(title_a, title_b)):
        key = find(a) if find(a) == find(b) else f"__par__{i}"
        ids.append(seen.setdefault(key, len(seen)))
    return np.asarray(ids)


def paired_bootstrap(y, pa, pb, cid, n=10000, seed=42):
    """IC bootstrap pareado sobre ΔF1, reamostrando CLUSTERS (não linhas).

    Reamostrar linhas trata variantes do mesmo produto como observações
    independentes e estreita o IC artificialmente. Como F1 só depende de
    contagens tp/fp/fn e cada cluster contribui com uma tripla fixa, a
    reamostragem vira uma multinomial sobre as contribuições — exata e vetorizável.
    """
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    k = int(cid.max()) + 1
    agg = lambda p, m: np.bincount(cid, weights=m(y, p).astype(float), minlength=k)
    tp = lambda y_, p: (y_ == 1) & (p == 1)
    fp = lambda y_, p: (y_ == 0) & (p == 1)
    fn = lambda y_, p: (y_ == 1) & (p == 0)

    A = [agg(pa, tp), agg(pa, fp), agg(pa, fn)]
    B = [agg(pb, tp), agg(pb, fp), agg(pb, fn)]

    rng = np.random.default_rng(seed)
    c = rng.multinomial(k, np.full(k, 1 / k), size=n).astype(float)
    d = _f1(c @ A[0], c @ A[1], c @ A[2]) - _f1(c @ B[0], c @ B[1], c @ B[2])

    fa = float(_f1(A[0].sum(), A[1].sum(), A[2].sum()))
    fb = float(_f1(B[0].sum(), B[1].sum(), B[2].sum()))
    return dict(
        f1_a=round(fa, 4), f1_b=round(fb, 4), delta_f1=round(fa - fb, 4),
        ci_lo=round(float(np.percentile(d, 2.5)), 4),
        ci_hi=round(float(np.percentile(d, 97.5)), 4),
        significativo=bool(np.percentile(d, 2.5) > 0),
        n_clusters=k,
    )


def mcnemar(y, pa, pb):
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    b = int(((pa == y) & (pb != y)).sum())
    c = int(((pa != y) & (pb == y)).sum())
    n = b + c
    if n == 0:
        return dict(only_a=b, only_b=c, p_value=1.0)
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return dict(only_a=b, only_b=c, p_value=round(min(2 * tail, 1.0), 6))


def tabela_significancia() -> list[dict]:
    dfs = {m: pd.read_csv(DADOS / f) for m, f in MODELOS.items()}
    base = pd.read_csv(DADOS / "baseline_predictions.csv")
    ref = dfs["Gemini 3.1 Flash-Lite"]
    y = ref["label"].values
    cid = _clusters(ref["title_a"], ref["title_b"], y)

    preds = {f"{m} (Weighted)": d["weighted_pred"].values for m, d in dfs.items()}
    preds["TF-IDF palavra"] = base["pred_tfidf_word"].values
    preds["TF-IDF caractere"] = base["pred_tfidf_char"].values

    a = "Gemini 3.1 Flash-Lite (Weighted)"
    linhas = []
    for b_name, pb in preds.items():
        if b_name == a:
            continue
        r = paired_bootstrap(y, preds[a], pb, cid)
        m = mcnemar(y, preds[a], pb)
        linhas.append(dict(
            metodo_a=a, metodo_b=b_name,
            f1_a=r["f1_a"], f1_b=r["f1_b"],
            delta_f1_pp=round(r["delta_f1"] * 100, 1),
            ci95_lo_pp=round(r["ci_lo"] * 100, 1), ci95_hi_pp=round(r["ci_hi"] * 100, 1),
            bootstrap_significativo=r["significativo"],
            mcnemar_so_a_acerta=m["only_a"], mcnemar_so_b_acerta=m["only_b"],
            mcnemar_p=m["p_value"], mcnemar_significativo=m["p_value"] < 0.05,
            n_pares=len(y), n_positivos=int(y.sum()), n_clusters=r["n_clusters"],
        ))
    return linhas


# ── cobertura de schema AI-PAVE-Br ───────────────────────────────────
META_ANOT = {"analista", "data", "revisor", "observacao", "observacoes", "status"}


def _mapeia(col: str, frouxo: bool = False):
    """Campo do TAIL correspondente, ou None.

    Estrito exige mesma semântica. O modo frouxo acrescenta 'tipo*', que é onde a
    correspondência fica discutível (tipo_de_degelo, tipo_de_veiculo não são
    tipo de produto) — reportado só para mostrar o teto otimista.
    """
    c = col.strip().lower()
    if c == "marca": return "marca"
    if c == "cor": return "cor"
    if c == "voltagem": return "voltagem"
    if c in ("material", "material_da_estrutura", "material_do_tampo"): return "material"
    if c.startswith("capacidade"): return "capacidade_volume"
    if c == "quantidade" or c.startswith("quantidade_de"): return "quantidade"
    if frouxo and (c == "tipo" or c.startswith("tipo_de")): return "tipo_produto"
    return None


def tabela_cobertura(pave_dir: str) -> list[dict]:
    linhas = []
    for p in sorted(glob.glob(os.path.join(pave_dir, "*.csv"))):
        cat = os.path.basename(p)[:-4]
        # linha 0 = rótulos de grupo mesclados; linha 1 = nomes reais das colunas
        df = pd.read_csv(p, header=1, low_memory=False)
        cols = df.columns.tolist()
        gold = [c for c in cols
                if not c.endswith("_ol") and not c.endswith("_llm")
                and c not in {"id", "product_type", "extracted_attributes"}
                and not str(c).startswith("Unnamed")
                and c.strip().lower() not in META_ANOT]
        ns = sum(1 for c in gold if _mapeia(c))
        nl = sum(1 for c in gold if _mapeia(c, frouxo=True))
        linhas.append(dict(
            categoria=cat, n_itens=len(df), n_slots_gold=len(gold),
            n_mapeados_estrito=ns, n_mapeados_frouxo=nl, n_nao_mapeados=len(gold) - ns,
            tem_product_type="product_type" in cols,
            gold_sem_correspondente=";".join(c for c in gold if not _mapeia(c)),
        ))
    return linhas


def _escreve(nome: str, linhas: list[dict]) -> None:
    caminho = OUT / nome
    with caminho.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"  -> {caminho.relative_to(REPO)}  ({len(linhas)} linhas)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pave-dir", help="pasta Dataset_in_CSVs do AI-PAVE-Br")
    args = ap.parse_args()

    sig = tabela_significancia()
    print(f"Significância par-a-par ({sig[0]['n_pares']} pares, {sig[0]['n_positivos']} positivos):")
    _escreve("T3_significancia.csv", sig)
    for r in sig:
        print(f"     vs {r['metodo_b']:34s} {r['delta_f1_pp']:+6.1f} pp  "
              f"IC95%[{r['ci95_lo_pp']:+.1f},{r['ci95_hi_pp']:+.1f}]  "
              f"McNemar p={r['mcnemar_p']:.5g}  "
              f"{'SIGNIFICATIVO' if r['bootstrap_significativo'] else 'nao significativo'}")

    if args.pave_dir:
        print("\nCobertura de schema AI-PAVE-Br:")
        cob = tabela_cobertura(args.pave_dir)
        _escreve("T2_cobertura_schema_aipavebr.csv", cob)
        ti = sum(r["n_itens"] for r in cob); tg = sum(r["n_slots_gold"] for r in cob)
        ts = sum(r["n_mapeados_estrito"] for r in cob); tl = sum(r["n_mapeados_frouxo"] for r in cob)
        print(f"     {len(cob)} categorias | {ti} itens | {tg} slots gold")
        print(f"     estrito {ts}/{tg} = {ts/tg:.1%} | frouxo {tl}/{tg} = {tl/tg:.1%}")
