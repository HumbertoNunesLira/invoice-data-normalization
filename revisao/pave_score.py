#!/usr/bin/env python3
"""Métrica PAVE aplicada aos três sistemas sobre o Golden Set do AI-PAVE-Br.

IMPORTANTE — sobre reprodutibilidade dos números deles: o repositório publicado
traz 22 categorias / 8.274 itens, enquanto o artigo reporta ~7.700 itens em 20
categorias. Não é possível reconstruir o subconjunto exato que geraram, e por
isso **não** tentamos casar com os valores publicados (F1 0,7468 / coverage
71,96%). Em vez disso recomputamos os TRÊS sistemas com esta mesma
implementação, sobre os mesmos itens. A comparação é internamente consistente e
reproduzível; a divergência contra o publicado está reportada no README.

Métrica (seguindo a descrição deles: "set-based F1 para atributos multivalorados,
token-level F1 entre valores"):

* Valores multivalorados vêm como ``[Noz, Avelã]`` — separados por vírgula ou
  " e ", entre colchetes.
* Cada valor é normalizado (minúsculas, sem acento, ``8 KG``→``8kg``,
  ``40 litros``→``40l``, vírgula decimal→ponto) e quebrado em tokens.
* P/R/F1 micro sobre a interseção de multiconjuntos de tokens.
* Coverage = fração das células avaliáveis em que o sistema produziu saída
  não vazia (``Não informado`` conta como vazio).

Só entram células com gold não vazio: não há como pontuar contra gold ausente.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PAVE = Path(__file__).resolve().parent / "pave"
csv.field_size_limit(10 ** 7)

NULOS = {"não informado", "nao informado", "n/a", "na", "none", "null", "-", "", "nan"}
STOP = {"de", "da", "do", "e", "com", "para", "em", "a", "o"}

# Campo do pipeline -> nomes de atributo gold que ele deve responder.
MAPA = {
    "marca": {"marca"},
    "cor": {"cor"},
    "voltagem": {"voltagem"},
    "material": {"material", "material_da_estrutura", "material_do_tampo"},
    "capacidade_volume": {"capacidade", "capacidade_de_lavagem", "capacidade_liquida_total",
                          "capacidade_liquida_de_armazenamento", "capacidade_de_refrigeracao",
                          "capacidade_do_hd", "capacidade_do_ssd"},
    "quantidade": {"quantidade", "quantidade_de_portas", "quantidade_de_gavetas",
                   "quantidade_de_pecas", "quantidade_de_pares", "quantidade_de_pneus",
                   "quantidade_de_lugares", "quantidade_de_bocas"},
    "tipo_produto": {"product_type"},
}
GOLD2TAIL = {g: t for t, gs in MAPA.items() for g in gs}

# Recorte "schema alinhado", decidido A PRIORI pela definição dos campos do TAIL —
# não pelos resultados. Excluímos os atributos gold que o schema do TAIL não tem
# como responder, por dois motivos estruturais:
#
#  (a) conceito ausente: `capacidade_volume` do pipeline é volume/massa física (o prompt
#      manda converter para ml/kg); armazenamento digital (HD/SSD) não é isso.
#  (b) pergunta indistinguível: o pipeline tem UM campo genérico `quantidade`, enquanto
#      o gold deles tem sete contagens distintas (portas, gavetas, bocas, peças,
#      pares, pneus, lugares). Um campo não responde sete perguntas diferentes.
#
# Este recorte mede o que o TAIL se propõe a extrair; o escopo completo mede o
# custo de um schema fixo cross-categoria frente a um schema por categoria.
NAO_MODELADOS = {
    "capacidade_do_hd", "capacidade_do_ssd",              # (a)
    "quantidade_de_portas", "quantidade_de_gavetas",      # (b)
    "quantidade_de_bocas", "quantidade_de_pecas",
    "quantidade_de_pares", "quantidade_de_pneus",
    "quantidade_de_lugares", "quantidade",
}


def eh_nulo(v) -> bool:
    return v is None or str(v).strip().lower() in NULOS


# Só colapsa dígito+espaço+UNIDADE física ("8 kg"->"8kg"). Colapsar dígito+palavra
# qualquer quebraria "1 Porta" em "1porta", que não casa com o "1" das predições.
_UNIDADES = (r"kg|kgs|g|mg|t|l|lt|ml|litros?|quilos?|v|volts?|w|kw|watts?|"
             r"cm|mm|m|pol|polegadas?|btus?|hz|ghz|mhz|gb|tb|mb|rpm|ah?")
_UNI = re.compile(rf"(\d)\s+(?={_UNIDADES}\b)")
_DEC = re.compile(r"(\d),(\d)")                   # "8,5"  -> "8.5"
_MIL = re.compile(r"(\d)\.(?=\d{3}\b)")           # "38.000" -> "38000"


def normaliza(txt: str) -> str:
    s = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode().lower()
    s = s.strip().strip("[]")
    s = _MIL.sub(r"\1", s)
    s = _DEC.sub(r"\1.\2", s)
    s = _UNI.sub(r"\1", s)
    # sinônimos de unidade que aparecem no gold vs nas predições
    s = re.sub(r"\b(\d+(?:\.\d+)?)(litros|litro|lt)\b", r"\1l", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)(volts|volt)\b", r"\1v", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)(quilos|quilo|kgs)\b", r"\1kg", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)(btus|btu/h|btu)\b", r"\1btu", s)
    return s


# Equivalência de unidade: o prompt do TAIL canonicaliza volume para ml e massa
# para kg ("converter SEMPRE para ml"), enquanto o gold deles usa litros. `40L` e
# `40000ml` são a mesma quantidade física. Convertemos ambos a uma base única
# antes de comparar — aplicado aos TRÊS sistemas, simetricamente.
_QTD = re.compile(r"\b(\d+(?:\.\d+)?)(ml|l|g|kg|mg)\b")
_BASE = {"ml": ("l", 0.001), "l": ("l", 1.0), "mg": ("kg", 1e-6), "g": ("kg", 0.001), "kg": ("kg", 1.0)}


def _canon_unidade(s: str) -> str:
    def sub(m):
        val, un = float(m.group(1)), m.group(2)
        base, fator = _BASE[un]
        v = val * fator
        return f"{v:g}{base}"
    return _QTD.sub(sub, s)


def tokens(valor, unidade_equivalente: bool = False) -> Counter:
    """Multiconjunto de tokens de um valor, possivelmente multivalorado."""
    if eh_nulo(valor):
        return Counter()
    s = normaliza(valor)
    if unidade_equivalente:
        s = _canon_unidade(s)
    partes = re.split(r",|\be\b|/", s)
    toks = []
    for p in partes:
        for t in re.split(r"[^a-z0-9.]+", p):
            t = t.strip(".")
            if t and t not in STOP:
                toks.append(t)
    return Counter(toks)


def pontua(pares, unidade_equivalente: bool = False) -> dict:
    """pares: iterável de (gold, pred). Micro P/R/F1 + coverage."""
    tp = fp = fn = 0
    aval = naovazio = 0
    for gold, pred in pares:
        if eh_nulo(gold):
            continue
        aval += 1
        if not eh_nulo(pred):
            naovazio += 1
        g = tokens(gold, unidade_equivalente)
        p = tokens(pred, unidade_equivalente)
        inter = sum((g & p).values())
        tp += inter
        fp += sum(p.values()) - inter
        fn += sum(g.values()) - inter
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(precision=round(P, 4), recall=round(R, 4), f1=round(F, 4),
                coverage=round(naovazio / aval, 4) if aval else 0.0,
                n_avaliados=aval, tp=tp, fp=fp, fn=fn)


def carrega_tail(path: Path) -> dict:
    pred = {}
    if not path.is_file():
        return pred
    with path.open(encoding="utf-8") as fh:
        for linha in fh:
            try:
                r = json.loads(linha)
            except Exception:
                continue
            pred[r["item_uid"]] = r.get("cleaned") or {}
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", default=str(PAVE / "tail_pred.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()

    with (PAVE / "gold_longo.csv").open(encoding="utf-8") as fh:
        longo = list(csv.DictReader(fh))
    with (PAVE / "itens.csv").open(encoding="utf-8") as fh:
        itens = {r["item_uid"]: r for r in csv.DictReader(fh)}
    tail = carrega_tail(Path(a.tail))
    if not tail:
        sys.exit(f"sem predições do TAIL em {a.tail} — rode pave_extract.py antes")

    # Escopo avaliável: atributos que o TAIL se propõe a responder, itens extraídos,
    # gold não vazio. `product_type` fica fora (é taxonomia do catálogo, e os
    # sistemas deles não o predizem); reportado à parte.
    aval = [r for r in longo
            if r["atributo"] in GOLD2TAIL and r["item_uid"] in tail and not eh_nulo(r["gold"])]

    SIST = {
        "aipave": ("AI-PAVE-Br (Gemini 1.5 Flash)", lambda r: r["pred_llm"]),
        "terceira": ("Baseline empresa terceira", lambda r: r["pred_ol"]),
        "tail": ("TAIL (Gemini 3.1 Flash-Lite)", lambda r: tail[r["item_uid"]].get(GOLD2TAIL[r["atributo"]])),
    }
    VARIANTES = [("literal", False), ("equiv. de unidade", True)]

    # ── T4: global, dois escopos × duas métricas ──────────────
    alinhado = [r for r in aval if r["atributo"] not in NAO_MODELADOS]
    glob_rows = []
    for escopo, dados in [("todos os atributos mapeados", aval),
                          ("schema alinhado", alinhado)]:
        for var, eq in VARIANTES:
            for tag, (nome, fn) in SIST.items():
                m = pontua(((r["gold"], fn(r)) for r in dados), eq)
                glob_rows.append(dict(escopo=escopo, metrica=var, sistema=nome, **m))

    # ── T5: por atributo gold ─────────────────────────────────
    por_at = defaultdict(list)
    for r in aval:
        por_at[r["atributo"]].append(r)
    attr_rows = []
    for at in sorted(por_at, key=lambda x: -len(por_at[x])):
        sub = por_at[at]
        linha = dict(atributo_gold=at, campo_tail=GOLD2TAIL[at], n_avaliados=len(sub))
        for var, eq in VARIANTES:
            suf = "" if not eq else "_eq"
            for tag, (nome, fn) in SIST.items():
                m = pontua(((r["gold"], fn(r)) for r in sub), eq)
                linha[f"{tag}_f1{suf}"] = m["f1"]
                if not eq:
                    linha[f"{tag}_cov"] = m["coverage"]
        attr_rows.append(linha)

    # ── T6: por categoria ─────────────────────────────────────
    por_cat = defaultdict(list)
    for r in aval:
        por_cat[r["categoria"]].append(r)
    cat_rows = []
    for cat in sorted(por_cat):
        sub = por_cat[cat]
        linha = dict(categoria=cat, n_avaliados=len(sub))
        for tag, (nome, fn) in SIST.items():
            m = pontua(((r["gold"], fn(r)) for r in sub), True)
            linha[f"{tag}_f1_eq"] = m["f1"]
            linha[f"{tag}_cov"] = m["coverage"]
        cat_rows.append(linha)

    out = Path(a.out)
    for nome, rows in [("T4_pave_global.csv", glob_rows),
                       ("T5_pave_por_atributo.csv", attr_rows),
                       ("T6_pave_por_categoria.csv", cat_rows)]:
        with (out / nome).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"-> revisao/{nome}")

    print(f"\nitens extraídos: {len(tail)}   células avaliáveis: {len(aval)}\n")
    print(f"{'escopo':28s} {'métrica':18s} {'sistema':32s} {'P':>7s} {'R':>7s} {'F1':>7s} {'cov':>7s} {'n':>7s}")
    for r in glob_rows:
        print(f"{r['escopo']:28s} {r['metrica']:18s} {r['sistema']:32s} {r['precision']:7.3f} "
              f"{r['recall']:7.3f} {r['f1']:7.3f} {r['coverage']:7.1%} {r['n_avaliados']:7d}")

    pt = [r for r in longo if r["item_uid"] in tail]
    pts = [(itens[u]["product_type"], tail[u].get("tipo_produto")) for u in tail
           if itens[u].get("product_type")]
    if pts:
        m = pontua(pts)
        print(f"\n(à parte) TAIL em product_type: F1={m['f1']:.3f} cov={m['coverage']:.1%} "
              f"n={m['n_avaliados']} — taxonomia do catálogo, não extraída do título")


if __name__ == "__main__":
    main()
