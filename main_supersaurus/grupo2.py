"""Grupo 2 — experimentos novos com poucas chamadas à API.

Usado pelas Seções 11–13 do ``benchmark.ipynb``:

* **11 — Ablação do pipeline (Gemini)**: rodada completa (reprodução + JSON cru),
  sem pré-normalização, sem pós-processamento (derivada do JSON cru da rodada
  completa, sem chamadas novas) e com o prompt sem as regras 5 a 9.
* **12 — Comparação direta**: o Gemini decide, par a par (lotes de 5), se os dois
  títulos são o mesmo produto.
* **13 — Linha de base com embeddings**: similaridade do cosseno com modelos
  multilíngues de sentenças, rodando localmente.

Pré-normalização, prompt, parse e limpeza vêm de ``inference_harness.normalization``,
que é idêntico ao ``classifier_api.ipynb`` (conferido: mesmo SYSTEM_PROMPT, mesma
pré-normalização nos 2.257 títulos, mesma limpeza/nome canônico).

Toda chamada à API é salva em JSONL (um lote por linha) em ``dados/grupo2/``.
Re-executar retoma de onde parou e não repete lotes já respondidos; lotes que
falharam são refeitos.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inference_harness.normalization import (  # noqa: E402
    GEMINI_USER_TEMPLATE,
    SYSTEM_PROMPT,
    build_canonical_name,
    clean_llm_output,
    parse_llm_json,
    pre_normalize_title,
)

logger = logging.getLogger("grupo2")

DADOS = HERE / "dados"
PAIRS_INPUT = DADOS / "pairs_to_label.csv"
EXP_DIR = DADOS / "grupo2"
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
BATCH_SIZE = 5
STRUCT_FIELDS = ["tipo_produto", "marca", "modelo", "capacidade_volume", "voltagem", "quantidade"]

_HASH_SUFFIX = re.compile(r"\s*#[a-z0-9]{4,}\s*$", re.I)


# ═══════════════════════════════════════════════════════════════════
#  Dados
# ═══════════════════════════════════════════════════════════════════

def strip_hash(title: str) -> str:
    """Remove o sufixo '#abcd' (mesma regra do Passo 6 do classifier_api)."""
    return _HASH_SUFFIX.sub("", str(title)).strip()


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(PAIRS_INPUT, dtype=str)
    df["label"] = df["label"].str.strip().astype(int)
    df["a_clean"] = [strip_hash(t) for t in df["title_a"]]
    df["b_clean"] = [strip_hash(t) for t in df["title_b"]]
    return df


def unique_titles(df: pd.DataFrame) -> list[str]:
    """Mesma ordem do Passo 6 (A depois B, sem repetição) — mantém a composição dos lotes."""
    return list(dict.fromkeys(df["a_clean"].tolist() + df["b_clean"].tolist()))


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for d in (HERE, REPO):
        if (d / ".env").is_file():
            load_dotenv(d / ".env", override=True)
            return


# ═══════════════════════════════════════════════════════════════════
#  Prompts
# ═══════════════════════════════════════════════════════════════════

def prompt_sem_regras_5_9(base: str = SYSTEM_PROMPT) -> str:
    """SYSTEM_PROMPT sem os blocos 'REGRAS CRÍTICAS DE LIMPEZA' (5–7) e
    'REGRAS DE DESAMBIGUAÇÃO SEMÂNTICA' (8–9). Schema, regras 1–4 e formato de
    saída ficam intactos."""
    linhas = base.split("\n")
    i_ini = next(i for i, l in enumerate(linhas) if l.strip().startswith("REGRAS CRÍTICAS DE LIMPEZA")) - 1
    i_fim = next(i for i, l in enumerate(linhas) if l.strip().startswith("FORMATO DE SAÍDA")) - 1
    removido = "\n".join(linhas[i_ini:i_fim])
    novo = "\n".join(linhas[:i_ini] + linhas[i_fim:])
    for n in range(5, 10):
        assert re.search(rf"^\s*{n}\. ", removido, re.M), f"regra {n} não estava no bloco removido"
        assert not re.search(rf"^\s*{n}\. ", novo, re.M), f"regra {n} ainda no prompt"
    for n in range(1, 5):
        assert re.search(rf"^\s*{n}\. ", novo, re.M), f"regra {n} sumiu do prompt"
    return novo


# Peeters & Bizer (2023), prompt "general" traduzido, em lote. Sem regras de
# domínio de propósito: é a linha de base "pergunte direto ao LLM".
PROMPT_COMPARACAO_DIRETA = """
Você receberá um ou mais pares de títulos de anúncios de produtos de e-commerce brasileiro.
Para CADA par, responda: os dois títulos se referem ao mesmo produto do mundo real?

Retorne SOMENTE um array JSON válido, sem texto extra e sem markdown, com um objeto por par:
  {"idx": <int, o índice recebido>, "mesmo_produto": true | false}
"""


# ═══════════════════════════════════════════════════════════════════
#  Cliente Gemini + cache em JSONL
# ═══════════════════════════════════════════════════════════════════

class GeminiJSON:
    """Mesma configuração do classifier_api (JSON mime, temperatura 0, 2048 tokens)."""

    def __init__(self, system_prompt: str, model_name: str = MODEL_NAME,
                 temperature: float = 0.0, max_output_tokens: int = 2048):
        import google.generativeai as genai

        load_env()
        key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("GOOGLE_API_KEY ausente (.env na raiz ou em main_supersaurus/).")
        genai.configure(api_key=key)
        self.model_name = model_name
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

    def call(self, user_prompt: str, tentativas: int = 8) -> tuple[str, dict, bool]:
        """Devolve (texto, uso de tokens, bloqueado).

        Bloqueio do filtro de segurança do provedor (ex.: PROHIBITED_CONTENT) é
        determinístico: não adianta repetir. Vira resposta vazia válida — o mesmo
        que o classifier_api faz (ValueError -> [{}] para o lote todo).
        """
        ultimo = None
        for t in range(tentativas):
            try:
                resp = self._model.generate_content(user_prompt)
                um = resp.usage_metadata
                uso = {"entrada": int(getattr(um, "prompt_token_count", 0) or 0),
                       "saida": int(getattr(um, "candidates_token_count", 0) or 0)}
                fb = getattr(resp, "prompt_feedback", None)
                if not resp.candidates or (fb is not None and getattr(fb, "block_reason", 0)):
                    logger.warning("lote bloqueado pelo provedor: %s", getattr(fb, "block_reason", "?"))
                    return "", uso, True
                return resp.text, uso, False
            except Exception as e:  # noqa: BLE001 — 429/5xx/timeout: espera e tenta de novo
                ultimo = e
                espera = min(60, 5 * 2 ** t)
                logger.warning("falha na chamada (%s/%s): %s — nova tentativa em %ss",
                               t + 1, tentativas, str(e)[:160], espera)
                time.sleep(espera)
        raise RuntimeError(f"chamada falhou após {tentativas} tentativas: {ultimo}")


def _hash(linhas: list[str]) -> str:
    return hashlib.sha1("\n".join(linhas).encode("utf-8")).hexdigest()[:16]


def _carrega_cache(path: Path) -> dict[int, dict]:
    feitos = {}
    if path.is_file():
        for linha in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(linha)
            except json.JSONDecodeError:
                continue
            if r.get("erro") is None:
                feitos[r["inicio"]] = r      # o último registro válido vence
    return feitos


def rodar_lotes(client: GeminiJSON, itens: list, formata: Callable[[int, object], str],
                cache_path: Path, batch_size: int = BATCH_SIZE, pausa: float = 0.5,
                max_lotes: int | None = None, desc: str = "") -> dict[int, dict]:
    """Envia `itens` em lotes indexados; devolve {inicio_do_lote: registro}.

    Cada registro: inicio, n, hash das linhas, texto cru, uso de tokens, erro.
    Lotes já respondidos (mesmo hash) não são reenviados.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feitos = _carrega_cache(cache_path)
    inicios = list(range(0, len(itens), batch_size))
    if max_lotes is not None:
        inicios = inicios[:max_lotes]

    pendentes = []
    for ini in inicios:
        linhas = [formata(ini + j, it) for j, it in enumerate(itens[ini:ini + batch_size])]
        r = feitos.get(ini)
        if r is None or r.get("hash") != _hash(linhas):
            pendentes.append((ini, linhas))

    if pendentes:
        from tqdm.auto import tqdm
        with cache_path.open("a", encoding="utf-8") as fh:
            for ini, linhas in tqdm(pendentes, desc=desc or cache_path.stem, unit="lote"):
                reg = {"inicio": ini, "n": len(linhas), "hash": _hash(linhas),
                       "modelo": client.model_name, "texto": None, "uso": None,
                       "bloqueado": False, "erro": None}
                try:
                    reg["texto"], reg["uso"], reg["bloqueado"] = client.call("\n".join(linhas))
                except Exception as e:  # noqa: BLE001
                    reg["erro"] = str(e)[:500]
                fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
                fh.flush()
                if reg["erro"] is None:
                    feitos[ini] = reg
                time.sleep(pausa)

    faltando = [ini for ini in inicios if ini not in feitos]
    if faltando:
        logger.warning("%d lote(s) sem resposta válida em %s — rode de novo para refazer", len(faltando), cache_path.name)
    return {ini: feitos[ini] for ini in inicios if ini in feitos}


def _casa_por_idx(raw_list: list, inicio: int, n: int) -> list[dict]:
    """Mesma regra do classifier_api: casa por 'idx'; senão, pela posição."""
    out = []
    for local in range(n):
        alvo = inicio + local
        achado = next((r for r in raw_list if isinstance(r, dict) and r.get("idx") == alvo), None)
        if achado is None and local < len(raw_list) and isinstance(raw_list[local], dict):
            achado = raw_list[local]
        out.append(achado or {})
    return out


def uso_total(registros: dict[int, dict]) -> dict:
    e = sum((r.get("uso") or {}).get("entrada", 0) for r in registros.values())
    s = sum((r.get("uso") or {}).get("saida", 0) for r in registros.values())
    b = sum(1 for r in registros.values() if r.get("bloqueado"))
    return {"chamadas": len(registros), "tokens_entrada": e, "tokens_saida": s, "lotes_bloqueados": b}


# ═══════════════════════════════════════════════════════════════════
#  11 — Ablação do pipeline
# ═══════════════════════════════════════════════════════════════════

# Rodadas que chamam a API. "sem_pos" não está aqui: sai do JSON cru de "completa".
RODADAS_ABLACAO = {
    "completa":       {"pre": True,  "prompt": "completo"},
    "sem_pre":        {"pre": False, "prompt": "completo"},
    "sem_regras_5_9": {"pre": True,  "prompt": "sem_5_9"},
}
VARIANTES_ABLACAO = {
    "completa":       ("completa", True),        # (rodada de origem, aplica pós-processamento?)
    "sem_pre":        ("sem_pre", True),
    "sem_pos":        ("completa", False),
    "sem_regras_5_9": ("sem_regras_5_9", True),
}


def _prompt(nome: str) -> str:
    return SYSTEM_PROMPT if nome == "completo" else prompt_sem_regras_5_9()


def cache_ablacao(rodada: str) -> Path:
    return EXP_DIR / f"ablacao_{rodada}.jsonl"


def rodar_ablacao(rodada: str, max_lotes: int | None = None, df: pd.DataFrame | None = None) -> dict[int, dict]:
    cfg = RODADAS_ABLACAO[rodada]
    df = load_pairs() if df is None else df
    titulos = unique_titles(df)
    entrada = [pre_normalize_title(t) for t in titulos] if cfg["pre"] else titulos
    client = GeminiJSON(_prompt(cfg["prompt"]))
    fmt = lambda i, t: GEMINI_USER_TEMPLATE.format(idx=i, categoria="", titulo=t)
    return rodar_lotes(client, entrada, fmt, cache_ablacao(rodada), max_lotes=max_lotes,
                       desc=f"ablação {rodada}")


def raw_por_titulo(rodada: str, df: pd.DataFrame | None = None) -> tuple[dict[str, dict], dict]:
    """{título limpo: JSON cru} a partir do cache da rodada + resumo de cobertura."""
    df = load_pairs() if df is None else df
    titulos = unique_titles(df)
    regs = _carrega_cache(cache_ablacao(rodada))
    raw = {}
    for ini in range(0, len(titulos), BATCH_SIZE):
        n = min(BATCH_SIZE, len(titulos) - ini)
        r = regs.get(ini)
        lista = parse_llm_json(r["texto"]) if r and r.get("texto") else []
        for t, j in zip(titulos[ini:ini + n], _casa_por_idx(lista, ini, n)):
            raw[t] = j
    vazios = sum(1 for t in titulos if not raw.get(t))
    resumo = {"titulos": len(titulos), "lotes_respondidos": len(regs),
              "lotes_total": -(-len(titulos) // BATCH_SIZE), "titulos_sem_json": vazios,
              **uso_total(regs)}
    return raw, resumo


def sem_limpeza(raw: dict) -> dict:
    """Saída do LLM sem a camada simbólica: só descarta 'idx' e nulos.
    Nada de minúsculas, unidades, voltagem ou colapso de quantidade implícita."""
    out = {}
    for k, v in (raw or {}).items():
        if k == "idx":
            continue
        s = None if v is None else str(v).strip()
        out[k] = None if (s is None or s == "" or s.lower() == "null") else s
    return out


def montar_pares(df: pd.DataFrame, raw: dict[str, dict], pos: bool) -> pd.DataFrame:
    """Mesmo schema de dados/pairs_normalized.csv (Passo 6 do classifier_api)."""
    limpo = {t: (clean_llm_output(j) if pos else sem_limpeza(j)) for t, j in raw.items()}
    canon = {t: build_canonical_name(c) if c else "" for t, c in limpo.items()}
    out = df[["title_a", "title_b", "label"]].copy()
    out["item_a_normalized"] = [canon.get(t, "") for t in df["a_clean"]]
    out["item_b_normalized"] = [canon.get(t, "") for t in df["b_clean"]]
    for f in STRUCT_FIELDS:
        out[f"a_{f}"] = [limpo.get(t, {}).get(f) for t in df["a_clean"]]
        out[f"b_{f}"] = [limpo.get(t, {}).get(f) for t in df["b_clean"]]
    return out


def pares_variante(variante: str, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Monta e salva dados/grupo2/pairs_normalized_gemini_<variante>.csv."""
    df = load_pairs() if df is None else df
    rodada, pos = VARIANTES_ABLACAO[variante]
    raw, resumo = raw_por_titulo(rodada, df)
    pares = montar_pares(df, raw, pos)
    pares.to_csv(EXP_DIR / f"pairs_normalized_gemini_{variante}.csv", index=False, encoding="utf-8-sig")
    return pares, resumo


# ═══════════════════════════════════════════════════════════════════
#  12 — Comparação direta par a par
# ═══════════════════════════════════════════════════════════════════

CACHE_DIRETA = EXP_DIR / "comparacao_direta.jsonl"


def _fmt_par(i: int, par: tuple[str, str]) -> str:
    return f'[{i}] Título A: "{par[0]}" | Título B: "{par[1]}"'


def rodar_comparacao_direta(max_lotes: int | None = None, df: pd.DataFrame | None = None) -> dict[int, dict]:
    df = load_pairs() if df is None else df
    pares = list(zip(df["a_clean"], df["b_clean"]))
    client = GeminiJSON(PROMPT_COMPARACAO_DIRETA)
    return rodar_lotes(client, pares, _fmt_par, CACHE_DIRETA, max_lotes=max_lotes, desc="comparação direta")


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in {"true", "sim", "yes", "1"}:
        return True
    if isinstance(v, str) and v.strip().lower() in {"false", "nao", "não", "no", "0"}:
        return False
    return None


def predicoes_comparacao_direta(df: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    """Predição 0/1 por par. Par sem resposta válida conta como 0 (não-match) e é reportado."""
    df = load_pairs() if df is None else df
    regs = _carrega_cache(CACHE_DIRETA)
    n = len(df)
    pred = np.zeros(n, int)
    sem_resposta = 0
    for ini in range(0, n, BATCH_SIZE):
        k = min(BATCH_SIZE, n - ini)
        r = regs.get(ini)
        lista = parse_llm_json(r["texto"]) if r and r.get("texto") else []
        for local, j in enumerate(_casa_por_idx(lista, ini, k)):
            v = _bool(j.get("mesmo_produto")) if j else None
            if v is None:
                sem_resposta += 1
            pred[ini + local] = int(bool(v))
    resumo = {"pares": n, "lotes_respondidos": len(regs), "lotes_total": -(-n // BATCH_SIZE),
              "pares_sem_resposta": sem_resposta, **uso_total(regs)}
    return pred, resumo


# ═══════════════════════════════════════════════════════════════════
#  13 — Embeddings (local)
# ═══════════════════════════════════════════════════════════════════

MODELOS_EMBEDDING = {
    # SBERT multilíngue clássico (Reimers & Gurevych, 2020).
    "SBERT paraphrase-multilingual-mpnet": {"id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "prefixo": ""},
    # E5 multilíngue (Wang et al., 2024) — exige o prefixo "query: " em tarefas simétricas.
    "multilingual-e5-base": {"id": "intfloat/multilingual-e5-base", "prefixo": "query: "},
}


def similaridades_embedding(nome: str, df: pd.DataFrame | None = None, batch_size: int = 64) -> np.ndarray:
    """Cosseno entre os embeddings de A e B (títulos sem o sufixo de hash). Cache em CSV."""
    df = load_pairs() if df is None else df
    cfg = MODELOS_EMBEDDING[nome]
    cache = EXP_DIR / f"emb_{re.sub(r'[^a-z0-9]+', '_', nome.lower())}.csv"
    if cache.is_file():
        c = pd.read_csv(cache, dtype={"sim": float})
        if len(c) == len(df) and (c["title_a"].values == df["title_a"].values).all():
            return c["sim"].values

    from sentence_transformers import SentenceTransformer

    titulos = unique_titles(df)
    modelo = SentenceTransformer(cfg["id"], device="cpu")
    emb = modelo.encode([cfg["prefixo"] + t for t in titulos], batch_size=batch_size,
                        normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    pos = {t: i for i, t in enumerate(titulos)}
    ea = emb[[pos[t] for t in df["a_clean"]]]
    eb = emb[[pos[t] for t in df["b_clean"]]]
    sims = (ea * eb).sum(axis=1)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"title_a": df["title_a"], "title_b": df["title_b"], "sim": sims}).to_csv(cache, index=False)
    return sims


# ═══════════════════════════════════════════════════════════════════
#  Limiar (para linhas de base com escore contínuo)
# ═══════════════════════════════════════════════════════════════════

def _f1(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)


def limiar_otimo(y: np.ndarray, sims: np.ndarray, grade: np.ndarray | None = None) -> tuple[float, float]:
    """Mesmo critério do baseline.ipynb: limiar que maximiza F1 no conjunto todo
    (grade de 200 pontos em [0, 1]). Otimista por construção."""
    grade = np.linspace(0, 1, 200) if grade is None else grade
    y = np.asarray(y)
    P = sims[None, :] >= grade[:, None]
    f1 = _f1((P & (y == 1)).sum(1), (P & (y == 0)).sum(1), (~P & (y == 1)).sum(1))
    j = int(np.argmax(f1))
    return float(grade[j]), float(f1[j])


def cv_limiar(y: np.ndarray, sims: np.ndarray, cid: np.ndarray, grade: np.ndarray | None = None,
              bloqueado: np.ndarray | None = None, n_rep: int = 200, k: int = 5, seed: int = 42) -> np.ndarray:
    """F1 com limiar escolhido só nos folds de treino (folds por cluster, estratificados
    por clusters positivos). Devolve o F1 de cada repetição."""
    grade = np.linspace(0, 1, 200) if grade is None else grade
    y = np.asarray(y)
    bloq = np.zeros(len(y), bool) if bloqueado is None else np.asarray(bloqueado, bool)
    P = (sims[None, :] >= grade[:, None]) & ~bloq[None, :]
    TPm, FPm, FNm = P & (y == 1), P & (y == 0), ~P & (y == 1)
    pos_c = np.unique(cid[y == 1])
    neg_c = np.setdiff1d(np.unique(cid), pos_c)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        fold_c = np.empty(cid.max() + 1, int)
        for grupo in (pos_c, neg_c):
            g = rng.permutation(grupo)
            fold_c[g] = (np.arange(len(g)) + rng.integers(k)) % k
        fold = fold_c[cid]
        tp = fp = fn = 0
        for f in range(k):
            tr, te = fold != f, fold == f
            f1_tr = _f1(TPm[:, tr].sum(1), FPm[:, tr].sum(1), FNm[:, tr].sum(1))
            emp = np.flatnonzero(f1_tr == f1_tr.max())
            j = emp[len(emp) // 2]
            tp += TPm[j, te].sum(); fp += FPm[j, te].sum(); fn += FNm[j, te].sum()
        out.append(float(_f1(tp, fp, fn)))
    return np.array(out)


# ═══════════════════════════════════════════════════════════════════
#  CLI — roda as chamadas fora do notebook (mesmo cache)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("experimento", choices=[*RODADAS_ABLACAO, "direta", "embeddings"])
    ap.add_argument("--max-lotes", type=int, default=None, help="limita o nº de lotes (teste)")
    a = ap.parse_args()
    if a.experimento in RODADAS_ABLACAO:
        regs = rodar_ablacao(a.experimento, a.max_lotes)
        print(uso_total(regs))
    elif a.experimento == "direta":
        regs = rodar_comparacao_direta(a.max_lotes)
        print(uso_total(regs))
    else:
        for nome in MODELOS_EMBEDDING:
            s = similaridades_embedding(nome)
            print(nome, f"sim média={s.mean():.3f}")
