"""Grupo 2 — experimentos novos com chamadas à API (três modelos).

Usado pelas Seções 11–13 do ``benchmark.ipynb``:

* **11 — Rodadas de hoje: referência + ablação do pipeline**, para Gemini 3.1
  Flash-Lite (1 execução, é determinístico), GPT-5.4-mini e DeepSeek V4 Flash
  (3 execuções cada). Variantes: completa (referência), (a) sem pré-normalização,
  (b) sem pós-processamento — derivada do JSON cru da completa, sem chamadas —
  e (c) prompt sem as regras 5 a 9.
* **12 — Comparação direta**: cada modelo decide, par a par (lotes de 5), se os dois
  títulos são o mesmo produto. 3 execuções por modelo. O prompt foi fixado antes de
  qualquer resultado e é idêntico para os três.
* **13 — Linha de base com embeddings** (local, sem API).

Protocolo de cada modelo = exatamente o do ``classifier_api.ipynb``:

=================  =======================================  ==================  ======================
modelo             template do usuário                      formato JSON        extra
=================  =======================================  ==================  ======================
Gemini             ``[i] Categoria: "" | Título: "..."``    response_mime_type  —
GPT-5.4-mini       ``[i] Título: "..."``                    nenhum              —
DeepSeek V4 Flash  ``[i] Título: "..."``                    json_object         thinking desabilitado
=================  =======================================  ==================  ======================

Todos com o mesmo SYSTEM_PROMPT, temperatura 0, 2048 tokens de saída, lotes de 5.
Pré-normalização, parse e limpeza vêm de ``inference_harness.normalization``
(idêntico ao notebook — conferido).

Toda chamada é salva em JSONL (um lote por linha) em ``dados/grupo2/``.
Re-executar retoma de onde parou e não repete lotes já respondidos; falhas
transitórias são refeitas. Bloqueio do filtro de conteúdo do provedor é
determinístico e vira resposta vazia, como no notebook original.

CLI:  python grupo2.py plano [--modelos gemini gpt deepseek] [--so-direta] [--max-lotes N]
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
BATCH_SIZE = 5
MAX_TOKENS = 2048
STRUCT_FIELDS = ["tipo_produto", "marca", "modelo", "capacidade_volume", "voltagem", "quantidade"]

OPENAI_USER_TEMPLATE = '[{idx}] Título: "{titulo}"'

MODELOS = {
    "gemini": {"nome": "Gemini 3.1 Flash-Lite", "provedor": "gemini",
               "modelo": os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview"),
               "env": "GOOGLE_API_KEY", "execucoes": 1, "execucoes_referencia": 3},
    "gpt": {"nome": "GPT-5.4-mini", "provedor": "openai", "modelo": "gpt-5.4-mini",
            "env": "OPENAI_API_KEY", "base_url": None, "json_object": False, "extra_body": None,
            "execucoes": 3, "execucoes_referencia": 5},
    "deepseek": {"nome": "DeepSeek V4 Flash", "provedor": "openai", "modelo": "deepseek-v4-flash",
                 "env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com", "json_object": True,
                 "extra_body": {"thinking": {"type": "disabled"}}, "execucoes": 3, "execucoes_referencia": 5},
}
# "execucoes" = execuções de cada variante da ablação; "execucoes_referencia" = execuções
# da rodada completa (referência), usadas também para medir a instabilidade de hoje.
EXECUCOES_DIRETA = 3

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
# FIXADO antes da primeira chamada; o mesmo texto para os três modelos. Não alterar.
PROMPT_COMPARACAO_DIRETA = """
Você receberá um ou mais pares de títulos de anúncios de produtos de e-commerce brasileiro.
Para CADA par, responda: os dois títulos se referem ao mesmo produto do mundo real?

Retorne SOMENTE um array JSON válido, sem texto extra e sem markdown, com um objeto por par:
  {"idx": <int, o índice recebido>, "mesmo_produto": true | false}
"""


# ═══════════════════════════════════════════════════════════════════
#  Clientes (devolvem texto, uso de tokens, bloqueado)
# ═══════════════════════════════════════════════════════════════════

def _espera(t: int) -> float:
    return min(60, 5 * 2 ** t)


class GeminiJSON:
    """Mesma configuração do classifier_api (JSON mime, temperatura 0, 2048 tokens)."""

    def __init__(self, system_prompt: str, cfg: dict = MODELOS["gemini"]):
        import google.generativeai as genai

        load_env()
        key = (os.getenv(cfg["env"]) or "").strip()
        if not key:
            raise RuntimeError(f"{cfg['env']} ausente (.env na raiz ou em main_supersaurus/).")
        genai.configure(api_key=key)
        self.model_name = cfg["modelo"]
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", temperature=0.0, max_output_tokens=MAX_TOKENS),
        )

    def call(self, user_prompt: str, tentativas: int = 8) -> tuple[str, dict, bool]:
        """Bloqueio do filtro de segurança (ex.: PROHIBITED_CONTENT) é determinístico:
        vira resposta vazia válida, como no classifier_api (ValueError -> [{}])."""
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
            except Exception as e:  # noqa: BLE001 — 429/5xx/timeout
                ultimo = e
                logger.warning("[%s] falha (%s/%s): %s", self.model_name, t + 1, tentativas, str(e)[:160])
                time.sleep(_espera(t))
        raise RuntimeError(f"chamada falhou após {tentativas} tentativas: {ultimo}")


class OpenAICompatJSON:
    """GPT-5.4-mini e DeepSeek V4 Flash, com os parâmetros do classifier_api (Passos 9 e 10)."""

    def __init__(self, system_prompt: str, cfg: dict):
        from openai import OpenAI

        load_env()
        key = (os.getenv(cfg["env"]) or "").strip()
        if not key:
            raise RuntimeError(f"{cfg['env']} ausente (.env na raiz ou em main_supersaurus/).")
        self.cfg = cfg
        self.model_name = cfg["modelo"]
        self.system_prompt = system_prompt
        self._client = OpenAI(api_key=key, base_url=cfg.get("base_url"), timeout=180, max_retries=0)

    def call(self, user_prompt: str, tentativas: int = 8) -> tuple[str, dict, bool]:
        kw = dict(model=self.model_name,
                  messages=[{"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt}],
                  temperature=0.0, max_completion_tokens=MAX_TOKENS)
        if self.cfg.get("json_object"):
            kw["response_format"] = {"type": "json_object"}
        if self.cfg.get("extra_body"):
            kw["extra_body"] = self.cfg["extra_body"]
        ultimo = None
        for t in range(tentativas):
            try:
                r = self._client.chat.completions.create(**kw)
                u = r.usage
                uso = {"entrada": int(getattr(u, "prompt_tokens", 0) or 0),
                       "saida": int(getattr(u, "completion_tokens", 0) or 0)}
                ch = r.choices[0]
                if ch.message.content is None:
                    # No notebook isto virava ValueError -> lote vazio. Resposta válida, vazia.
                    logger.warning("[%s] content=None (finish_reason=%s)", self.model_name, ch.finish_reason)
                    return "", uso, True
                return ch.message.content, uso, False
            except Exception as e:  # noqa: BLE001
                ultimo = e
                logger.warning("[%s] falha (%s/%s): %s", self.model_name, t + 1, tentativas, str(e)[:160])
                time.sleep(_espera(t))
        raise RuntimeError(f"chamada falhou após {tentativas} tentativas: {ultimo}")


def cliente(modelo: str, system_prompt: str):
    cfg = MODELOS[modelo]
    return GeminiJSON(system_prompt, cfg) if cfg["provedor"] == "gemini" else OpenAICompatJSON(system_prompt, cfg)


def parse_resposta(modelo: str, texto: str) -> list[dict]:
    """Parse idêntico ao do notebook para cada provedor."""
    if MODELOS[modelo]["provedor"] == "gemini":
        return parse_llm_json(texto)
    raw = (texto or "").strip()                       # _call / _call_ds do classifier_api
    if not raw:
        return []
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            listas = [v for v in parsed.values() if isinstance(v, list)]
            parsed = listas[0] if len(listas) == 1 else (max(listas, key=len) if listas else [parsed])
        parsed = parsed if isinstance(parsed, list) else [parsed]
        return [p for p in parsed if isinstance(p, dict)]
    except json.JSONDecodeError:
        out = []
        for h in re.findall(r"\{[^{}]+\}", raw, re.DOTALL):
            try:
                out.append(json.loads(h))
            except json.JSONDecodeError:
                pass
        return out


def template_usuario(modelo: str) -> Callable[[int, str], str]:
    if MODELOS[modelo]["provedor"] == "gemini":
        return lambda i, t: GEMINI_USER_TEMPLATE.format(idx=i, categoria="", titulo=t)
    return lambda i, t: OPENAI_USER_TEMPLATE.format(idx=i, titulo=t)


# ═══════════════════════════════════════════════════════════════════
#  Lotes + cache JSONL
# ═══════════════════════════════════════════════════════════════════

def _hash(linhas: list[str]) -> str:
    return hashlib.sha1("\n".join(linhas).encode("utf-8")).hexdigest()[:16]


def carrega_cache(path: Path) -> dict[int, dict]:
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


def rodar_lotes(cli, itens: list, formata: Callable[[int, object], str], cache_path: Path,
                batch_size: int = BATCH_SIZE, pausa: float = 0.5, max_lotes: int | None = None,
                desc: str = "", barra: bool = True,
                valida: Callable[[str, int, int], bool] | None = None,
                max_invalidas: int = 5) -> dict[int, dict]:
    """Envia `itens` em lotes indexados; devolve {inicio_do_lote: registro}.
    Lotes já respondidos (mesmo hash de entrada) não são reenviados.

    `valida(texto, inicio, n)`: resposta que não tem o formato pedido (ex.: o
    DeepSeek em modo json_object às vezes devolve só '{"type": "json_object"}')
    é tratada como falha transitória e repetida até `max_invalidas` vezes. Lotes
    já em cache que não passam na validação também são refeitos. O registro
    guarda quantas respostas inválidas precederam a final ('invalidas')."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feitos = carrega_cache(cache_path)
    inicios = list(range(0, len(itens), batch_size))
    if max_lotes is not None:
        inicios = inicios[:max_lotes]

    pendentes = []
    for ini in inicios:
        linhas = [formata(ini + j, it) for j, it in enumerate(itens[ini:ini + batch_size])]
        r = feitos.get(ini)
        if r is None or r.get("hash") != _hash(linhas):
            pendentes.append((ini, linhas, 0))
        elif valida and not r.get("bloqueado") and not valida(r.get("texto") or "", ini, len(linhas)):
            pendentes.append((ini, linhas, 1 + int(r.get("invalidas", 0))))

    if pendentes:
        if barra:
            from tqdm.auto import tqdm
            it = tqdm(pendentes, desc=desc or cache_path.stem, unit="lote")
        else:
            it = pendentes
            logger.info("%s: %d lotes pendentes", desc or cache_path.stem, len(pendentes))
        with cache_path.open("a", encoding="utf-8") as fh:
            for k, (ini, linhas, invalidas) in enumerate(it, 1):
                reg = {"inicio": ini, "n": len(linhas), "hash": _hash(linhas), "modelo": cli.model_name,
                       "texto": None, "uso": None, "bloqueado": False, "erro": None, "invalidas": invalidas}
                try:
                    while True:
                        reg["texto"], reg["uso"], reg["bloqueado"] = cli.call("\n".join(linhas))
                        if (not valida or reg["bloqueado"] or valida(reg["texto"], ini, len(linhas))
                                or reg["invalidas"] >= max_invalidas):
                            break
                        reg["invalidas"] += 1
                        logger.warning("%s: resposta fora do formato no lote %d (%d) — repetindo",
                                       desc or cache_path.stem, ini, reg["invalidas"])
                except Exception as e:  # noqa: BLE001
                    reg["erro"] = str(e)[:500]
                fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
                fh.flush()
                if reg["erro"] is None:
                    feitos[ini] = reg
                if not barra and k % 100 == 0:
                    logger.info("%s: %d/%d lotes", desc or cache_path.stem, k, len(pendentes))
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
    rep = sum(1 for r in registros.values() if r.get("invalidas"))
    return {"chamadas": len(registros), "tokens_entrada": e, "tokens_saida": s, "lotes_bloqueados": b,
            "lotes_repetidos_formato": rep}


# ═══════════════════════════════════════════════════════════════════
#  11 — Referência de hoje + ablação do pipeline
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


def cache_ablacao(modelo: str, rodada: str, execucao: int) -> Path:
    return EXP_DIR / f"ablacao_{modelo}_{rodada}_r{execucao}.jsonl"


def rodar_ablacao(modelo: str, rodada: str, execucao: int = 1, max_lotes: int | None = None,
                  df: pd.DataFrame | None = None, barra: bool = True) -> dict[int, dict]:
    cfg = RODADAS_ABLACAO[rodada]
    df = load_pairs() if df is None else df
    titulos = unique_titles(df)
    entrada = [pre_normalize_title(t) for t in titulos] if cfg["pre"] else titulos
    cli = cliente(modelo, SYSTEM_PROMPT if cfg["prompt"] == "completo" else prompt_sem_regras_5_9())
    return rodar_lotes(cli, entrada, template_usuario(modelo), cache_ablacao(modelo, rodada, execucao),
                       max_lotes=max_lotes, desc=f"{modelo} {rodada} r{execucao}", barra=barra)


def raw_por_titulo(modelo: str, rodada: str, execucao: int = 1,
                   df: pd.DataFrame | None = None) -> tuple[dict[str, dict], dict]:
    """{título limpo: JSON cru} a partir do cache + resumo de cobertura."""
    df = load_pairs() if df is None else df
    titulos = unique_titles(df)
    regs = carrega_cache(cache_ablacao(modelo, rodada, execucao))
    raw = {}
    for ini in range(0, len(titulos), BATCH_SIZE):
        n = min(BATCH_SIZE, len(titulos) - ini)
        r = regs.get(ini)
        lista = parse_resposta(modelo, r["texto"]) if r and r.get("texto") else []
        for t, j in zip(titulos[ini:ini + n], _casa_por_idx(lista, ini, n)):
            raw[t] = j
    resumo = {"titulos": len(titulos), "lotes_respondidos": len(regs),
              "lotes_total": -(-len(titulos) // BATCH_SIZE),
              "titulos_sem_json": sum(1 for t in titulos if not raw.get(t)), **uso_total(regs)}
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


def pares_variante(modelo: str, variante: str, execucao: int = 1,
                   df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Monta e salva dados/grupo2/pairs_normalized_<modelo>_<variante>_r<k>.csv."""
    df = load_pairs() if df is None else df
    rodada, pos = VARIANTES_ABLACAO[variante]
    raw, resumo = raw_por_titulo(modelo, rodada, execucao, df)
    pares = montar_pares(df, raw, pos)
    pares.to_csv(EXP_DIR / f"pairs_normalized_{modelo}_{variante}_r{execucao}.csv", index=False, encoding="utf-8-sig")
    return pares, resumo


# ═══════════════════════════════════════════════════════════════════
#  12 — Comparação direta par a par
# ═══════════════════════════════════════════════════════════════════

def cache_direta(modelo: str, execucao: int) -> Path:
    return EXP_DIR / f"direta_{modelo}_r{execucao}.jsonl"


def _fmt_par(i: int, par: tuple[str, str]) -> str:
    return f'[{i}] Título A: "{par[0]}" | Título B: "{par[1]}"'


def rodar_comparacao_direta(modelo: str, execucao: int = 1, max_lotes: int | None = None,
                            df: pd.DataFrame | None = None, barra: bool = True) -> dict[int, dict]:
    df = load_pairs() if df is None else df
    pares = list(zip(df["a_clean"], df["b_clean"]))
    cli = cliente(modelo, PROMPT_COMPARACAO_DIRETA)
    return rodar_lotes(cli, pares, _fmt_par, cache_direta(modelo, execucao), max_lotes=max_lotes,
                       desc=f"{modelo} direta r{execucao}", barra=barra, valida=_valida_direta(modelo))


def _valida_direta(modelo: str) -> Callable[[str, int, int], bool]:
    """Resposta válida = todos os pares do lote com 'mesmo_produto' interpretável."""
    def valida(texto: str, ini: int, n: int) -> bool:
        casados = _casa_por_idx(parse_resposta(modelo, texto), ini, n)
        return all(j and _bool(j.get("mesmo_produto")) is not None for j in casados)
    return valida


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


def predicoes_comparacao_direta(modelo: str, execucao: int = 1,
                                df: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    """Predição 0/1 por par. Par sem resposta válida conta como 0 (não-match) e é reportado."""
    df = load_pairs() if df is None else df
    regs = carrega_cache(cache_direta(modelo, execucao))
    n = len(df)
    pred = np.zeros(n, int)
    sem_resposta = 0
    for ini in range(0, n, BATCH_SIZE):
        k = min(BATCH_SIZE, n - ini)
        r = regs.get(ini)
        lista = parse_resposta(modelo, r["texto"]) if r and r.get("texto") else []
        for local, j in enumerate(_casa_por_idx(lista, ini, k)):
            v = _bool(j.get("mesmo_produto")) if j else None
            if v is None:
                sem_resposta += 1
            pred[ini + local] = int(bool(v))
    resumo = {"pares": n, "lotes_respondidos": len(regs), "lotes_total": -(-n // BATCH_SIZE),
              "pares_sem_resposta": sem_resposta, **uso_total(regs)}
    return pred, resumo


def pares_sem_resposta_direta(modelo: str, execucao: int = 1, df: pd.DataFrame | None = None) -> np.ndarray:
    """Máscara booleana: True nos pares da comparação direta sem resposta válida
    (lote ausente, bloqueado, fora do formato ou 'mesmo_produto' não interpretável)."""
    df = load_pairs() if df is None else df
    regs = carrega_cache(cache_direta(modelo, execucao))
    n = len(df)
    sem = np.zeros(n, bool)
    for ini in range(0, n, BATCH_SIZE):
        k = min(BATCH_SIZE, n - ini)
        r = regs.get(ini)
        lista = parse_resposta(modelo, r["texto"]) if r and r.get("texto") else []
        for local, j in enumerate(_casa_por_idx(lista, ini, k)):
            sem[ini + local] = (_bool(j.get("mesmo_produto")) if j else None) is None
    return sem


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
#  Estatística
# ═══════════════════════════════════════════════════════════════════

def _f1(tp, fp, fn):
    tp, fp, fn = (np.asarray(x, float) for x in (tp, fp, fn))
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)


def f1_boot(y, preds: list, cid: np.ndarray, n: int = 10000, seed: int = 42) -> tuple[float, np.ndarray]:
    """(F1 observado, distribuição bootstrap) da MÉDIA do F1 de várias execuções.

    Mesma semente e mesmo nº de clusters => mesmos pesos em toda chamada, então
    qualquer contraste entre distribuições (Δ, diferença de diferenças) é pareado.
    """
    y = np.asarray(y)
    k = int(cid.max()) + 1
    rng = np.random.default_rng(seed)
    W = rng.multinomial(k, np.full(k, 1 / k), size=n).astype(float)
    obs, dist = [], []
    for p in preds:
        p = np.asarray(p)
        c = [np.bincount(cid, weights=m.astype(float), minlength=k)
             for m in ((y == 1) & (p == 1), (y == 0) & (p == 1), (y == 1) & (p == 0))]
        obs.append(float(_f1(c[0].sum(), c[1].sum(), c[2].sum())))
        dist.append(_f1(W @ c[0], W @ c[1], W @ c[2]))
    return float(np.mean(obs)), np.mean(dist, axis=0)


def ic95(dist: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


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
#  Plano completo (paralelo por provedor) — CLI
# ═══════════════════════════════════════════════════════════════════

def tarefas_plano(modelos: list[str], so_direta: bool = False, so_referencia: bool = False) -> list[tuple]:
    """Ordem = prioridade: referência, comparação direta, ablações."""
    tarefas = []
    for m in modelos:
        n = MODELOS[m]["execucoes"]
        if not so_direta:
            tarefas += [("ablacao", m, "completa", r) for r in range(1, MODELOS[m]["execucoes_referencia"] + 1)]
        if so_referencia:
            continue
        tarefas += [("direta", m, None, r) for r in range(1, EXECUCOES_DIRETA + 1)]
        if not so_direta:
            tarefas += [("ablacao", m, rod, r) for rod in ("sem_pre", "sem_regras_5_9") for r in range(1, n + 1)]
    return tarefas


def executar_plano(modelos: list[str], workers_por_modelo: int = 9, so_direta: bool = False,
                   max_lotes: int | None = None, so_referencia: bool = False) -> None:
    df = load_pairs()
    tarefas = tarefas_plano(modelos, so_direta, so_referencia)
    logger.info("%d tarefas: %s", len(tarefas), tarefas)

    def roda(t):
        tipo, m, rod, r = t
        if tipo == "ablacao":
            regs = rodar_ablacao(m, rod, r, max_lotes=max_lotes, df=df, barra=False)
        else:
            regs = rodar_comparacao_direta(m, r, max_lotes=max_lotes, df=df, barra=False)
        return t, uso_total(regs)

    pools = {m: ThreadPoolExecutor(max_workers=workers_por_modelo, thread_name_prefix=m) for m in modelos}
    futuros = [pools[t[1]].submit(roda, t) for t in tarefas]
    for f in as_completed(futuros):
        try:
            t, u = f.result()
            logger.info("CONCLUÍDA %s -> %s", t, u)
        except Exception as e:  # noqa: BLE001
            logger.error("tarefa falhou: %s", e)
    for p in pools.values():
        p.shutdown()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description="Grupo 2 — experimentos com API")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plano", help="referência + comparação direta + ablações, em paralelo")
    p.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    p.add_argument("--workers", type=int, default=9, help="execuções simultâneas por modelo")
    p.add_argument("--so-direta", action="store_true")
    p.add_argument("--so-referencia", action="store_true", help="só a rodada completa (referência)")
    p.add_argument("--max-lotes", type=int, default=None, help="limita lotes por tarefa (teste)")
    sub.add_parser("embeddings", help="similaridades com os modelos de embedding")
    a = ap.parse_args()
    if a.cmd == "plano":
        executar_plano(a.modelos, a.workers, a.so_direta, a.max_lotes, a.so_referencia)
    else:
        for nome in MODELOS_EMBEDDING:
            s = similaridades_embedding(nome)
            print(nome, f"sim média={s.mean():.3f}")
