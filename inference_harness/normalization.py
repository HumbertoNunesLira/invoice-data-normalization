"""Camada simbólica canônica — fonte única de pré/pós-processamento.

Extraído fielmente de ``main_supersaurus/classifier_api.ipynb`` (caminho do
Gemini, cuja saída o ``benchmark.ipynb`` já consome em ``output/normalized.csv``).

Regra do harness: TODO backend (gemini, qwen, ...) importa daqui. A única coisa
que muda entre rodadas é o modelo — pré-normalização, parse de JSON, limpeza
determinística e construção da match key são SEMPRE estas funções.
"""
from __future__ import annotations

import json
import re

# ═══════════════════════════════════════════════════════════════════
#  Prompts (contrato do modelo) — por backend, schema JSON idêntico
# ═══════════════════════════════════════════════════════════════════
# O system prompt é parte do contrato de cada modelo: o Gemini usa o do
# classifier_api; o Qwen FT usa exatamente o prompt com que foi treinado
# (extraído da JSONL de fine-tuning). Ambos produzem o MESMO schema de JSON,
# por isso `clean_llm_output`/`build_match_key` abaixo são compartilhados.

SYSTEM_PROMPT = """
Você é um especialista em extração e padronização (normalização) estrita de atributos de produtos de e-commerce brasileiro. A sua saída será usada para comparar se dois títulos representam EXACTAMENTE o mesmo produto no mundo real.

══════════════════════════════════════════════════════════════════
TAREFA
══════════════════════════════════════════════════════════════════
Você receberá um ou mais títulos de produtos. Para CADA título, extraia APENAS dados técnicos e reais. Ignore textos de marketing, condições de uso ou contexto. 

Chaves obrigatórias (use null se o atributo não existir no título):
  "idx"                : int    — o índice recebido
  "marca"              : string — ex: "Electrolux", "Samsung", "Tramontina"
  "modelo"             : string — ex: "LTD12", "Galaxy S21", null
  "tipo_produto"       : string — ex: "máquina de lavar", "geladeira", "furadeira". SEMPRE resuma a 1 a 3 palavras no máximo.
  "capacidade_volume"  : string — ex: "12kg", "500ml", "2000ml", null
  "voltagem"           : string — "110v", "220v" ou "bivolt"; null se ausente
  "cor"                : string — ex: "branco", "preto", null
  "material"           : string — ex: "inox", "plástico", null
  "quantidade"         : string — ex: "kit 3", "1", null (Se for pacote, extraia só o número)
  "outros_atributos"   : string — null para a maioria. Apenas se tiver uma ESPECIFICAÇÃO DE HARDWARE (ex: "classe 10", "DDR4", "21v"). NÃO coloque adjetivos aqui.

══════════════════════════════════════════════════════════════════
REGRAS DE NORMALIZAÇÃO DE VALORES
══════════════════════════════════════════════════════════════════
1. Volume → converter TODOS para ml sem espaço: "1/2 litro", "0,5 litro" → "500ml" | "1l" → "1000ml"
2. Voltagem → "110 v", "110 volts" → "110v" | "bi-volt" → "bivolt"
3. Peso/Capacidade → sem espaço: "12 kilos", "12 kg" → "12kg" | "500 gr" → "500g"
4. Minúsculo: Texto em minúsculo para todos os valores, exceto marcas.

══════════════════════════════════════════════════════════════════
REGRAS CRÍTICAS DE LIMPEZA (FILTRO DE LIXO)
══════════════════════════════════════════════════════════════════
5. OMITA ABSOLUTAMENTE adjetivos de marketing, texto promocional ou menção a público-alvo e eventos.
   - IGNORE palavras como: "gamer", "profissional", "dia das mães", "promoção", "original", "uso diário", "premium".
6. OMITA conectivos ("de", "para", "e", "com") e mantenha a essência direta.
   - Errado: "pote de vidro" -> Certo: "pote vidro"
   - Errado: "capa para celular" -> Certo: "capa celular"
   - Errado: "furadeira e lixadeira" -> Certo: "furadeira lixadeira"
7. DESCRIÇÕES REPETITIVAS: Se o título usa duas palavras quase sinônimas para a mesma coisa, simplifique para a mais comum.
   - Errado: "tênis sapatilha" -> Certo: "tênis"

══════════════════════════════════════════════════════════════════
REGRAS DE DESAMBIGUAÇÃO SEMÂNTICA
══════════════════════════════════════════════════════════════════
8. SINÔNIMOS DE TIPO (use a forma genérica curta ao invés de nomes longos ou gírias da área):
   - "lavadora" = "máquina de lavar" → "máquina lavar"
   - "refrigerador" = "geladeira"
   - "tv" = "televisão" = "smart tv" → "televisão"
   - "pendrive ultra slim" = "pen drive flash" → "pen drive"
   - "ração gatos castrados" = "comida gato adulto" → "ração para gato" (simplifique o descritivo de utilidade)

9. MARCA — unifique variações: "Coca Cola" = "coca cola" = "Coca-Cola" → "Coca-Cola".
   NÃO invente marcas. Se não tiver certeza se uma palavra é a fabricante oficial, prefira "null".

══════════════════════════════════════════════════════════════════
FORMATO DE SAÍDA (BATCH)
══════════════════════════════════════════════════════════════════
Retorne SOMENTE um array JSON válido. Sem texto extra, sem markdown.
"""


# Prompt EXATO com que o Qwen2.5 foi fine-tunado (messages[0] da JSONL).
# Difere do SYSTEM_PROMPT do Gemini apenas em espaçamento; mantido literal
# para o modelo rodar in-distribution.
QWEN_SYSTEM_PROMPT = """
Você é um especialista em extração de atributos de produtos de e-commerce brasileiro.

══════════════════════════════════════════════════════════════════
TAREFA
══════════════════════════════════════════════════════════════════
Você receberá um ou mais títulos de produtos, cada um identificado por um
índice numérico. Para CADA título, retorne um objeto JSON com as chaves
abaixo. Retorne um ARRAY JSON (lista) contendo um objeto por título, na
mesma ordem recebida. Se receber apenas 1 título, retorne um array com 1
objeto.

Chaves obrigatórias (use null se o atributo não existir no título):
  "idx"                : int    — o índice recebido
  "marca"              : string — ex: "Electrolux", "Samsung", "Tramontina"
  "modelo"             : string — ex: "LTD12", "Galaxy S21", null
  "tipo_produto"       : string — ex: "máquina de lavar", "geladeira"
  "capacidade_volume"  : string — ex: "12kg", "500ml", "2000ml", null
  "voltagem"           : string — "110v", "220v" ou "bivolt"; null se ausente
  "cor"                : string — ex: "branco", "preto", null
  "material"           : string — ex: "inox", "plástico", null
  "quantidade"         : string — ex: "kit 3 peças", "par", null
  "outros_atributos"   : string — outros atributos relevantes, null se não houver

══════════════════════════════════════════════════════════════════
REGRAS DE NORMALIZAÇÃO (unidades e formato)
══════════════════════════════════════════════════════════════════

1. Volume → converter SEMPRE para ml sem espaço:
   "1/2 litro", "meio litro", "0,5 litro" → "500ml"
   "1 litro", "1l" → "1000ml"   |   "1,5 litro" → "1500ml"
   "2 litros" → "2000ml"

2. Voltagem → formato padronizado:
   "110 v", "110 volts", "110V" → "110v"
   "220 v", "220 volts" → "220v"
   "bivolt", "bi-volt", "Bi Volt" → "bivolt"

3. Peso/capacidade → sem espaço:
   "12 kilos", "12 kg", "12KG" → "12kg"
   "500 g", "500gr", "500 gramas" → "500g"

4. Texto em minúsculo para todos os valores, exceto marcas (1ª letra maiúscula).

══════════════════════════════════════════════════════════════════
REGRAS DE DESAMBIGUAÇÃO SEMÂNTICA
══════════════════════════════════════════════════════════════════

5. SINÔNIMOS DE TIPO — use sempre a forma CANÔNICA:
   "lavadora" = "máquina de lavar" → use "máquina de lavar"
   "geladeira" = "refrigerador" → use "geladeira"
   "liq." = "liquidificador" → use "liquidificador"
   "TV" = "televisão" = "televisor" → use "televisão"
   "notebook" = "laptop" → use "notebook"
   "celular" = "smartphone" = "telefone celular" → use "celular"
   "fone" = "fone de ouvido" = "headphone" = "earphone" → use "fone de ouvido"

6. MARCA — unifique variações gráficas da mesma marca:
   "Coca-Cola" = "Coca Cola" = "CocaCola" = "coca cola" → "Coca-Cola"
   Sempre capitalize a 1ª letra de cada palavra da marca.

7. Se o título contém apenas nome genérico sem marca, use null para marca.
   NÃO invente marcas.

8. Se um atributo é ambíguo, prefira a interpretação mais provável para
   o contexto do produto.

══════════════════════════════════════════════════════════════════
FORMATO DE SAÍDA (BATCH)
══════════════════════════════════════════════════════════════════

Retorne SOMENTE um array JSON válido. Sem texto extra, sem markdown.
Cada objeto deve conter o campo "idx" com o índice recebido.
"""

# Templates de user message — por backend (contrato do modelo).
# Gemini: classifier_api manda `Categoria: "" |`; Qwen FT foi treinado sem isso.
GEMINI_USER_TEMPLATE = '[{idx}] Categoria: "{categoria}" | Título: "{titulo}"'
QWEN_USER_TEMPLATE = '[{idx}] Título: "{titulo}"'


# ═══════════════════════════════════════════════════════════════════
#  1. Pré-normalização do título (Ditto-inspired) — ANTES do LLM
# ═══════════════════════════════════════════════════════════════════
# Princípio: regras CONSERVADORAS — preferir não-normalizar a normalizar
# errado. Bug histórico: o `k` solto matcheava "4K" em "Smart TV 4K",
# produzindo "Smart TV 4kg". A regra de quilos agora exige a palavra
# completa (kg/kilo/kilograma) para evitar essa classe de regressão.

_PRE_RULES = [
    # ── 1. Frações de litro textuais → ml (mais específico primeiro) ──
    (re.compile(r"\b1/2\s*l(?:itro)?s?\b", re.I),         "500ml"),
    (re.compile(r"\bmeio\s+litro", re.I),                  "500ml"),

    # ── 2. Volume em litros → ml ──
    # Aceita inteiro ou decimal (vírgula ou ponto). Ex: 0,9L / 1.5l / 2 litros
    (re.compile(r"(\d+(?:[,.]\d+)?)\s*l(?:itro[s]?)?\b", re.I),
        lambda m: (
            f"{int(float(m.group(1).replace(',', '.')) * 1000)}ml"
            if float(m.group(1).replace(',', '.')) * 1000
               == int(float(m.group(1).replace(',', '.')) * 1000)
            else f"{float(m.group(1).replace(',', '.')) * 1000}ml"
        )),

    # ── 3. Mililitros → padronização sem espaço ──
    (re.compile(r"(\d+(?:[,.]\d+)?)\s*ml\b", re.I),
        lambda m: f"{m.group(1).replace(',', '.')}ml"),

    # ── 4. Quilos → kg ──
    # ATENÇÃO: exige palavra completa (kg, kilo, kilograma).  Caso contrário
    # consume "4K"/"8K" (resoluções de TV).  Regressão histórica documentada.
    (re.compile(r"(\d+(?:[,.]\d+)?)\s*(?:kg|kilos?|kilogramas?)\b", re.I),
        lambda m: f"{m.group(1).replace(',', '.')}kg"),

    # ── 5. Gramas → g ──
    # Exige "gramas?" ou "gr" para não colidir com "5G" (rede móvel) ou "8GB".
    (re.compile(r"(\d+(?:[,.]\d+)?)\s*(?:gramas?|gr)\b", re.I),
        lambda m: f"{m.group(1).replace(',', '.')}g"),

    # ── 6. Voltagem ──
    (re.compile(r"\bbi[\s-]?volt(?:s)?\b", re.I),         "bivolt"),
    # Aceita "110v", "110V", "110volt", "110volts" (regra anterior exigia "vol").
    (re.compile(r"(\d+)\s*v(?:olts?)?\b", re.I),
        lambda m: f"{m.group(1)}v"),

    # ── 7. Watts → padronização sem espaço ──
    # Cobre "1000w", "1000 W", "1000 watt", "1000 watts".
    (re.compile(r"(\d+(?:[,.]\d+)?)\s*w(?:atts?)?\b", re.I),
        lambda m: f"{m.group(1).replace(',', '.')}w"),

    # ── 8. Polegadas → "Xpol" ──
    # Cobre `55"`, `55 polegadas`, `55 pol.` (display de TVs e form-factors).
    (re.compile(r'(\d+(?:[,.]\d+)?)\s*(?:"|polegadas?|pol\.?)(?=\s|$|[^a-z0-9])', re.I),
        lambda m: f"{m.group(1).replace(',', '.')}pol"),

    # ── 9. Espaços excessivos ──
    (re.compile(r"\s{2,}"), " "),
]


def pre_normalize_title(title: str) -> str:
    """Aplica regras de normalização no título ANTES de enviar ao LLM."""
    text = title.strip()
    for pattern, replacement in _PRE_RULES:
        text = pattern.sub(replacement, text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
#  2. Filtro de distância de edição ponderada (pré-LLM)
# ═══════════════════════════════════════════════════════════════════
# Penaliza fortemente diferenças em dígitos: "1L" vs "2L" são produtos
# distintos mesmo com 1 caractere diferente.  Mantém custo baixo para
# variações ortográficas e abreviações — o LLM resolve essa camada.
#
# NB: esta é a função do classifier_api (filtro pré-LLM, threshold 0.25),
# DISTINTA da métrica homônima em benchmark/weighted_distance.py.

EDIT_DISTANCE_THRESHOLD = 0.25

_COST_LETTER_SUB   = 0.1   # letra ↔ letra (substituição) — barato
_COST_LETTER_INDEL = 0.2   # letra inserir/deletar — barato (cobre abreviações)
_COST_DIGIT_SUB    = 10.0  # qualquer sub envolvendo dígito — proibitivo
_COST_DIGIT_INDEL  = 8.0   # dígito inserir/deletar — alto (quantidade fantasma)
_COST_OTHER        = 0.3   # punct/espaço (sub e indel sem dígito) — intermediário


def _indel_cost(c: str) -> float:
    if c.isdigit():
        return _COST_DIGIT_INDEL
    if c.isalpha():
        return _COST_LETTER_INDEL
    return _COST_OTHER


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    # Qualquer substituição envolvendo dígito é cara — caso contrário o DP
    # acha caminhos baratos do tipo "insert dígito + sub dígito↔letra (other)"
    # que burlam a penalidade de dígito↔dígito.
    if a.isdigit() or b.isdigit():
        return _COST_DIGIT_SUB
    if a.isalpha() and b.isalpha():
        return _COST_LETTER_SUB
    return _COST_OTHER


def weighted_edit_distance(s1: str, s2: str) -> float:
    """Levenshtein ponderada por tipo de caractere. Case-insensitive."""
    s1, s2 = s1.lower(), s2.lower()
    m, n = len(s1), len(s2)
    if m == 0:
        return sum(_indel_cost(c) for c in s2)
    if n == 0:
        return sum(_indel_cost(c) for c in s1)

    prev = [0.0] * (n + 1)
    for j in range(1, n + 1):
        prev[j] = prev[j - 1] + _indel_cost(s2[j - 1])

    for i in range(1, m + 1):
        curr = [prev[0] + _indel_cost(s1[i - 1])]
        for j in range(1, n + 1):
            curr.append(min(
                curr[j - 1] + _indel_cost(s2[j - 1]),             # insert
                prev[j]     + _indel_cost(s1[i - 1]),             # delete
                prev[j - 1] + _sub_cost(s1[i - 1], s2[j - 1]),    # substitute
            ))
        prev = curr
    return prev[n]


def normalized_weighted_distance(s1: str, s2: str) -> float:
    """Distância ponderada normalizada por max(len(s1), len(s2)). 0.0 se ambos vazios."""
    denom = max(len(s1), len(s2))
    if denom == 0:
        return 0.0
    return weighted_edit_distance(s1, s2) / denom


# ═══════════════════════════════════════════════════════════════════
#  3. Parse do JSON do LLM (strip markdown + json.loads + fallback regex)
# ═══════════════════════════════════════════════════════════════════

def parse_llm_json(raw_text: str) -> list[dict]:
    """Parseia a resposta do LLM para uma lista de dicts.

    Reproduz o parser do classifier_api (`extract_batch_llm`) e do finetune
    (`parse_assistant_output`): remove cerca de markdown, tenta ``json.loads``
    e, se falhar, extrai cada ``{...}`` individualmente. Sempre retorna lista
    (possivelmente vazia); um dict único vira lista de 1.
    """
    raw_text = (raw_text or "").strip()

    # Limpa markdown se houver
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?", "", raw_text).rstrip("`").strip()

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)]
        return []
    except json.JSONDecodeError:
        # Fallback: tentar extrair cada {...} individualmente
        matches = re.findall(r"\{[^{}]+\}", raw_text, re.DOTALL)
        results = []
        for m in matches:
            try:
                results.append(json.loads(m))
            except json.JSONDecodeError:
                continue
        return results


# ═══════════════════════════════════════════════════════════════════
#  4. Limpeza determinística (camada simbólica pós-LLM)
# ═══════════════════════════════════════════════════════════════════

_RE_SPACES     = re.compile(r"\s+")
_RE_VOLUME_ML  = re.compile(r"(\d+(?:[.,]\d+)?)\s*ml", re.I)
_RE_VOLUME_L   = re.compile(r"(\d+(?:[.,]\d+)?)\s*l(?:itro[s]?)?(?=\b|$)", re.I)
_RE_WEIGHT_KG  = re.compile(r"(\d+(?:[.,]\d+)?)\s*k(?:g|ilos?|ilogramas?)", re.I)
_RE_WEIGHT_G   = re.compile(r"(\d+(?:[.,]\d+)?)\s*g(?:r|ramas?)?(?=\b|$)", re.I)
_RE_VOLTAGE    = re.compile(r"(\d+)\s*v(?:olts?)?(?=\b|$)", re.I)
_RE_BIVOLT     = re.compile(r"bi[\s-]?volt", re.I)


def _clean_str(value) -> str:
    if not value or not isinstance(value, str):
        return ""
    return _RE_SPACES.sub(" ", value.strip()).lower()


def _normalize_volume(value: str) -> str:
    if not value:
        return value
    value = value.replace(",", ".")
    m = _RE_VOLUME_ML.search(value)
    if m:
        qty = float(m.group(1))
        return f"{int(qty)}ml" if qty == int(qty) else f"{qty}ml"
    m = _RE_VOLUME_L.search(value)
    if m:
        return f"{int(float(m.group(1)) * 1000)}ml"
    return value


def _normalize_weight(value: str) -> str:
    if not value:
        return value
    value = value.replace(",", ".")
    m = _RE_WEIGHT_KG.search(value)
    if m:
        qty = float(m.group(1))
        return f"{int(qty)}kg" if qty == int(qty) else f"{qty}kg"
    m = _RE_WEIGHT_G.search(value)
    if m:
        qty = float(m.group(1))
        return f"{int(qty)}g" if qty == int(qty) else f"{qty}g"
    return value


def _normalize_voltage(value: str) -> str:
    if not value:
        return value
    if _RE_BIVOLT.search(value):
        return "bivolt"
    m = _RE_VOLTAGE.search(value)
    return f"{m.group(1)}v" if m else value


def _normalize_quantidade(value: str):
    """Unidade única (1, 01, "- 1") vira None -- sem isso a LLM às vezes
    extrai null e às vezes "1" pro mesmo caso implícito (nenhuma quantidade
    mencionada no título), gerando instabilidade entre execuções sem ganho
    de informação real. Espelha classifier_api.ipynb (célula 290a5546).
    """
    if not value:
        return None
    m = re.search(r"\d+", str(value))
    if m and int(m.group()) == 1:
        return None
    return value


def _normalize_capacity(value: str) -> str:
    if not value:
        return value
    v = value.lower()
    if any(u in v for u in ["ml", "litro", " l", "l "]):
        return _normalize_volume(v)
    if any(u in v for u in ["kg", "kilo", "grama", " g", "g "]):
        return _normalize_weight(v)
    return _clean_str(value)


def clean_llm_output(raw_json: dict) -> dict:
    if not raw_json:
        return {}
    cleaned = {}
    for key, value in raw_json.items():
        if key == "idx":
            continue
        if value is None or value == "null":
            cleaned[key] = None
            continue
        s = str(value).strip()
        if key == "marca":
            cleaned[key] = _RE_SPACES.sub(" ", s).title() or None
        elif key == "voltagem":
            cleaned[key] = _normalize_voltage(_clean_str(s)) or None
        elif key == "capacidade_volume":
            cleaned[key] = _normalize_capacity(_clean_str(s)) or None
        elif key == "quantidade":
            cleaned[key] = _normalize_quantidade(_clean_str(s))
        else:
            cleaned[key] = _clean_str(s) or None
    return cleaned


# ═══════════════════════════════════════════════════════════════════
#  5. Match key (estável, consumida pelo benchmark) vs canonical (display)
# ═══════════════════════════════════════════════════════════════════
# A constancy do benchmark é sensível a QUALQUER divergência entre variantes
# do mesmo `original`. Por isso a match key usa só atributos estáveis.

_STABLE_FIELDS = (
    "tipo_produto",
    "marca",
    "modelo",
    "capacidade_volume",
    "voltagem",
)


def build_match_key(cleaned_json: dict) -> str:
    """Chave estável usada pelo benchmark de constância (coluna `normalized`)."""
    if not cleaned_json:
        return ""
    parts = [cleaned_json.get(f) for f in _STABLE_FIELDS]
    return " | ".join(p for p in parts if p)


def build_canonical_name(cleaned_json: dict) -> str:
    """Formato display: ${nome} ${marca} ${cor} ${outros} - ${quantidade} ${unidade}"""
    nome  = cleaned_json.get("tipo_produto") or ""
    marca = cleaned_json.get("marca") or ""
    cor   = cleaned_json.get("cor") or ""

    outros_parts = [
        cleaned_json.get("modelo"),
        cleaned_json.get("material"),
        cleaned_json.get("voltagem"),
        cleaned_json.get("outros_atributos"),
    ]
    outros = ", ".join(p for p in outros_parts if p)

    cap = cleaned_json.get("capacidade_volume") or ""
    m_cap = re.match(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", cap)
    cap_str = f"{m_cap.group(1)} {m_cap.group(2)}" if m_cap else cap

    # quantidade era extraída pela LLM mas nunca chegava no nome final -- bug
    # que colapsava produtos com quantidades diferentes no mesmo nome
    # canônico. Entra como componente próprio, separado de capacidade_volume.
    qtd = cleaned_json.get("quantidade") or ""

    main = " ".join(p for p in [nome, marca, cor, outros] if p)
    tail = " ".join(p for p in [qtd, cap_str] if p)
    if tail:
        return f"{main} - {tail}".strip()
    return main
