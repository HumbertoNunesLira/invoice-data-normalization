"""Testes do harness de inferência.

Dois focos:
  1. Paridade da camada simbólica (normalization.py reproduz o classifier_api).
  2. Dry-run do runner com um backend stub → os dois CSVs saem no schema exato
     que benchmark.benchmark.run e benchmark.pairwise_f1.run aceitam.

Rode da raiz do repo:  python -m pytest tests/test_inference_harness.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from inference_harness import (
    build_consistency_csv,
    build_pairwise_csv,
    get_backend,
    normalize,
    normalize_many,
)
from inference_harness.backends import Backend
from inference_harness.normalization import (
    build_canonical_name,
    build_match_key,
    clean_llm_output,
    normalized_weighted_distance,
    parse_llm_json,
    pre_normalize_title,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Backend stub: JSON determinístico por título, sem rede/GPU ──────────
class StubBackend(Backend):
    name = "stub"

    def generate(self, titles_pre: list[str]) -> list[dict]:
        out = []
        for t in titles_pre:
            tokens = t.lower().split()
            out.append({
                "idx": 0,
                "tipo_produto": " ".join(tokens[:2]) if tokens else None,
                "marca": tokens[2] if len(tokens) > 2 else None,
                "modelo": None,
                "capacidade_volume": None,
                "voltagem": None,
                "cor": None,
                "material": None,
                "quantidade": None,
                "outros_atributos": None,
            })
        return out


# ═══════════════════════════════════════════════════════════════════
#  1. Paridade da pré-normalização (regressões conhecidas do cell 6)
# ═══════════════════════════════════════════════════════════════════

def test_pre_normalize_4k_nao_vira_4kg():
    out = pre_normalize_title("Smart TV Samsung 55 4K Crystal Bivolt")
    assert "4kg" not in out.lower()
    assert "bivolt" in out.lower()


def test_pre_normalize_unidades():
    assert pre_normalize_title("Coca Cola meio litro") == "Coca Cola 500ml"
    assert pre_normalize_title("Liquidificador Mondial 110 Volts").endswith("110v")
    assert pre_normalize_title('TV 50" 4K HDR') == "TV 50pol 4K HDR"


def test_pre_normalize_preserva_gb_e_g_de_rede():
    assert "8gb" in pre_normalize_title("Memória 8GB DDR4").lower()
    assert "5g" in pre_normalize_title("Roteador 5G Modem").lower()


# ═══════════════════════════════════════════════════════════════════
#  2. Filtro de distância de edição (DROP em dígito, KEEP em ortografia)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("a,b", [
    ("1L coca", "2L coca"),
    ("500ml leite", "750ml leite"),
    ("110v", "220v"),
    ("Smart TV 4K", "Smart TV 8K"),
])
def test_filtro_dropa_mudanca_de_digito(a, b):
    assert normalized_weighted_distance(a, b) > 0.25


@pytest.mark.parametrize("a,b", [
    ("leite integral", "leite desnatado"),
    ("Liquidificador", "Liq."),
    ("Coca-Cola 500ml", "CocaCola 500ml"),
    ("Samsung Galaxy", "Samsugn Galaxy"),
])
def test_filtro_mantem_variacao_ortografica(a, b):
    assert normalized_weighted_distance(a, b) <= 0.25


# ═══════════════════════════════════════════════════════════════════
#  3. Limpeza + match key (5 campos estáveis, " | ", marca capitalizada)
# ═══════════════════════════════════════════════════════════════════

def test_clean_e_match_key():
    raw = {
        "idx": 0,
        "tipo_produto": "Geladeira",
        "marca": "electrolux",
        "modelo": "RD30",
        "capacidade_volume": "260 litros",
        "voltagem": "220 volts",
        "cor": "branco",
    }
    cleaned = clean_llm_output(raw)
    assert cleaned["marca"] == "Electrolux"          # Title-case
    assert cleaned["voltagem"] == "220v"
    mk = build_match_key(cleaned)
    parts = mk.split(" | ")
    assert len(parts) == 5                            # só os 5 campos estáveis
    assert parts[0] == "geladeira" and parts[1] == "Electrolux"
    assert "branco" not in mk                         # cor NÃO entra na match key


def test_match_key_vazio_para_json_vazio():
    assert build_match_key(clean_llm_output({})) == ""


def test_parse_llm_json_strip_markdown_e_fallback():
    assert parse_llm_json('```json\n[{"idx":0,"marca":"Lg"}]\n```') == [{"idx": 0, "marca": "Lg"}]
    assert parse_llm_json('{"idx":0}') == [{"idx": 0}]            # dict vira lista de 1
    assert parse_llm_json("lixo {\"a\": 1} mais lixo") == [{"a": 1}]  # fallback regex
    assert parse_llm_json("") == []


# ═══════════════════════════════════════════════════════════════════
#  4. Interface normalize() com backend
# ═══════════════════════════════════════════════════════════════════

def test_normalize_retorna_match_key():
    res = normalize("Geladeira Electrolux RD30", backend=StubBackend())
    assert set(res) == {"raw", "cleaned", "match_key", "canonical"}
    assert isinstance(res["match_key"], str)


def test_get_backend_factory():
    assert get_backend("gemini").name == "gemini"
    assert get_backend("qwen").name == "qwen"
    with pytest.raises(ValueError):
        get_backend("inexistente")


# ═══════════════════════════════════════════════════════════════════
#  5. Dry-run dos dois CSVs → aceitos pelos módulos reais do benchmark
# ═══════════════════════════════════════════════════════════════════

def test_consistency_csv_aceito_pelo_benchmark(tmp_path):
    from benchmark.benchmark import run as run_consistency

    attack = tmp_path / "attack.csv"
    with attack.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "original", "tipo_variacao", "attacked"])
        w.writerow(["0", "Geladeira Electrolux RD30", "Abreviacao", "Gelad. Electrolux RD30"])
        w.writerow(["0", "Geladeira Electrolux RD30", "Sinonimo", "Refrigerador Electrolux RD30"])
        w.writerow(["1", "Notebook Dell Inspiron", "Abreviacao", "Note. Dell Inspiron"])

    out_csv = tmp_path / "normalized.csv"
    df = build_consistency_csv(StubBackend(), attack_csv=attack, out_csv=out_csv)
    assert list(df.columns) == ["original", "attacked", "normalized", "display_name"]

    res = run_consistency(out_csv, tmp_path / "normalized.json")
    assert res["metrics"]["num_originals"] == 2


def test_pairwise_csv_aceito_pelo_pairwise_f1(tmp_path):
    from benchmark.pairwise_f1 import run as run_pairwise

    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["title_a", "title_b", "label"])
        w.writerow(["Geladeira Electrolux RD30", "Gelad. Electrolux RD30", "1"])
        w.writerow(["Geladeira Electrolux RD30", "Notebook Dell Inspiron", "0"])

    out_csv = tmp_path / "pairs_normalized.csv"
    df = build_pairwise_csv(StubBackend(), pairs_csv=pairs, out_csv=out_csv)
    assert list(df.columns) == ["title_a", "title_b", "label", "normalized_a", "normalized_b"]

    res = run_pairwise(str(out_csv), str(tmp_path / "pairwise_results.json"))
    assert res["metrics"]["n_pairs"] == 2
