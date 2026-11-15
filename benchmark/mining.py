import argparse
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def similarity(a: str, b: str) -> float:
    
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def mine_by_prefixes(
    df: pd.DataFrame,
    prefixes: list[str],
    items_per_prefix: int = 30,
    language_filter: str = None,
) -> pd.DataFrame:
   
    all_pairs = []

    for prefix in prefixes:
        prefix_clean = prefix.strip().lower()
        mask = df["title"].str.lower().str.startswith(prefix_clean)
        group = df[mask].copy()

        
        if language_filter and "language" in group.columns:
            group = group[group["language"] == language_filter]

        group = group.drop_duplicates(subset=["title"]).head(items_per_prefix).reset_index(drop=True)

        if len(group) < 2:
            logger.warning("Prefixo '%s': menos de 2 títulos encontrados, pulando.", prefix)
            continue

        logger.info("Prefixo '%s': %d títulos selecionados → %d pares.",
                    prefix, len(group), len(group) * (len(group) - 1) // 2)

        for i, j in combinations(range(len(group)), 2):
            title_a = group.loc[i, "title"]
            title_b = group.loc[j, "title"]

            all_pairs.append({
                "title_a":    title_a,
                "title_b":    title_b,
                "category_a": group.loc[i, "category"] if "category" in group.columns else "",
                "category_b": group.loc[j, "category"] if "category" in group.columns else "",
                "prefix":     prefix_clean,
                "similarity": round(similarity(title_a, title_b), 4),
                "label":      "",   # ← preencha: 1 = mesmo produto, 0 = diferente
            })

    if not all_pairs:
        return pd.DataFrame()

    result = pd.DataFrame(all_pairs)

    result = result.sort_values(
        ["prefix", "similarity"], ascending=[True, False]
    ).reset_index(drop=True)

    return result

def run(
    input_path: str = "data/train.csv",
    out_path: str = "data/pairs_raw.csv",
    prefixes: list[str] = None,
    items_per_prefix: int = 30,
    language_filter: str = None,
) -> pd.DataFrame:
   
    if not prefixes:
        logger.error("Informe ao menos um prefixo com --prefixes.")
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    logger.info("Dataset carregado: %d linhas", len(df))
    df = df.dropna(subset=["title"]).reset_index(drop=True)

    result = mine_by_prefixes(
        df,
        prefixes=prefixes,
        items_per_prefix=items_per_prefix,
        language_filter=language_filter,
    )

    if result.empty:
        logger.error("Nenhum par gerado. Verifique os prefixos.")
        return result

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info("Total de pares gerados : %d", len(result))
    for prefix in result["prefix"].unique():
        n = (result["prefix"] == prefix).sum()
        logger.info("  %-20s → %d pares", prefix, n)
    logger.info("Salvos em: %s", out_path)
    logger.info("")
    logger.info("Próximo passo: abra o CSV e preencha a coluna 'label'")
    logger.info("   1  = mesmo produto")
    logger.info("   0  = produtos diferentes")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mineração de pares parecidos para rotulagem manual"
    )
    parser.add_argument("--input",            default="data/train.csv",     help="Dataset Mercado Livre")
    parser.add_argument("--out",              default="data/pairs_raw.csv", help="CSV de saída")
    parser.add_argument("--prefixes",         required=True,                help="Prefixos separados por vírgula: 'coca,geladeira,ventilador'")
    parser.add_argument("--items-per-prefix", default=30, type=int,         help="Títulos por prefixo (padrão: 30)")
    parser.add_argument("--language",         default=None,                 help="Filtrar idioma: portuguese ou spanish")
    args = parser.parse_args()

    prefixes = [p.strip() for p in args.prefixes.split(",")]

    run(
        input_path=args.input,
        out_path=args.out,
        prefixes=prefixes,
        items_per_prefix=args.items_per_prefix,
        language_filter=args.language,
    )
