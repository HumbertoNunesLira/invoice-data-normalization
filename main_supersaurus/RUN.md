# Como executar — main_supersaurus

Pipeline de normalização de títulos de produtos (e-commerce / NF-e) com LLM, teste de robustez via *attack augmentation* e benchmark de consistência.

**Pipeline modular (recomendado):**

| Notebook | Saída |
|----------|--------|
| `attack_augmentation.ipynb` | `data/attack.csv` |
| `classifier_api.ipynb` | `output/normalized.csv` |
| `benchmark.ipynb` | `output/normalized.json`, `output/by_attack_type.json` |

**Monolítico:** `model_v2_1.ipynb` (tudo em um arquivo)  
**Legado:** `model_v2.ipynb`

---

## Visão geral

```
data/train.csv
      │
      ▼  (amostragem + attack LLM)
data/attack.csv          ← 6 variantes adversariais por título original
      │
      ▼  (pré-normalização simbólica + extração LLM + limpeza)
output/normalized.csv    ← original | attacked | normalized | display_name
      │
      ▼
output/normalized.json   ← métricas de constância (`consistency_benchmark.ipynb` / `benchmark/`)
```

**Arquitetura (5 passos):**

1. Ingestão e amostragem
2. Pré-normalização simbólica do título (antes do LLM)
3. Extração via LLM com batch prompting
4. Limpeza determinística (camada simbólica / regex)
5. Exportação + avaliação estruturada

---

## Pré-requisitos

| Item | Detalhe |
|------|---------|
| Python | 3.11+ |
| Jupyter | Notebook ou JupyterLab |
| Chave de API | `GOOGLE_API_KEY` (Google AI / Gemini) |
| GPU | Não necessária para o pipeline principal |
| GPU (Colab) | L4 ou superior para `finetune_nfe_colab.ipynb` |

---

## Setup

### 1. Clonar o repositório e entrar na pasta

```bash
cd /caminho/para/invoice-data-normalizer/main_supersaurus
```

### 2. Ambiente virtual (recomendado)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
pip install jupyter ipywidgets   # opcional, remove warning do tqdm no notebook
```

Dependências principais (`../requirements.txt`):

- `pandas`, `numpy`, `tqdm`, `python-dotenv`, `google-generativeai`

### 3. Variáveis de ambiente

Crie `.env` na raiz de `main_supersaurus/`:

```bash
GOOGLE_API_KEY=sua_chave_aqui
```

> `.env` e `data/train.csv` estão no `.gitignore` — não são versionados.

### 4. Dados de entrada

Coloque o CSV de títulos em:

```
main_supersaurus/data/train.csv
```

Colunas esperadas (mínimo):

| Coluna | Uso |
|--------|-----|
| `title` | Título original do produto (obrigatória) |

Colunas adicionais (`category`, `language`, etc.) são ignoradas pelo pipeline principal, mas úteis em células de teste avançadas.

### 5. Diretórios de saída

```bash
mkdir -p data output
```

---

## Pipeline principal — `model_v2_1.ipynb`

Abra o notebook e execute as células **em ordem**, a partir da raiz do repositório ou de `main_supersaurus/`:

```bash
jupyter notebook model_v2_1.ipynb
# ou
jupyter lab model_v2_1.ipynb
```

### Configuração (célula de config)

Valores padrão:

| Variável | Default | Descrição |
|----------|---------|-----------|
| `INPUT_PATH` | `data/train.csv` | CSV de entrada |
| `INPUT_COLUMN` | `title` | Coluna com títulos |
| `ATTACK_PATH` | `data/attack.csv` | Saída do stage de attack |
| `NORMALIZED_PATH` | `output/normalized.csv` | Saída final |
| `BATCH_SIZE` | `5` | Títulos por chamada LLM |
| `RANDOM_SEED` | `42` | Reprodutibilidade da amostra |
| `ATTACK_MODEL` | `gemini-2.5-flash` | Gera variantes adversariais |
| `NORMALIZATION_MODEL` | `gemini-3.1-flash-lite-preview` | Normaliza títulos atacados |
| `TEMPERATURE` | `0.0` | Determinismo máximo |

### Amostragem

Por padrão o notebook amostra **100 títulos** (`df.sample(n=100, ...)`). Ajuste `n` conforme custo e tempo desejados:

- 100 originais × 6 attacks = **600 linhas** em `attack.csv` / `normalized.csv`
- Cada batch de 5 títulos = 1 chamada LLM (attack + normalização separados)

Para processar o dataset inteiro, remova ou comente a célula de `sample()`.

### Saídas esperadas

| Arquivo | Colunas principais |
|---------|-------------------|
| `data/attack.csv` | `original`, `attacked` |
| `output/normalized.csv` | `original`, `attacked`, `normalized`, `display_name` |

- **`normalized`**: *match key* estável — usado pelo benchmark de constância
- **`display_name`**: nome canônico legível — inspeção humana

### Tempo e custo (estimativa)

Com 100 originais e `BATCH_SIZE=5`:

- ~20 chamadas LLM para attack + ~120 para normalização (600 títulos / 5)
- Tempo típico: **15–45 min** (depende de rate limit e latência da API)

---

## Benchmark de consistência

O benchmark oficial fica em `benchmark/` na **raiz do repositório**. Execute a partir da raiz:

```bash
cd /caminho/para/invoice-data-normalizer

python -m benchmark.benchmark \
  --input main_supersaurus/output/normalized.csv \
  --out main_supersaurus/output/results.json
```

Métricas principais em `results.json`:

| Métrica | Significado |
|---------|-------------|
| `mean_constancy` | Média de `dominant_count / group_size` por `original` |
| `weighted_constancy` | Média ponderada pelo tamanho do grupo |
| `best_grouping` / `worst_grouping` | Produto com maior / menor consistência entre variantes |

> A última célula de benchmark em `model_v2_1.ipynb` chama o mesmo módulo via `subprocess`. Funciona se o kernel Jupyter foi iniciado na **raiz do repo**; caso contrário, use o comando acima.

---

## Avaliação complementar — `eval.py`

Script com métricas adicionais de qualidade JSON, completude de atributos, corner-cases e robustez.

### Relatório de robustez (attack vs normalized)

Requer CSV com colunas `original`, `attacked`, `normalized` (ex.: `output/normalized.csv`):

```bash
cd main_supersaurus

python -c "
import pandas as pd
from eval import gerar_relatorio_robusto
df = pd.read_csv('output/normalized.csv')
gerar_relatorio_robusto(df)
"
```

### Métricas detalhadas (JSON, atributos, agrupamento)

Requer CSV enriquecido com colunas como `json_limpo_python`, `json_llm_bruto`, `nome_canonico_final`, `categoria_original` — gerado nas células avançadas do notebook (seção de teste / `resultado_pipeline_v2.csv`).

```python
import pandas as pd
from eval import (
    avaliar_qualidade_json,
    avaliar_completude_atributos,
    avaliar_agrupamento,
    detectar_corner_cases,
    avaliar_correcoes_simbolicas,
)

df = pd.read_csv("output/resultado_pipeline_v2.csv")
print(avaliar_qualidade_json(df))
print(avaliar_completude_atributos(df))
```

---

## Pipeline legado — `model_v2.ipynb`

Mesma arquitetura, diferenças principais:

| Aspecto | `model_v2` | `model_v2_1` |
|---------|------------|--------------|
| Attack output | `output/attack.csv` | `data/attack.csv` |
| Modelo LLM | único (`gemini-3.1-flash-lite-preview`) | dois modelos (attack + normalização) |
| `normalized` | nome canônico completo | *match key* estável + `display_name` |
| Benchmark | células inline / eval local | `benchmark.benchmark` canônico |

Use `model_v2_1` para novos experimentos.

---

## Fine-tuning (Colab) — `finetune_nfe_colab.ipynb`

Destila o pipeline Gemini num SLM (Qwen2.5-3B) via QLoRA / Unsloth.

**Onde rodar:** Google Colab Pro (GPU L4 recomendada) — **não** roda localmente sem GPU adequada.

### Arquivos necessários (upload manual no Colab)

1. `finetune_dataset_augmented.jsonl` — ~1400 pares (200 originais × 7 variantes)
2. `benchmark/benchmark.py` — módulo de benchmark do repo
3. `resultado_pipeline_v2.csv` *(opcional)* — títulos originais reais

### Fluxo no notebook

1. Sanity check GPU
2. Install (Unsloth, TRL, PEFT, …)
3. Split **por original** (180 train / 20 eval) — evita leakage entre variantes
4. Treino QLoRA (~3 épocas, ~30–45 min no L4)
5. Inferência no holdout
6. Benchmark: Gemini (upper bound) vs Qwen fine-tuned vs Qwen zero-shot

Saídas em `outputs/qwen25-3b-nfe-r16/` (nome configurável):

- `eval_gemini_upperbound.csv`, `eval_finetuned.csv`, `eval_zeroshot.csv`
- `results_*.json`
- `final_adapter/` — adapter LoRA treinado

---

## Troubleshooting

### `Arquivo não encontrado: data/train.csv`

O dataset não vem versionado. Copie ou gere um CSV com coluna `title` em `data/train.csv`.

### `GOOGLE_API_KEY` ausente ou inválida

Verifique `.env` na pasta `main_supersaurus/` e reinicie o kernel Jupyter após criar/editar o arquivo.

### `ModuleNotFoundError: benchmark`

Execute o benchmark a partir da **raiz do repositório** (`python -m benchmark.benchmark ...`), não de dentro de `main_supersaurus/`.

### Rate limit / quota da API Gemini

- Reduza a amostra (`n=20` ou menos)
- Aumente o `time.sleep` entre batches (se presente no notebook)
- Use modelos mais leves para testes iniciais

### Warning `google.generativeai` deprecated

O notebook usa `google-generativeai`. A Google recomenda migrar para `google-genai` no futuro; o pipeline atual ainda funciona com o pacote instalado via `requirements.txt`.

### Kernel Jupyter iniciado no diretório errado

Paths relativos (`data/`, `output/`) dependem do diretório de trabalho. Inicie o Jupyter em `main_supersaurus/` ou ajuste os paths na célula de config.

---

## Referência rápida de comandos

```bash
# Setup
cd main_supersaurus
python -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
echo "GOOGLE_API_KEY=..." > .env
mkdir -p data output

# Pipeline (interativo)
jupyter notebook model_v2_1.ipynb

# Benchmark (da raiz do repo)
cd ..
python -m benchmark.benchmark \
  --input main_supersaurus/output/normalized.csv \
  --out main_supersaurus/output/results.json

# Relatório de robustez
cd main_supersaurus
python -c "import pandas as pd; from eval import gerar_relatorio_robusto; gerar_relatorio_robusto(pd.read_csv('output/normalized.csv'))"
```
