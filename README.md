# TAIL — Title Attribute Inference Layer

Zero-shot LLM pipeline for product title normalization in Brazilian e-commerce catalogs.

This repository contains the code, data, and evaluation scripts for the paper:

> **Zero-Shot Product Title Normalization with LLMs: A Multi-Model Evaluation**

## Project Structure

```
.
├── inference_harness/       # Core pipeline: backends, normalization, runner
│   ├── backends.py          # GeminiBackend, QwenBackend
│   ├── normalization.py     # System prompt, pre/post normalization, match key
│   ├── core.py              # normalize() / normalize_many()
│   └── runner.py            # build_consistency_csv(), build_pairwise_csv()
├── benchmark/               # Evaluation metrics
│   ├── benchmark.py         # Constancy metric
│   ├── pairwise_f1.py       # Pairwise F1 with blocking
│   ├── blocking.py          # Unit-based blocking + similarity
│   ├── weighted_distance.py # Weighted edit distance constancy
│   ├── by_attack_type.py    # Per-attack-type analysis
│   ├── mining.py            # Pair mining for labeling
│   └── labeled_pairs/       # Annotated dataset (1,200 pairs)
├── scripts/
│   └── reproduce.py         # End-to-end reproduction script
├── main_supersaurus/        # Notebooks and evaluation scripts
└── poc_*/                   # Early proof-of-concept notebooks
```

## Requirements

- Python 3.11+
- API keys for the models you want to evaluate

## Installation

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, the core dependencies are:

```bash
pip install pandas scikit-learn google-generativeai
```

## Environment Variables

Set the API keys for the backends you want to use:

```bash
export GOOGLE_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
```

## Reproducing the Experiments

The `scripts/reproduce.py` script runs the full pipeline end-to-end:

```bash
# Run with Gemini (default, single execution):
python scripts/reproduce.py --backends gemini

# Run all three backends:
python scripts/reproduce.py --backends gemini deepseek gpt

# Run 3 executions per backend (stability analysis, Table 6):
python scripts/reproduce.py --backends gemini deepseek gpt --runs 3
```

### What it does

For each backend and each run:

1. **Normalization**: Processes all attacked titles through the pipeline (`build_consistency_csv`)
2. **Pairwise matching**: Normalizes both sides of labeled pairs (`build_pairwise_csv`)
3. **Pairwise F1**: Computes precision, recall, F1 over 1,200 annotated pairs (`pairwise_f1.run`)
4. **Constancy**: Computes group constancy metric (`benchmark.run`)

### Outputs

```
output/<backend>/run_<N>/
├── normalized.csv           # original, attacked, normalized, display_name
├── pairs_normalized.csv     # title_a, title_b, label, normalized_a, normalized_b
├── pairwise_results.json    # precision, recall, F1, error examples
└── constancy_results.json   # mean/min/max constancy, best/worst groups
```

A consolidated summary is written to `output/reproduction_summary.json`.

## Running Individual Benchmarks

```bash
# Constancy benchmark
python -m benchmark.benchmark --input output/gemini/run_1/normalized.csv --out results.json

# Pairwise F1
python -m benchmark.pairwise_f1 --input output/gemini/run_1/pairs_normalized.csv --out pairwise.json

# Weighted distance constancy
python -m benchmark.weighted_distance --input output/gemini/run_1/normalized.csv --out weighted.json

# Blocking-based constancy
python -m benchmark.blocking --input output/gemini/run_1/normalized.csv --out blocking.json
```

## Dataset

The annotated dataset of 1,200 pairs is in `benchmark/labeled_pairs/`:

- `pairs_raw.csv` — Original mined pairs with labels (title_a, title_b, label, similarity, prefix)
- `pairs_normalized.csv` — Pairs with normalized names (used by pairwise F1)
- `pairs_positive.csv` — Subset of positive pairs (label=1)

## System Prompt

The full extraction prompt is defined in `inference_harness/normalization.py` (variable `SYSTEM_PROMPT`). It is also reproduced in the paper's Appendix A.

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{tail2026,
  title={Zero-Shot Product Title Normalization with LLMs: A Multi-Model Evaluation},
  author={...},
  booktitle={...},
  year={2026}
}
```

## License

This project is for academic research purposes.
