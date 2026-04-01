# CoEvo

**CoEvo: Continual Evolution of Symbolic Solutions using Large Language Models**

Official implementation for the AAAI 2026 paper. CoEvo is built on [EvoToolkit](https://github.com/pgg3/evotoolkit) and uses multi-layer idea generation with Pareto-based selection for symbolic regression and scientific discovery.

> Original standalone implementation: [`v1`](../../tree/v1) branch

## Project Structure

```
CoEvo/                  # Repository root
├── main_run.py         # Main entry point (eoh / coevo modes)
├── model_config.json   # LLM API configuration (fill in your key)
├── data/               # Datasets
│   ├── bactgrow/
│   ├── oscillation_1/
│   ├── oscillation_2/
│   └── stress_strain/
├── CoEvo/              # Python package root
│   ├── pyproject.toml
│   └── src/coevo/
│       ├── core/       # CoEvo algorithm, interface, state, NDS, summarizer
│       ├── tasks/      # Task definitions (one per dataset)
│       └── utils/      # String formatting utilities
├── assets/             # Figures and visual assets
└── paper_res/          # Paper result analysis
```

## Installation

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
cd CoEvo/CoEvo
uv sync
```

To enable the embedding-based summarizer (requires PyTorch):

```bash
uv sync --extra summarizer
```

## LLM Configuration

CoEvo requires access to an OpenAI-compatible LLM API. Configure it via `model_config.json` in the repository root:

```json
{
  "host": "https://api.openai.com",
  "key": "your-api-key-here",
  "model": "gpt-4o",
  "url": "/v1/chat/completions",
  "timeout": 120
}
```

Alternatively, set environment variables (used as fallback when `model_config.json` is absent):

```bash
export API_URL="https://api.openai.com/v1/chat/completions"
export API_KEY="your-key"
export MODEL="gpt-4o"
```

## Usage

All commands run from the `CoEvo/CoEvo/` directory (the Python package root):

```bash
cd CoEvo/CoEvo
```

### CoEvo Mode (default)

Full CoEvo algorithm with multi-representation prompting, chain-based evolution, and NDS selection:

```bash
uv run python ../main_run.py --task oscillation_1 --max_gen 97 --pop_size 2
```

### EoH Mode

Use EvoToolkit's built-in EoH algorithm as a baseline:

```bash
uv run python ../main_run.py --task oscillation_1 --mode eoh --max_gen 97 --pop_size 2
```

### Available Tasks

| Task | Dataset | Description |
|------|---------|-------------|
| `oscillation_1` | `data/oscillation_1/train.csv` | Oscillation system discovery (type 1) |
| `oscillation_2` | `data/oscillation_2/train.csv` | Oscillation system discovery (type 2) |
| `bactgrow` | `data/bactgrow/train.csv` | Bacterial growth modeling |
| `stress_strain` | `data/stress_strain/train.csv` | Stress-strain relationship discovery |

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `oscillation_1` | Task name (see table above) |
| `--mode` | `coevo` | Algorithm mode: `coevo` or `eoh` |
| `--max_gen` | `97` | Maximum number of generations |
| `--pop_size` | `2` | Population size for NDS selection |

### Output

Results are saved to `CoEvo/CoEvo/results/<task>_<mode>/`:

- `run_state.json` — Algorithm state checkpoint
- `history/` — Per-generation solution history (JSON)
- `summary/` — Summarizer idea pool snapshots

## Algorithm Overview

CoEvo extends the [EoH](https://github.com/pgg3/evotoolkit) framework with:

1. **Multi-representation prompting** — Solutions are described in natural language, Python code, and mathematical formulas simultaneously
2. **Chain-based evolution** — Each solution goes through init → continue layers, forming a reasoning chain
3. **Non-dominated sorting (NDS)** — Pareto selection on multiple objectives (e.g., MSE + complexity)
4. **Embedding-based summarizer** — Uses GPT-2 embeddings + DBSCAN clustering to maintain a diverse idea pool that inspires future generations

## Citation

```bibtex
@inproceedings{guo2026coevo,
  title={Coevo: Continual evolution of symbolic solutions using large language models},
  author={Guo, Ping and Zhang, Qingfu and Lin, Xi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={3},
  pages={1810--1818},
  year={2026}
}
```

## Contact

For questions or issues, contact: pingguo5-c@my.cityu.edu.hk
