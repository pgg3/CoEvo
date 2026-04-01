# CoEvo

**CoEvo: Continual Evolution of Symbolic Solutions using Large Language Models**

Official implementation for the AAAI 2026 paper. CoEvo is built on [EvoToolkit](https://github.com/pgg3/evotoolkit) and uses multi-layer idea generation with Pareto-based selection for symbolic regression and scientific discovery.

> Original standalone implementation: [`v1`](../../tree/v1) branch

## Installation

Requires Python >= 3.10.

```bash
pip install uv
cd CoEvo
uv sync
```

## Usage

```bash
uv run python main_run.py --task oscillation_1 --max_gen 97 --pop_size 2
```

Set environment variables for LLM API:
```bash
export API_URL="https://api.openai.com/v1"
export API_KEY="your-key"
export MODEL="gpt-4o"
```

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
