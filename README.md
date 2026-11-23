# Few-Shot Prompting vs. LoRA under Small Label Budgets

Code for the HW2 paper *Evaluating the Limits of Few-Shot Prompting and Adaptation in Resource-Scarce Specialized Domains* (CSE4057, Hanyang University).

We compare zero-shot prompting, random few-shot prompting, dynamic (retrieved) few-shot prompting, and LoRA for sentence-level section classification on PubMed 200k RCT (20k version), using Qwen2.5-3B-Instruct.

## Environment

- Google Colab, NVIDIA T4 GPU (16 GB), Python 3 (Colab default)
- Install:

```bash
pip install -r requirements.txt
```

The model (`Qwen/Qwen2.5-3B-Instruct`), the encoder (`sentence-transformers/all-MiniLM-L6-v2`) and the dataset (`armanc/pubmed-rct20k`) are downloaded automatically from Hugging Face. No access token is needed.

## Files

| File | Purpose |
|---|---|
| `data.py` | Samples the test set (1,000 sentences) and dev set (500), and the class-balanced labeled pools for B = 10, 50, 100 |
| `prompt_eval.py` | Zero-shot, random few-shot and dynamic few-shot prompting |
| `lora_train.py` | LoRA training on a labeled pool, then zero-shot-prompt inference |
| `run_experiments.py` | Runs every experiment in the paper (skips runs that already exist) |
| `score.py` | Builds all tables, the figure, bootstrap CIs and random samples from `results/` |

## Reproducing the paper

Run from the repository root, in this order:

```bash
python data.py              # 1. data subsets and labeled pools   (~1 min)
python run_experiments.py   # 2. all 40 runs                       (~2 h on a T4)
python score.py             # 3. tables and figure -> paper_outputs/ (~1 min, no GPU)
```

| Paper item | Produced by | Output file |
|---|---|---|
| Table 1 (main results) | `score.py` | `paper_outputs/table1_main.tex` |
| Table 2 (stability) | `score.py` | `paper_outputs/table2_stability.tex` |
| Table 3 (per-class F1) | `score.py` | `paper_outputs/table3_perclass.tex` |
| Figure 1 (label budget) | `score.py` | `paper_outputs/fig1_budget.pdf` |
| H1 bootstrap CIs | `score.py` | `paper_outputs/h1_bootstrap.txt` |
| Error examples (Sec. 6.4) | `score.py` | `paper_outputs/samples.txt` |

A single run can also be reproduced directly, for example:

```bash
python prompt_eval.py --method dynamic --k 3 --budget 100 --seed 0
python lora_train.py --budget 10 --seed 0
```

## Random seeds

| What | Seed(s) |
|---|---|
| Test / dev subset sampling | 0 |
| Labeled pools (`data.py` builds 0–4; the paper uses 0, 1, 2) | 0, 1, 2 |
| Random few-shot support set | same as the pool seed |
| LoRA training (`torch`, `random`) | same as the pool seed |
| Bootstrap and example sampling (`score.py`) | 0 |

Decoding is greedy. Small numerical differences can still occur across GPU types and library versions.

## Notes

- Results are appended to `results/summary.jsonl`; `score.py` keeps the last entry per run.
- On Colab, run everything inside a Google Drive folder so results survive disconnects.