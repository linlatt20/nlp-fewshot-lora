"""
Build fixed evaluation subsets and label-budget training pools.

Outputs (in data/):
  test.jsonl                 N_TEST sentences sampled from the official test split (seed 0)
  dev.jsonl                  N_DEV sentences sampled from the official validation split (seed 0)
  pool_B{B}_s{S}.jsonl       class-balanced training pool of size B (B/5 per label), seed S

Usage:
  python data.py
"""
import json, os, random
from collections import defaultdict, Counter
from datasets import load_dataset

N_TEST = 1000
N_DEV = 500
BUDGETS = [10, 50, 100]
SEEDS = [0, 1, 2, 3, 4]
OUT = "data"


def rows(split):
    return [{"text": t, "label": l.upper()} for t, l in zip(split["text"], split["label"])]


def write(path, items):
    with open(path, "w") as f:
        for r in items:
            f.write(json.dumps(r) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset("armanc/pubmed-rct20k")

    test = rows(ds["test"])
    dev = rows(ds["validation"])
    train = rows(ds["train"])

    # Natural-distribution subsets (macro-F1 handles the imbalance).
    test_sub = random.Random(0).sample(test, N_TEST)
    dev_sub = random.Random(0).sample(dev, N_DEV)
    write(f"{OUT}/test.jsonl", test_sub)
    write(f"{OUT}/dev.jsonl", dev_sub)
    print("test:", Counter(r["label"] for r in test_sub))
    print("dev :", Counter(r["label"] for r in dev_sub))

    by_label = defaultdict(list)
    for r in train:
        by_label[r["label"]].append(r)
    labels = sorted(by_label)
    print("labels:", labels)

    # Class-balanced pools: guarantees every label appears even at B=10.
    for B in BUDGETS:
        per = B // len(labels)
        for S in SEEDS:
            rng = random.Random(1000 * B + S)
            pool = []
            for lab in labels:
                pool += rng.sample(by_label[lab], per)
            rng.shuffle(pool)
            write(f"{OUT}/pool_B{B}_s{S}.jsonl", pool)
    print("pools written:", len(BUDGETS) * len(SEEDS))


if __name__ == "__main__":
    main()
