"""
Prompting conditions: zero-shot, random few-shot, dynamic (retrieved) few-shot.

  zero    : instruction + test sentence
  random  : k demonstrations drawn ONCE per seed from the pool (a fixed support set)
  dynamic : for each test sentence, the k most similar pool sentences
            (all-MiniLM-L6-v2 embeddings, cosine similarity); most similar placed last

Every condition with a pool uses data/pool_B{B}_s{S}.jsonl, so random and dynamic
see exactly the same labeled data for a given (B, S).

Usage:
  python prompt_eval.py --method zero
  python prompt_eval.py --method random  --k 3 --budget 100 --seed 0
  python prompt_eval.py --method dynamic --k 3 --budget 100 --seed 0
  (add --split dev to evaluate on the dev subset)

Output: results/{split}_{method}_k{k}_B{B}_s{S}.jsonl  (one line per sentence)
        results/summary.jsonl                           (one line per run)
"""
import argparse, json, os, random, re, time
import torch
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
LABELS = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
INSTRUCTION = (
    "Classify the following sentence from a biomedical abstract into exactly one section label: "
    + ", ".join(LABELS)
    + ".\nAnswer with the label only."
)


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def build_prompt(sentence, demos):
    parts = [INSTRUCTION]
    if demos:
        parts.append("Examples:")
        for d in demos:
            parts.append(f"Sentence: {d['text']}\nLabel: {d['label']}")
    parts.append(f"Sentence: {sentence}\nLabel:")
    return "\n\n".join(parts)


def parse(raw):
    up = raw.upper()
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", up):
            return lab
    return "INVALID"


def select_demos(method, k, eval_rows, pool, seed):
    """Return a list (one entry per eval sentence) of demonstration lists."""
    if method == "zero":
        return [[] for _ in eval_rows]
    if method == "random":
        fixed = random.Random(seed).sample(pool, k)
        return [fixed for _ in eval_rows]
    if method == "dynamic":
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer(EMBEDDER, device="cuda" if torch.cuda.is_available() else "cpu")
        P = emb.encode([p["text"] for p in pool], convert_to_tensor=True, normalize_embeddings=True)
        Q = emb.encode([r["text"] for r in eval_rows], convert_to_tensor=True, normalize_embeddings=True)
        sims = Q @ P.T                                  # cosine (normalized)
        top = sims.topk(k, dim=1).indices.tolist()      # most similar first
        del emb
        return [[pool[j] for j in reversed(idx)] for idx in top]  # most similar last
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["zero", "random", "dynamic"], required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    eval_rows = load(f"data/{args.split}.jsonl")
    pool = [] if args.method == "zero" else load(f"data/pool_B{args.budget}_s{args.seed}.jsonl")
    demos = select_demos(args.method, args.k, eval_rows, pool, args.seed)

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"                           # required for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto")
    model.eval()

    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(r["text"], d)}],
            add_generation_prompt=True, tokenize=False,
        )
        for r, d in zip(eval_rows, demos)
    ]

    raws, n_prompt_tokens = [], 0
    t0 = time.time()
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        n_prompt_tokens += int(enc["attention_mask"].sum())
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        raws += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    elapsed = time.time() - t0

    preds = [parse(r) for r in raws]
    golds = [r["label"] for r in eval_rows]
    macro = f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0)
    acc = accuracy_score(golds, preds)

    tag = f"{args.split}_{args.method}_k{args.k}_B{args.budget}_s{args.seed}"
    os.makedirs("results", exist_ok=True)
    with open(f"results/{tag}.jsonl", "w") as f:
        for r, d, raw, p in zip(eval_rows, demos, raws, preds):
            f.write(json.dumps({"text": r["text"], "gold": r["label"], "pred": p, "raw": raw,
                                "demos": [x["label"] for x in d]}) + "\n")
    summary = {"tag": tag, **vars(args), "macro_f1": macro, "accuracy": acc,
               "invalid": preds.count("INVALID"), "n": len(preds),
               "infer_sec": elapsed, "avg_prompt_tokens": n_prompt_tokens / len(preds)}
    with open("results/summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
