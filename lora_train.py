"""
LoRA condition: train low-rank adapters on a label-budget pool, then predict
with the zero-shot prompt (no demonstrations).

Training target: the label tokens only (prompt tokens are masked with -100).
Fixed hyperparameters (no tuning; report in the appendix):
  r=8, alpha=16, dropout=0.05, targets q/k/v/o_proj, AdamW lr=2e-4, batch size 4,
  steps = max(MIN_STEPS, EPOCHS * ceil(B / batch)), fp16 autocast, grad clip 1.0

Usage:
  python lora_train.py --budget 100 --seed 0
  (add --split dev to evaluate on the dev subset)

Output: results/{split}_lora_k0_B{B}_s{S}.jsonl and a line in results/summary.jsonl
"""
import argparse, json, math, os, random, time
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_eval import MODEL, LABELS, build_prompt, parse, load

LR = 2e-4
BATCH = 4
EPOCHS = 5
MIN_STEPS = 50
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")


def chat_prompt(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": build_prompt(text, [])}],
                                   add_generation_prompt=True, tokenize=False)


def encode(tok, ex):
    p = tok(chat_prompt(tok, ex["text"]), add_special_tokens=False)["input_ids"]
    a = tok(ex["label"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p + a, [-100] * len(p) + a


def collate(items, pad_id, device):
    L = max(len(x) for x, _ in items)
    ids = torch.full((len(items), L), pad_id)
    lab = torch.full((len(items), L), -100)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for i, (x, y) in enumerate(items):          # right padding for training
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["test", "dev"], default="test")
    ap.add_argument("--batch_size", type=int, default=16)   # inference batch
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pool = load(f"data/pool_B{args.budget}_s{args.seed}.jsonl")
    eval_rows = load(f"data/{args.split}.jsonl")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**LORA))
    for p in model.parameters():                 # adapters in fp32 for stable fp16 training
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    # ---- train ----
    data = [encode(tok, ex) for ex in pool]
    steps = max(MIN_STEPS, EPOCHS * math.ceil(len(data) / BATCH))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(args.seed)
    model.train()
    t0 = time.time()
    for step in range(steps):
        batch = collate(rng.sample(data, min(BATCH, len(data))), tok.pad_token_id, model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}")
    train_sec = time.time() - t0

    # ---- predict (zero-shot prompt, batched, left padding) ----
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    prompts = [chat_prompt(tok, r["text"]) for r in eval_rows]
    raws = []
    t1 = time.time()
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        raws += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    infer_sec = time.time() - t1

    preds = [parse(r) for r in raws]
    golds = [r["label"] for r in eval_rows]
    tag = f"{args.split}_lora_k0_B{args.budget}_s{args.seed}"
    os.makedirs("results", exist_ok=True)
    with open(f"results/{tag}.jsonl", "w") as f:
        for r, raw, p in zip(eval_rows, raws, preds):
            f.write(json.dumps({"text": r["text"], "gold": r["label"], "pred": p, "raw": raw}) + "\n")
    summary = {"tag": tag, "method": "lora", "k": 0, "budget": args.budget, "seed": args.seed,
               "split": args.split,
               "macro_f1": f1_score(golds, preds, labels=LABELS, average="macro", zero_division=0),
               "accuracy": accuracy_score(golds, preds), "invalid": preds.count("INVALID"),
               "n": len(preds), "train_steps": steps, "train_sec": train_sec, "infer_sec": infer_sec}
    with open("results/summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
