import json, os
import numpy as np
import pandas as pd

LABELS = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]
L2I = {l: i for i, l in enumerate(LABELS)}
OUT = "paper_outputs"
N_BOOT = 1000
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- loading
def load_summary():
    rows, bad = [], 0
    with open("results/summary.jsonl") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[warn] skipped {bad} broken line(s) in summary.jsonl")
    df = pd.DataFrame(rows).drop_duplicates("tag", keep="last")
    df = df[df["split"] == "test"].copy()
    for c in ["avg_prompt_tokens", "infer_sec", "train_sec"]:
        if c not in df:
            df[c] = np.nan
    return df


def load_preds(tag):
    gold, pred = [], []
    with open(f"results/{tag}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            gold.append(L2I[r["gold"]])
            pred.append(L2I.get(r["pred"], -1))   # INVALID -> -1 (always wrong)
    return np.array(gold), np.array(pred)


def per_class_f1(g, p):
    out = []
    for c in range(len(LABELS)):
        tp = np.sum((p == c) & (g == c))
        fp = np.sum((p == c) & (g != c))
        fn = np.sum((p != c) & (g == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(out)


def macro_f1(g, p):
    return per_class_f1(g, p).mean()


def name(row):
    if row["method"] == "zero":
        return "Zero-shot"
    if row["method"] == "lora":
        return f"LoRA ($B$={row['budget']})"
    m = "Random" if row["method"] == "random" else "Dynamic"
    return f"{m} $k$={row['k']} ($B$={row['budget']})"


def pm(mean, std, n):
    return f"{100*mean:.1f}" if n <= 1 or np.isnan(std) else f"{100*mean:.1f} $\\pm$ {100*std:.1f}"


# ---------------------------------------------------------------- tables
df = load_summary()
print(f"{len(df)} test runs found")
print(df.groupby(["method", "budget", "k"]).size().rename("n_seeds").to_string(), "\n")

g = (df.groupby(["method", "budget", "k"])
       .agg(f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
            acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
            n=("macro_f1", "size"), invalid=("invalid", "mean"),
            tokens=("avg_prompt_tokens", "mean"), infer=("infer_sec", "mean"),
            train=("train_sec", "mean"))
       .reset_index())
order = {"zero": 0, "random": 1, "dynamic": 2, "lora": 3}
g = g.sort_values(by=["method", "budget", "k"], key=lambda s: s.map(order) if s.name == "method" else s)
g["name"] = g.apply(name, axis=1)
g.to_csv(f"{OUT}/all_conditions.csv", index=False)
print(g[["name", "n", "f1_mean", "f1_std", "acc_mean", "invalid", "tokens", "infer", "train"]]
      .to_string(index=False), "\n")

lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lcccr}", r"\toprule",
         r"Condition & Seeds & Macro-F1 & Acc. & Tokens \\", r"\midrule"]
prev = None
for _, r in g.iterrows():
    if prev is not None and r["method"] != prev:
        lines.append(r"\midrule")
    prev = r["method"]
    tok = "--" if np.isnan(r["tokens"]) else f"{r['tokens']:.0f}"
    lines.append(f"{r['name']} & {r['n']} & {pm(r['f1_mean'], r['f1_std'], r['n'])} & "
                 f"{pm(r['acc_mean'], r['acc_std'], r['n'])} & {tok} \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{TODO: write caption (test set, mean $\pm$ std over seeds, tokens = avg prompt tokens).}",
          r"\label{tab:main}", r"\end{table}"]
open(f"{OUT}/table1_main.tex", "w").write("\n".join(lines))

# Table 2: stability (H3)
st = (df[df["method"].isin(["random", "dynamic"])]
      .groupby(["method", "budget", "k"])["macro_f1"]
      .agg(["std", "min", "max", "size"]).reset_index())
lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{llcccc}", r"\toprule",
         r"$B$ & $k$ & Method & Std & Min & Max \\", r"\midrule"]
for _, r in st.sort_values(["budget", "k", "method"]).iterrows():
    lines.append(f"{r['budget']} & {r['k']} & {r['method'].capitalize()} & {100*r['std']:.2f} & "
                 f"{100*r['min']:.1f} & {100*r['max']:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{TODO: macro-F1 variability across support-set seeds.}",
          r"\label{tab:stability}", r"\end{table}"]
open(f"{OUT}/table2_stability.tex", "w").write("\n".join(lines))
print(st.to_string(index=False), "\n")

# ---------------------------------------------------------------- H1 paired bootstrap
rng = np.random.default_rng(0)
out = []
for (B, k), _ in df[df["method"] == "dynamic"].groupby(["budget", "k"]):
    pairs = []
    for s in sorted(df[(df["method"] == "dynamic") & (df["budget"] == B) & (df["k"] == k)]["seed"]):
        td, tr = f"test_dynamic_k{k}_B{B}_s{s}", f"test_random_k{k}_B{B}_s{s}"
        if os.path.exists(f"results/{tr}.jsonl"):
            gd, pd_ = load_preds(td)
            _, pr = load_preds(tr)
            pairs.append((gd, pd_, pr))
    if not pairs:
        continue
    N = len(pairs[0][0])
    point = np.mean([macro_f1(g_, a) - macro_f1(g_, b) for g_, a, b in pairs])
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, N, N)               # same resampled sentences for both methods
        boots[i] = np.mean([macro_f1(g_[idx], a[idx]) - macro_f1(g_[idx], b[idx]) for g_, a, b in pairs])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out.append(f"B={B} k={k} seeds={len(pairs)}: dynamic - random = {100*point:+.2f} "
               f"[95% CI {100*lo:+.2f}, {100*hi:+.2f}]  {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")
open(f"{OUT}/h1_bootstrap.txt", "w").write("\n".join(out) + "\n")
print("H1 paired bootstrap (macro-F1 points):\n" + "\n".join(out), "\n")

# ---------------------------------------------------------------- per-class F1 (best config per method)
best = g.loc[g.groupby("method")["f1_mean"].idxmax()]
lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{l" + "c" * len(LABELS) + "}",
         r"\toprule", "Condition & " + " & ".join(l.capitalize()[:4] + "." for l in LABELS) + r" \\",
         r"\midrule"]
for _, r in best.sort_values("method", key=lambda s: s.map(order)).iterrows():
    tags = df[(df["method"] == r["method"]) & (df["budget"] == r["budget"]) & (df["k"] == r["k"])]["tag"]
    pcs = np.mean([per_class_f1(*load_preds(t)) for t in tags], axis=0)
    lines.append(r["name"] + " & " + " & ".join(f"{100*v:.1f}" for v in pcs) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{TODO: per-class F1 (mean over seeds) for the best configuration of each method.}",
          r"\label{tab:perclass}", r"\end{table}"]
open(f"{OUT}/table3_perclass.tex", "w").write("\n".join(lines))

# ---------------------------------------------------------------- Figure 1: budget curve (H2)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.2, 3.0))
budgets = sorted(b for b in df["budget"].unique() if b > 0)
z = g[g["method"] == "zero"]
if len(z):
    ax.axhline(100 * z["f1_mean"].iloc[0], ls="--", color="gray", lw=1, label="Zero-shot")
styles = {"random": ("o", "Random"), "dynamic": ("s", "Dynamic"), "lora": ("^", "LoRA")}
for m, (mk, lab) in styles.items():
    sub = g[g["method"] == m]
    if sub.empty:
        continue
    if m != "lora":                               # fixed k=3 at every budget (no test-set selection)
        sub = sub[sub["k"] == 3]
        lab += " ($k$=3)"
    sub = sub.sort_values("budget")
    ax.errorbar(sub["budget"], 100 * sub["f1_mean"], yerr=100 * sub["f1_std"].fillna(0),
                marker=mk, capsize=3, lw=1.5, label=lab)
ax.set_xscale("log")
ax.set_xticks(budgets)
ax.set_xticklabels([str(b) for b in budgets])
ax.set_xlabel("Labeled examples $B$")
ax.set_ylabel("Macro-F1 (%)")
ax.legend(fontsize=7, frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_budget.pdf")
fig.savefig(f"{OUT}/fig1_budget.png", dpi=200)

# ---------------------------------------------------------------- random samples for Analysis
rs = np.random.default_rng(0)
with open(f"{OUT}/samples.txt", "w") as f:
    for _, r in best.iterrows():
        tag = df[(df["method"] == r["method"]) & (df["budget"] == r["budget"]) &
                 (df["k"] == r["k"])].sort_values("seed")["tag"].iloc[0]
        rows = [json.loads(l) for l in open(f"results/{tag}.jsonl")]
        f.write(f"===== {tag} : 8 random sentences =====\n")
        for i in rs.choice(len(rows), 8, replace=False):
            x = rows[i]
            mark = "OK " if x["gold"] == x["pred"] else "ERR"
            f.write(f"[{mark}] gold={x['gold']:<12} pred={x['pred']:<12} raw={x['raw']!r}\n      {x['text']}\n")
        f.write("\n")

print(f"done -> {OUT}/")