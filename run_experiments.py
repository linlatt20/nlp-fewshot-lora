import os, subprocess, time

def tag(m, k, B, s):
    return f"test_{m}_k{k}_B{B}_s{s}"

jobs = []
# 1순위: H1/H3 핵심 (B=100, k=3)
for s in range(3):
    jobs += [("random", 3, 100, s), ("dynamic", 3, 100, s)]
# 2순위: LoRA (H2)
for B in [10, 50, 100]:
    for s in range(3):
        jobs.append(("lora", 0, B, s))
# 3순위: 작은 예산에서의 프롬프팅 (H2 곡선)
for B in [10, 50]:
    for s in range(3):
        jobs += [("random", 3, B, s), ("dynamic", 3, B, s)]
# 4순위: shot 수 변화 (k=1, 5)
for k in [1, 5]:
    for s in range(3):
        jobs += [("random", k, 100, s), ("dynamic", k, 100, s)]

todo = [j for j in jobs if not os.path.exists(f"results/{tag(*j)}.jsonl")]
print(f"{len(todo)} runs to do")
for i, (m, k, B, s) in enumerate(todo, 1):
    t0 = time.time()
    if m == "lora":
        cmd = ["python", "lora_train.py", "--budget", str(B), "--seed", str(s)]
    else:
        cmd = ["python", "prompt_eval.py", "--method", m, "--k", str(k), "--budget", str(B), "--seed", str(s)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"[{i}/{len(todo)}] {tag(m,k,B,s)}  {status}  {time.time()-t0:.0f}s", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])