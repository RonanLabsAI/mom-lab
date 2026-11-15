"""Merge six specialists into one model (baseline: does a merged model match routing?).

Methods (task vectors tau_i = theta_i - theta_base):
  avg  : theta_base + mean_i tau_i                         (model soup / task arithmetic with lambda=1/6*6... i.e. mean)
  ties : theta_base + lambda * TIES(tau_1..6, density k)    (trim to top-k% magnitude, elect sign by summed mass, disjoint mean)
usage: python3 scripts/merge_specialists.py <base_dir> <out_dir> <method> <spec_dir>... [--density 0.2] [--lam 1.0]
Runs on CPU (Threadripper). Copies tokenizer/config from the first specialist.
"""
import sys, os, json, shutil, torch
from safetensors.torch import load_file, save_file
args = sys.argv[1:]
dens = float(args[args.index("--density") + 1]) if "--density" in args else 0.2
lam = float(args[args.index("--lam") + 1]) if "--lam" in args else 1.0
pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
base_dir, out_dir, method, specs = pos[0], pos[1], pos[2], pos[3:]


def load(d):
    sd = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(d, f)))
    return sd


base = load(base_dir)
taus = [{k: v.float() - base[k].float() for k, v in load(s).items()} for s in specs]
out = {}
# tied embeddings (Qwen3-1.7B): HF base ships lm_head.weight, our trained checkpoints don't; skip keys a
# specialist lacks only if the config ties them (vLLM rebuilds lm_head from embed_tokens)
tied = json.load(open(os.path.join(specs[0], "config.json"))).get("tie_word_embeddings", False)
for k, b in base.items():
    if any(k not in t for t in taus):
        if tied and k == "lm_head.weight":
            continue
        raise SystemExit(f"key {k} missing from a specialist")
    T = torch.stack([t[k] for t in taus])  # [n, ...]
    if method == "avg":
        m = T.mean(0)
    elif method == "ties":
        flat = T.reshape(T.shape[0], -1).abs()
        kth = max(1, int(flat.shape[1] * dens))
        thr = flat.kthvalue(flat.shape[1] - kth + 1, dim=1).values.view(-1, *([1] * (T.dim() - 1)))
        Tt = torch.where(T.abs() >= thr, T, torch.zeros_like(T))
        sign = torch.sign(Tt.sum(0))
        agree = (torch.sign(Tt) == sign) & (Tt != 0)
        m = (Tt * agree).sum(0) / agree.sum(0).clamp(min=1)
        m = lam * m
    else:
        raise SystemExit("method must be avg or ties")
    out[k] = (b.float() + m).to(torch.bfloat16)
os.makedirs(out_dir, exist_ok=True)
save_file(out, os.path.join(out_dir, "model.safetensors"))
for f in os.listdir(specs[0]):
    if f.endswith((".json", ".jinja")) and f not in ("train_args.json",):
        shutil.copy(os.path.join(specs[0], f), out_dir)
json.dump({"method": method, "density": dens, "lambda": lam, "base": base_dir, "specs": specs},
          open(os.path.join(out_dir, "merge_args.json"), "w"), indent=1)
print("MERGE_DONE", out_dir)
