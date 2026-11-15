"""Part B memory report: serving memory per system = weights + adapters + router + KV cache (bf16) at a fixed
batch x context. Weights/adapters = on-disk safetensors bytes (measured on station-b); router = the
multilingual-e5-small checkpoint (fp32, as served on CPU or GPU); KV = batch * ctx * 2 * layers * kv_heads * head_dim
* 2 bytes. Primary operating point: batch 1 x 8,192 tokens (max-model-len); secondary: batch 16 x 4,096.
Cross-check: vLLM "Model loading took X GiB" lines from servers started this session.
usage: .venv/bin/python -P scripts/memory_report.py   -> results/analysis/memory_report.json
"""
import json, os, subprocess

HOST = "station-b"
MODELS = {"1.7B": "Qwen__Qwen3-1.7B", "4B": "Qwen__Qwen3-4B", "8B": "Qwen__Qwen3-8B",
          "4B-AWQ": "Qwen__Qwen3-4B-AWQ", "8B-AWQ": "Qwen__Qwen3-8B-AWQ", "4B-FP8": "Qwen__Qwen3-4B-FP8", "8B-FP8": "Qwen__Qwen3-8B-FP8"}
ADAPTERS = {"1.7B": ["pt17-lora16-code-lr2e-4", "pt17-lora16-sql-lr2e-4", "pt17-lora16-tools-lr2e-4"],
            "4B": ["pt4-lora16-sql-lr2e-4", "pt4-lora16-tools-lr2e-4"]}
POINTS = {"b1x8192": (1, 8192), "b16x4096": (16, 4096)}


def sh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd], capture_output=True, text=True, check=True).stdout


def size(path_glob):
    return int(sh(f"du -bc {path_glob} | tail -1").split()[0])


cfg = {k: json.loads(sh(f"cat /models/hf/{m}/config.json")) for k, m in MODELS.items() if "-" not in k}
W = {k: size(f"/models/hf/{m}/*.safetensors") for k, m in MODELS.items()}
AD = {k: sum(size(f"~/mom-lab/adapters/{a}/adapter_model.safetensors") for a in v) for k, v in ADAPTERS.items()}
ROUTER = os.path.getsize(os.path.expanduser("~/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots/"
                                             "614241f622f53c4eeff9890bdc4f31cfecc418b3/model.safetensors"))  # resolves the blob link


def kv(size_key, b, t):
    c = cfg[size_key.split("-")[0]]
    return b * t * 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]) * 2


SYSTEMS = {  # name -> (weights key, adapters key or None, router?)
    "Qwen3-1.7B": ("1.7B", None, False), "Qwen3-1.7B + 3 LoRA (code,sql,tools) + router": ("1.7B", "1.7B", True),
    "Qwen3-1.7B + 3 full-FT replay specialists + router": ("1.7B", "x3", True),
    "Qwen3-4B": ("4B", None, False), "Qwen3-4B + 2 LoRA (sql,tools) + router": ("4B", "4B", True),
    "Qwen3-8B": ("8B", None, False),
    "Qwen3-4B AWQ": ("4B-AWQ", None, False), "Qwen3-4B AWQ + 2 LoRA + router": ("4B-AWQ", "4B", True),
    "Qwen3-8B AWQ": ("8B-AWQ", None, False), "Qwen3-4B FP8": ("4B-FP8", None, False), "Qwen3-8B FP8": ("8B-FP8", None, False),
}
rep = {"bytes": {"weights": W, "adapters": AD, "router": ROUTER}, "systems": {}}
for name, (wk, ak, r) in SYSTEMS.items():
    wt = W[wk] * (4 if ak == "x3" else 1)
    ad = AD.get(ak, 0) if ak and ak != "x3" else 0
    row = {"weights": wt, "adapters": ad, "router": ROUTER if r else 0}
    for p, (b, t) in POINTS.items():
        row[f"kv_{p}"] = kv(wk, b, t)
        row[f"total_{p}_GB"] = round((wt + ad + row["router"] + row[f"kv_{p}"]) / 1e9, 3)
    rep["systems"][name] = row
ref = rep["systems"]["Qwen3-8B"]["total_b1x8192_GB"]; ref_awq = rep["systems"]["Qwen3-8B AWQ"]["total_b1x8192_GB"]
rep["h2_ratio_bf16"] = rep["systems"]["Qwen3-4B + 2 LoRA (sql,tools) + router"]["total_b1x8192_GB"] / ref
rep["h2_ratio_awq"] = rep["systems"]["Qwen3-4B AWQ + 2 LoRA + router"]["total_b1x8192_GB"] / ref_awq
os.makedirs("results/analysis", exist_ok=True)
json.dump(rep, open("results/analysis/memory_report.json", "w"), indent=1)
print(f"{'system':52s} {'b1x8192 GB':>11s} {'b16x4096 GB':>12s}")
for n, r in rep["systems"].items():
    print(f"{n:52s} {r['total_b1x8192_GB']:11.2f} {r['total_b16x4096_GB']:12.2f}")
print(f"H2 memory ratio (4B+LoRA+router / 8B, b1x8192): bf16 {rep['h2_ratio_bf16']:.3f}; AWQ {rep['h2_ratio_awq']:.3f}")
print("router bytes", ROUTER, "; adapters", AD)
