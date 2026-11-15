"""METAR schema gate launcher (preregs/2026-11-14-metar-schema-gate.md). Run on the Threadripper from the worktree.

  python3 -P scripts/gate/run_gate.py check    # freeze hashes, 6000 free, routes present, claim held (no GPU use)
  python3 -P scripts/gate/run_gate.py smoke    # dev, first 30 items, every schema job; then scripts/gate/smoke_check.py
  python3 -P scripts/gate/run_gate.py main     # test split, configs/gate/gate_jobs.txt in order (systemd user unit)
  python3 -P scripts/gate/run_gate.py status | stop

Serving: one vLLM container `mom-gate-8691` on Station LEFT RTX PRO 6000, pinned by UUID (deviation 1), batch-invariant,
xgrammar backend.
Requests go only through LiteLLM route names at 127.0.0.1:4000 (routes in docs/gate/litellm-routes.yaml). The only
direct contact with the Station is ssh for docker and a localhost health probe run on the Station itself.
Parity: after gate-qwen3-8b-zs-free, outputs must equal results/partdC/base-qwen3-8b (main checkout) on every item,
otherwise the queue stops (prereg "Parity control").
The fleet claim is taken and released by hand with ~/bin/fleet-claims (prereg "Claim").
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get("MOM_MAIN", str(Path(__file__).resolve().parents[2])))
PY = str(MAIN / ".venv/bin/python")  # the harness's own frozen venv (httpx etc.)
HOST, PORT, NAME = "station-a", 8691, "mom-gate-8691"
# Deviation 1 (content/PART_G_DEVIATIONS.md): the GPU is chosen by UUID, verified by name, never by index.
GPU_UUID = "GPU-00000000-0000-0000-0000-2cd434a6b8f5"   # NVIDIA RTX PRO 6000 Blackwell Max-Q (research-only)
GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
GB300_UUID = "GPU-00000000-0000-0000-0000-42943b2791b3"  # production glmf-daily: never touched
IMAGE = "vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b"
ROUTER = "http://127.0.0.1:4001/v1"  # research-only router (deviation 1); production :4000 is never used
KEYFILE = os.path.expanduser("~/.config/litellm/key")
JOBS = WT / "configs/gate/gate_jobs.txt"
SCHEMA = WT / "configs/gate/metar_schema.json"
FREEZE = WT / "artifacts/GATE_FREEZE.sha256"
EV = WT / "logs/queue/gate.events.log"
CLAIM_PREFIX = "mom-gate-left6000"
WS_FLAG = WT / "configs/gate/disable_any_whitespace"  # exists only after the smoke rule fires (deviation)
FIX1 = WT / "artifacts/GATE_FREEZE_fix1.sha256"
SMOKE_OK = WT / "logs/queue/gate.smoke.ok"
PROMPT_HASHES = WT / "configs/gate/schema_prompts.sha256"
ROUTER_CFG = Path("/srv/release-data/services/litellm-research/cfg/config.yaml")
PARITY_REF = MAIN / "results/partdC/base-qwen3-8b/gen/aviation/metar.test.jsonl"
SSH = ["ssh", "-o", "BatchMode=yes", HOST]


def log(msg):
    EV.parent.mkdir(parents=True, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S %Z") + " " + msg
    print(line, flush=True)
    with EV.open("a") as f:
        f.write(line + "\n")


def sh(cmd, check=True, **kw):
    r = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if check and r.returncode:
        raise RuntimeError(f"{cmd!r} -> {r.returncode}: {r.stderr[-500:]}")
    return r


def read_jobs():
    jobs = []
    for line in JOBS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            jobs.append(dict(kv.split("=", 1) for kv in line.split()))
    return jobs


def router_models():
    req = urllib.request.Request(ROUTER + "/models",
                                 headers={"Authorization": "Bearer " + Path(KEYFILE).read_text().strip()})
    with urllib.request.urlopen(req, timeout=10) as r:
        return {m["id"] for m in json.load(r)["data"]}


def gpu_used_mib():
    out = sh(SSH + [f"nvidia-smi -i {GPU_UUID} --query-gpu=name,uuid,memory.used --format=csv,noheader,nounits"]).stdout
    name, uuid, used = [x.strip() for x in out.strip().split(",")]
    if name != GPU_NAME or uuid != GPU_UUID:
        raise RuntimeError(f"GPU identity mismatch: {name} {uuid}")
    return int(used)


def gb300_procs() -> set:
    out = sh(SSH + ["nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader"]).stdout
    return {l.split(",")[1].strip() for l in out.splitlines() if l.strip().startswith(GB300_UUID)}


def verify_placement(before_gb300: set):
    """After the vLLM starts: it must hold memory on the 6000 and add no process to the GB300; else stop at once."""
    used = gpu_used_mib()
    after = gb300_procs()
    if used < 2000 or after - before_gb300:
        stop_server()
        raise RuntimeError(f"placement check FAILED (6000 used {used} MiB, new GB300 pids {after - before_gb300})")
    log(f"placement OK: 6000 {GPU_UUID} {used} MiB; GB300 process set unchanged")


def claim_held():
    out = sh(["fleet-claims", "active"]).stdout
    data = json.loads(out)
    if isinstance(data, dict):
        data = data.get("claims", data)
    items = list(data.values()) if isinstance(data, dict) else data
    return any(str(c.get("id", "")).startswith(CLAIM_PREFIX) and not c.get("released") for c in items)


def check(need_routes=True) -> bool:
    ok = True
    if FREEZE.exists():
        # files superseded by a logged deviation are checked against its manifest instead (deviation 1)
        devs = sorted((WT / "artifacts").glob("GATE_FREEZE_dev*.sha256"))
        superseded = {l.split(None, 1)[1].strip() for d in devs for l in d.read_text().splitlines() if l.strip()}
        keep = "".join(l + "\n" for l in FREEZE.read_text().splitlines()
                       if l.strip() and l.split(None, 1)[1].strip() not in superseded)
        r = sh(["sha256sum", "-c", "--quiet", "-"], check=False, cwd=WT, input=keep)
        r2 = sh(["sha256sum", "-c", "--quiet"] + [str(d) for d in devs], check=False, cwd=WT) if devs else r
        good = r.returncode == 0 and r2.returncode == 0
        print("freeze hashes", f"OK ({len(superseded)} file(s) under {len(devs)} deviation manifest(s))" if good
              else "FAIL\n" + r.stdout + r.stderr + r2.stdout + r2.stderr)
        ok &= good
    else:
        print("freeze manifest missing (not frozen yet)"); ok = False
    r = sh(["sha256sum", "-c", "--quiet", str(PROMPT_HASHES)], check=False, cwd=WT)
    print("schema-arm prompt files", "OK" if r.returncode == 0 else "FAIL (run scripts/gate/build_schema_prompts.py)")
    ok &= r.returncode == 0
    ok &= route_cfg_ok()
    ok &= xgrammar_ok()
    used = gpu_used_mib()
    print(f"{HOST} {GPU_NAME} {GPU_UUID} memory.used={used} MiB", "free" if used < 2000 else "BUSY")
    ok &= used < 2000
    try:
        have = router_models()
        need = {j["route"] for j in read_jobs()}
        print("routes:", {n: (n in have) for n in sorted(need)})
        ok &= need <= have or not need_routes
    except Exception as e:  # router down or key unreadable
        print("router check failed:", type(e).__name__); ok = False
    held = claim_held()
    print("fleet claim", CLAIM_PREFIX + "*", "held" if held else "NOT held")
    ok &= held
    return ok


def route_cfg_ok() -> bool:
    """The two research routes must exist in the live router config with drop_params: false (prereg Serving)."""
    try:
        cfg = ROUTER_CFG.read_text()
    except OSError:
        print("router config unreadable"); return False
    lines = [l for l in cfg.splitlines() if "research-mom-gate" in l or "hosted_vllm/qwen3-" in l]
    names = {j["route"] for j in read_jobs()}
    have = all(any(f"model_name: {n}" in l for l in lines) for n in names)
    nodrop = sum("drop_params: false" in l for l in lines) >= len(names)
    print("router config: routes", "present" if have else "ABSENT", "| drop_params false", "yes" if nodrop else "NO")
    return have and nodrop


def xgrammar_ok() -> bool:
    """All 1,300 METAR golds must be in the grammar xgrammar 0.2.7 compiles from schema v1 (CPU, in the image)."""
    golds = "".join(json.dumps(json.loads(l)["reference"]) + "\n" for sp in ("dev", "test")
                    for l in (WT / f"data_partd_c/eval/aviation/metar/{sp}.jsonl").open() if l.strip())
    tmp = WT / "logs/queue/gate.golds.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(golds)
    d = "/tmp/mom_gate_xg"
    sh(SSH + [f"mkdir -p {d}"])
    sh(["scp", "-q", "-o", "BatchMode=yes", str(tmp), str(SCHEMA), str(WT / "scripts/gate/xgrammar_gold_check.py"),
        f"{HOST}:{d}/"])
    flag = " --no-whitespace" if WS_FLAG.exists() else ""
    r = sh(SSH + [f"docker run --rm -v {d}/metar_schema.json:/schema.json:ro -v {d}/gate.golds.jsonl:/golds.jsonl:ro "
                  f"-v {d}/xgrammar_gold_check.py:/c.py:ro --entrypoint python3 {IMAGE} /c.py{flag}; "
                  f"rc=$?; rm -rf {d}; exit $rc"], check=False)
    print("xgrammar gold check:", (r.stdout.strip().splitlines() or ["no output"])[-1])
    return r.returncode == 0


def data_root(j) -> Path:
    return WT / j["root"] if j["root"].startswith("data_gate") else MAIN / j["root"]


def server_cmd(j) -> str:
    so = {"backend": "xgrammar", "enable_in_reasoning": False}
    if WS_FLAG.exists():
        so["disable_any_whitespace"] = True
    args = [
        "docker", "run", "-d", "--name", NAME, "-e", "VLLM_BATCH_INVARIANT=1", "--gpus", f"device={GPU_UUID}",
        "-e", f"CUDA_VISIBLE_DEVICES={GPU_UUID}",
        "--shm-size", "16g", "-p", f"{PORT}:8000", "-v", "/models/hf:/models/hf:ro", "--entrypoint", "vllm", IMAGE,
        "serve", j["model"], "--served-model-name", j["served"], "--dtype", "bfloat16",
        "--max-model-len", j["mml"], "--gpu-memory-utilization", "0.85", "--seed", "0",
        "--structured-outputs-config", json.dumps(so, separators=(",", ":")),
    ]
    if j.get("rparser"):
        args += ["--reasoning-parser", j["rparser"]]
    return " ".join(shlex.quote(a) for a in args)


def server_key(j):
    return (j["model"], j["served"], j["mml"], j.get("rparser", ""), WS_FLAG.exists())


def stop_server():
    sh(SSH + [f"docker rm -f {NAME} >/dev/null 2>&1 || true"], check=False)


def start_server(j, timeout_s=1800):
    stop_server()
    before = gb300_procs()
    log(f"serve {j['served']} mml={j['mml']} rparser={j.get('rparser', '')} ws_off={WS_FLAG.exists()}")
    sh(SSH + [server_cmd(j)])
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = sh(SSH + [f"docker inspect -f '{{{{.State.Status}}}}' {NAME}"], check=False).stdout.strip()
        if st != "running":
            tail = sh(SSH + [f"docker logs --tail 30 {NAME} 2>&1"], check=False).stdout
            raise RuntimeError(f"server exited ({st}): {tail[-1500:]}")
        if sh(SSH + [f"curl -s -o /dev/null -w '%{{http_code}}' -m 3 localhost:{PORT}/health"],
              check=False).stdout.strip() == "200":
            log(f"server ready after {time.time() - t0:.0f} s")
            verify_placement(before)
            return
        time.sleep(10)
    raise RuntimeError("server not ready in time")


def gen_args(j, split, step, limit=None):
    a = [PY, "-P", "scripts/gate/schema_generate.py", "--endpoint", ROUTER, "--model", j["route"],
         "--api-key-file", KEYFILE, "--run-id", j["run"], "--step", step, "--split", split,
         "--data-root", str(data_root(j)), "--results-root", str(WT / "results"),
         "--domains", "aviation", "--benchmarks", "metar", "--concurrency", "128",
         "--max-tokens-override", j["maxtok"], "--timeout", "3600",
         "--hardware", "RTX PRO 6000 batch-invariant (Station LEFT)",
         "--revision", "vllm-0.30.1rc1.dev245+gaf7f9488c xgrammar-0.2.7"]
    if split == "test":
        a.append("--allow-test")
    if limit:
        a += ["--limit", str(limit)]
    if j["schema"] == "1":
        a += ["--schema", str(SCHEMA)]
    if j["think"] == "1":
        a += ["--thinking", "--temperature", j["temp"], "--top-p", j["top_p"], "--top-k", j["top_k"],
              "--seed", j["gseed"]]
    return a


def run_job(j, split, step, limit=None):
    log(f"gen {step}/{j['run']} split={split} limit={limit}")
    r = sh(gen_args(j, split, step, limit), check=False, cwd=WT)
    if r.returncode:
        log(f"FAIL gen {j['run']}: {r.stderr[-800:]}")
        return False
    if split == "test":
        r = sh([PY, "-P", "-m", "harness_d.aggregate", "--run-id", j["run"], "--step", step, "--split", "test",
                "--allow-test", "--benchmarks", "metar", "--data-root", str(data_root(j)),
                "--results-root", str(WT / "results")], check=False, cwd=WT,
               env=dict(os.environ, PYTHONPATH=str(WT)))  # -P drops cwd from sys.path (deviation 3)
        if r.returncode:
            log(f"FAIL aggregate {j['run']}: {r.stderr[-800:]}")
            return False
    log(f"DONE {step}/{j['run']}")
    return True


def parity() -> dict:
    new = WT / "results/partdG/gate-qwen3-8b-zs-free/gen/aviation/metar.test.jsonl"
    a = {json.loads(l)["id"]: json.loads(l)["output"] for l in PARITY_REF.open() if l.strip()}
    b = {json.loads(l)["id"]: json.loads(l)["output"] for l in new.open() if l.strip()}
    same = sum(1 for k in a if b.get(k) == a[k])
    res = {"reference": str(PARITY_REF), "n_ref": len(a), "n_new": len(b), "identical": same,
           "ok": same == len(a) == len(b) == 1000}
    (WT / "results/partdG/parity.json").write_text(json.dumps(res, indent=2))
    return res


def queue(split, step, limit, schema_only):
    current = None
    try:
        for j in read_jobs():
            if schema_only and j["schema"] != "1":
                continue
            if server_key(j) != current:
                start_server(j)
                current = server_key(j)
            if not run_job(j, split, step, limit):
                log("queue stopped after a failure")
                return 1
            if split == "test" and j["run"] == "gate-qwen3-8b-zs-free":
                p = parity()
                log(f"parity identical {p['identical']}/{p['n_ref']} ok={p['ok']}")
                if not p["ok"]:
                    log("PARITY FAILED: queue stopped before any schema arm (prereg stop rule)")
                    return 2
        return 0
    finally:
        stop_server()
        log("server removed")


def launch(unit, sub):
    if sh(["systemctl", "--user", "is-active", "--quiet", unit], check=False).returncode == 0:
        sys.exit(f"{unit} already running")
    out = WT / f"logs/queue/{unit}.out"
    out.parent.mkdir(parents=True, exist_ok=True)
    sh(["systemd-run", "--user", f"--unit={unit}", "--collect", f"--working-directory={WT}",
        "-p", f"StandardOutput=append:{out}", "-p", f"StandardError=append:{out}",
        "python3", "-P", str(WT / "scripts/gate/run_gate.py"), sub])
    print(f"started {unit}; events: {EV}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        sys.exit(0 if check() else 1)
    if cmd in ("smoke", "main"):
        if not check():
            sys.exit("check failed: not launching")
        if WS_FLAG.exists() and not FIX1.exists():
            sys.exit("disable_any_whitespace is set but artifacts/GATE_FREEZE_fix1.sha256 is missing (deviation first)")
        if cmd == "main" and not SMOKE_OK.exists():
            sys.exit("smoke has not passed (logs/queue/gate.smoke.ok missing)")
        launch(f"mom-gate-{cmd}", f"_{cmd}")
    elif cmd == "_smoke":
        SMOKE_OK.unlink(missing_ok=True)
        rc = queue("dev", "partdG-smoke", 30, schema_only=True)
        if rc:
            sys.exit(rc)
        r = sh([str(WT / ".venv-gate/bin/python"), "-P", "scripts/gate/smoke_check.py"], check=False, cwd=WT)
        log("smoke_check exit " + str(r.returncode) + "\n" + r.stdout)
        if r.returncode == 0:
            SMOKE_OK.write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n" + r.stdout)
        elif r.returncode == 3 and not WS_FLAG.exists():
            WS_FLAG.write_text("smoke rule fired; deviation entry and artifacts/GATE_FREEZE_fix1.sha256 required\n")
            log("SMOKE RULE FIRED: disable_any_whitespace set; write the deviation + fix1 manifest, then rerun smoke")
        sys.exit(r.returncode)
    elif cmd == "_main":
        sys.exit(queue("test", "partdG", None, schema_only=False))
    elif cmd == "status":
        for u in ("mom-gate-smoke", "mom-gate-main"):
            print(u, sh(["systemctl", "--user", "is-active", u], check=False).stdout.strip())
        if EV.exists():
            print("".join(EV.read_text().splitlines(True)[-15:]))
    elif cmd == "stop":
        for u in ("mom-gate-smoke", "mom-gate-main"):
            sh(["systemctl", "--user", "stop", u], check=False)
        stop_server()
        print("stopped; container removed")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
