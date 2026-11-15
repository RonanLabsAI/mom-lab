"""Evaluation queue: trained models on the Stations -> Spark eval farm (protocol v2).

For every training run marked finished with exit 0 in logs/queue/<trainq>.state.json (or listed in --extra),
on a free Spark: stream the model dir from its Station to the Spark (tar over ssh, epoch* dirs excluded),
serve it with vLLM batch-invariant mode, run all six domains on dev (greedy, 4096 max tokens), aggregate,
stop the server. One worker thread per Spark. Results: results/<step>/<run>/scores.json.
Events: logs/queue/eval-<step>.events.log. Done markers: logs/queue/eval-<step>.done (one run name per line).
usage: python3 scripts/evalq.py <trainq> <step> [--extra name@host:path ...]
An extra may end in +lora=<container path of a LoRA adapter>: the path is served as the base with that adapter
loaded under the run name (Station slots only). AWQ checkpoints are served in float16 (the AWQ kernels reject bf16).
"""
import json, os, subprocess, sys, threading, time
TRAINQ, STEP = sys.argv[1], sys.argv[2]
EXTRA = [a for a in sys.argv[3:] if "@" in a]
SPARKS = [("contact@example.invalid", 8651), ("contact@example.invalid", 8652), ("contact@example.invalid", 8653)]
STATION_SLOTS = [s for s in os.environ.get("EVAL_STATION_SLOTS", "").split(",") if s]  # e.g. station-a:0
STATION_IP = {"station-a": "host-3cc546d0.invalid", "station-b": "host-b693d1b8.invalid"}
IMG = "vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b"
DONE = f"logs/queue/eval-{STEP}.done"; EVF = f"logs/queue/eval-{STEP}.events.log"
lock = threading.Lock(); claimed = set(); fails = {}
DOMS = os.environ.get("EVAL_DOMAINS", "math,code,sql,tools,translate,general,mixed")  # Station slots only
ONLY = os.environ.get("EVAL_ONLY")  # optional regex: evaluate only matching run names (extras always included)
PORT0 = int(os.environ.get("EVAL_PORT_BASE", "8671"))
SPLIT = os.environ.get("EVAL_SPLIT", "dev")  # "test" only for the single frozen Part B test pass
SPLIT_ARGS = f"--split {SPLIT}" + (" --allow-test" if SPLIT == "test" else "")


def log(msg):
    with lock, open(EVF, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def run(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def done_set():
    return set(open(DONE).read().split()) if os.path.exists(DONE) else set()


def candidates():
    st = json.load(open(f"logs/queue/{TRAINQ}.state.json"))
    jobs = {}
    for name, info in st.get("finished", {}).items():
        if info.get("exit") == 0 and (not ONLY or __import__("re").search(ONLY, name)):
            host = info["slot"].split(":")[0]
            jobs[name] = (host, f"mom-lab/runs/{name}")
    for e in EXTRA:
        name, rest = e.split("@", 1); host, path = rest.split(":", 1); jobs[name] = (host, path)
    return jobs


def evaluate(spark, port, name, host, path):
    t0 = time.time()
    src = os.path.dirname(path) if path.startswith("/") else "~/" + os.path.dirname(path)
    r = run(f"ssh {host} 'tar c -C {src} --exclude=epoch* {os.path.basename(path)}' | "
            f"ssh {spark} 'mkdir -p ~/models/runs && rm -rf ~/models/runs/{name} && tar x -C ~/models/runs && "
            f"[ \"{os.path.basename(path)}\" = \"{name}\" ] || mv ~/models/runs/{os.path.basename(path)} ~/models/runs/{name}'",
            timeout=3600)
    if r.returncode:
        log(f"FAIL copy {name} {r.stderr[-300:]}"); return False
    run(f"ssh {spark} 'docker rm -f mom-eval-{port} >/dev/null 2>&1; docker run -d --name mom-eval-{port} "
        f"-e VLLM_BATCH_INVARIANT=1 --gpus all --ipc host -p {port}:8000 -v $HOME/models:/models --entrypoint vllm {IMG} "
        f"serve /models/runs/{name} --served-model-name {name} --dtype bfloat16 --max-model-len 8192 "
        f"--gpu-memory-utilization 0.75 --seed 0'")
    ip = spark.split("@")[1]
    for _ in range(120):
        if run(f"curl -s -m 3 http://{ip}:{port}/v1/models").stdout.startswith("{"):
            break
        if "running" not in run(f"ssh {spark} 'docker inspect -f {{{{.State.Status}}}} mom-eval-{port}'").stdout:
            log(f"FAIL serve {name} (container exited)"); return False
        time.sleep(10)
    else:
        log(f"FAIL serve {name}"); return False
    env = "HARNESS_SANDBOX=docker-batch"
    g = run(f"{env} .venv/bin/python -m harness.generate --endpoint http://{ip}:{port}/v1 --model {name} --run-id {name} "
            f"--step {STEP} {SPLIT_ARGS} --concurrency 128 --max-tokens-override 4096 --hardware 'DGX Spark batch-invariant' "
            f"> logs/{STEP}/{name}.gen.log 2>&1", timeout=6 * 3600)
    a = run(f"{env} .venv/bin/python -m harness.aggregate --run-id {name} --step {STEP} {SPLIT_ARGS} "
            f"> logs/{STEP}/{name}.agg.log 2>&1", timeout=3600)
    run(f"ssh {spark} 'docker rm -f mom-eval-{port} >/dev/null 2>&1'")
    ok = g.returncode == 0 and a.returncode == 0
    log(f"{'DONE' if ok else 'FAIL'} {name} on {ip} in {int(time.time() - t0)} s")
    return ok


def evaluate_station(slot, port, name, host, path):
    """Serve on a Station GPU (mom-lab image) — model copied into ~/mom-lab/evalcache unless already local."""
    t0 = time.time(); shost, gpu = slot.split(":")
    path, _, lora = path.partition("+lora=")
    if host == shost and not path.startswith("/"):
        mpath = f"/lab/{path.split('mom-lab/', 1)[1]}"
    elif host == shost:
        mpath = path
    else:
        src = os.path.dirname(path) if path.startswith("/") else "~/" + os.path.dirname(path)
        r = run(f"ssh {host} 'tar c -C {src} --exclude=epoch* {os.path.basename(path)}' | "
                f"ssh {shost} 'mkdir -p ~/mom-lab/evalcache && rm -rf ~/mom-lab/evalcache/{name} && tar x -C ~/mom-lab/evalcache && "
                f"[ \"{os.path.basename(path)}\" = \"{name}\" ] || mv ~/mom-lab/evalcache/{os.path.basename(path)} ~/mom-lab/evalcache/{name}'", timeout=3600)
        if r.returncode:
            log(f"FAIL copy {name} {r.stderr[-300:]}"); return False
        mpath = f"/lab/evalcache/{name}"
        # a root-owned 0600 weight file is skipped by tar with only a warning: fail loudly instead
        if run(f"ssh {shost} 'ls ~/mom-lab/evalcache/{name}/*.safetensors'").returncode:
            log(f"FAIL copy {name} (no weights after copy)"); return False
    # ensure <|im_end|> (151645) is a stop token (Base-init checkpoints only stop on <|endoftext|>)
    gpath = mpath.replace("/lab/", "~/mom-lab/", 1) if mpath.startswith("/lab/") else mpath
    run(f"ssh {shost} \"python3 -c 'import json,os;p=os.path.expanduser(\\\"{gpath}/generation_config.json\\\");d=json.load(open(p));e=d.get(\\\"eos_token_id\\\");e=e if isinstance(e,list) else [e];d[\\\"eos_token_id\\\"]=e if 151645 in e else [151645]+e;json.dump(d,open(p,\\\"w\\\"))'\"")
    run(f"ssh {shost} 'docker rm -f mom-eval-{port} >/dev/null 2>&1; docker run -d --name mom-eval-{port} -e VLLM_BATCH_INVARIANT=1 "
        f"--gpus device={gpu} --shm-size 16g -p {port}:8000 -v /models/hf:/models/hf -v $HOME/mom-lab:/lab --entrypoint vllm mom-lab:v1 "
        f"serve {mpath} --served-model-name {name + '-base' if lora else name} --dtype {'float16' if 'AWQ' in path.upper() else 'bfloat16'} "
        f"--max-model-len 8192 --gpu-memory-utilization 0.85 --seed 0"
        f"{f' --enable-lora --max-lora-rank 16 --lora-modules {name}={lora}' if lora else ''}'")
    ip = STATION_IP[shost]
    for _ in range(120):
        if run(f"curl -s -m 3 http://{ip}:{port}/v1/models").stdout.startswith("{"):
            break
        if "running" not in run(f"ssh {shost} 'docker inspect -f {{{{.State.Status}}}} mom-eval-{port}'").stdout:
            log(f"FAIL serve {name} (container exited) on {slot}"); return False
        time.sleep(10)
    else:
        log(f"FAIL serve {name} on {slot}"); return False
    env = "HARNESS_SANDBOX=docker-batch"
    g = run(f"{env} .venv/bin/python -m harness.generate --endpoint http://{ip}:{port}/v1 --model {name} --run-id {name} "
            f"--step {STEP} {SPLIT_ARGS} --domains {DOMS} --concurrency 128 --max-tokens-override 4096 --hardware 'RTX PRO 6000 batch-invariant' "
            f"{'--domain-tag ' if 'tagged' in name else ''}"
            f"> logs/{STEP}/{name}.gen.log 2>&1", timeout=6 * 3600)
    a = run(f"{env} .venv/bin/python -m harness.aggregate --run-id {name} --step {STEP} {SPLIT_ARGS} > logs/{STEP}/{name}.agg.log 2>&1", timeout=3600)
    run(f"ssh {shost} 'docker rm -f mom-eval-{port} >/dev/null 2>&1'")
    ok = g.returncode == 0 and a.returncode == 0
    log(f"{'DONE' if ok else 'FAIL'} {name} on {slot} in {int(time.time() - t0)} s")
    return ok


def worker(spark, port, station_slot=None):
    while True:
        with lock:
            pending = [(n, h, p) for n, (h, p) in sorted(candidates().items(), key=lambda kv: (0 if kv[0].startswith('pt17') else 1, 0 if 'pooled' in kv[0] else 1, kv[0])) if n not in done_set() and n not in claimed]
            job = pending[0] if pending else None
            if job:
                claimed.add(job[0])
        if not job:
            time.sleep(60); continue
        if station_slot:
            sh_, g_ = station_slot.split(":")
            if int(run(f"ssh {sh_} 'nvidia-smi -i {g_} --query-gpu=memory.used --format=csv,noheader,nounits'").stdout.strip() or 10**9) > 2000:
                with lock:
                    claimed.discard(job[0])
                time.sleep(60); continue
        ok = evaluate_station(station_slot, port, *job) if station_slot else evaluate(spark, port, *job)
        if ok:
            with lock, open(DONE, "a") as f:
                f.write(job[0] + "\n")
        else:  # release for retry (a failed claim used to stick until restart); give up after 3 attempts
            with lock:
                fails[job[0]] = fails.get(job[0], 0) + 1
                if fails[job[0]] < 3:
                    claimed.discard(job[0])
            time.sleep(60)


os.makedirs(f"logs/{STEP}", exist_ok=True)
for s, p in (SPARKS if os.environ.get("EVAL_SPARKS", "1") == "1" else []):
    threading.Thread(target=worker, args=(s, p), daemon=True).start()
for i, slot in enumerate(STATION_SLOTS):
    threading.Thread(target=worker, args=(None, PORT0 + i, slot), daemon=True).start()
while True:
    time.sleep(3600)
