"""Part D evaluation queue (Station 6000s only). Derived from scripts/evalq.py (frozen; serving and generation
commands are identical: mom-lab:v1, VLLM_BATCH_INVARIANT=1, greedy via the harness, 4,096 max new tokens).

Differences from evalq.py:
- Jobs come from a file re-read every loop (so jobs can be added without restarting), plus auto-jobs for finished
  training runs of a trainq queue.
- Each job carries its own step (results/<step>/<run>), data root, domains, harness package and priority, so one
  queue can run every Part D phase in order.

jobs file lines:   name@host:path[+lora=<container path>] key=val ...
autojobs lines:    <regex on finished trainq run names> key=val ...      (path = mom-lab/runs/<name> on its host)
keys: step (required), domains (default core+mixed+para), root (data root; default data/), harness (harness|harness_d),
      prio (int, lower first; default 5), lora_rank (max LoRA rank to allow; default 16)
Done markers: logs/queue/eval-<step>.done (same as evalq.py, so phases already done by evalq.py are not redone).
usage: python3 -P scripts/evalq_p.py <trainq> <jobsfile> <autojobsfile>     (env EVAL_STATION_SLOTS=station-a:0,...)
"""
import json, os, re, subprocess, sys, threading, time
TRAINQ, JOBF, AUTOF = sys.argv[1], sys.argv[2], sys.argv[3]
STATION_SLOTS = [s for s in os.environ.get("EVAL_STATION_SLOTS", "").split(",") if s]
STATION_IP = {"station-a": "host-3cc546d0.invalid", "station-b": "host-b693d1b8.invalid"}
EVF = "logs/queue/eval-partd-p.events.log"
DEF_DOMS = "math,code,sql,tools,translate,general,mixed,para"
lock = threading.Lock(); claimed = set(); fails = {}


def log(msg):
    with lock, open(EVF, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def run(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def done_set(step):
    p = f"logs/queue/eval-{step}.done"
    return set(open(p).read().split()) if os.path.exists(p) else set()


def parse_opts(toks):
    o = dict(t.split("=", 1) for t in toks)
    o.setdefault("domains", DEF_DOMS); o.setdefault("harness", "harness"); o["prio"] = int(o.get("prio", 5))
    o["lora_rank"] = int(o.get("lora_rank", 16)); o.setdefault("root", "")
    return o


def candidates():
    jobs = {}
    if os.path.exists(JOBF):
        for l in open(JOBF):
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            t = l.split(); name, rest = t[0].split("@", 1); host, path = rest.split(":", 1); o = parse_opts(t[1:])
            jobs[(o["step"], name)] = dict(name=name, host=host, path=path, **o)
    st = json.load(open(f"logs/queue/{TRAINQ}.state.json")) if os.path.exists(f"logs/queue/{TRAINQ}.state.json") else {}
    autos = [(re.compile(l.split()[0]), parse_opts(l.split()[1:])) for l in open(AUTOF)
             if l.strip() and not l.startswith("#")] if os.path.exists(AUTOF) else []
    for name, info in st.get("finished", {}).items():
        if info.get("exit") != 0:
            continue
        for rx, o in autos:
            if rx.search(name):
                jobs.setdefault((o["step"], name), dict(name=name, host=info["slot"].split(":")[0], path=f"mom-lab/runs/{name}", **o))
    return jobs


def evaluate_station(slot, port, j):
    t0 = time.time(); shost, gpu = slot.split(":"); name, host, step = j["name"], j["host"], j["step"]
    path, _, lora = j["path"].partition("+lora=")
    if host == shost and not path.startswith("/"):
        mpath = f"/lab/{path.split('mom-lab/', 1)[1]}"
    elif host == shost or path.startswith("/models/hf/"):
        mpath = path  # /models/hf is mirrored on both Stations
    else:
        src = os.path.dirname(path) if path.startswith("/") else "~/" + os.path.dirname(path)
        r = run(f"ssh {host} 'tar c -C {src} --exclude=epoch* {os.path.basename(path)}' | "
                f"ssh {shost} 'mkdir -p ~/mom-lab/evalcache && rm -rf ~/mom-lab/evalcache/{name} && tar x -C ~/mom-lab/evalcache && "
                f"[ \"{os.path.basename(path)}\" = \"{name}\" ] || mv ~/mom-lab/evalcache/{os.path.basename(path)} ~/mom-lab/evalcache/{name}'", timeout=3600)
        if r.returncode:
            log(f"FAIL copy {step}/{name} {r.stderr[-300:]}"); return False
        mpath = f"/lab/evalcache/{name}"
        if run(f"ssh {shost} 'ls ~/mom-lab/evalcache/{name}/*.safetensors'").returncode:
            log(f"FAIL copy {step}/{name} (no weights after copy)"); return False
    if lora and host != shost and run(f"ssh {shost} 'ls {lora.replace('/lab/', '~/mom-lab/', 1)}/adapter_model.safetensors'").returncode:
        log(f"FAIL {step}/{name}: adapter {lora} missing on {shost}"); return False
    lora_flags = f" --enable-lora --max-lora-rank {j['lora_rank']} --lora-modules {name}={lora}" if lora else ""
    gpath = mpath.replace("/lab/", "~/mom-lab/", 1) if mpath.startswith("/lab/") else mpath
    if not gpath.startswith("/models/hf/"):  # never touch the shared base-model dirs
        run(f"ssh {shost} \"python3 -c 'import json,os;p=os.path.expanduser(\\\"{gpath}/generation_config.json\\\");d=json.load(open(p));e=d.get(\\\"eos_token_id\\\");e=e if isinstance(e,list) else [e];d[\\\"eos_token_id\\\"]=e if 151645 in e else [151645]+e;json.dump(d,open(p,\\\"w\\\"))'\"")
    run(f"ssh {shost} 'docker rm -f mom-eval-{port} >/dev/null 2>&1; docker run -d --name mom-eval-{port} -e VLLM_BATCH_INVARIANT=1 "
        f"--gpus device={gpu} --shm-size 16g -p {port}:8000 -v /models/hf:/models/hf -v $HOME/mom-lab:/lab --entrypoint vllm mom-lab:v1 "
        f"serve {mpath} --served-model-name {name + '-base' if lora else name} --dtype {'float16' if 'AWQ' in path.upper() else 'bfloat16'} "
        f"--max-model-len 8192 --gpu-memory-utilization 0.85 --seed 0"
        f"{lora_flags}'")
    ip = STATION_IP[shost]
    for _ in range(120):
        if run(f"curl -s -m 3 http://{ip}:{port}/v1/models").stdout.startswith("{"):
            break
        if "running" not in run(f"ssh {shost} 'docker inspect -f {{{{.State.Status}}}} mom-eval-{port}'").stdout:
            log(f"FAIL serve {step}/{name} (container exited) on {slot}"); return False
        time.sleep(10)
    else:
        log(f"FAIL serve {step}/{name} on {slot}"); return False
    sargs = "--split test --allow-test" + (f" --data-root {j['root']}" if j["root"] else "")
    env = "HARNESS_SANDBOX=docker-batch"
    os.makedirs(f"logs/{step}", exist_ok=True)
    g = run(f"{env} .venv/bin/python -m {j['harness']}.generate --endpoint http://{ip}:{port}/v1 --model {name} --run-id {name} "
            f"--step {step} {sargs} --domains {j['domains']} --concurrency 128 --max-tokens-override 4096 --hardware 'RTX PRO 6000 batch-invariant' "
            f"> logs/{step}/{name}.gen.log 2>&1", timeout=6 * 3600)
    a = run(f"{env} .venv/bin/python -m {j['harness']}.aggregate --run-id {name} --step {step} {sargs} > logs/{step}/{name}.agg.log 2>&1", timeout=3600)
    run(f"ssh {shost} 'docker rm -f mom-eval-{port} >/dev/null 2>&1'")
    ok = g.returncode == 0 and a.returncode == 0
    log(f"{'DONE' if ok else 'FAIL'} {step}/{name} on {slot} in {int(time.time() - t0)} s")
    return ok


def worker(port, slot):
    while True:
        with lock:
            jobs = candidates()
            pending = sorted((k for k in jobs if k[1] not in done_set(k[0]) and k not in claimed),
                             key=lambda k: (jobs[k]["prio"], k[0], k[1]))
            key = pending[0] if pending else None
            if key:
                claimed.add(key)
        if not key:
            time.sleep(60); continue
        sh_, g_ = slot.split(":")
        if int(run(f"ssh {sh_} 'nvidia-smi -i {g_} --query-gpu=memory.used --format=csv,noheader,nounits'").stdout.strip() or 10**9) > 2000:
            with lock:
                claimed.discard(key)
            time.sleep(60); continue
        ok = evaluate_station(slot, port, jobs[key])
        if ok:
            with lock, open(f"logs/queue/eval-{key[0]}.done", "a") as f:
                f.write(key[1] + "\n")
        else:
            with lock:
                fails[key] = fails.get(key, 0) + 1
                if fails[key] < 3:
                    claimed.discard(key)
            time.sleep(60)


for i, slot in enumerate(STATION_SLOTS):
    threading.Thread(target=worker, args=(8681 + i, slot), daemon=True).start()
while True:
    time.sleep(3600)
