"""Tiny GPU job queue for the Stations (runs on the Threadripper, polls every 60 s).

jobs file (JSONL): {"name","prefer":["station-b:1",...] or [], "model","data":[...],"args":[...]}
A slot (host:gpu) is free when no mom-train/mom-serve container uses it and the GPU has < 2 GB in use
(so foreign workloads are never overwritten). Finished = line "EXIT <code> <name>" in host:~/mom-lab/runs/done.txt.
State: logs/queue/<queue>.state.json; events appended to logs/queue/<queue>.events.log (one line per start/finish).
"""
import json, subprocess, sys, time, os
Q = sys.argv[1]
JOBS = f"configs/queues/{Q}.jsonl"
SLOTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["station-a:0", "station-a:1", "station-b:0", "station-b:1"]
# SOFT_SLOTS (env, e.g. station-a:0,station-b:0): eval GPUs lent to training only while the eval backlog
# is empty, i.e. every finished run (plus SOFT_EXTRA names) is listed in EVAL_DONE. A run being evaluated is not
# yet in EVAL_DONE, so training never takes a soft slot from an in-flight or waiting eval.
SOFT = [x for x in os.environ.get("SOFT_SLOTS", "").split(",") if x]
SOFT_EXTRA = [x for x in os.environ.get("SOFT_EXTRA", "").split(",") if x]
EVAL_DONE = os.environ.get("EVAL_DONE", "")
SLOTS = SLOTS + [x for x in SOFT if x not in SLOTS]
os.makedirs("logs/queue", exist_ok=True)
STATE = f"logs/queue/{Q}.state.json"; EV = open(f"logs/queue/{Q}.events.log", "a")


def sh(host, cmd, timeout=60):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, cmd],
                           capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.stdout
    except subprocess.TimeoutExpired:
        return ""


def gpu_used_mb(host, gpu):
    out = sh(host, f"nvidia-smi -i {gpu} --query-gpu=memory.used --format=csv,noheader,nounits")
    try:
        return int(out.strip().splitlines()[0])
    except Exception:
        return 10 ** 9


def eval_backlog(st):
    done = set(open(EVAL_DONE).read().split()) if EVAL_DONE and os.path.exists(EVAL_DONE) else set()
    want = {n for n, i in st["finished"].items() if i.get("exit") == 0} | set(SOFT_EXTRA)
    return len(want - done)


def done_names(host):
    return {l.split()[2]: int(l.split()[1]) for l in sh(host, "cat ~/mom-lab/runs/done.txt 2>/dev/null").splitlines()
            if l.startswith("EXIT ")}


st = json.load(open(STATE)) if os.path.exists(STATE) else {"running": {}, "finished": {}}
while True:
    jobs = [json.loads(l) for l in open(JOBS) if l.strip()]
    started = set(st["running"]) | set(st["finished"])
    # reap finished
    for host in {s.split(":")[0] for s in SLOTS}:
        d = done_names(host)
        for name, slot in list(st["running"].items()):
            if slot.startswith(host) and name in d:
                code = d[name]
                # judge by the run's own log, not the wrapper's exit code (wrapper artefacts gave exit 127)
                last = sh(host, f"tail -1 ~/mom-lab/runs/{name}/train_log.jsonl 2>/dev/null; ls ~/mom-lab/runs/{name}/model.safetensors 2>/dev/null")
                try:
                    rec = json.loads(last.splitlines()[0]); complete = rec["step"] == rec["of"] and "model.safetensors" in last
                except Exception:
                    complete = False
                code = 0 if complete else (code or 1)
                st["finished"][name] = {"slot": slot, "exit": code, "wrapper_exit": d[name], "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
                del st["running"][name]
                EV.write(f"{time.strftime('%H:%M:%S')} FINISH {name} exit={code} wrapper={d[name]} slot={slot}\n"); EV.flush()
    busy = set(st["running"].values())
    pending = [j for j in jobs if j["name"] not in started]
    backlog = eval_backlog(st) if SOFT else 0
    for slot in SLOTS:
        if slot in busy or not pending:
            continue
        if slot in SOFT and backlog:
            continue
        host, gpu = slot.split(":")
        if gpu_used_mb(host, gpu) > 2000:
            continue
        # first pending job that prefers this slot, else first job with no preference
        job = next((j for j in pending if slot in j.get("prefer", [])), None) or \
              next((j for j in pending if not j.get("prefer")), None)
        if not job:
            continue
        # guard: never launch a run whose container already exists on any Station
        if any(f"mom-train-{job['name']}" in sh(h, "docker ps -a --format '{{.Names}}'") for h in {s.split(':')[0] for s in SLOTS}):
            EV.write(f"{time.strftime('%H:%M:%S')} SKIP {job['name']} already has a container\n"); EV.flush()
            pending.remove(job); continue
        args = " ".join(job["args"])
        data = " ".join(job["data"])
        cmd = f"cd ~/mom-lab && setsid nohup ./train_run.sh {gpu} {job['name']} {job['model']} {data} -- {args} > /dev/null 2>&1 < /dev/null &"
        sh(host, cmd)
        st["running"][job["name"]] = slot; busy.add(slot); pending.remove(job)
        EV.write(f"{time.strftime('%H:%M:%S')} START {job['name']} slot={slot}\n"); EV.flush()
    json.dump(st, open(STATE, "w"), indent=1)
    if not pending and not st["running"]:
        EV.write("QUEUE_EMPTY\n"); EV.flush(); break
    time.sleep(60)
