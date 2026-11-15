"""code: humaneval_plus, mbpp_plus, livecodebench. pass@1 greedy.

Item schema assumed (CONTRACT leaves it to meta):
  meta.test         str   test code appended after the solution (HumanEval/MBPP style asserts).
                          If it defines `check(candidate)` and meta.entry_point is set, `check(entry_point)` is appended.
  meta.test_imports list[str] optional import lines prepended
  meta.io_tests     list[{"input": str, "output": str}]  stdin/stdout tests (LiveCodeBench stdin style)
  meta.func_name    str   with io_tests: LiveCodeBench functional style; `input` is one JSON arg per line,
                          `output` is the JSON expected return; solution must define class Solution.
  reference         canonical solution (used only by tests / sanity runs)

Sandbox: HARNESS_SANDBOX=subprocess (default; rlimits, temp dir, process-group kill, scrubbed env)
or HARNESS_SANDBOX=docker (throwaway --network none container, python image HARNESS_DOCKER_IMAGE).
The subprocess mode does NOT block network access; use docker mode for untrusted bulk runs.
"""
from __future__ import annotations

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..common import strip_think
from . import output_of

MEM_BYTES = 4 * 1024 ** 3
DEFAULT_TIMEOUT = 10.0
_FENCE = re.compile(r"```[ \t]*([\w+#-]*)[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    text = strip_think(text)
    blocks = _FENCE.findall(text)
    py = [c for lang, c in blocks if lang.lower() in ("python", "py", "python3", "")]
    if py:
        return py[-1]
    if blocks:
        return blocks[-1][1]
    m = re.search(r"```[ \t]*\w*\n(.*)$", text, re.S)  # unterminated fence (truncated output)
    if m:
        return m.group(1)
    return text


_BOOT = (
    "import resource,runpy\n"
    "resource.setrlimit(resource.RLIMIT_AS,(%d,%d))\n"
    "resource.setrlimit(resource.RLIMIT_FSIZE,(67108864,67108864))\n"
    "resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
    "runpy.run_path('prog.py', run_name='__main__')\n" % (MEM_BYTES, MEM_BYTES)
)


def run_sandboxed(script: str, stdin: str = "", timeout: float = DEFAULT_TIMEOUT,
                  mode: str | None = None) -> tuple[bool, str, str]:
    """Run python source. Returns (ok, stdout, stderr); ok False on non-zero exit or timeout."""
    mode = mode or os.environ.get("HARNESS_SANDBOX", "subprocess")
    d = tempfile.mkdtemp(prefix="harness_code_")
    name = None
    try:
        with open(os.path.join(d, "prog.py"), "w", encoding="utf-8") as f:
            f.write(script)
        if mode == "docker":
            name = "harness-" + uuid.uuid4().hex[:12]
            os.chmod(d, 0o755)
            cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none",
                   "--memory", "2g", "--pids-limit", "256", "--cpus", "1", "--read-only",
                   "--tmpfs", "/tmp:size=64m", "-v", f"{d}:/w:ro", "-w", "/w",
                   os.environ.get("HARNESS_DOCKER_IMAGE", "python:3.12-slim"),
                   "python", "-I", "/w/prog.py"]
            kw = {}
            timeout += 5  # container start overhead
        else:
            cmd = [sys.executable, "-I", "-c", _BOOT]
            kw = {"start_new_session": True, "cwd": d,
                  "env": {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONHASHSEED": "0", "TMPDIR": d}}
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
        try:
            out, err = p.communicate(stdin.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            if mode == "docker":
                subprocess.run(["docker", "kill", name], capture_output=True)
            else:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            p.kill()
            p.communicate()
            return False, "", "TIMEOUT"
        if mode != "docker":
            try:
                os.killpg(p.pid, signal.SIGKILL)  # reap stray children
            except (ProcessLookupError, PermissionError):
                pass
        return p.returncode == 0, out.decode(errors="replace")[:100000], err.decode(errors="replace")[-2000:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _norm_out(s: str) -> list[str]:
    return [ln.rstrip() for ln in s.strip().splitlines()]


_FUNC_RUNNER = '''
import json as _json, sys as _sys
_args = [_json.loads(l) for l in _sys.stdin.read().splitlines() if l.strip()]
_r = Solution().{fn}(*_args)
print(_json.dumps(_r))
'''


def passes(code: str, meta: dict, timeout: float = DEFAULT_TIMEOUT) -> bool:
    imports = "\n".join(meta.get("test_imports") or [])
    if meta.get("io_tests"):
        fn = meta.get("func_name")
        prog = imports + "\n" + code + ("\n" + _FUNC_RUNNER.format(fn=fn) if fn else "")
        for t in meta["io_tests"]:
            ok, out, _ = run_sandboxed(prog, t["input"], timeout)
            if not ok:
                return False
            if fn:
                try:
                    if json.loads(out) != json.loads(t["output"]):
                        return False
                except ValueError:
                    return False
            elif _norm_out(out) != _norm_out(t["output"]):
                return False
        return True
    test = meta.get("test") or ""
    ep = meta.get("entry_point")
    prog = imports + "\n" + code + "\n\n" + test
    if ep and "def check(" in test and f"check({ep})" not in test:
        prog += f"\n\ncheck({ep})\n"
    ok, _, _ = run_sandboxed(prog, "", timeout)
    return ok


def _one(args) -> float:
    out, meta, timeout = args
    if out is None or not out.strip():
        return 0.0
    try:
        return float(passes(extract_code(out), meta, timeout))
    except Exception:
        return 0.0


def _score_in_container(items, gens, workers, timeout):
    """Run the whole code-scoring batch once inside a network-less container (HARNESS_SANDBOX=docker-batch);
    inside it, each program runs in the subprocess sandbox. One container start instead of one per program."""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    io = tempfile.mkdtemp(prefix="harness_codebatch_")
    os.chmod(io, 0o777)
    sub = {it["id"]: gens[it["id"]] for it in items if it["id"] in gens}
    with open(os.path.join(io, "in.json"), "w") as f:
        json.dump({"items": items, "gens": sub, "workers": workers, "timeout": timeout}, f)
    code = ("import json,sys; sys.path.insert(0,'/r'); from harness.score.code import score_code; "
            "d=json.load(open('/io/in.json')); r=score_code(d['items'],d['gens'],d['workers'],d['timeout']); "
            "json.dump(r,open('/io/out.json','w'))")
    cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "96g", "--pids-limit", "8192",
           "--cpus", str(workers), "-e", "HARNESS_SANDBOX=subprocess", "-v", f"{repo}:/r:ro", "-v", f"{io}:/io",
           os.environ.get("HARNESS_DOCKER_IMAGE", "mom-sandbox:v1"), "python", "-c", code]
    subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
    with open(os.path.join(io, "out.json")) as f:
        r = json.load(f)
    shutil.rmtree(io, ignore_errors=True)
    return r


def score_code(items, gens, workers=None, timeout: float = DEFAULT_TIMEOUT):
    workers = workers or min(48, os.cpu_count() or 4)
    if os.environ.get("HARNESS_SANDBOX") == "docker-batch":
        return _score_in_container(items, gens, workers, timeout)
    args = [(output_of(gens, it["id"]), it.get("meta") or {}, timeout) for it in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        vals = list(ex.map(_one, args))
    return {"ids": [it["id"] for it in items], "correct": vals,
            "score": 100.0 * sum(vals) / max(1, len(vals))}
