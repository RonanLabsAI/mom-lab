#!/usr/bin/env python3
"""Refuse files that must never reach a public artifact (master plan Review Focus 5; spec section 5 L2).

usage: python3 -I scripts/release/check_public_safe.py [--allow FILE] PATH [PATH ...]
  PATH may be a file or a directory (walked; .git, .venv*, __pycache__ skipped).
Exit 0 = every file passed. Exit 1 = at least one file refused; stderr has one line per reason:
  REFUSE <path>: <reason> (line N)
--allow FILE: reviewed exceptions, one per line "<path-glob> <rule_token>  # reason" (rule token = a rule name
below with spaces as underscores, e.g. proprietary_reference; the
proprietary-path rule cannot be allowed). Default: scripts/release/public_safe_allow.txt if it exists.
Matched values are never printed, only the rule name and line number.

Rules:
  proprietary path       the file's own path contains FlyCurrent or RetenzAI
  proprietary reference  the content mentions a FlyCurrent or RetenzAI path (/FlyCurrent/, /RetenzAI/)
  owner email            any email outside the neutral example.invalid domain (review public fixtures)
  ssn                    \\b\\d{3}-\\d{2}-\\d{4}\\b
  mrn                    \\bMRN[:# ]*\\d{5,}
  dob                    \\bDOB[: ]
  phone                  US phone numbers: (925) 555-0142, 925-555-0142, 925.555.0142, +1 925 555 0142
These are cheap screens, not a PHI audit. A pass means none of the patterns matched; the source-level rules in
docs/eval-set-plan.md (public sources only, never patient data) still apply.
"""
from __future__ import annotations

import fnmatch
import os
import re
import sys

PROPRIETARY_PATH = re.compile(r"(?i)(flycurrent|retenzai)")
RULES = [
    ("proprietary reference", re.compile(r"(?i)/(?:flycurrent|retenzai)(?:/|\b)")),
    # Screen personal addresses generically; do not embed an actual identity in source.
    ("owner email", re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@(?!example\.invalid\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("mrn", re.compile(r"\bMRN[:# ]*\d{5,}")),
    ("dob", re.compile(r"\bDOB[: ]")),
    ("phone", re.compile(r"(?<![\w.-])(?:\+1[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[.-])\d{3}[.-]\d{4}(?![\w-])")),
]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules"}
MAX_BYTES = 64 << 20


def scan_text(text: str) -> list[tuple[str, int]]:
    """Return (rule, 1-based line) for every rule hit in text."""
    hits = []
    for name, rx in RULES:
        for m in rx.finditer(text):
            hits.append((name, text.count("\n", 0, m.start()) + 1))
    return hits


def iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith(".venv"))
                for f in sorted(files):
                    yield os.path.join(root, f)
        else:
            yield p


DEFAULT_ALLOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public_safe_allow.txt")


def load_allow(path: str | None) -> list[tuple[str, str]]:
    if not path or not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        glob, rule = line.rsplit(None, 1) if " " in line else (line, "")
        rule = rule.replace("_", " ")  # rule tokens use underscores: proprietary_reference, owner_email
        if rule == "proprietary path":
            raise SystemExit("allowlist may not exempt the proprietary path rule")
        out.append((glob.strip(), rule))
    return out


def allowed(path: str, reason: str, allow: list[tuple[str, str]]) -> bool:
    rule = reason.split(" (line", 1)[0]
    rel = os.path.relpath(path)
    return any(rule == r and (fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(path, g)) for g, r in allow)


def check_file(path: str) -> list[str]:
    reasons = []
    if PROPRIETARY_PATH.search(os.path.abspath(path)):
        reasons.append("proprietary path")
    try:
        if os.path.getsize(path) > MAX_BYTES:
            return reasons + [f"too large to scan (> {MAX_BYTES >> 20} MiB); review by hand"]
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        return reasons + [f"unreadable ({e.__class__.__name__})"]
    if b"\0" in raw[:8192]:
        return reasons  # binary: path rule only
    for name, line in scan_text(raw.decode("utf-8", errors="replace")):
        reasons.append(f"{name} (line {line})")
    return reasons


def main(argv: list[str]) -> int:
    allow_path = DEFAULT_ALLOW
    if argv[:1] == ["--allow"]:
        allow_path, argv = argv[1], argv[2:]
    if not argv:
        print(__doc__.split("\n\n")[0], file=sys.stderr)
        return 2
    allow = load_allow(allow_path)
    n = bad = 0
    for path in iter_files(argv):
        n += 1
        reasons = [r for r in check_file(path) if not allowed(path, r, allow)]
        if reasons:
            bad += 1
            for r in reasons:
                print(f"REFUSE {path}: {r}", file=sys.stderr)
    print(f"checked {n} files, refused {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
