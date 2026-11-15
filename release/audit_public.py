#!/usr/bin/env python3
"""Screen tracked release files without echoing matched values; CPU/offline only."""
from collections import Counter
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
C=load('public_safe','scripts/release/check_public_safe.py')
R=load('prep','scripts/release/prepare_public.py')
files=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode().split('\0')[:-1]
allow=C.load_allow(str(ROOT/'scripts/release/public_safe_allow.txt'))
refused=[];reviewed=[];residual=[];stats=Counter();files_checked=0
for path in sorted(set(files)):
    p=ROOT/path
    if not p.is_file():continue
    files_checked+=1
    reasons=C.check_file(str(p))
    for reason in reasons:
        # C.allowed uses cwd-relative paths; run from repository root.
        dst=reviewed if C.allowed(str(p),reason,allow) else refused
        dst.append(dict(path=path,rule=reason))
    try:s=p.read_text()
    except UnicodeError:continue
    for name,rx in [('email',R.EMAIL),('credential',R.SECRET),('proprietary_name',re.compile(r'(?i)flycurrent|retenzai')),('home_path',R.HOME),('private_data_path',R.DATA),('station_hostname',R.HOST),('serial_hostname',R.SERIAL)]:
        matches=list(rx.finditer(s))
        if name=='email':matches=[m for m in matches if not m.group().endswith('@example.invalid')]
        if matches:
            stats[name]+=len(matches)
            if path.startswith('data_partc/'):
                disposition='unchanged public upstream benchmark fixtures; see RELEASE-CHECKLIST.md'
            elif name=='proprietary_name':
                disposition='exclusion policy, scanner rule or synthetic test; no proprietary source material'
            else:disposition='review required'
            residual.append(dict(path=path,rule=name,hits=len(matches),disposition=disposition))
    private=sum(R.private_ip(m.group()) for m in R.IPV4.finditer(s))
    if private:stats['private_ipv4']+=private;residual.append(dict(path=path,rule='private_ipv4',hits=private,disposition='review required'))
report=dict(generated_pt=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),files_checked=files_checked,refused=refused,reviewed_screen_exceptions=reviewed,residual_pattern_counts=dict(stats),residual_inventory=residual,
            limitation='Pattern screening is not an independent PHI audit. Public MedCalc clinical notes retain source provenance and CC BY-SA terms. Raw downloads, ignored generations/adapters and private git history are outside this tracked-tree audit and must be screened separately before uploads.')
if '--write' in sys.argv:
    (ROOT/'release/PUBLIC-AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(files_checked=files_checked,refused=len(refused),reviewed_exceptions=len(reviewed),residual_pattern_counts=dict(stats))))
if refused:print(json.dumps(refused,indent=2))
raise SystemExit(bool(refused))
