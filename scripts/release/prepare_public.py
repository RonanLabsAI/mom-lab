#!/usr/bin/env python3
"""Offline release scrub and provenance verification. Never reads credentials.

Apply to a clean release-prep branch with --apply --base COMMIT. Only files
tracked at that commit are scrubbed. --verify checks the recorded release tree.
Original SHA-256s remain in the manifest; original sensitive strings do not.
Public benchmark items are immutable and are screened separately.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'release'
# Operational run logs (queue events, Docker inspect dumps, host summaries) are left out of the
# public export (RELEASE-CHECKLIST.md). Their scrubbed hashes stay in SCRUB-MANIFEST.json.
# logs/analysis/ (analysis text outputs cited by the notebook) is published.
def excluded_from_export(path):
    return path.startswith('logs/') and not path.startswith(('logs/analysis/','logs/README.md'))
IPV4 = re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)')
# Only non-routable fleet ranges; public IPs in programming benchmarks are data.
NETWORKS = tuple(ipaddress.ip_network(n) for n in ((0x0A000000,8),(0xAC100000,12),(0xC0A80000,16),(0x64400000,10)))
HOME = re.compile(r'(?<![\w/])/home/[A-Za-z0-9_.-]+')
DATA = re.compile(r'(?<![\w/])/data/(?=(?:sentinel|services|mom-lab-archive)(?:/|\b))')
HOST = re.compile(r'\bstation-(left|right)\b')
SERIAL = re.compile(r'(?i)\bsn\d{5,}[\w-]*\b')
GPU = re.compile(r'GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}')
MAC = re.compile(r'(?i)(?<![\w:])(?:[a-f0-9]{2}:){5}[a-f0-9]{2}(?![\w:])')
EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
SECRET = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b')
# Matching results never print the values. This script's synthetic rule strings
# and checker fixtures are allowed; no credentials are embedded in either.
RULES = {'home_path':HOME, 'private_data_path':DATA, 'station_hostname':HOST,
         'serial_hostname':SERIAL, 'gpu_identifier':GPU, 'mac_address':MAC}


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.DEVNULL)


def sha(b):return hashlib.sha256(b).hexdigest()


def private_ip(s):
    try:return any(ipaddress.ip_address(s) in n for n in NETWORKS)
    except ValueError:return False


def scrub(s):
    counts=Counter()
    def replace(name,rx,fn):
        nonlocal s
        def sub(m):
            new=fn(m)
            if new!=m.group():counts[name]+=1
            return new
        s=rx.sub(sub,s)
    replace('private_ipv4',IPV4,lambda m:'host-'+sha(m.group().encode())[:8]+'.invalid' if private_ip(m.group()) else m.group())
    replace('home_path',HOME,lambda m:'/opt/release-user')
    replace('private_data_path',DATA,lambda m:'/srv/release-data/')
    replace('station_hostname',HOST,lambda m:'station-a' if m[1]=='left' else 'station-b')
    replace('serial_hostname',SERIAL,lambda m:'node-'+sha(m.group().encode())[:8]+'.invalid')
    replace('gpu_identifier',GPU,lambda m:m.group() if m.group().startswith('GPU-00000000-0000-0000-0000-') else 'GPU-00000000-0000-0000-0000-'+sha(m.group().encode())[:12])
    replace('mac_address',MAC,lambda m:m.group() if m.group().lower().startswith('02:00:00:') else '02:00:00:'+':'.join(sha(m.group().encode())[i:i+2] for i in (0,2,4)))
    replace('email',EMAIL,lambda m:'contact@example.invalid')
    if SECRET.search(s):
        replace('credential',SECRET,lambda m:'REDACTED_CREDENTIAL')
    return s,counts


def scrub_text(path,s):
    new,counts=scrub(s)
    if path=='scripts/release/check_public_safe.py':
        lines=new.splitlines(keepends=True)
        for i,line in enumerate(lines):
            if '    ("owner email", re.compile(' in line:
                lines[i]='    ("owner email", re.compile(r"(?i)\\b[A-Za-z0-9._%+-]+@(?!example\\.invalid\\b)[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b")),\n'
            if 'built from parts so the literal address' in line:
                lines[i]='    # Screen personal addresses generically; do not embed an actual identity in source.\n'
        revised=''.join(lines)
        if revised!=new:counts['embedded_identity_reference']+=1
        new=revised
    elif path=='tests/test_check_public_safe.py':
        revised=re.sub(r'    addr = [^\n]+','    addr = "private-user" + "@" + "example.org"',new)
        if revised!=new:counts['embedded_identity_reference']+=1
        new=revised
    return new,counts


def numeric_leaves(obj,path=()):
    """Numbers and bools, independent of string endpoint/path metadata."""
    out={}
    if isinstance(obj,dict):
        for k,v in obj.items():out.update(numeric_leaves(v,path+(k,)))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):out.update(numeric_leaves(v,path+(i,)))
    elif isinstance(obj,(int,float,bool)) or obj is None:out[path]=obj
    return out


def frozen_entries(base,renames):
    records=[]
    for manifest in git('ls-tree','-r','--name-only',base,'artifacts').decode().splitlines():
        if not manifest.endswith('.sha256'):continue
        content=git('show',f'{base}:{manifest}').decode()
        for line_no,line in enumerate(content.splitlines(),1):
            line=line.strip()
            if not line or line.startswith('#'):continue
            m=re.match(r'([0-9a-f]{64})\s+(.+)',line)
            remote=re.match(r'station:(\S+)\s+([0-9a-f]{64})\s+(.+)',line)
            if remote:expected=remote[2];path=f'{remote[1]}:{remote[3]}'
            elif m:expected=m[1];path=m[2].lstrip('*')
            else:
                clean,_=scrub(line)
                records.append(dict(manifest=manifest,line=line_no,status='environment_receipt',release_description=clean,annotation='Image digest, pipfreeze digest or driver version; requires original host/environment to verify'))
                continue
            clean,_=scrub(path)
            record=dict(manifest=manifest,line=line_no,original_expected_sha256=expected,release_path=clean)
            if ':' in path:
                record['status']='external_unavailable'
            elif path.startswith('/'):
                original=Path(path)
                if original.is_file():
                    actual=sha(original.read_bytes())
                    record.update(original_actual_sha256=actual,status='external_local_match' if actual==expected else 'preexisting_external_mismatch')
                else:record['status']='external_unavailable'
            else:
                try:old=git('show',f'{base}:{path}')
                except subprocess.CalledProcessError:
                    local=ROOT/path
                    if local.is_file():
                        actual=sha(local.read_bytes())
                        record.update(original_actual_sha256=actual,release_sha256=actual,status='untracked_local_match' if actual==expected else 'preexisting_untracked_mismatch')
                    else:record['status']='untracked_or_unavailable'
                else:
                    newpath=renames.get(path,path);new=(ROOT/newpath).read_bytes()
                    record.update(original_actual_sha256=sha(old),release_sha256=sha(new),release_path=newpath,
                                  status='original_match' if sha(old)==expected else 'preexisting_mismatch',
                                  changed_for_release=old!=new)
            records.append(record)
    return records


def apply(base):
    base=git('rev-parse',base).decode().strip()
    if git('rev-parse','HEAD').decode().strip()!=base:
        raise SystemExit('Apply only at the declared baseline HEAD; use --verify after committing.')
    files=git('ls-tree','-r','--name-only','-z',base).decode().split('\0')[:-1]
    records=[];totals=Counter();renames={};frozen=[];bench=[]
    for path in files:
        p=ROOT/path;old=git('show',f'{base}:{path}')
        if path.startswith(('data_partc/','data_partd_c/eval/','data_partd_f/eval/')):
            bench.append(dict(path=path,sha256=sha(old)))
            continue
        try:s=old.decode('utf-8')
        except UnicodeDecodeError:continue
        new,c=scrub_text(path,s);newpath,pc=scrub(path);c.update(pc)
        if not c:continue
        # Refuse to overwrite user work.
        if p.read_bytes() not in (old,new.encode()):raise SystemExit(f'Preexisting edit: {path}')
        mode=p.stat().st_mode
        p.chmod(mode | 0o200)
        p.write_bytes(new.encode())
        p.chmod(mode)
        target=ROOT/newpath
        if newpath!=path:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise SystemExit('Rename collision')
            p.rename(target);renames[path]=newpath
        totals.update(c)
        records.append(dict(path=newpath,original_sha256=sha(old),release_sha256=sha(new.encode()),rules=dict(c),
                            renamed=newpath!=path))
        if path.startswith('results/') and path.endswith('.json'):
            if numeric_leaves(json.loads(s))!=numeric_leaves(json.loads(new)):
                raise SystemExit('Result numerical values changed: '+path)
    frozen=frozen_entries(base,renames)
    RELEASE.mkdir(exist_ok=True)
    report=dict(baseline_commit=base,generated_pt=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
                rules='Neutral nonresolving endpoints; generic host aliases, storage roots and hardware IDs. No benchmark edits.',
                changed_files=len(records),replacement_counts=dict(totals),files=records,
                benchmark_files_unchanged=bench,
                provenance='Original expected hashes are retained. Release hashes attest scrubbed bytes, not original experiment inputs. Original contents remain in the private baseline commit. A public release must export a clean tree without that private history.')
    (RELEASE/'SCRUB-MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n')
    (RELEASE/'FREEZE-PROVENANCE.json').write_text(json.dumps(dict(baseline_commit=base,entries=frozen),indent=2)+'\n')
    print(json.dumps(dict(changed_files=len(records),replacement_counts=dict(totals),freeze_entries=len(frozen),freeze_statuses=dict(Counter(r['status'] for r in frozen)),immutable_benchmarks=len(bench))))


def seal():
    """Record final reviewed release edits without rewriting original hashes."""
    p=RELEASE/'SCRUB-MANIFEST.json';report=json.loads(p.read_text())
    for r in report['files']:
        if excluded_from_export(r['path']):
            # Not in the public export; the scrubbed hash stays as the provenance record.
            r['excluded_from_public_export']=True
            continue
        new=sha((ROOT/r['path']).read_bytes())
        if new!=r['release_sha256']:
            r['additional_release_edit']='Portability/documentation update after mechanical scrub; original hash retained.'
            r['release_sha256']=new
    p.write_text(json.dumps(report,indent=2)+'\n')
    # Manifest path aliases can be recovered without storing private source paths.
    base=report['baseline_commit']
    originals=git('ls-tree','-r','--name-only',base).decode().splitlines()
    renames={old:scrub(old)[0] for old in originals if old!=scrub(old)[0]}
    entries=frozen_entries(base,renames)
    (RELEASE/'FREEZE-PROVENANCE.json').write_text(json.dumps(dict(baseline_commit=base,entries=entries),indent=2)+'\n')
    files=git('ls-files','-z','--cached','--others','--exclude-standard').decode().split('\0')[:-1]
    sums=[]
    for path in sorted(set(files)):
        current=ROOT/path
        if current.is_file() and path!='release/RELEASE-SHA256SUMS':
            sums.append(sha(current.read_bytes())+'  '+path+'\n')
    (RELEASE/'RELEASE-SHA256SUMS').write_text(''.join(sums))
    print(json.dumps(dict(freeze_entries=len(entries),freeze_statuses=dict(Counter(r['status'] for r in entries)))))


def verify():
    report=json.loads((RELEASE/'SCRUB-MANIFEST.json').read_text());bad=[];counts=Counter()
    checksums=RELEASE/'RELEASE-SHA256SUMS'
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            expected,path=line.split('  ',1)
            current=ROOT/path
            if not current.is_file() or sha(current.read_bytes())!=expected:bad.append(path+': release inventory hash differs')
    for r in report['files']:
        p=ROOT/r['path']
        if r.get('excluded_from_public_export'):
            if p.exists():bad.append(r['path']+': excluded from the public export but present')
            continue
        if not p.exists() or sha(p.read_bytes())!=r['release_sha256']:bad.append(r['path']+': release hash differs')
    for r in report['benchmark_files_unchanged']:
        if sha((ROOT/r['path']).read_bytes())!=r['sha256']:bad.append(r['path']+': benchmark bytes differ')
    frozen=json.loads((RELEASE/'FREEZE-PROVENANCE.json').read_text())['entries']
    checked_frozen=0
    for r in frozen:
        if 'release_sha256' not in r:continue
        local=ROOT/r['release_path']
        if not local.is_file() or sha(local.read_bytes())!=r['release_sha256']:
            bad.append(r['release_path']+': frozen release receipt differs')
        checked_frozen+=1
    files=git('ls-files','-z','--cached','--others','--exclude-standard').decode().split('\0')[:-1]
    for path in sorted(set(files)):
        p=ROOT/path
        if not p.exists():continue
        try:s=p.read_text()
        except (UnicodeDecodeError,OSError):continue
        if path.startswith(('data_partc/','data_partd_c/eval/')):continue
        hits={}
        for name,rx in RULES.items():
            values=rx.findall(s)
            if name=='gpu_identifier':values=[v for v in values if not v.startswith('GPU-00000000-0000-0000-0000-')]
            if name=='mac_address':values=[v for v in values if not v.lower().startswith('02:00:00:')]
            if values:hits[name]=len(values)
        n=sum(private_ip(m.group()) for m in IPV4.finditer(s+'\n'+path))
        if n:hits['private_ipv4']=n
        n=sum(not m.group().endswith('@example.invalid') for m in EMAIL.finditer(s))
        if n:hits['email']=n
        if SECRET.search(s):hits['credential']=len(SECRET.findall(s))
        # Rules in this scanner are expressed as regexes, not actual private values.
        if hits:counts.update(hits);bad.append(path+': '+','.join(hits))
    print(json.dumps(dict(verified_scrubbed_files=len(report['files']),verified_benchmark_files=len(report['benchmark_files_unchanged']),verified_frozen_receipts=checked_frozen,remaining=dict(counts),errors=bad),indent=2))
    return 1 if bad else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');p.add_argument('--base');p.add_argument('--verify',action='store_true');p.add_argument('--seal',action='store_true');args=p.parse_args()
    if args.apply and args.base:apply(args.base)
    elif args.seal:seal()
    elif args.verify:raise SystemExit(verify())
    else:p.error('use --apply --base COMMIT or --verify')
