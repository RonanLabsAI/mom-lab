"""Exploratory (not preregistered): per-domain x per-set tables for every test-split system."""
import json, os, sys
SETS = {
 'core': {'math':['gsm8k','math500'],'code':['humaneval_plus','mbpp_plus','livecodebench'],'sql':['bird_dev'],'tools':['bfcl'],'trans':['flores200'],'gen':['mmlu_pro','ifeval']},
 'para': {'math':['gsm8k_pp'],'code':['mbpp_pp'],'sql':['bird_pp'],'tools':['bfcl_pp'],'trans':['flores_pp'],'gen':['mmlupro_pp']},
 'mixed':{'math':['gsm8k_ml'],'code':['mbpp_ml'],'sql':['bird_ml'],'tools':['bfcl_ml']},
}
def load(run, root):
    p = f'results/{root}/{run}/scores.json'
    return json.load(open(p))['benchmarks'] if os.path.exists(p) else None
def dom(runs, root, bs):
    v = []
    for r in runs:
        s = load(r, root)
        if s and all(b in s for b in bs): v.append(sum(s[b]['score'] for b in bs)/len(bs))
    return sum(v)/len(v) if v else None
S3 = ['', '-s1', '-s2']
def seeds(n): return [n+s for s in S3]
def systems(size):
    if size == '1.7B':
        base='base-qwen3-1.7b'; p='pt17'; routes=['code','sql','tools']
        sy = {'base':{d:[base] for d in SETS['core']}}
        sy['pooled'] = 'ALL:'+'|'.join(seeds(f'{p}-pooled-lr2e-5'))
        sy['distill'] = 'ALL:'+'|'.join(seeds(f'{p}-distill-lr2e-5'))
        for k,tmpl in [('lora',f'{p}-lora16-{{d}}-lr2e-4'),('replay',f'{p}-replay-{{d}}-r20-lr2e-5')]:
            sy[k+'-routed'] = {d:(seeds(tmpl.format(d=d)) if d in routes else [base]) for d in SETS['core']}
            for d in routes: sy[f'{k}-{d} always-on'] = 'ALL:'+'|'.join(seeds(tmpl.format(d=d)))
    else:
        base='base-qwen3-4b'; p='pt4'; routes=['sql','tools']
        sy = {'base':{d:[base] for d in SETS['core']}}
        sy['pooled'] = 'ALL:'+'|'.join(seeds(f'{p}-pooled-lr2e-5'))
        sy['pooled-from-Base'] = 'ALL:b4-pooled-lr2e-5'
        sy['distill'] = 'ALL:'+'|'.join(seeds(f'{p}-distill-lr2e-5'))
        sy['lora-routed'] = {d:(seeds(f'{p}-lora16-{d}-lr2e-4') if d in routes else [base]) for d in SETS['core']}
        for d in routes: sy[f'lora-{d} always-on'] = 'ALL:'+'|'.join(seeds(f'{p}-lora16-{d}-lr2e-4'))
        sy['8B'] = 'ALL:base-qwen3-8b'; sy['8B-awq'] = 'ALL:base-qwen3-8b-awq'; sy['8B-fp8']='ALL:base-qwen3-8b-fp8'
        sy['4B-awq'] = 'ALL:base-qwen3-4b-awq'; sy['4B-fp8']='ALL:base-qwen3-4b-fp8'
        sy['awq4+lora-routed'] = {d:(seeds(f'awq4-lora16-{d}-lr2e-4') if d in routes else ['base-qwen3-4b-awq']) for d in SETS['core']}
    return sy
def runs_for(spec, d):
    if isinstance(spec, str): return spec[4:].split('|')
    return spec[d]
def table(size, root, setname):
    out = {}
    for name, spec in systems(size).items():
        row = {}
        for d, bs in SETS[setname].items():
            if setname=='para' and root=='partc': pass
            row[d] = dom(runs_for(spec, d), root, bs)
        vals = [v for v in row.values() if v is not None]
        row['ALL'] = sum(vals)/len(vals) if len(vals)==len(row) else None
        out[name] = row
    return out
if __name__ == '__main__':
    res = {}
    for size in ['1.7B','4B']:
        for root, setname, lab in [('partbtest','core','core'),('partbtest','para','para590'),('partc','para','para1562'),('partbtest','mixed','mixed')]:
            t = table(size, root, setname); res[f'{size} {lab}'] = t
            cols = list(SETS[setname])+['ALL']
            print(f'\n== {size} {lab}'); print(f"{'':24s}"+''.join(f'{c:>8s}' for c in cols))
            for n, r in t.items():
                print(f'{n:24s}'+''.join(f"{r[c]:8.1f}" if r[c] is not None else f"{'-':>8s}" for c in cols))
    json.dump(res, open('results/analysis/explore_tables.json','w'), indent=1)
