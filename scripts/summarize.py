import json,glob,sys
pat=sys.argv[1] if len(sys.argv)>1 else 'results/step1/*/scores.json'
for f in sorted(glob.glob(pat)):
    d=json.load(open(f)); b=d['benchmarks']
    bs=" ".join(f"{k}={v['score']:.1f}" if isinstance(v,dict) else f"{k}={v:.1f}" for k,v in b.items())
    A=d.get('A'); print(f"{d['run_id']:32s} A={A if A is None else round(A,2)}  {bs}")
