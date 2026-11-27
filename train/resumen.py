"""
Tabla de resultados de train/runs/log.jsonl, agrupando corridas que solo difieren en --fold.
La configuración se identifica por el nombre sin el sufijo _f0/_f1/_f2.

  python train/resumen.py            # todas
  python train/resumen.py cv_        # solo las que empiezan con cv_
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

rows = [json.loads(l) for l in open(Path(__file__).resolve().parent / 'runs' / 'log.jsonl')]
pref = sys.argv[1] if len(sys.argv) > 1 else ''
g = defaultdict(dict)
for r in rows:
    if r['name'].startswith(pref) and 'val' in r:
        g[re.sub(r'_f\d$', '', r['name'])][r['args'].get('fold', 0)] = r
print(f"{'configuración':28s} {'f0':>7s} {'f1':>7s} {'f2':>7s} {'media':>7s}  RMSE f0 (RH T U V)   min")
for k, f in sorted(g.items(), key=lambda kv: -sum(r['val'] for r in kv[1].values()) / len(kv[1])):
    s = [f[i]['val'] if i in f else None for i in range(3)]
    m = sum(x for x in s if x is not None) / sum(x is not None for x in s)
    v = f.get(0, next(iter(f.values())))['per_var_rmse']
    print(f"{k:28s} " + ' '.join(f'{x:7.4f}' if x is not None else '      -' for x in s) + f' {m:7.4f}  '
          + ' '.join(f'{v[x]:.2f}' for x in ('RH', 'T', 'U', 'V')) + f"  {sum(r['minutes'] for r in f.values()):.1f}")
