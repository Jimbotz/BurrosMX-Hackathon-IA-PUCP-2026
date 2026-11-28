"""
Escribe un script bash que entrena con TODO el train (--full) cada modelo del ensamble que todavía no
exista en train/runs/, usando los argumentos exactos con que se validó (fold 0 en train/runs/log.jsonl),
y al final arma la submission con los pesos del ensamble.

  python train/generar_final.py train/runs/ensamble.json train/rondas/final_v3.sh
  nohup bash train/rondas/final_v3.sh > train/runs/final_v3.out 2>&1 < /dev/null &
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ens_path, out = sys.argv[1], sys.argv[2]
ens = json.load(open(ens_path))
rows = [json.loads(l) for l in open(ROOT / 'train' / 'runs' / 'log.jsonl')]
args = {r['name'][:-3]: r['args'] for r in rows if r['name'].endswith('_f0')}
falta = [n[len('full_'):] for n in ens if not (ROOT / 'train' / 'runs' / n / 'model.pt').exists()]

L = ['#!/bin/bash', f'# Generado por train/generar_final.py desde {ens_path}: modelos del ensamble con todo el train.', 'P=.venv/bin/python']
for c in falta:
    assert c in args, f'no encuentro {c}_f0 en log.jsonl'
    parts = []
    for k, v in args[c].items():
        if k in ('name', 'fold', 'full', 'explain') or v is None or v is False or v == []: continue
        parts.append(f'--{k}' if v is True else f'--{k} ' + (' '.join(map(str, v)) if isinstance(v, list) else str(v)))
    L.append(f'$P train/exp.py --name full_{c} --full ' + ' '.join(parts) + f' > train/runs/full_{c}.log 2>&1 || echo "FALLÓ full_{c}"')
L.append(f'$P train/ensemble.py train/runs/sub_{Path(ens_path).stem}.csv {ens_path}')
open(out, 'w').write('\n'.join(L) + '\n')
print(f'{out}: {len(falta)} modelos por entrenar', falta)
