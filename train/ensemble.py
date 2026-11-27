"""
Promedia las predicciones de varias corridas de train/exp.py.

  python train/ensemble.py val  cfg1 cfg2 ...             # score CV del promedio: cfg = nombre sin _f0/_f1/_f2
  python train/ensemble.py greedy cfg1 cfg2 ...          # selección greedy por variable (Caruana) sobre los 3 folds
  python train/ensemble.py sub.csv run1 run2 ...         # submission de Kaggle (test_pred.npy, promedio simple)
  python train/ensemble.py sub.csv run1:2 run2:1 ...     # ... con pesos (repeticiones)
  python train/ensemble.py sub.csv train/runs/ensamble.json   # pesos por variable que escribe greedy
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'train' / 'runs'
VARS = ['RH', 'T', 'U', 'V']
FOLDS = {0: ('2024061600', '2024070100'), 1: ('2024060100', '2024061600'), 2: ('2024051600', '2024060100')}
mode, names = sys.argv[1], sys.argv[2:]


def fold_data(f):  # misma partición y referencia que exp.py
    tr = np.load(ROOT / 'data' / 'train.npz'); t, Y = tr['times'].astype(str), tr['Y']
    TS = pd.to_datetime(t, format='%Y%m%d%H'); h = TS.hour.values
    lo, hi = (pd.to_datetime(s, format='%Y%m%d%H') for s in FOLDS[f])
    val = np.asarray((TS >= lo) & (TS < hi))
    trn = ~np.asarray((TS >= lo - pd.Timedelta(hours=24)) & (TS < hi + pd.Timedelta(hours=24)))
    clim = np.stack([Y[trn & (h == k)].mean(0) for k in range(24)])
    return Y[val], np.sqrt(((clim[h[val]] - Y[val]) ** 2).mean((0, 1, 2)))


def ratios(P, Y, ref): return np.sqrt(((P - Y) ** 2).mean((0, 1, 2))) / ref   # (4,) RMSE / RMSE_clim


if mode in ('val', 'greedy'):
    F = {f: fold_data(f) for f in FOLDS}
    preds = {n: {f: np.load(RUNS / f'{n}_f{f}' / 'val_pred.npy') for f in FOLDS} for n in names}
    def cv(members_per_var):  # members_per_var: lista (por variable) de listas de nombres (con repetición = peso)
        out = []
        for f, (Y, ref) in F.items():
            r = [ratios(np.mean([preds[n][f][..., j:j + 1] for n in members_per_var[j]], 0), Y[..., j:j + 1], ref[j:j + 1])[0]
                 for j in range(4)]
            out.append(r)
        return np.array(out)  # (folds, 4)
    if mode == 'val':
        r = cv([names] * 4)
        print('score por fold', np.round(1 - r.mean(1), 4).tolist(), '· media', round(float(1 - r.mean()), 4),
              '· ratio por variable', dict(zip(VARS, np.round(r.mean(0), 3).tolist())))
        for n in names:
            rn = cv([[n]] * 4); print(f'  {n:24s} {1 - rn.mean():.4f}')
    else:  # Caruana: agrega (con reemplazo) el modelo que más mejora cada variable, hasta 20 pasos
        chosen = []
        for j, v in enumerate(VARS):
            sel, best = [], 9.
            for _ in range(20):
                cand = {n: cv([sel + [n]] * 4)[:, j].mean() for n in names}
                n, s = min(cand.items(), key=lambda kv: kv[1])
                if s >= best - 1e-5: break
                sel.append(n); best = s
            chosen.append(sel)
            print(f'{v}: ratio CV {best:.4f} · pesos {dict(Counter(sel))}')
        r = cv(chosen)
        print('ensamble greedy · score por fold', np.round(1 - r.mean(1), 4).tolist(), '· media', round(float(1 - r.mean()), 4))
        # pesos por variable para los modelos finales (entrenados con --full como full_<cfg>): lo usan la submission y modelo.Ensamble
        cnt = [Counter(s) for s in chosen]
        pesos = {f'full_{n}': [c.get(n, 0) for c in cnt] for n in names if any(c.get(n) for c in cnt)}
        json.dump(pesos, open(RUNS / 'ensamble.json', 'w'), indent=1)
        print('escrito train/runs/ensamble.json con', len(pesos), 'modelos')
else:  # submission: "run:k" = peso k para las 4 variables, o un ensamble.json con pesos por variable
    if names[0].endswith('.json'):
        w = {n: np.asarray(v, np.float32) for n, v in json.load(open(names[0])).items()}
        P = sum(np.load(RUNS / n / 'test_pred.npy') * v for n, v in w.items()) / sum(w.values())
    else:
        w = [(n.split(':')[0], int(n.split(':')[1]) if ':' in n else 1) for n in names]
        P = sum(k * np.load(RUNS / n / 'test_pred.npy') for n, k in w) / sum(k for _, k in w)
    tte = np.load(ROOT / 'data' / 'test.npz')['times'].astype(str)
    sample = pd.read_csv(ROOT / 'data' / 'sample_submission.csv')
    parts = sample['id'].str.split('_', expand=True)
    ti = parts[0].map({k: i for i, k in enumerate(tte)})
    assert ti.notna().all(), 'hay horas del sample_submission que no están en test.npz'
    r, c = parts[1].astype(int).values, parts[2].astype(int).values
    sub = pd.DataFrame({'id': sample['id']})
    for j, v in enumerate(VARS): sub[v] = P[ti.values, r, c, j].round(3)
    sub.to_csv(mode, index=False)
    print(mode, sub.shape, 'promedio de', w)
