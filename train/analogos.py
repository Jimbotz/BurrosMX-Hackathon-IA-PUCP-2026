"""
Modelo simple de "análogos" (k vecinos): para cada hora, busca en el train las horas cuyo GFS más se
parecía (misma hora UTC ± h_win) y promedia cómo corrigió el WRF a la base en esos casos.
  pred(t) = base(t) + promedio ponderado de [Y − base] de los k análogos
La "situación" se describe con el GFS en t y ventanas de días (media ±12 h, ±3.5 días y tendencia),
reducidas con PCA. Mismo formato de salida que exp.py (train/runs/<name>_f{0,1,2} y full_<name>).

  python train/analogos.py --name knn --k 20
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelo import grid, interp, hour  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--k', type=int, default=20)
p.add_argument('--h_win', type=int, default=1, help='acepta análogos a ± estas horas UTC')
p.add_argument('--ncomp', type=int, default=30)
p.add_argument('--ctx', nargs='*', default=['t', 'dia', 'semana', 'tendencia'], help='qué describe la situación')
p.add_argument('--bias_days', type=int, default=30)
a = p.parse_args()
VARS = ['RH', 'T', 'U', 'V']
FOLDS = {0: ('2024061600', '2024070100'), 1: ('2024060100', '2024061600'), 2: ('2024051600', '2024060100')}

tr, te, st = (np.load(ROOT / 'data' / f) for f in ('train.npz', 'test.npz', 'static.npz'))
X = np.concatenate([tr['X'], te['X']]); t = np.concatenate([tr['times'], te['times']]).astype(str)
ntr = len(tr['X']); Y = tr['Y']
TS = pd.to_datetime(t, format='%Y%m%d%H'); h = hour(t)
I = interp(X, grid(st))                                    # GFS a 3 km (n,70,35,4)


def running_mean(lo, hi):  # media del GFS en [t+lo, t+hi] horas (usa train y test: es solo GFS)
    pos = {k: i for i, k in enumerate(TS)}
    acc, cnt = np.zeros_like(X), np.zeros((len(X), 1, 1, 1), np.float32)
    for o in range(lo, hi + 1):
        idx = np.array([pos.get(k + pd.Timedelta(hours=o), -1) for k in TS]); ok = idx >= 0
        acc[ok] += X[idx[ok]]; cnt[ok] += 1
    return acc / np.maximum(cnt, 1)


blocks = {'t': X}
if 'dia' in a.ctx: blocks['dia'] = running_mean(-12, 12)
if 'semana' in a.ctx or 'tendencia' in a.ctx:
    past, fut = running_mean(-84, 0), running_mean(0, 84)
    if 'semana' in a.ctx: blocks['semana'] = (past + fut) / 2
    if 'tendencia' in a.ctx: blocks['tendencia'] = fut - past   # ¿los próximos 3.5 días serán más húmedos/cálidos que los pasados?


def run(fold):
    """fold None = todo el train para predecir el test."""
    if fold is None:
        val = np.zeros(ntr, bool); trn = np.ones(ntr, bool); END = TS[ntr - 1] + pd.Timedelta(hours=1)
    else:
        lo, hi = (pd.to_datetime(s, format='%Y%m%d%H') for s in FOLDS[fold])
        T = TS[:ntr]; val = np.asarray((T >= lo) & (T < hi))
        trn = ~np.asarray((T >= lo - pd.Timedelta(hours=24)) & (T < hi + pd.Timedelta(hours=24))); END = lo
    # base = GFS interpolado + sesgo por hora de los últimos `bias_days` días (igual que exp.py)
    D = Y - I[:ntr]
    rec = trn & np.asarray((TS[:ntr] < END) & (TS[:ntr] >= END - pd.Timedelta(days=a.bias_days)))
    bias = np.stack([D[(rec if (rec & (h[:ntr] == k)).any() else trn) & (h[:ntr] == k)].mean(0) for k in range(24)])
    base = I + bias[h]
    R = Y - base[:ntr]                                          # lo que hay que corregir
    # descriptor de la situación: bloques normalizados -> PCA ajustada en trn
    F = []
    for b in blocks.values():
        z = (b - b[:ntr][trn].mean((0, 1, 2))) / (b[:ntr][trn].std((0, 1, 2)) + 1e-6)
        F.append(z.reshape(len(z), -1) / np.sqrt(z[0].size))   # cada bloque pesa lo mismo
    F = np.concatenate(F, 1)
    mu = F[:ntr][trn].mean(0); U, S, Vt = np.linalg.svd(F[:ntr][trn] - mu, full_matrices=False)
    Z = (F - mu) @ Vt[:a.ncomp].T                               # (n, ncomp)
    targets = np.where(val)[0] if fold is not None else np.arange(ntr, len(X))
    cand = np.where(trn)[0]
    P = np.empty((len(targets), 70, 35, 4), np.float32)
    for n, i in enumerate(targets):
        dh = np.abs(h[cand] - h[i]); dh = np.minimum(dh, 24 - dh)
        c = cand[dh <= a.h_win]
        d = np.sqrt(((Z[c] - Z[i]) ** 2).sum(1))
        nn_ = np.argsort(d)[:a.k]; w = 1 / (d[nn_] + 1e-3); w /= w.sum()
        P[n] = base[i] + np.tensordot(w, R[c[nn_]], 1)
    P[..., 0] = P[..., 0].clip(0, 100)
    return P, val, trn


for fold in (0, 1, 2, None):
    t0 = time.time()
    P, val, trn = run(fold)
    name = f'{a.name}_f{fold}' if fold is not None else f'full_{a.name}'
    out = ROOT / 'train' / 'runs' / name; out.mkdir(parents=True, exist_ok=True)
    row = dict(name=name, args={**vars(a), 'fold': fold if fold is not None else 0, 'full': fold is None},
               minutes=round((time.time() - t0) / 60, 1))
    if fold is None:
        np.save(out / 'test_pred.npy', P)
    else:
        np.save(out / 'val_pred.npy', P)
        hh = h[:ntr]
        clim = np.stack([Y[trn & (hh == k)].mean(0) for k in range(24)])
        ref = np.sqrt(((clim[hh[val]] - Y[val]) ** 2).mean((0, 1, 2)))
        rmse = np.sqrt(((P - Y[val]) ** 2).mean((0, 1, 2)))
        row.update(val=round(float(1 - (rmse / ref).mean()), 4), per_var_rmse=dict(zip(VARS, rmse.round(3).tolist())))
        print(name, row['val'], row['per_var_rmse'], f"{row['minutes']} min", flush=True)
    with open(ROOT / 'train' / 'runs' / 'log.jsonl', 'a') as f: f.write(json.dumps(row) + '\n')
