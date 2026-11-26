"""
Promedia las predicciones de varias corridas de train/exp.py.

  python train/ensemble.py val  run1 run2 ...          # score local del promedio (val_pred.npy)
  python train/ensemble.py sub.csv run1 run2 ...       # submission de Kaggle (test_pred.npy)
"""
import sys
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out, names = sys.argv[1], sys.argv[2:]
VARS = ['RH', 'T', 'U', 'V']
load = lambda f: np.mean([np.load(ROOT / 'train' / 'runs' / n / f) for n in names], 0)

if out == 'val':  # misma validación que exp.py: últimas 2 semanas de junio
    tr = np.load(ROOT / 'data' / 'train.npz'); t, Y = tr['times'].astype(str), tr['Y']
    h = np.array([int(k[-2:]) for k in t]); val = t >= '2024061600'; trn = ~val
    clim = np.stack([Y[trn & (h == k)].mean(0) for k in range(24)])
    ref = np.sqrt(((clim[h[val]] - Y[val]) ** 2).mean((0, 1, 2)))
    rmse = np.sqrt(((load('val_pred.npy') - Y[val]) ** 2).mean((0, 1, 2)))
    print(f'val {1 - (rmse / ref).mean():.4f}', dict(zip(VARS, rmse.round(3).tolist())))
else:
    P = load('test_pred.npy')
    tte = np.load(ROOT / 'data' / 'test.npz')['times'].astype(str)
    sample = pd.read_csv(ROOT / 'data' / 'sample_submission.csv')
    parts = sample['id'].str.split('_', expand=True)
    ti = parts[0].map({k: i for i, k in enumerate(tte)})
    assert ti.notna().all(), 'hay horas del sample_submission que no están en test.npz'
    r, c = parts[1].astype(int).values, parts[2].astype(int).values
    sub = pd.DataFrame({'id': sample['id']})
    for j, v in enumerate(VARS): sub[v] = P[ti.values, r, c, j].round(3)
    sub.to_csv(out, index=False)
    print(out, sub.shape, 'promedio de', names)
