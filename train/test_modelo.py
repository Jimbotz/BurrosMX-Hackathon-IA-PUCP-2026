"""Chequeo rápido del modelo:  python train/test_modelo.py"""
import sys
from pathlib import Path

import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelo import build, static_feats  # noqa: E402

st = dict(np.load(ROOT / 'data' / 'static.npz')); S = static_feats(st['dem'])
lo = torch.randn(2, 7 * 4, 17, 13); hh = torch.randn(2, 2); hi = torch.randn(2, 4, 70, 35)
lo_fut = lo.clone(); lo_fut[:, 4 * 4:] += 5.0          # altera solo las horas t+1, t+2, t+3
for mode, debe_cambiar in (('causal', False), ('full', True)):
    torch.manual_seed(0)
    m = build(dict(feats=[], lags=[-3, -2, -1, 0, 1, 2, 3], width=32, tattn=mode), st, S).eval()
    with torch.no_grad(): d = (m(lo, hh, hi) - m(lo_fut, hh, hi)).abs().max().item()
    assert (d > 1e-4) == debe_cambiar, f'tattn={mode}: cambio {d}'
print('ok: la atención causal no ve el futuro; la completa sí')
