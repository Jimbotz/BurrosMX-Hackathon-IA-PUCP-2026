"""
Explicabilidad de la atención cruzada (--xattn): para varios lugares de la grilla de 3 km, ¿a qué celdas
del GFS (27 km) mira el modelo? Promedia los pesos de atención sobre las horas del test (solo usa GFS).

  python analysis/atencion.py train/runs/full_e_xattn_post
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelo import Corrida  # noqa: E402

c = Corrida(sys.argv[1])
assert c.cfg.get('xattn'), 'la corrida no usa --xattn'
te = np.load(ROOT / 'data' / 'test.npz'); X, t = te['X'], te['times'].astype(str)
st = c.st
c.net.keep_attn = True
A = 0
for i in range(0, len(X), 64):                   # una pasada por bloque para quedarnos con sus pesos
    c.predecir(X[i:i + 64], t[i:i + 64], bs=64)
    A = A + c.net.last_attn.float().sum(0).cpu().numpy()
A /= len(X)                                       # (162 zonas 18x9, 221 celdas GFS 17x13)

lugares = {'Huaraz (valle, ~3,050 m)': (-9.53, -77.53), 'Nevado Huascarán': (-9.12, -77.60),
           'Ladera costera': (-9.90, -77.95), 'Vertiente amazónica': (-8.90, -77.18)}
wl, wo, gl, go = st['wrf_lats'], st['wrf_lons'], st['gfs_lats'], st['gfs_lons']
fig, ax = plt.subplots(1, len(lugares), figsize=(4.2 * len(lugares), 5))
ext = [go[0] - .125, go[-1] + .125, gl[-1] - .125, gl[0] + .125]   # GFS: fila 0 = norte
for k, (nombre, (la, lo)) in enumerate(lugares.items()):
    r, col = np.abs(wl - la).argmin(), np.abs(wo - lo).argmin()       # celda WRF (fila 0 = sur)
    q = (r // 4) * 9 + col // 4                                       # zona 18x9 tras dos pooling 2x2
    im = ax[k].imshow(A[q].reshape(17, 13), extent=ext, cmap='viridis')
    ax[k].plot([wo.min(), wo.max(), wo.max(), wo.min(), wo.min()], [wl.min(), wl.min(), wl.max(), wl.max(), wl.min()], 'w--', lw=1)
    ax[k].plot(lo, la, 'r*', ms=14); ax[k].set_title(nombre, fontsize=10); ax[k].set_xlabel('longitud')
    fig.colorbar(im, ax=ax[k], shrink=.6)
ax[0].set_ylabel('latitud')
fig.suptitle('¿A qué celdas del GFS mira el modelo? (atención cruzada, promedio del test) · ★ lugar · - - dominio de 3 km')
fig.tight_layout(); out = ROOT / 'docs' / 'analisis' / 'atencion.png'; fig.savefig(out, dpi=110); print('guardado', out)
ent = -(A * np.log(A + 1e-9)).sum(1) / np.log(A.shape[1])
print(f'entropía media de la atención (1 = mira todo por igual): {ent.mean():.3f}')
