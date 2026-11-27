"""
Figuras de explicabilidad de una corrida hecha con `exp.py --explain`:
importancia por permutación, mapa de error relativo y error por altura / hora.

  python analysis/explicar.py b_b1_explica_f0
"""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
OUT = ROOT / 'docs' / 'analisis'; OUT.mkdir(parents=True, exist_ok=True)
row = [json.loads(l) for l in open(ROOT / 'train' / 'runs' / 'log.jsonl') if json.loads(l)['name'] == name][-1]
VARS = ['RH', 'T', 'U', 'V']
dem = np.load(ROOT / 'data' / 'static.npz')['dem']

# 1. importancia por permutación (caída del score al romper cada entrada)
imp = row['importancia']
k = list(imp)[::-1]
fig, ax = plt.subplots(figsize=(7, 0.28 * len(k) + 1))
ax.barh(k, [imp[x] for x in k], color=['#c0392b' if 'GFS' in x else '#2471a3' if 'base' in x else '#7d8c3c' if 'relieve' in x else '#888' for x in k])
ax.axvline(0, c='k', lw=.5); ax.set_xlabel('caída del score en validación')
ax.set_title(f'¿Qué usa el modelo? (permutación · {name})'); fig.tight_layout()
fig.savefig(OUT / 'importancia.png', dpi=110); plt.close(fig)

# 2. dónde se equivoca: error relativo a la climatología por píxel
rel = np.load(ROOT / 'train' / 'runs' / name / 'error_rel_pixel.npy')
fig, ax = plt.subplots(1, 5, figsize=(17, 5))
ax[0].imshow(dem, origin='lower', cmap='terrain'); ax[0].set_title('Relieve (m)')
for j, v in enumerate(VARS):
    im = ax[j + 1].imshow(rel[..., j], origin='lower', cmap='magma_r', vmin=0, vmax=1.2); ax[j + 1].set_title(f'{v}: RMSE / RMSE clim')
fig.colorbar(im, ax=ax[1:], shrink=.7); fig.suptitle('Error del modelo por píxel (1 = igual que la climatología)')
fig.savefig(OUT / 'error_pixel.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# 3. error por banda de altura y por hora del día
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
bands = list(row['error_por_altura']); E = np.array([row['error_por_altura'][b] for b in bands])
for j, v in enumerate(VARS): ax[0].plot(bands, E[:, j], 'o-', label=v)
ax[0].set_ylabel('RMSE / RMSE clim'); ax[0].set_title('Error por altura'); ax[0].legend()
hrs = sorted(row['error_por_hora_utc'], key=int); H = np.array([row['error_por_hora_utc'][h] for h in hrs])
for j, v in enumerate(VARS): ax[1].plot([(int(h) - 5) % 24 for h in hrs], H[:, j], 'o', label=v)
ax[1].set_xlabel('hora de Lima'); ax[1].set_title('Error por hora del día'); fig.tight_layout()
fig.savefig(OUT / 'error_altura_hora.png', dpi=110); plt.close(fig)
print(json.dumps(dict(importancia=imp, error_por_altura=row['error_por_altura']), ensure_ascii=False)[:1500])
