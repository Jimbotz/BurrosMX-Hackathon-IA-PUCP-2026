"""
Análisis de la data oficial: limpieza, cambio de temporada train→test, desfase GFS↔WRF,
correlaciones y sesgos. Escribe figuras en docs/analisis/ y un resumen en docs/analisis/datos.json.

  python analysis/datos.py
"""
import json
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'analisis'; OUT.mkdir(parents=True, exist_ok=True)
VARS = ['RH', 'T', 'U', 'V']
tr, te, st = (np.load(ROOT / 'data' / f) for f in ('train.npz', 'test.npz', 'static.npz'))
X, Y, t = tr['X'], tr['Y'], tr['times'].astype(str)
Xte, tte = te['X'], te['times'].astype(str)
dem = st['dem']
ts, tst = pd.to_datetime(t, format='%Y%m%d%H'), pd.to_datetime(tte, format='%Y%m%d%H')
h = ts.hour.values
R = {}

# ---------- 1. limpieza ----------
R['nan'] = dict(X=int(np.isnan(X).sum()), Y=int(np.isnan(Y).sum()), Xte=int(np.isnan(Xte).sum()))
R['horas_duplicadas'] = int(ts.duplicated().sum())
gaps = np.diff(ts.values).astype('timedelta64[h]').astype(int)
R['huecos_train'] = {str(ts[i]): int(g) for i, g in enumerate(gaps) if g > 1}
R['huecos_test'] = {str(tst[i]): int(g) for i, g in enumerate(np.diff(tst.values).astype('timedelta64[h]').astype(int)) if g > 1}
same = [str(ts[i]) for i in range(1, len(Y)) if np.allclose(Y[i], Y[i - 1])]
R['frames_Y_repetidos'] = same
# frames atípicos: media del dominio muy lejos de la climatología de su hora (|z| > 5)
fm = Y.mean((1, 2))                                                  # (n,4)
cz = np.stack([fm[h == k].mean(0) for k in range(24)])[h]
sz = np.stack([fm[h == k].std(0) for k in range(24)])[h]
z = (fm - cz) / sz
R['frames_atipicos_z5'] = {str(ts[i]): np.round(z[i], 1).tolist() for i in np.where((np.abs(z) > 5).any(1))[0]}
R['rango_Y'] = {v: [float(np.percentile(Y[..., j], q)) for q in (0, 0.1, 50, 99.9, 100)] for j, v in enumerate(VARS)}

# ---------- 2. GFS interpolado a 3 km ----------
LAT, LON = np.meshgrid(st['wrf_lats'], st['wrf_lons'], indexing='ij')
gl, go = st['gfs_lats'], st['gfs_lons']
G = torch.tensor(np.stack([(LON - go[0]) / (go[-1] - go[0]) * 2 - 1, (LAT - gl[0]) / (gl[-1] - gl[0]) * 2 - 1], -1)[None], dtype=torch.float32)
def interp(A):
    x = torch.tensor(A).permute(0, 3, 1, 2)
    return F.grid_sample(x, G.expand(len(x), -1, -1, -1), align_corners=True).permute(0, 2, 3, 1).numpy()
I, Ite = interp(X), interp(Xte)

# ---------- 3. cambio de temporada: medias mensuales sobre el dominio WRF ----------
mon = pd.DataFrame({'mes': ts.month})
gfs_m = pd.DataFrame(I.mean((1, 2)), columns=VARS).groupby(ts.month).mean()
wrf_m = pd.DataFrame(Y.mean((1, 2)), columns=VARS).groupby(ts.month).mean()
gte_m = pd.DataFrame(Ite.mean((1, 2)), columns=VARS).groupby(tst.month).mean()
R['medias_mensuales'] = dict(gfs_train=gfs_m.round(2).to_dict(), wrf_train=wrf_m.round(2).to_dict(), gfs_test=gte_m.round(2).to_dict())
fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
for j, v in enumerate(VARS):
    ax[j].plot(gfs_m.index, gfs_m[v], 'o-', label='GFS train'); ax[j].plot(wrf_m.index, wrf_m[v], 's-', label='WRF train')
    ax[j].plot(gte_m.index, gte_m[v], 'o--', c='C0', alpha=.6, label='GFS test'); ax[j].set_title(v); ax[j].set_xlabel('mes')
ax[0].legend(); fig.suptitle('Medias mensuales en el dominio WRF: el test (jul–ago) es otra temporada'); fig.tight_layout()
fig.savefig(OUT / 'temporada.png', dpi=110); plt.close(fig)

# ---------- 4. desfase: correlación GFS(t+k) vs WRF(t), con anomalías horarias ----------
pos = {k: i for i, k in enumerate(ts)}
def anom(A):
    c = np.stack([A[h == k].mean(0) for k in range(24)]); return A - c[h]
Ya, Ia = anom(Y), anom(I)
lagcorr = {}
for k in range(-6, 7):
    idx = np.array([pos.get(tt + pd.Timedelta(hours=k), -1) for tt in ts]); ok = idx >= 0
    a, b = Ia[idx[ok]], Ya[ok]
    lagcorr[k] = [float(np.corrcoef(a[..., j].ravel(), b[..., j].ravel())[0, 1]) for j in range(4)]
R['correlacion_por_desfase'] = lagcorr
fig, ax = plt.subplots(figsize=(6, 3.5))
for j, v in enumerate(VARS): ax.plot(list(lagcorr), [lagcorr[k][j] for k in lagcorr], 'o-', label=v)
ax.axvline(0, c='gray', lw=.5); ax.set_xlabel('k (horas): GFS(t+k) vs WRF(t)'); ax.set_ylabel('correlación de anomalías')
ax.legend(); ax.set_title('¿Qué hora del GFS explica mejor al WRF?'); fig.tight_layout(); fig.savefig(OUT / 'desfase.png', dpi=110); plt.close(fig)

# ---------- 5. correlación por píxel (anomalías) y relación con la altura ----------
def corrmap(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A * B).sum(0) / np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
C = np.stack([corrmap(Ia[..., j], Ya[..., j]) for j in range(4)], -1)   # (70,35,4)
R['corr_pixel_media'] = dict(zip(VARS, C.mean((0, 1)).round(3).tolist()))
R['corr_pixel_vs_altura'] = dict(zip(VARS, [float(np.corrcoef(C[..., j].ravel(), dem.ravel())[0, 1]) for j in range(4)]))
fig, ax = plt.subplots(1, 5, figsize=(17, 5))
ax[0].imshow(dem, origin='lower', cmap='terrain'); ax[0].set_title('Relieve (m)')
for j, v in enumerate(VARS):
    im = ax[j + 1].imshow(C[..., j], origin='lower', cmap='RdYlGn', vmin=-.2, vmax=1); ax[j + 1].set_title(f'corr GFS↔WRF · {v}')
fig.colorbar(im, ax=ax[1:], shrink=.7); fig.suptitle('Dónde el GFS interpolado sigue al WRF (anomalías horarias)')
fig.savefig(OUT / 'correlacion_pixel.png', dpi=110, bbox_inches='tight'); plt.close(fig)

# ---------- 6. sesgo WRF − GFS interpolado (6 a. m. y 2 p. m. Lima = 11 y 19 UTC) ----------
B = Y - I
fig, ax = plt.subplots(2, 4, figsize=(15, 9))
for r, hh in enumerate((11, 19)):
    for j, v in enumerate(VARS):
        m = B[h == hh, ..., j].mean(0); lim = np.abs(m).max()
        im = ax[r, j].imshow(m, origin='lower', cmap='coolwarm', vmin=-lim, vmax=lim)
        ax[r, j].set_title(f'{v} · {hh - 5:02d}:00 Lima'); fig.colorbar(im, ax=ax[r, j], shrink=.6)
fig.suptitle('Sesgo medio WRF − GFS interpolado: lo que el relieve le agrega al GFS'); fig.tight_layout()
fig.savefig(OUT / 'sesgo_hora.png', dpi=110); plt.close(fig)
R['sesgo_T_vs_altura'] = float(np.corrcoef(B[..., 1].mean(0).ravel(), dem.ravel())[0, 1])
slope = np.polyfit(dem.ravel(), B[..., 1].mean(0).ravel(), 1)[0]
R['gradiente_T_K_por_km'] = round(float(slope * 1000), 2)

# ---------- 7. relaciones entre variables y con el relieve (WRF) ----------
spd = np.hypot(Y[..., 2], Y[..., 3])
R['corr_WRF'] = dict(
    RH_T=float(np.corrcoef(Y[..., 0].ravel(), Y[..., 1].ravel())[0, 1]),
    velocidad_altura=float(np.corrcoef(spd.mean(0).ravel(), dem.ravel())[0, 1]),
    T_altura=float(np.corrcoef(Y[..., 1].mean(0).ravel(), dem.ravel())[0, 1]),
    RH_altura=float(np.corrcoef(Y[..., 0].mean(0).ravel(), dem.ravel())[0, 1]))
# ciclo diurno del dominio
fig, ax = plt.subplots(1, 4, figsize=(16, 3.2))
for j, v in enumerate(VARS):
    ax[j].plot(range(24), [Y[h == k, ..., j].mean() for k in range(24)], 'o-', label='WRF')
    ax[j].plot(range(24), [I[h == k, ..., j].mean() for k in range(24)], 's--', label='GFS interp'); ax[j].set_title(v); ax[j].set_xlabel('hora UTC')
ax[0].legend(); fig.suptitle('Ciclo diurno (media del dominio)'); fig.tight_layout(); fig.savefig(OUT / 'ciclo_diurno.png', dpi=110); plt.close(fig)

json.dump(R, open(OUT / 'datos.json', 'w'), indent=1, ensure_ascii=False, default=float)
for k, v in R.items():
    s = json.dumps(v, ensure_ascii=False, default=float); print(f'{k}: {s[:400]}')
