"""
Tablas de resultados (markdown) a partir de train/runs/log.jsonl (GPU, protocolo de 3 folds) y
train/runs/log_mac.jsonl (primeras pruebas en la Mac, protocolo anterior). Las usa docs/RESULTADOS.md.

  python analysis/tablas_resultados.py > /tmp/tablas.md
"""
import json, re
from collections import defaultdict
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'train' / 'runs'
VARS = ['RH', 'T', 'U', 'V']
KAGGLE = np.array([25.360, 2.0652, 2.6776, 2.0876])        # denominadores de la métrica de Kaggle (test)
FOLDS = {0: ('2024061600', '2024070100'), 1: ('2024060100', '2024061600'), 2: ('2024051600', '2024060100')}

tr = np.load(ROOT / 'data' / 'train.npz'); t, Y = tr['times'].astype(str), tr['Y']
TS = pd.to_datetime(t, format='%Y%m%d%H'); h = TS.hour.values


def ref(f, purge=True):  # RMSE de la climatología en el fold (misma cuenta que exp.py)
    lo, hi = (pd.to_datetime(s, format='%Y%m%d%H') for s in FOLDS[f])
    val = np.asarray((TS >= lo) & (TS < hi))
    trn = ~np.asarray((TS >= lo - pd.Timedelta(hours=24)) & (TS < hi + pd.Timedelta(hours=24))) if purge else ~val
    clim = np.stack([Y[trn & (h == k)].mean(0) for k in range(24)])
    return np.sqrt(((clim[h[val]] - Y[val]) ** 2).mean((0, 1, 2)))


REF = {f: ref(f) for f in FOLDS}
REF_MAC = ref(0, purge=False)


def rm(r): return np.array([r['per_var_rmse'][v] for v in VARS])
def lb(r): return 1 - (rm(r) / KAGGLE).mean()                   # leaderboard estimado con el RMSE del fold 0
def f(x, d=4): return '—' if x is None else f'{x:.{d}f}'


def describe(a):
    if 'k' in a: return f"k={a['k']} · situación: " + '+'.join(a['ctx']) + f" · ±{a['h_win']} h · {a['ncomp']} comp. PCA"
    s = [a.get('base', 'clim')]
    lags = a.get('lags', [-2, -1, 0])
    s.append('t' + (f'{min(lags):+d}…{max(lags):+d}' if lags == list(range(min(lags), max(lags) + 1)) else ','.join(f'{x:+d}' for x in lags)))
    if a.get('width', 64) != 64: s.append(f"ancho {a['width']}")
    if a.get('bias_days'): s.append(f"sesgo {a['bias_days']}d")
    if a.get('recent_days'): s.append(f"recientes {a['recent_days']}d×{a.get('recent_rep')}")
    if a.get('feats'): s.append('+' + '+'.join(a['feats']))
    if a.get('static_plus'): s.append('+relieve')
    if a.get('enc_blocks', 1) > 1: s.append(f"enc×{a['enc_blocks']}")
    if a.get('attn'): s.append('atención bottleneck')
    if a.get('xattn'): s.append('atención cruzada')
    if a.get('tattn'): s.append(f"atención temporal {a['tattn']}")
    if a.get('postnorm'): s.append('post-norm')
    if a.get('hour_onehot'): s.append('hora one-hot')
    if a.get('init'): s.append('GFS 00 UTC')
    if a.get('drop'): s.append(f"dropout {a['drop']}")
    if a.get('vw', [1, 1, 1, 1]) != [1, 1, 1, 1]: s.append('pesos ' + '/'.join(str(x) for x in a['vw']))
    if a.get('lr', 2e-3) != 2e-3: s.append(f"lr {a['lr']}")
    if a.get('epochs', 30) != 15: s.append(f"{a['epochs']} ép")
    if a.get('wd', 1e-4) != 1e-4: s.append(f"wd {a['wd']}")
    if a.get('loss', 'mse') != 'metric': s.append(f"pérdida {a.get('loss', 'mse')}")
    if not a.get('amp'): s.append('fp32')
    if a.get('seed', 0): s.append(f"semilla {a['seed']}")
    return ' · '.join(s)


rows = [json.loads(l) for l in open(RUNS / 'log.jsonl')]
g = defaultdict(dict)
for r in rows:
    if 'val' in r and not r['name'].startswith('full_'):
        g[re.sub(r'_f\d$', '', r['name'])][r['args'].get('fold', 0)] = r

una_sola = {k: v for k, v in g.items() if len(v) < 3}          # corridas de un solo fold (explicabilidad, benchmark)
redes = {k: v for k, v in g.items() if not k.startswith('knn') and len(v) == 3}
knn = {k: v for k, v in g.items() if k.startswith('knn') and len(v) == 3}


def tabla_principal(grupo, titulo):
    print(f'\n#### {titulo}\n')
    print('| # | Configuración | Qué es | f0 | f1 | f2 | **Media** | RH | T | U | V | LB estimado | Media sin SWA (mejor época) | min |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    orden = sorted(grupo.items(), key=lambda kv: -np.mean([r['val'] for r in kv[1].values()]))
    for i, (k, fs) in enumerate(orden, 1):
        sc = [fs[x]['val'] if x in fs else None for x in range(3)]
        m = np.mean([x for x in sc if x is not None])
        rat = np.mean([rm(fs[x]) / REF[x] for x in fs], 0)        # RMSE / RMSE clim, media de folds
        bv = np.mean([fs[x]['best_val'] for x in fs]) if all('best_val' in fs[x] and fs[x]['best_val'] > -1 for x in fs) else None
        a = next(iter(fs.values()))['args']
        print(f"| {i} | `{k}` | {describe(a)} | " + ' | '.join(f(x) for x in sc) + f' | **{m:.4f}** | '
              + ' | '.join(f'{x:.3f}' for x in rat) + f" | {f(lb(fs[0]), 3) if 0 in fs else '—'} | {f(bv)} | "
              + f"{sum(r['minutes'] for r in fs.values()):.1f} |")


def tabla_rmse(grupo):
    print('\n| Configuración | Fold | Score | RMSE RH (%) | RMSE T (K) | RMSE U (m/s) | RMSE V (m/s) |')
    print('|---|---|---|---|---|---|---|')
    for k, fs in sorted(grupo.items()):
        for x in sorted(fs):
            r = rm(fs[x]); print(f"| `{k}` | f{x} | {fs[x]['val']:.4f} | {r[0]:.2f} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} |")


print('<!-- generado por analysis/tablas_resultados.py -->')
print('\n### Referencia: RMSE de la climatología por fold (denominadores locales)\n')
print('| Fold | Periodo | RH (%) | T (K) | U (m/s) | V (m/s) |\n|---|---|---|---|---|---|')
for x, (lo, hi) in FOLDS.items():
    print(f'| f{x} | {lo[:8]} → {hi[:8]} (excl.) | ' + ' | '.join(f'{v:.3f}' for v in REF[x]) + ' |')
print('| Kaggle (test, jul–ago) | 2 jul → 4 ago | ' + ' | '.join(f'{v:.4f}' for v in KAGGLE) + ' |')

tabla_principal(redes, 'Redes neuronales (U-Net y variantes), 3 folds')
tabla_principal(knn, 'Análogos (k vecinos, sin red), 3 folds')

print('\n#### Corridas de un solo fold (explicabilidad y benchmark de GPU; no comparables con la media de 3 folds)\n')
print('| Corrida | Qué es | Score f0 | RMSE RH | T | U | V | LB estimado |\n|---|---|---|---|---|---|---|---|')
for k, fs in una_sola.items():
    r = fs[0]; print(f"| `{k}` | {describe(r['args'])} | {r['val']:.4f} | " + ' | '.join(f'{x:.3f}' for x in rm(r)) + f' | {lb(r):.3f} |')

print('\n### RMSE absoluto por fold y variable (todas las corridas)')
tabla_rmse({**redes, **knn})

# primeras pruebas en la Mac (protocolo anterior: fold 0 sin purga)
mac = [json.loads(l) for l in open(RUNS / 'log_mac.jsonl')]
print('\n### Primeras pruebas en la Mac (protocolo anterior: solo fold 0, sin purga de 24 h)\n')
print('| Corrida | Qué es | Score (mejor época) | Score honesto (SWA) | RMSE RH | T | U | V | LB estimado |\n|---|---|---|---|---|---|---|---|---|')
for r in mac:
    if r['args'].get('full'): continue
    print(f"| `{r['name']}` | {describe({**r['args'], 'amp': True})} | {r['best_val']:.4f} | {f(r.get('val'))} | "
          + ' | '.join(f'{x:.3f}' for x in rm(r)) + f' | {lb(r):.3f} |')

# explicabilidad
for r in rows:
    if 'importancia' not in r: continue
    print(f"\n### Explicabilidad · `{r['name']}` (fold 0, score {r['val']:.4f})\n")
    print(f"Modelo: {describe(r['args'])}\n")
    print('| Entrada | Caída del score al permutarla |\n|---|---|')
    for k, v in r['importancia'].items(): print(f'| {k} | {v:+.4f} |')
    print('\n| Altura | RH | T | U | V |\n|---|---|---|---|---|')
    for k, v in r['error_por_altura'].items(): print(f'| {k} | ' + ' | '.join(f'{x:.3f}' for x in v) + ' |')
    print('\n| Hora UTC (Lima) | RH | T | U | V |\n|---|---|---|---|---|')
    for k, v in sorted(r['error_por_hora_utc'].items(), key=lambda kv: int(kv[0])):
        print(f"| {int(k):02d} ({(int(k) - 5) % 24:02d}) | " + ' | '.join(f'{x:.3f}' for x in v) + ' |')

# pesos del ensamble final
ens = json.load(open(ROOT / 'models' / 'ensamble.json'))
tot = np.sum([v for v in ens.values()], 0)
print('\n### Ensamble final: pesos por variable (`models/ensamble.json`)\n')
print('Cada número es cuántas veces la selección greedy eligió ese modelo para esa variable; el peso efectivo es ese número dividido por el total de la columna.\n')
print('| Modelo (entrenado con todo el train) | RH | T | U | V |\n|---|---|---|---|---|')
for k, v in sorted(ens.items(), key=lambda kv: -sum(kv[1])):
    print(f'| `{k}` | ' + ' | '.join(f'{int(x)}' if x else '·' for x in v) + ' |')
print('| **Total** | ' + ' | '.join(f'**{int(x)}**' for x in tot) + ' |')

# análisis de datos
D = json.load(open(ROOT / 'docs' / 'analisis' / 'datos.json'))
print('\n### Datos: correlación de anomalías GFS(t+k) ↔ WRF(t), media del dominio\n')
print('| k (horas) | RH | T | U | V |\n|---|---|---|---|---|')
for k, v in D['correlacion_por_desfase'].items(): print(f'| {int(k):+d} | ' + ' | '.join(f'{x:.3f}' for x in v) + ' |')
print('\n### Datos: medias mensuales en el dominio WRF\n')
print('| Fuente | Variable | ' + ' | '.join(['feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago']) + ' |\n|---|---|' + '---|' * 7)
for fuente, nombre in (('gfs_train', 'GFS (train)'), ('wrf_train', 'WRF (train)'), ('gfs_test', 'GFS (test)')):
    for v in VARS:
        d = D['medias_mensuales'][fuente][v]
        print(f'| {nombre} | {v} | ' + ' | '.join(f"{d[m]:.2f}" if m in d else '' for m in ['2', '3', '4', '5', '6', '7', '8']) + ' |')
print('\n### Datos: otras métricas\n')
print('| Métrica | RH | T | U | V |\n|---|---|---|---|---|')
print('| Correlación media por píxel GFS↔WRF (anomalías) | ' + ' | '.join(f"{D['corr_pixel_media'][v]:.3f}" for v in VARS) + ' |')
print('| Correlación de esa correlación con la altura | ' + ' | '.join(f"{D['corr_pixel_vs_altura'][v]:.3f}" for v in VARS) + ' |')
c = D['corr_WRF']
print(f"\nRelaciones en el WRF: T vs altura {c['T_altura']:.3f} · RH vs altura {c['RH_altura']:.3f} · RH vs T {c['RH_T']:.3f} · "
      f"velocidad del viento vs altura {c['velocidad_altura']:.3f}. Sesgo de T (WRF − GFS) vs altura: {D['sesgo_T_vs_altura']:.3f} "
      f"({D['gradiente_T_K_por_km']} K/km).")
print(f"\nLimpieza: valores faltantes {D['nan']} · horas duplicadas {D['horas_duplicadas']} · frames del WRF repetidos "
      f"{len(D['frames_Y_repetidos'])} · horas atípicas (|z|>5) {list(D['frames_atipicos_z5'])}.")
print('\nHuecos en el train (inicio → horas):', ', '.join(f'{k[:13]} → {v} h' for k, v in D['huecos_train'].items()),
      '· en el test:', ', '.join(f'{k[:13]} → {v} h' for k, v in D['huecos_test'].items()))
