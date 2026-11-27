"""
Experimentos BurrosMX: una corrida = un modelo. Resultados en train/runs/log.jsonl.

  python train/exp.py --name prueba --fold 0                     # valida en 16–30 jun (folds: 0, 1, 2)
  python train/exp.py --name final --full --epochs 15 --swa_from 8   # entrena con todo (sin val)
  python train/ensemble.py sub.csv final                         # submission (promedia si das varias corridas)
  python train/resumen.py                                        # tabla de resultados por configuración

Las opciones nuevas valen por defecto lo que hacía la versión anterior, salvo la purga
de 24 h alrededor de la validación (evita fuga por autocorrelación en los bordes).
"""
import argparse, json, time
from pathlib import Path

import sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modelo import coarse_feats, windows, hour, hour_feats, grid, interp, static_feats, build  # noqa: E402
p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--fold', type=int, default=0, help='0: 16–30 jun · 1: 1–15 jun · 2: 16–31 may')
p.add_argument('--full', action='store_true', help='entrena con todo el train (sin validación)')
p.add_argument('--base', default='clim', choices=['clim', 'interp'])  # sobre qué predice la anomalía
p.add_argument('--bias_days', type=int, default=0, help='sesgo de la base interp con los N días previos al periodo a predecir (0 = todo)')
p.add_argument('--recent_days', type=int, default=0, help='repite en cada época los N días más recientes...')
p.add_argument('--recent_rep', type=int, default=1, help='...esta cantidad de veces')
p.add_argument('--lags', type=int, nargs='+', default=[-2, -1, 0])     # horas GFS relativas a t
p.add_argument('--feats', nargs='*', default=[], choices=['wind', 'vap', 'daily', 'semana'], help='variables compuestas del GFS')
p.add_argument('--static_plus', action='store_true', help='más relieve: TPI 3/21, pendiente, orientación')
# el WRF se reinicia cada día a las 00 UTC desde el GFS (salto en Y a las 00 UTC, ver analysis/): la hora UTC es
# el tiempo de pronóstico de esa corrida y el GFS de las 00 UTC del día es su condición inicial
p.add_argument('--hour_onehot', action='store_true', help='hora UTC en one-hot (24) además de sin/cos')
p.add_argument('--init', action='store_true', help='agrega el GFS de las 00 UTC del mismo día (condición inicial del WRF)')
p.add_argument('--width', type=int, default=64)
p.add_argument('--enc_blocks', type=int, default=1, help='bloques conv del encoder a 27 km')
p.add_argument('--attn', action='store_true', help='transformer en el nivel más bajo de la U-Net (18x9)')
p.add_argument('--xattn', action='store_true', help='atención cruzada: la grilla fina (18x9) mira a todo el GFS (17x13)')
p.add_argument('--drop', type=float, default=0.0)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--bs', type=int, default=32)
p.add_argument('--wd', type=float, default=1e-4)
p.add_argument('--loss', default='mse', choices=['mse', 'metric'])  # metric = promedio de RMSE_v / RMSE_clim_v, como Kaggle
p.add_argument('--vw', type=float, nargs=4, default=[1, 1, 1, 1], help='peso de RH T U V en la pérdida (especialistas)')
p.add_argument('--swa_from', type=int, default=0, help='promedia los pesos desde esta época (SWA); 0 = elegir la mejor época en val')
p.add_argument('--amp', action='store_true', help='precisión mixta fp16 (solo CUDA)')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--explain', action='store_true', help='importancia por permutación y errores por altura/hora en val')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT = ROOT / 'train' / 'runs' / a.name; OUT.mkdir(parents=True, exist_ok=True)
VARS = ['RH', 'T', 'U', 'V']
FOLDS = {0: ('2024061600', '2024070100'), 1: ('2024060100', '2024061600'), 2: ('2024051600', '2024060100')}

# ---------- data y partición temporal ----------
tr, te, st = (np.load(ROOT / 'data' / f) for f in ('train.npz', 'test.npz', 'static.npz'))
Xtr, Ytr, ttr = tr['X'], tr['Y'], tr['times'].astype(str)
Xte, tte = te['X'], te['times'].astype(str)
TS = pd.to_datetime(ttr, format='%Y%m%d%H')
htr, hte = hour(ttr), hour(tte)
if a.full:
    val = np.zeros(len(ttr), bool); trn = ~val; END = TS.max() + pd.Timedelta(hours=1)
else:
    lo, hi = (pd.Timestamp(pd.to_datetime(s, format='%Y%m%d%H')) for s in FOLDS[a.fold])
    val = np.asarray((TS >= lo) & (TS < hi))
    purge = np.asarray((TS >= lo - pd.Timedelta(hours=24)) & (TS < hi + pd.Timedelta(hours=24)))
    trn = ~purge; END = lo   # END = inicio del periodo a predecir


def recent(days):  # máscara de los `days` días de trn más cercanos (antes) al periodo a predecir
    return trn & np.asarray((TS < END) & (TS >= END - pd.Timedelta(days=days)))


# ---------- variables del GFS (27 km): ver modelo.py ----------
Ctr, Cte = coarse_feats(Xtr, ttr, a.feats), coarse_feats(Xte, tte, a.feats)
NC = Ctr.shape[-1]
NL = len(a.lags) + a.init          # horas del GFS que ve el modelo
GRID = grid(st)


# ---------- base: climatología por hora, o GFS interpolado + sesgo medio por (hora, píxel) ----------
if a.base == 'clim':
    clim = np.stack([Ytr[trn & (htr == k)].mean(0) for k in range(24)])
    Btr, Bte = clim[htr], clim[hte]
else:
    Itr, Ite = interp(Xtr, GRID), interp(Xte, GRID)
    D = Ytr - Itr
    all_b = np.stack([D[trn & (htr == k)].mean(0) for k in range(24)])
    if a.bias_days:  # la temporada cambia el sesgo (RH: +2.9 en marzo, −5.0 en junio): usar días recientes
        m = recent(a.bias_days)
        bias = np.stack([D[m & (htr == k)].mean(0) if (m & (htr == k)).any() else all_b[k] for k in range(24)])
    else:
        bias = all_b
    Btr, Bte = Itr + bias[htr], Ite + bias[hte]
scale = (Ytr[trn] - Btr[trn]).std((0, 1, 2))           # por variable
mu_x, sd_x = Ctr[trn].mean((0, 1, 2)), Ctr[trn].std((0, 1, 2)) + 1e-6
mu_y, sd_y = Ytr[trn].mean((0, 1, 2)), Ytr[trn].std((0, 1, 2))

S = static_feats(st['dem'], a.static_plus)   # relieve a 3 km


def t32(x): return torch.tensor(x, dtype=torch.float32, device=dev)


def prep(C, t, B):
    W = (windows(C, t, a.lags, a.init) - mu_x) / sd_x                           # (n,L,17,13,C)
    lo = t32(W).permute(0, 1, 4, 2, 3).flatten(1, 2)                             # (n,C*L,17,13): canal = lag*C + var
    hh = t32(hour_feats(t, a.hour_onehot))                                       # (n,2 ó 26)
    hi = t32((B - mu_y) / sd_y).permute(0, 3, 1, 2)                              # base normalizada (n,4,70,35)
    return lo, hh, hi


LOtr, HHtr, HItr = prep(Ctr, ttr, Btr)
LOte, HHte, HIte = prep(Cte, tte, Bte)
Btr_t, Bte_t, SC = t32(Btr), t32(Bte), t32(scale)
Ytr_t = t32(Ytr)
use_amp = a.amp and dev == 'cuda'


def predict(model, LO, HH, HI, B, bs=128):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast('cuda', torch.float16, enabled=use_amp):
        for i in range(0, len(LO), bs):
            out.append((B[i:i + bs] + model(LO[i:i + bs], HH[i:i + bs], HI[i:i + bs]).float() * SC).cpu().numpy())
    P = np.concatenate(out); P[..., 0] = P[..., 0].clip(0, 100)
    return P


# score local: misma fórmula que Kaggle, con la climatología (de trn) como referencia en val
clim_trn = np.stack([Ytr[trn & (htr == k)].mean(0) for k in range(24)])
if val.any():
    ref = np.sqrt(((clim_trn[htr[val]] - Ytr[val]) ** 2).mean((0, 1, 2)))


def score(P, Y):
    rmse = np.sqrt(((P - Y) ** 2).mean((0, 1, 2)))
    return float(1 - (rmse / ref).mean()), dict(zip(VARS, rmse.round(3).tolist()))


if val.any():
    print('base sola en val:', score(Btr[val], Ytr[val]))

model = build(vars(a), st, S).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
REF = t32(np.sqrt(((clim_trn[htr[trn]] - Ytr[trn]) ** 2).mean((0, 1, 2))))  # RMSE de la climatología en trn


VW = t32(np.array(a.vw) / np.mean(a.vw))


def loss_fn(pred, y):
    if a.loss == 'mse': return ((pred - y).pow(2).mean((0, 1, 2)) * VW).mean()
    return (((pred - y) * SC).pow(2).mean((0, 1, 2)).add(1e-8).sqrt() / REF * VW).mean()


itr = np.where(trn)[0]
if a.recent_rep > 1 and a.recent_days:  # sobremuestrea lo más parecido (en temporada) al periodo a predecir
    itr = np.concatenate([itr] + [np.where(recent(a.recent_days))[0]] * (a.recent_rep - 1))
steps = a.epochs * int(np.ceil(len(itr) / a.bs))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
swa = torch.optim.swa_utils.AveragedModel(model) if a.swa_from else None
vi = torch.tensor(np.where(val)[0], device=dev)
best, best_ep, best_state, t0 = -9e9, -1, None, time.time()
for ep in range(a.epochs):
    model.train(); np.random.shuffle(itr)
    for i in range(0, len(itr), a.bs):
        b = torch.tensor(itr[i:i + a.bs], device=dev)
        y = (Ytr_t[b] - Btr_t[b]) / SC
        with torch.autocast('cuda', torch.float16, enabled=use_amp):
            pred = model(LOtr[b], HHtr[b], HItr[b])
        loss = loss_fn(pred.float(), y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    if val.any():
        sc, pv = score(predict(model, LOtr[vi], HHtr[vi], HItr[vi], Btr_t[vi]), Ytr[val])
        if sc > best: best, best_ep, best_state = sc, ep + 1, {k: v.clone() for k, v in model.state_dict().items()}
        print(f'ep {ep + 1:02d} · val {sc:.4f} · {pv} · {time.time() - t0:.0f}s', flush=True)
    if swa and ep + 1 >= a.swa_from: swa.update_parameters(model)
if swa: model = swa.module                       # pesos promediados: sin mirar val, sirve igual en --full
elif best_state: model.load_state_dict(best_state)  # ponytail: elegir la mejor época con val es optimista
torch.save(model.state_dict(), OUT / 'model.pt')
# lo que el producto necesita para reproducir este modelo con el GFS de Open-Meteo (ver app/modelo.py)
np.savez(OUT / 'stats.npz', mu_x=mu_x, sd_x=sd_x, mu_y=mu_y, sd_y=sd_y, scale=scale, static=S,
         base=clim if a.base == 'clim' else bias)   # (24,70,35,4): climatología o sesgo por hora
json.dump(vars(a), open(OUT / 'args.json', 'w'))
np.save(OUT / 'test_pred.npy', predict(model, LOte, HHte, HIte, Bte_t))   # submission: train/ensemble.py

row = dict(name=a.name, args=vars(a), best_val=round(best, 4), best_epoch=best_ep, minutes=round((time.time() - t0) / 60, 1))
if val.any():
    Pv = predict(model, LOtr[vi], HHtr[vi], HItr[vi], Btr_t[vi]); np.save(OUT / 'val_pred.npy', Pv)
    sc, row['per_var_rmse'] = score(Pv, Ytr[val]); row['val'] = round(sc, 4)   # score del modelo guardado

# ---------- explicabilidad (sobre val) ----------
if a.explain and val.any():
    rng = np.random.default_rng(0); perm = torch.tensor(rng.permutation(len(vi)), device=dev)
    LOv, HHv, HIv, Bv = LOtr[vi], HHtr[vi], HItr[vi], Btr_t[vi]
    base_sc = score(predict(model, LOv, HHv, HIv, Bv), Ytr[val])[0]
    def drop(LO=LOv, HH=HHv, HI=HIv):
        return round(base_sc - score(predict(model, LO, HH, HI, Bv), Ytr[val])[0], 4)
    names = VARS + (['vel', 'dir_u', 'dir_v'] if 'wind' in a.feats else []) + (['e', 'Td'] if 'vap' in a.feats else []) \
        + ([f'{v}_24h' for v in VARS] if 'daily' in a.feats else []) \
        + ([f'{v}_semana' for v in VARS] + [f'{v}_tendencia' for v in VARS] if 'semana' in a.feats else [])
    imp = {}
    for j, n in enumerate(names):      # una variable del GFS, en todas sus horas
        L = LOv.clone(); ch = [l * NC + j for l in range(NL)]; L[:, ch] = LOv[perm][:, ch]; imp[f'GFS {n}'] = drop(LO=L)
    for l, lag in enumerate([f't{x:+d}' for x in a.lags] + (['00 UTC del día'] if a.init else [])):   # una hora del GFS
        L = LOv.clone(); ch = list(range(l * NC, (l + 1) * NC)); L[:, ch] = LOv[perm][:, ch]; imp[f'GFS hora {lag}'] = drop(LO=L)
    imp['hora del día'] = drop(HH=HHv[perm])
    for j, v in enumerate(VARS):       # canal de la base a 3 km
        Hh = HIv.clone(); Hh[:, j] = HIv[perm][:, j]; imp[f'base {v}'] = drop(HI=Hh)
    S0 = model.S.clone()
    snames = ['altura', 'dx', 'dy', 'TPI9'] + (['TPI3', 'TPI21', 'pendiente', 'orient_x', 'orient_y'] if a.static_plus else [])
    for k, n in enumerate(snames):     # relieve: se anula el canal (ablación)
        model.S.copy_(S0); model.S[:, k] = 0; imp[f'relieve {n}'] = drop()
    model.S.copy_(S0)
    row['importancia'] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    # errores por píxel, por banda de altura y por hora
    err = (Pv - Ytr[val]) ** 2
    rel = np.sqrt(err.mean(0)) / ref                                            # (70,35,4) error relativo a la climatología
    dem = st["dem"]; bands = [0, 1000, 2000, 3000, 4000, 6000]
    row['error_por_altura'] = {f'{bands[i]}-{bands[i + 1]} m': np.round(np.sqrt(err[:, (dem >= bands[i]) & (dem < bands[i + 1])].mean((0, 1))) / ref, 3).tolist()
                               for i in range(len(bands) - 1)}
    hv = htr[val]
    row['error_por_hora_utc'] = {int(k): np.round(np.sqrt(err[hv == k].mean((0, 1, 2))) / ref, 3).tolist() for k in np.unique(hv)}
    np.save(OUT / 'error_rel_pixel.npy', rel)

with open(ROOT / 'train' / 'runs' / 'log.jsonl', 'a') as f: f.write(json.dumps(row) + '\n')
print('RESULTADO', json.dumps(row))
