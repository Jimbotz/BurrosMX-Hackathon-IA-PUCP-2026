"""
Experimentos BurrosMX: una corrida = un modelo, validado igual que el starter
(últimas 2 semanas de junio). Resultados en train/runs/log.jsonl.

  python train/exp.py --name unet_clim
  python train/exp.py --name unet_interp_fut --base interp --lags -2 -1 0 1 2
  python train/exp.py --name final --full --swa_from 8 --epochs 15   # entrena con todo (sin val)
  python train/ensemble.py sub.csv final        # submission (promedia si das varias corridas)
"""
import argparse, json, time
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--base', default='clim', choices=['clim', 'interp'])  # sobre qué predice la anomalía
p.add_argument('--lags', type=int, nargs='+', default=[-2, -1, 0])     # horas GFS relativas a t
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--width', type=int, default=64)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--bs', type=int, default=32)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--wd', type=float, default=1e-4)
p.add_argument('--loss', default='mse', choices=['mse', 'metric'])  # metric = promedio de RMSE_v / RMSE_clim_v, como Kaggle
p.add_argument('--swa_from', type=int, default=0, help='promedia los pesos desde esta época (SWA); 0 = elegir la mejor época en val')
p.add_argument('--full', action='store_true', help='entrena con todo el train (sin validación)')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT = ROOT / 'train' / 'runs' / a.name; OUT.mkdir(parents=True, exist_ok=True)
VARS = ['RH', 'T', 'U', 'V']

# ---------- data ----------
tr, te, st = (np.load(ROOT / 'data' / f) for f in ('train.npz', 'test.npz', 'static.npz'))
Xtr, Ytr, ttr = tr['X'], tr['Y'], tr['times'].astype(str)
Xte, tte = te['X'], te['times'].astype(str)
hour = lambda t: np.array([int(k[-2:]) for k in t])
val = np.zeros(len(ttr), bool) if a.full else ttr >= '2024061600'
trn = ~val


def windows(X, t, lags):  # (n, L, 17, 13, 4); si falta una hora, usa la hora t (como el starter)
    ts = pd.to_datetime(t, format='%Y%m%d%H'); pos = {k: i for i, k in enumerate(ts)}
    out = np.empty((len(X), len(lags)) + X.shape[1:], np.float32)
    for i, k in enumerate(ts):
        for j, l in enumerate(lags):
            out[i, j] = X[pos.get(k + pd.Timedelta(hours=l), i)]
    return out


# grilla WRF dentro de la grilla GFS, para grid_sample (igual que el starter)
LAT, LON = np.meshgrid(st['wrf_lats'], st['wrf_lons'], indexing='ij')
glat, glon = st['gfs_lats'], st['gfs_lons']
gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
GRID = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32)


def interp(X):  # GFS (n,17,13,4) -> (n,70,35,4) bilineal, en CPU
    x = torch.tensor(X).permute(0, 3, 1, 2)
    return F.grid_sample(x, GRID.expand(len(x), -1, -1, -1), align_corners=True).permute(0, 2, 3, 1).numpy()


# base: climatología por hora, o GFS interpolado + sesgo medio por (hora, píxel)
htr, hte = hour(ttr), hour(tte)
if a.base == 'clim':
    clim = np.stack([Ytr[trn & (htr == k)].mean(0) for k in range(24)])
    Btr, Bte = clim[htr], clim[hte]
else:
    Itr, Ite = interp(Xtr), interp(Xte)
    bias = np.stack([(Ytr - Itr)[trn & (htr == k)].mean(0) for k in range(24)])
    Btr, Bte = Itr + bias[htr], Ite + bias[hte]
scale = (Ytr[trn] - Btr[trn]).std((0, 1, 2))           # por variable
mu_x, sd_x = Xtr[trn].mean((0, 1, 2)), Xtr[trn].std((0, 1, 2))
mu_y, sd_y = Ytr[trn].mean((0, 1, 2)), Ytr[trn].std((0, 1, 2))

# estáticos a 3 km: relieve, pendientes y relieve local
dem = st['dem'].astype(np.float32)
k = 9; dp = np.pad(dem, k // 2, mode='edge')
smooth = np.mean([dp[i:i + 70, j:j + 35] for i in range(k) for j in range(k)], 0)
dy, dx = np.gradient(dem)
S = np.stack([dem, dx, dy, dem - smooth]); S = (S - S.mean((1, 2), keepdims=True)) / S.std((1, 2), keepdims=True)
STATIC = torch.tensor(S[None], dtype=torch.float32, device=dev)
GRID = GRID.to(dev)


def t32(x): return torch.tensor(x, dtype=torch.float32, device=dev)


def prep(X, t, B):
    W = (windows(X, t, a.lags) - mu_x) / sd_x                                   # (n,L,17,13,4)
    lo = t32(W).permute(0, 1, 4, 2, 3).flatten(1, 2)                             # (n,4L,17,13)
    ang = 2 * np.pi * hour(t) / 24
    hh = t32(np.stack([np.sin(ang), np.cos(ang)], 1))                            # (n,2)
    hi = t32((B - mu_y) / sd_y).permute(0, 3, 1, 2)                              # base normalizada (n,4,70,35)
    return lo, hh, hi


# ---------- modelo ----------
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())


class UNet(nn.Module):
    """Encoder a 27 km -> grid_sample a 3 km -> U-Net de 3 niveles con relieve y la base."""
    def __init__(s, L, w):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv2d(4 * L + 2, w, 3, padding=1), nn.GELU(), block(w, w))
        s.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        s.d1, s.d2, s.d3 = block(w + 4 + 4 + 8 + 2, w), block(w, 2 * w), block(2 * w, 4 * w)
        s.u2, s.u1 = block(6 * w, 2 * w), block(3 * w, w)
        s.out = nn.Conv2d(w, 4, 1)

    def forward(s, lo, hh, hi):
        B = lo.shape[0]
        H = hh[..., None, None]
        f = s.enc(torch.cat([lo, H.expand(-1, -1, 17, 13)], 1))
        f = F.grid_sample(f, GRID.expand(B, -1, -1, -1), align_corners=True)
        x = torch.cat([f, hi, STATIC.expand(B, -1, -1, -1), s.static.expand(B, -1, -1, -1), H.expand(-1, -1, 70, 35)], 1)
        e1 = s.d1(x)
        e2 = s.d2(F.avg_pool2d(e1, 2, ceil_mode=True))
        e3 = s.d3(F.avg_pool2d(e2, 2, ceil_mode=True))
        up = lambda z, ref: F.interpolate(z, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        u2 = s.u2(torch.cat([up(e3, e2), e2], 1))
        u1 = s.u1(torch.cat([up(u2, e1), e1], 1))
        return s.out(u1).permute(0, 2, 3, 1)  # anomalía normalizada (B,70,35,4)


LOtr, HHtr, HItr = prep(Xtr, ttr, Btr)
LOte, HHte, HIte = prep(Xte, tte, Bte)
Btr_t, Bte_t, SC = t32(Btr), t32(Bte), t32(scale)
Ytr_t = t32(Ytr)


def predict(model, LO, HH, HI, B, bs=128):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(LO), bs):
            out.append((B[i:i + bs] + model(LO[i:i + bs], HH[i:i + bs], HI[i:i + bs]) * SC).cpu().numpy())
    P = np.concatenate(out); P[..., 0] = P[..., 0].clip(0, 100)
    return P


# score local: misma fórmula que Kaggle, con la climatología (de trn) como referencia en val
if val.any():
    clim_ref = np.stack([Ytr[trn & (htr == k)].mean(0) for k in range(24)])
    ref = np.sqrt(((clim_ref[htr[val]] - Ytr[val]) ** 2).mean((0, 1, 2)))


def score(P, Y):
    rmse = np.sqrt(((P - Y) ** 2).mean((0, 1, 2)))
    return float(1 - (rmse / ref).mean()), dict(zip(VARS, rmse.round(3).tolist()))


if val.any():
    print('base sola en val:', score(Btr[val], Ytr[val]))

model = UNet(len(a.lags), a.width).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
clim_trn = np.stack([Ytr[trn & (htr == k)].mean(0) for k in range(24)])
REF = t32(np.sqrt(((clim_trn[htr[trn]] - Ytr[trn]) ** 2).mean((0, 1, 2))))  # RMSE de la climatología en trn


def loss_fn(pred, y):
    if a.loss == 'mse': return F.mse_loss(pred, y)
    return (((pred - y) * SC).pow(2).mean((0, 1, 2)).add(1e-8).sqrt() / REF).mean()
itr = np.where(trn)[0]
steps = a.epochs * int(np.ceil(len(itr) / a.bs))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
swa = torch.optim.swa_utils.AveragedModel(model) if a.swa_from else None
best, best_ep, best_state, t0 = -9e9, -1, None, time.time()
for ep in range(a.epochs):
    model.train(); np.random.shuffle(itr)
    for i in range(0, len(itr), a.bs):
        b = torch.tensor(itr[i:i + a.bs], device=dev)
        y = (Ytr_t[b] - Btr_t[b]) / SC
        loss = loss_fn(model(LOtr[b], HHtr[b], HItr[b]), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if val.any():
        vi = torch.tensor(np.where(val)[0], device=dev)
        sc, pv = score(predict(model, LOtr[vi], HHtr[vi], HItr[vi], Btr_t[vi]), Ytr[val])
        if sc > best: best, best_ep, best_state = sc, ep + 1, {k: v.clone() for k, v in model.state_dict().items()}
        print(f'ep {ep + 1:02d} · val {sc:.4f} · {pv} · {time.time() - t0:.0f}s', flush=True)
    if swa and ep + 1 >= a.swa_from: swa.update_parameters(model)
if swa: model = swa.module                       # pesos promediados: sin mirar val, sirve igual en --full
elif best_state: model.load_state_dict(best_state)  # ponytail: elegir la mejor época con val es optimista
torch.save(model.state_dict(), OUT / 'model.pt')
np.save(OUT / 'test_pred.npy', predict(model, LOte, HHte, HIte, Bte_t))   # submission: train/ensemble.py

row = dict(name=a.name, args=vars(a), best_val=round(best, 4), best_epoch=best_ep, minutes=round((time.time() - t0) / 60, 1))
if val.any():
    vi = torch.tensor(np.where(val)[0], device=dev)
    Pv = predict(model, LOtr[vi], HHtr[vi], HItr[vi], Btr_t[vi]); np.save(OUT / 'val_pred.npy', Pv)
    sc, row['per_var_rmse'] = score(Pv, Ytr[val]); row['val'] = round(sc, 4)   # score del modelo guardado
with open(ROOT / 'train' / 'runs' / 'log.jsonl', 'a') as f: f.write(json.dumps(row) + '\n')
print('RESULTADO', json.dumps(row))
