"""
Modelo BurrosMX (fuente única): variables del GFS, base y U-Net.
Lo usan train/exp.py para entrenar y app/ para el producto, así el producto corre exactamente
el mismo modelo que la submission de Kaggle.

  from modelo import Ensamble
  ens = Ensamble.desde_json('models/ensamble.json')      # varias corridas con pesos por variable
  Y = ens.predecir(X, times)                               # X: (H,17,13,4) GFS · times 'YYYYMMDDHH' UTC -> (H,70,35,4)
"""
import json
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

VARS = ['RH', 'T', 'U', 'V']
ROOT = Path(__file__).resolve().parent


def hour(t): return np.array([int(k[-2:]) for k in t])


def coarse_feats(X, t, feats):
    """GFS (n,17,13,4) -> (n,17,13,C): las 4 variables + compuestas opcionales."""
    RH, T, U, V = (X[..., j] for j in range(4))
    out = [X]
    if 'wind' in feats:                                     # velocidad y dirección (vector unitario)
        spd = np.hypot(U, V); out += [spd[..., None], (U / (spd + .1))[..., None], (V / (spd + .1))[..., None]]
    if 'vap' in feats:                                      # presión de vapor (hPa) y punto de rocío (°C)
        Tc = T - 273.15; e = np.clip(RH, 1, 100) / 100 * 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
        ln = np.log(e / 6.112); out += [e[..., None], (243.5 * ln / (17.67 - ln))[..., None]]
    if 'daily' in feats:                                    # media móvil de ±12 h: el "día" alrededor de t
        ts = pd.to_datetime(t, format='%Y%m%d%H'); pos = {k: i for i, k in enumerate(ts)}
        acc, cnt = np.zeros_like(X), np.zeros((len(X), 1, 1, 1), np.float32)
        for o in range(-12, 13):
            idx = np.array([pos.get(k + pd.Timedelta(hours=o), -1) for k in ts]); ok = idx >= 0
            acc[ok] += X[idx[ok]]; cnt[ok] += 1
        out += [acc / cnt]
    if 'semana' in feats:                                   # media de ±3.5 días y tendencia (próximos − pasados)
        ts = pd.to_datetime(t, format='%Y%m%d%H'); pos = {k: i for i, k in enumerate(ts)}
        def media(lo, hi):
            acc, cnt = np.zeros_like(X), np.zeros((len(X), 1, 1, 1), np.float32)
            for o in range(lo, hi + 1):
                idx = np.array([pos.get(k + pd.Timedelta(hours=o), -1) for k in ts]); ok = idx >= 0
                acc[ok] += X[idx[ok]]; cnt[ok] += 1
            return acc / np.maximum(cnt, 1)
        pas, fut = media(-84, 0), media(0, 84)
        out += [(pas + fut) / 2, fut - pas]
    return np.concatenate(out, -1).astype(np.float32)


def windows(X, t, lags, init=False):
    """(n, L[+1], 17, 13, C). Si falta una hora usa la hora t; con init agrega las 00 UTC del mismo día
    (si faltan, la primera hora disponible de ese día)."""
    ts = pd.to_datetime(t, format='%Y%m%d%H'); pos = {k: i for i, k in enumerate(ts)}
    idx = [[pos.get(k + pd.Timedelta(hours=l), i) for l in lags] for i, k in enumerate(ts)]
    if init:
        first = {}
        for i, k in enumerate(ts): first.setdefault(k.normalize(), i)
        idx = [r + [pos.get(k.normalize(), first[k.normalize()])] for r, k in zip(idx, ts)]
    return X[np.array(idx)]


def hour_feats(t, onehot=False):
    ang = 2 * np.pi * hour(t) / 24
    return np.stack([np.sin(ang), np.cos(ang)] + (list(np.eye(24)[hour(t)].T) if onehot else []), 1).astype(np.float32)


def grid(st):
    """Posición de cada celda WRF (70x35) dentro de la grilla GFS (17x13), para grid_sample."""
    LAT, LON = np.meshgrid(st['wrf_lats'], st['wrf_lons'], indexing='ij')
    glat, glon = st['gfs_lats'], st['gfs_lons']
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    return torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32)


def interp(X, G):
    """GFS (n,17,13,4) -> (n,70,35,4) bilineal, en CPU."""
    x = torch.tensor(X).permute(0, 3, 1, 2)
    return F.grid_sample(x, G.expand(len(x), -1, -1, -1), align_corners=True).permute(0, 2, 3, 1).numpy()


def static_feats(dem, plus=False):
    """Relieve a 3 km normalizado: altura, gradientes, TPI (y más si plus)."""
    dem = dem.astype(np.float32)
    def box(z, k):
        zp = np.pad(z, k // 2, mode='edge'); return np.mean([zp[i:i + 70, j:j + 35] for i in range(k) for j in range(k)], 0)
    dy, dx = np.gradient(dem)
    S = [dem, dx, dy, dem - box(dem, 9)]
    if plus:
        slope = np.hypot(dx, dy) + 1e-3
        S += [dem - box(dem, 3), dem - box(dem, 21), slope, dx / slope, dy / slope]
    S = np.stack(S)
    return ((S - S.mean((1, 2), keepdims=True)) / S.std((1, 2), keepdims=True)).astype(np.float32)


def block(i, o, drop=0.0):
    layers = [nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
              nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU()]
    return nn.Sequential(*layers, nn.Dropout2d(drop)) if drop else nn.Sequential(*layers)


class UNet(nn.Module):
    """Encoder a 27 km -> grid_sample a 3 km -> U-Net de 3 niveles con relieve, base y hora.
    Predice la anomalía normalizada respecto de la base."""
    def __init__(self, cin, nh, G, S, w=64, enc_blocks=1, attn=False, drop=0.0, xattn=False,
                 tattn=None, postnorm=False):
        super().__init__()
        self.register_buffer('G', G, persistent=False)          # no van al state_dict: se recalculan
        self.register_buffer('S', torch.as_tensor(S)[None], persistent=False)
        self.attn = attn
        self.enc = nn.Sequential(nn.Conv2d((tattn['d'] if tattn else cin) + nh, w, 3, padding=1), nn.GELU(), *[block(w, w, drop) for _ in range(enc_blocks)])
        self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        self.d1, self.d2, self.d3 = block(w + 4 + S.shape[0] + 8 + nh, w, drop), block(w, 2 * w, drop), block(2 * w, 4 * w, drop)
        if attn:   # 18x9 = 162 tokens: contexto de todo el dominio
            self.pos = nn.Parameter(torch.zeros(1, 18 * 9, 4 * w))
            self.tf = nn.TransformerEncoderLayer(4 * w, 8, 8 * w, dropout=drop, batch_first=True, norm_first=not postnorm)
        self.u2, self.u1 = block(6 * w, 2 * w, drop), block(3 * w, w, drop)
        self.out = nn.Conv2d(w, 4, 1)
        self.xattn = xattn
        if xattn:   # cada zona de la grilla fina (18x9) mira a las 17x13 celdas del GFS: ¿de qué flujo depende su viento?
            self.kv = nn.Linear(w, 4 * w)
            self.cpos = nn.Parameter(torch.zeros(1, 17 * 13, 4 * w))
            self.qpos = nn.Parameter(torch.zeros(1, 18 * 9, 4 * w))
            self.xnorm = nn.LayerNorm(4 * w)
            self.xa = nn.MultiheadAttention(4 * w, 8, dropout=drop, batch_first=True)
            self.keep_attn = False; self.last_attn = None   # para explicabilidad: pesos (B, 162, 221)
        self.postnorm = postnorm
        self.tattn = tattn
        if tattn:   # atención en el TIEMPO: cada celda GFS ve su secuencia de horas; causal = solo horas <= t
            L, C, d = tattn['L'], tattn['C'], tattn['d']
            self.tin = nn.Linear(C, d)
            self.tpos = nn.Parameter(torch.zeros(1, L, d))
            self.ttf = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=drop, batch_first=True, norm_first=not postnorm)
            self.register_buffer('tmask', nn.Transformer.generate_square_subsequent_mask(L) if tattn['causal'] else None, persistent=False)

    def forward(self, lo, hh, hi):
        B = lo.shape[0]
        H = hh[..., None, None]
        if self.tattn:   # (B, L*C, 17, 13) -> secuencias (B*221, L, C) -> token de la hora t -> (B, d, 17, 13)
            L, C, d, i0 = self.tattn['L'], self.tattn['C'], self.tattn['d'], self.tattn['i0']
            seq = lo.reshape(B, L, C, 17 * 13).permute(0, 3, 1, 2).reshape(B * 221, L, C)
            z = self.ttf(self.tin(seq) + self.tpos, src_mask=self.tmask, is_causal=self.tmask is not None)
            lo = z[:, i0].reshape(B, 221, d).transpose(1, 2).reshape(B, d, 17, 13)
        fc = self.enc(torch.cat([lo, H.expand(-1, -1, 17, 13)], 1))
        f = F.grid_sample(fc, self.G.expand(B, -1, -1, -1).to(fc.dtype), align_corners=True)
        x = torch.cat([f, hi, self.S.expand(B, -1, -1, -1).to(f.dtype), self.static.expand(B, -1, -1, -1).to(f.dtype),
                       H.expand(-1, -1, 70, 35).to(f.dtype)], 1)
        e1 = self.d1(x)
        e2 = self.d2(F.avg_pool2d(e1, 2, ceil_mode=True))
        e3 = self.d3(F.avg_pool2d(e2, 2, ceil_mode=True))
        if self.attn:
            b, c, hgt, wid = e3.shape
            e3 = self.tf(e3.flatten(2).transpose(1, 2) + self.pos).transpose(1, 2).reshape(b, c, hgt, wid)
        if self.xattn:
            b, c, hgt, wid = e3.shape
            q = e3.flatten(2).transpose(1, 2) + self.qpos
            kv = self.kv(fc.flatten(2).transpose(1, 2)) + self.cpos
            o, att = self.xa(q if self.postnorm else self.xnorm(q), kv, kv, need_weights=self.keep_attn)
            if self.keep_attn: self.last_attn = att.detach()
            r = e3.flatten(2).transpose(1, 2) + o                       # residual; post-norm normaliza después
            e3 = (self.xnorm(r) if self.postnorm else r).transpose(1, 2).reshape(b, c, hgt, wid)
        up = lambda z, ref: F.interpolate(z, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        u2 = self.u2(torch.cat([up(e3, e2), e2], 1))
        u1 = self.u1(torch.cat([up(u2, e1), e1], 1))
        return self.out(u1).permute(0, 2, 3, 1)  # (B,70,35,4)


def build(cfg, st, S):
    """UNet según los argumentos de exp.py (dict)."""
    dummy = np.broadcast_to(np.float32([50, 280, 0, 0]), (1, 17, 13, 4)).copy()   # solo para contar canales
    nc = coarse_feats(dummy, ['2024010100'], cfg['feats']).shape[-1]
    nl = len(cfg['lags']) + bool(cfg.get('init'))
    tattn = None
    if cfg.get('tattn'):
        assert not cfg.get('init') and sorted(cfg['lags']) == list(cfg['lags']) and 0 in cfg['lags'], 'tattn: lags ordenados con 0, sin init'
        tattn = dict(L=nl, C=nc, d=64, i0=list(cfg['lags']).index(0), causal=cfg['tattn'] == 'causal')
    return UNet(nc * nl, 26 if cfg.get('hour_onehot') else 2, grid(st), S, cfg['width'],
                cfg.get('enc_blocks', 1), cfg.get('attn', False), cfg.get('drop', 0.0), cfg.get('xattn', False),
                tattn, cfg.get('postnorm', False))


class Corrida:
    """Un modelo entrenado por exp.py (carpeta con args.json, stats.npz y model.pt), listo para predecir."""
    def __init__(self, carpeta, st=None, dev=None):
        carpeta = Path(carpeta)
        self.cfg = json.load(open(carpeta / 'args.json'))
        self.st = dict(st if st is not None else np.load(ROOT / 'data' / 'static.npz'))
        self.s = dict(np.load(carpeta / 'stats.npz'))
        self.dev = dev or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.net = build(self.cfg, self.st, self.s['static']).to(self.dev)
        self.net.load_state_dict(torch.load(carpeta / 'model.pt', map_location=self.dev)); self.net.eval()

    def predecir(self, X, t, bs=64):
        """X (n,17,13,4) en unidades físicas, t 'YYYYMMDDHH' UTC -> (n,70,35,4)."""
        c, s, t = self.cfg, self.s, np.asarray(t).astype(str)
        W = (windows(coarse_feats(X, t, c['feats']), t, c['lags'], c.get('init')) - s['mu_x']) / s['sd_x']
        lo = torch.tensor(W, dtype=torch.float32).permute(0, 1, 4, 2, 3).flatten(1, 2)
        hh = torch.tensor(hour_feats(t, c.get('hour_onehot')))
        base = s['base'][hour(t)] + (interp(X, grid(self.st)) if c['base'] == 'interp' else 0)
        hi = torch.tensor((base - s['mu_y']) / s['sd_y'], dtype=torch.float32).permute(0, 3, 1, 2)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                out.append(self.net(lo[i:i + bs].to(self.dev), hh[i:i + bs].to(self.dev), hi[i:i + bs].to(self.dev)).float().cpu().numpy())
        P = base + np.concatenate(out) * s['scale']
        P[..., 0] = P[..., 0].clip(0, 100)
        return P.astype(np.float32)


class Ensamble:
    """Promedio ponderado por variable de varias corridas: {"corrida": [peso_RH, peso_T, peso_U, peso_V], ...}."""
    def __init__(self, pesos, raiz=ROOT / 'models'):
        self.pesos = {k: np.asarray(v, np.float32) for k, v in pesos.items()}
        self.corridas = {k: Corrida(Path(raiz) / k) for k in self.pesos}

    @classmethod
    def desde_json(cls, ruta):
        ruta = Path(ruta); return cls(json.load(open(ruta)), ruta.parent)

    def predecir(self, X, t):
        tot = sum(self.pesos.values())
        return sum(self.corridas[k].predecir(X, t) * (w / tot) for k, w in self.pesos.items())
