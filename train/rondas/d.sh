#!/bin/bash
# Ronda D: ventanas de días (misma hora ±1/±2 días, media de ±3.5 días y tendencia) y atención cruzada
# para el viento (cada zona de 3 km mira a todo el GFS). Los análogos (train/analogos.py) corren aparte.
# nohup bash train/rondas/d.sh > train/runs/ronda_d.out 2>&1 < /dev/null &
P=.venv/bin/python
B1="--base interp --loss metric --epochs 15 --swa_from 8 --amp --width 96 --bias_days 30 --feats vap daily"
L3="--lags -3 -2 -1 0 1 2 3"
LD="--lags -48 -24 -3 -2 -1 0 1 2 3 24 48"
cv(){ n=$1; shift; for f in 0 1 2; do $P train/exp.py --name ${n}_f$f --fold $f "$@" > train/runs/${n}_f$f.log 2>&1 || echo "FALLÓ ${n}_f$f"; done; echo "listo $n"; }

cv d_xattn       $B1 $L3 --xattn
cv d_dias        $B1 $LD
cv d_semana      --base interp --loss metric --epochs 15 --swa_from 8 --amp --width 96 --bias_days 30 --feats vap daily semana $L3
cv d_xattn_dias  $B1 $LD --xattn
cv d_xattn_w128  $B1 $L3 --xattn --width 128
$P train/exp.py --name d_xattn_explica_f0 --fold 0 $B1 $L3 --xattn --explain > train/runs/d_xattn_explica_f0.log 2>&1
$P train/resumen.py d_
$P train/resumen.py knn
