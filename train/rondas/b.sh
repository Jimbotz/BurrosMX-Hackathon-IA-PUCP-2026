#!/bin/bash
# Ronda B: ruido entre semillas, combinación de lo que sumó en A, capacidad y especialistas de viento.
# nohup bash train/rondas/b.sh > train/runs/ronda_b.out 2>&1 &
P=.venv/bin/python
B0="--base interp --lags -2 -1 0 1 2 --loss metric --epochs 15 --swa_from 8 --amp"
B1="--base interp --lags -3 -2 -1 0 1 2 3 --loss metric --epochs 15 --swa_from 8 --amp --width 96 --bias_days 30 --feats vap daily"
cv(){ n=$1; shift; for f in 0 1 2; do $P train/exp.py --name ${n}_f$f --fold $f "$@" > train/runs/${n}_f$f.log 2>&1 || echo "FALLÓ ${n}_f$f"; done; echo "listo $n"; }

cv b_b0_s1      $B0 --seed 1
cv b_b0_s2      $B0 --seed 2
cv b_b1         $B1
cv b_b1_s1      $B1 --seed 1
cv b_b1_viento  $B1 --vw 0.5 0.5 1.5 1.5
cv b_b1_w128    $B1 --width 128
cv b_b1_suave   $B1 --lr 1e-3 --epochs 25 --swa_from 12
$P train/exp.py --name b_b1_explica_f0 --fold 0 $B1 --explain > train/runs/b_b1_explica_f0.log 2>&1
$P train/resumen.py b_
$P train/resumen.py a_b0
