#!/bin/bash
# Ronda C: el WRF se reinicia a las 00 UTC -> hora en one-hot (tiempo de pronóstico) y GFS de las 00 UTC del día.
# nohup bash train/rondas/c.sh > train/runs/ronda_c.out 2>&1 < /dev/null &
P=.venv/bin/python
B1="--base interp --lags -3 -2 -1 0 1 2 3 --loss metric --epochs 15 --swa_from 8 --amp --width 96 --bias_days 30 --feats vap daily"
cv(){ n=$1; shift; for f in 0 1 2; do $P train/exp.py --name ${n}_f$f --fold $f "$@" > train/runs/${n}_f$f.log 2>&1 || echo "FALLÓ ${n}_f$f"; done; echo "listo $n"; }

cv c_b1_oh        $B1 --hour_onehot
cv c_b1_init      $B1 --init
cv c_b1_oh_init   $B1 --hour_onehot --init
cv c_w128_oh_init $B1 --hour_onehot --init --width 128
$P train/exp.py --name c_explica_f0 --fold 0 $B1 --hour_onehot --init --explain > train/runs/c_explica_f0.log 2>&1
$P train/resumen.py c_
$P train/resumen.py b_b1
