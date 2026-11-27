#!/bin/bash
# Ronda E: atención en el TIEMPO sobre las horas del GFS (completa vs causal) y pre-norm vs post-norm.
# Causal: el WRF integra hacia adelante desde las 00 UTC, su estado en t solo depende del GFS hasta t.
# nohup bash train/rondas/e.sh > train/runs/ronda_e.out 2>&1 < /dev/null &
P=.venv/bin/python
B1="--base interp --loss metric --epochs 15 --swa_from 8 --amp --width 96 --bias_days 30 --feats vap daily --lags -3 -2 -1 0 1 2 3"
cv(){ n=$1; shift; for f in 0 1 2; do $P train/exp.py --name ${n}_f$f --fold $f "$@" > train/runs/${n}_f$f.log 2>&1 || echo "FALLÓ ${n}_f$f"; done; echo "listo $n"; }

$P train/test_modelo.py
cv e_tattn         $B1 --tattn full
cv e_tattn_causal  $B1 --tattn causal
cv e_tattn_post    $B1 --tattn full --postnorm
cv e_xattn_post    $B1 --xattn --postnorm
cv e_tattn_xattn   $B1 --tattn full --xattn
$P train/resumen.py e_
$P train/resumen.py d_
