#!/bin/bash
# Ronda A: un cambio a la vez sobre la referencia B0, en los 3 folds temporales.
# Correr desde la raíz del repo (bash, no zsh):  nohup bash train/rondas/a.sh > train/runs/ronda_a.out 2>&1 &
P=.venv/bin/python
B0="--base interp --lags -2 -1 0 1 2 --loss metric --epochs 15 --swa_from 8"
cv(){ n=$1; shift; for f in 0 1 2; do $P train/exp.py --name ${n}_f$f --fold $f $B0 "$@" > train/runs/${n}_f$f.log 2>&1 || echo "FALLÓ ${n}_f$f"; done; echo "listo $n"; }

cv a_b0_fp32
cv a_b0          --amp
cv a_bias30      --amp --bias_days 30
cv a_recent30x3  --amp --recent_days 30 --recent_rep 3
cv a_wind        --amp --feats wind
cv a_vap         --amp --feats vap
cv a_daily       --amp --feats daily
cv a_static      --amp --static_plus
cv a_attn        --amp --attn
cv a_enc2        --amp --enc_blocks 2
cv a_w96         --amp --width 96
cv a_drop10      --amp --drop 0.1
cv a_lags3       --amp --lags -3 -2 -1 0 1 2 3
.venv/bin/python train/resumen.py a_
