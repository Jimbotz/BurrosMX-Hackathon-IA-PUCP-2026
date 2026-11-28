#!/bin/bash
# Ensamble v2: entrena con TODO el train las configuraciones nuevas (rondas D y E) que eligió la selección greedy.
# Pesos: train/runs/ensamble.json (greedy sin análogos: 0.3884 CV; con análogos 0.3891, se dejan fuera porque
# necesitarían los datos de entrenamiento en el producto y no se pueden redistribuir).
# nohup bash train/rondas/final_v2.sh > train/runs/final_v2.out 2>&1 < /dev/null &
P=.venv/bin/python
$P train/exp.py --name full_d_semana --full --base interp --bias_days 30 --recent_days 0 --recent_rep 1 --lags -3 -2 -1 0 1 2 3 --feats vap daily semana --width 96 --enc_blocks 1 --drop 0.0 --epochs 15 --lr 0.002 --bs 32 --wd 0.0001 --loss metric --vw 1 1 1 1 --swa_from 8 --amp --seed 0 > train/runs/full_d_semana.log 2>&1 || echo "FALLÓ full_d_semana"
$P train/exp.py --name full_e_tattn_causal --full --base interp --bias_days 30 --recent_days 0 --recent_rep 1 --lags -3 -2 -1 0 1 2 3 --feats vap daily --width 96 --enc_blocks 1 --tattn causal --drop 0.0 --epochs 15 --lr 0.002 --bs 32 --wd 0.0001 --loss metric --vw 1 1 1 1 --swa_from 8 --amp --seed 0 > train/runs/full_e_tattn_causal.log 2>&1 || echo "FALLÓ full_e_tattn_causal"
$P train/exp.py --name full_e_tattn_post --full --base interp --bias_days 30 --recent_days 0 --recent_rep 1 --lags -3 -2 -1 0 1 2 3 --feats vap daily --width 96 --enc_blocks 1 --tattn full --postnorm --drop 0.0 --epochs 15 --lr 0.002 --bs 32 --wd 0.0001 --loss metric --vw 1 1 1 1 --swa_from 8 --amp --seed 0 > train/runs/full_e_tattn_post.log 2>&1 || echo "FALLÓ full_e_tattn_post"
$P train/exp.py --name full_e_xattn_post --full --base interp --bias_days 30 --recent_days 0 --recent_rep 1 --lags -3 -2 -1 0 1 2 3 --feats vap daily --width 96 --enc_blocks 1 --xattn --postnorm --drop 0.0 --epochs 15 --lr 0.002 --bs 32 --wd 0.0001 --loss metric --vw 1 1 1 1 --swa_from 8 --amp --seed 0 > train/runs/full_e_xattn_post.log 2>&1 || echo "FALLÓ full_e_xattn_post"
$P train/exp.py --name full_e_tattn_xattn --full --base interp --bias_days 30 --recent_days 0 --recent_rep 1 --lags -3 -2 -1 0 1 2 3 --feats vap daily --width 96 --enc_blocks 1 --xattn --tattn full --drop 0.0 --epochs 15 --lr 0.002 --bs 32 --wd 0.0001 --loss metric --vw 1 1 1 1 --swa_from 8 --amp --seed 0 > train/runs/full_e_tattn_xattn.log 2>&1 || echo "FALLÓ full_e_tattn_xattn"
$P train/ensemble.py train/runs/sub_ensamble_v2.csv train/runs/ensamble.json
