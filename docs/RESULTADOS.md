# Resultados de todos los modelos · BurrosMX · Hackathon IA PUCP 2026

> Documento para el equipo: **todos** los modelos que probamos, con **todas** las métricas, cómo leerlas y cómo seguir trabajando. Actualizado al 28 nov 2026.
> El reporte narrativo (hallazgos de datos y explicabilidad con figuras) está en [`docs/ANALISIS.md`](ANALISIS.md); la guía de setup en [`GUIA_EQUIPO.md`](../GUIA_EQUIPO.md).
> Las tablas del **Anexo** (sección 13) se generan con `python analysis/tablas_resultados.py`. Si agregan experimentos, regenérenlas y reemplacen el anexo.

---

## 0. Resumen en 30 segundos

| | Score local (CV, 3 folds) | Leaderboard Kaggle |
|---|---|---|
| Starter de la organización | 0.338 (solo fold 0) | **0.428** (real) |
| Nuestro modelo de referencia B0 | 0.352 | ~0.445 (estimado) |
| Nuestro mejor modelo individual (`e_xattn_post`) | **0.360** | ~0.463 (estimado) |
| **Nuestro ensamble final (18 redes, pesos por variable)** | **0.388** (0.393 en las horas que evalúa Kaggle) | **~0.48–0.49 (estimado)** |
| Primer lugar público (SciTeX, 28 nov) | — | 0.523 |

- **No hemos enviado nada a Kaggle todavía.** La submission está lista en la PC con GPU: `train/runs/sub_ensamble_v2.csv`, con 646,800 filas, ids iguales al `sample_submission` y sin vacíos.
- **La escala local y la de Kaggle son distintas.** Kaggle divide el error entre la climatología de julio–agosto, que es mucho peor que la de mayo–junio. Por eso los números del leaderboard son ~0.09 más altos. La conversión está explicada en la sección 2.4.
- **Lo que más sumó fue promediar modelos distintos:** +0.03. Ningún cambio de arquitectura movió más de ~0.005 a un modelo individual.
- **El cuello de botella es el viento.** Con los denominadores de Kaggle, el error relativo del viento V es ~0.74 y el de U ~0.62, contra ~0.44 en T y ~0.34 en RH.

---

## 1. Qué está hecho y dónde está cada cosa

| Qué | Dónde | ¿En GitHub? |
|---|---|---|
| Código del modelo (fuente única para entrenar y para el producto) | `modelo.py` | ✅ |
| Entrenar un experimento | `train/exp.py` | ✅ |
| Rondas de experimentos (comandos exactos) | `train/rondas/a.sh … e.sh` | ✅ |
| Modelos finales (con todo el train) | `train/rondas/final.sh`, `final_v2.sh`, generador `train/generar_final.py` | ✅ |
| Ensamble: score CV, selección greedy, submission | `train/ensemble.py` | ✅ |
| Tabla rápida por configuración | `train/resumen.py` | ✅ |
| Modelo simple de análogos (k vecinos) | `train/analogos.py` | ✅ |
| Test de la atención causal | `train/test_modelo.py` | ✅ |
| Pesos del ensamble final | `models/ensamble.json` | ✅ |
| Análisis de datos, explicabilidad, mapa de atención, estas tablas | `analysis/*.py`, figuras en `docs/analisis/` | ✅ |
| **Log de todas las corridas** | `train/runs/log.jsonl` (GPU) y `log_mac.jsonl` (Mac) | ❌ (`train/runs/` está en el `.gitignore`) |
| **Pesos de los modelos** (`full_*/model.pt`, 18 modelos = 272 MB) | `train/runs/full_*` en la PC con GPU | ❌ (se suben al entregar, ver sección 11) |
| Predicciones de validación/test por corrida (`val_pred.npy`, `test_pred.npy`) | `train/runs/<corrida>/` en la PC | ❌ |
| Submission lista | `train/runs/sub_ensamble_v2.csv` en la PC | ❌ |

> Si necesitan el log o los pesos, pídanselos a Jimbotz: están en su PC con la RTX 2060.

---

## 2. Las métricas (cómo leer las tablas)

### 2.1 Score (la métrica de Kaggle)

```
score = 1 − ( RMSE_RH/ref_RH + RMSE_T/ref_T + RMSE_U/ref_U + RMSE_V/ref_V ) / 4
```

- `ref_v` es el RMSE de la **climatología**: predecir, para cada píxel, el promedio histórico de esa hora del día. **0 = igual que la climatología, 1 = perfecto, mayor es mejor.**
- **Localmente**, `ref_v` es el RMSE de la climatología en cada fold de validación (tabla "Referencia" del anexo). En **Kaggle**, los denominadores son fijos: `25.360, 2.0652, 2.6776, 2.0876`.
- Las 4 variables pesan lo mismo, aunque tengan unidades distintas.

### 2.2 Ratio por variable: columnas RH, T, U, V de las tablas principales
`RMSE_v / ref_v`, promediado sobre los 3 folds. **Menos es mejor** (1 = igual que la climatología). Muestran *dónde* gana o pierde cada modelo. El score es `1 − promedio de los 4 ratios`.

### 2.3 RMSE absoluto
En unidades físicas: RH en %, T en K, U y V en m/s. Está en la tabla "RMSE absoluto por fold" del anexo.

### 2.4 Leaderboard estimado (columna "LB estimado")
```
LB_estimado = 1 − promedio( RMSE_fold0 / [25.360, 2.0652, 2.6776, 2.0876] )
```
Toma el error de nuestro modelo en el fold 0 (16–30 jun, lo más parecido al test) y lo divide entre los denominadores de Kaggle.
- **Calibración:** con el starter da **0.431**, y su score real en el leaderboard es **0.428**.
- **Supuesto:** que en julio–agosto nos equivocamos igual que a fines de junio. Solo un envío lo confirma.
- Para el ensamble no tenemos el RMSE del fold 0 en la Mac. Su estimado (~0.48–0.49) sale de sumar la ganancia del ensamble a los mejores individuales (~0.463).

### 2.5 "Media sin SWA (mejor época)"
El mejor score por época del modelo sin promediar pesos, **elegido mirando la validación**. Es **optimista** y solo sirve de referencia. El número que cuenta es la columna "Media", que viene del modelo con pesos promediados (SWA) y no mira la validación.

### 2.6 min
Minutos de entrenamiento sumando los 3 folds, en la RTX 2060 con fp16. En la Mac M4 es ~3 veces más.

---

## 3. Protocolo de validación (por qué confiar en los números)

- **3 folds temporales:** `f0` = 16–30 jun, `f1` = 1–15 jun, `f2` = 16–31 may. **Nunca usamos un split aleatorio**: horas consecutivas son casi idénticas y la validación saldría optimista.
- **Purga de 24 h** antes y después de cada fold. El entrenamiento no ve las horas pegadas a la validación.
- **Se decide con la media de los 3 folds.** El fold 0 es el más parecido al test (fines de junio). El f2 (mayo) es el menos parecido, pero protege contra el sobreajuste a junio.
- **Ruido:** el mismo modelo (B0) con 3 semillas da 0.3521, 0.3522 y 0.3541. **Diferencias menores a ~0.003 en la media no son concluyentes.**
- **Sin mirar la validación para elegir:** cada modelo se guarda con SWA (promedio de los pesos de las épocas 8–15). Es el mismo procedimiento que al entrenar con todo el train (`--full`), donde no hay validación.
- **Solo las horas de Kaggle:** Kaggle evalúa 00, 03, …, 21 UTC. Medido así, todos los scores suben ~+0.004–0.005 y **el orden de los modelos no cambia** (B0: 0.3519 → 0.3554; `e_xattn_post`: 0.3594 → 0.3639; ensamble: 0.3884 → 0.3931).

---

## 4. El modelo base y cómo leer la columna "Qué es"

**Arquitectura** (`modelo.py`, clase `UNet`):
1. Un **encoder a 27 km** (convoluciones sobre la grilla GFS de 17×13) recibe las horas del GFS (por defecto t−2…t+2) apiladas como canales, más el seno y coseno de la hora.
2. Un `grid_sample` lleva esas características a la **grilla de 3 km (70×35)**.
3. Una **U-Net de 3 niveles** a 3 km combina esas características con: la **base**, el **relieve** (altura, gradientes, TPI), un mapa estático aprendido y la hora.
4. La red predice la **corrección** sobre la base: `predicción = base + red × escala`.

**Base** (`interp`): el GFS interpolado a 3 km más el **sesgo medio WRF − GFS para cada (hora, píxel)**. La alternativa (`clim`) es la climatología por hora. `interp` le gana porque el test es otra temporada.

**Valores por defecto en las tablas** (salvo que la columna diga otra cosa): pérdida = la métrica (`metric`), 15 épocas, SWA desde la época 8, precisión mixta fp16, lr 2e-3, batch 32, weight decay 1e-4, ancho 64, semilla 0.

**B0** = `interp · t−2…+2` con esos valores por defecto.
**B1** = B0 + `t−3…+3` + `ancho 96` + `sesgo 30d` + `+vap+daily`.

| Etiqueta en "Qué es" | Significado | Opción de `exp.py` |
|---|---|---|
| `interp` / `clim` | Base: GFS interpolado + sesgo por (hora, píxel), o climatología | `--base` |
| `t-3…+3` | Horas del GFS que ve el modelo, relativas a t | `--lags` |
| `ancho N` | Canales de la U-Net (más ancho = más capacidad) | `--width` |
| `sesgo 30d` | El sesgo de la base se calcula con los 30 días previos al periodo a predecir (la temporada cambia el sesgo) | `--bias_days` |
| `recientes 30d×3` | Repite 3 veces los últimos 30 días en cada época | `--recent_days/--recent_rep` |
| `+vap` | Presión de vapor y punto de rocío del GFS | `--feats vap` |
| `+daily` | Media del GFS en ±12 h | `--feats daily` |
| `+wind` | Velocidad y dirección del viento del GFS | `--feats wind` |
| `+semana` | Media de ±3.5 días y tendencia (próximos − pasados) | `--feats semana` |
| `+relieve` | Más relieve: TPI a 3 y 21 celdas, pendiente, orientación | `--static_plus` |
| `enc×2` | Encoder a 27 km más profundo | `--enc_blocks 2` |
| `atención bottleneck` | Transformer en el nivel más bajo de la U-Net (18×9) | `--attn` |
| `atención cruzada` | Cada zona de 3 km "mira" las 221 celdas del GFS | `--xattn` |
| `atención temporal full/causal` | Transformer sobre la secuencia de horas del GFS; causal = solo horas ≤ t | `--tattn full/causal` |
| `post-norm` | LayerNorm después del residual (por defecto, pre-norm) | `--postnorm` |
| `hora one-hot` | La hora UTC en one-hot, además de sin/cos | `--hour_onehot` |
| `GFS 00 UTC` | Agrega el GFS de las 00 UTC del día (condición inicial del WRF) | `--init` |
| `dropout`, `pesos a/b/c/d`, `lr`, `ép`, `wd`, `semilla` | Regularización, peso de RH/T/U/V en la pérdida, tasa de aprendizaje, épocas, weight decay, semilla | `--drop`, `--vw`, `--lr`, `--epochs`, `--wd`, `--seed` |
| `pérdida mse`, `fp32` | Pérdida MSE en vez de la métrica; sin precisión mixta | `--loss mse`; sin `--amp` |
| Análogos: `k`, `situación`, `±h`, `comp. PCA` | Vecinos promediados; qué describe la situación (t, día, semana, tendencia); ventana de horas UTC; componentes PCA | `train/analogos.py` |

---

## 5. Resultados por ronda (resumen; detalle completo en el anexo)

| Ronda | Pregunta | Respuesta |
|---|---|---|
| Mac (protocolo anterior) | ¿Qué base y qué pérdida? | Base `interp` +0.022 sobre `clim`. La U-Net sola no mejora al starter. Pérdida tipo métrica ≈ mejor. 30 épocas sobreajusta (mejor época ~13). |
| **A** | Un cambio a la vez sobre B0 | Ancho 96 (+0.005), sesgo 30d, `vap` y t±3 (+0.003 cada uno). Nada más ayuda. Sobremuestrear lo reciente **empeora** (−0.008). fp16 = fp32. |
| **B** | Combinar lo bueno, capacidad, especialistas | B1 = 0.356. Ancho 128: 0.359. El especialista en viento y el entrenamiento suave no mejoran solos, pero **`b_b1_suave` es el miembro más usado del ensamble**. |
| **C** | El WRF se reinicia a las 00 UTC: ¿hora one-hot? ¿GFS de las 00 UTC? | No mejoran (la base por hora ya absorbe el salto). |
| **D** | Ventanas de días y atención cruzada | **Las ventanas de días empeoran (−0.02).** La atención cruzada con pre-norm ≈ B1. |
| **E** | Atención temporal causal vs completa, pre vs post-norm | **Causal > completa (+0.007)**, como predice la física. **La atención cruzada con post-norm es el mejor individual (0.360).** |
| Análogos | ¿Un modelo simple sin red? | 0.30 en el mejor caso: peor que la red. Con ventanas de semana empeora (0.236). |

---

## 6. Ensambles (todas las cifras son CV de 3 folds, todas las horas)

| Ensamble | f0 | f1 | f2 | **Media** | RH | T | U | V |
|---|---|---|---|---|---|---|---|---|
| Un solo modelo (B0, referencia) | 0.3546 | 0.3684 | 0.3333 | 0.3521 | 0.470 | 0.517 | 0.766 | 0.838 |
| Ronda A: promedio simple de 12 configuraciones | 0.3820 | 0.4021 | 0.3501 | 0.3781 | 0.445 | 0.496 | 0.739 | 0.808 |
| Ronda A: greedy por variable | 0.3873 | 0.4042 | 0.3512 | 0.3809 | 0.438 | 0.495 | 0.737 | 0.806 |
| A+B+C: promedio simple de 23 | 0.3913 | 0.4046 | 0.3543 | 0.3834 | 0.435 | 0.493 | 0.734 | 0.804 |
| Ensamble v1: greedy sobre 23 (18 modelos) | 0.3979 | 0.4058 | 0.3568 | 0.3868 | 0.428 | 0.491 | 0.733 | 0.802 |
| Todo (35, incluye D, E y análogos): promedio simple | 0.3927 | 0.4064 | 0.3531 | 0.3841 | 0.431 | 0.494 | 0.734 | 0.805 |
| Ensamble v2 con análogos | 0.4003 | 0.4114 | 0.3555 | 0.3891 | 0.425 | 0.490 | 0.730 | 0.800 |
| **Ensamble v2 sin análogos (FINAL)** | **0.4002** | **0.4090** | **0.3559** | **0.3884** | **0.426** | **0.490** | **0.731** | **0.801** |
| Final, solo horas de Kaggle (00, 03, …, 21 UTC) | — | — | — | 0.3931 | 0.425 | 0.472 | 0.730 | 0.800 |

**Primeras pruebas en la Mac** (protocolo anterior: fold 0 sin purga): promedio de 3 semillas eligiendo la mejor época = 0.3863 (optimista); 5 semillas con SWA (honesto) = 0.3766; 3 semillas con SWA = 0.3702.

**Cómo se arma.** La selección greedy de Caruana (`ensemble.py greedy`) agrega, con reemplazo, el modelo que más baja el error de **cada variable** por separado, mirando los 3 folds. Los pesos quedan en `models/ensamble.json` (tabla en el anexo). Después cada modelo elegido se reentrena con **todo** el train (`--full`), y la submission es el promedio ponderado de sus predicciones del test.

**Por qué los análogos quedaron fuera.** Aportan +0.0007 (ruido), y en el producto necesitarían los datos de entrenamiento en cada predicción. Esos datos no se pueden redistribuir según las reglas.

**Lectura clave:** el ensamble baja mucho el error de **humedad** (0.47 → 0.43) y casi nada el del **viento** (U 0.77 → 0.73, V 0.84 → 0.80). El error del viento no es varianza: es información que el GFS no trae.

---

## 7. Explicabilidad (resumen; tablas completas en el anexo, figuras en `docs/analisis/`)

- **Importancia por permutación** (`importancia.png`): hora del día > humedad (base y GFS) > viento U del GFS > medias de 24 h > altura. La temperatura del GFS casi no se usa: la temperatura a 3 km la explican la hora y el relieve.
- **Error por altura** (`error_altura_hora.png`): el viento falla más entre 1,000 y 3,000 m (valles y laderas). La temperatura falla más entre 1,000 y 2,000 m (inversión en la ladera costera).
- **Error por hora:** casi nulo a las 00 UTC (19:00 en Lima, el reinicio del WRF). Después crece. El viento es peor por la tarde (brisas de valle).
- **Atención cruzada** (`atencion.png`): todas las zonas miran las celdas del GFS al **sureste** (régimen del flujo del este en temporada seca), y cada zona mira además su celda local.
- **Datos** (`temporada.png`, `desfase.png`, `correlacion_pixel.png`, `sesgo_hora.png`, `ciclo_diurno.png`): el test se parece a junio, el WRF se reinicia a las 00 UTC, el viento V casi no se correlaciona con el GFS (0.26) y la temperatura de la ladera costera no sigue al GFS.

---

## 8. Qué funciona y qué no (para no repetir)

**Funciona:**
- Base `interp` + sesgo por hora de los **últimos 30 días**.
- Pérdida igual a la métrica.
- SWA (épocas 8–15) y 15 épocas.
- Ancho 96–128.
- t±3 horas y presión de vapor.
- Atención cruzada con post-norm.
- Atención temporal **causal**.
- **Ensamble de modelos distintos con pesos por variable.**

**No funciona** (o no supera el ruido):
- Ventanas de varios días (±1/±2 días, semana, tendencia).
- Sobremuestrear lo reciente.
- Hora one-hot y GFS de las 00 UTC.
- Atención en el bottleneck, sin post-norm.
- Atención temporal completa.
- Dropout, encoder más profundo y relieve extra.
- Especialista en viento solo.
- 30 épocas.
- Análogos solos.
- Regresión lineal por píxel (0.268).

---

## 9. Cómo trabajar sobre esto

### 9.1 Correr un experimento (con GPU: la PC, Kaggle Notebooks T4 o Colab)

```bash
# un fold
.venv/bin/python train/exp.py --name mi_idea_f0 --fold 0 --base interp --lags -3 -2 -1 0 1 2 3 \
    --loss metric --epochs 15 --swa_from 8 --amp --width 96 --bias_days 30 --feats vap daily
# los 3 folds (bash): copia la función cv() de train/rondas/e.sh y agrega una línea:
#   cv mi_idea $B1 --mi_opcion
.venv/bin/python train/resumen.py mi_idea          # tabla: f0, f1, f2, media
```

- Cada corrida guarda en `train/runs/<nombre>/`: `model.pt`, `stats.npz`, `args.json`, `val_pred.npy` y `test_pred.npy`, y agrega una línea a `train/runs/log.jsonl`.
- `--explain` agrega la importancia por permutación y los errores por altura y hora. Las figuras salen con `python analysis/explicar.py <corrida>`.
- **Regla para comparar:** una idea "sirve" si mejora la **media de 3 folds** en más de ~0.003 sobre su referencia, o si **entra al ensamble greedy**. Un modelo puede ser peor solo y aun así sumar diversidad: `d_semana` fue 0.331 solo y entró al ensamble.

### 9.2 Probar si un modelo nuevo mejora el ensamble

```bash
.venv/bin/python train/ensemble.py val cfg1 cfg2 ...       # score CV del promedio simple
.venv/bin/python train/ensemble.py greedy <todas las cfg>  # pesos por variable -> train/runs/ensamble.json
.venv/bin/python train/generar_final.py train/runs/ensamble.json train/rondas/final_v3.sh
bash train/rondas/final_v3.sh                              # entrena con todo lo que falte + escribe la submission
```

### 9.3 Usar el ensamble en el producto

```python
from modelo import Ensamble
ens = Ensamble.desde_json('models/ensamble.json')   # requiere models/full_*/ con model.pt, stats.npz y args.json
Y = ens.predecir(X, times)    # X (H,17,13,4) del GFS de Open-Meteo, times 'YYYYMMDDHH' UTC -> (H,70,35,4)
```

- Verificado: reproduce la submission con diferencias máximas de 0.03 % de RH, 0.003 K y 0.008 m/s (por fp16 vs fp32).
- Algunas configuraciones miran hasta ±3.5 días (`d_semana`) y ±3 h. Al pedir el pronóstico, incluyan horas pasadas (`past_hours`).

### 9.4 Regenerar este documento
`python analysis/tablas_resultados.py` imprime el anexo completo. Se necesita `train/runs/log.jsonl`.

---

## 10. Qué falta explorar (priorizado)

| # | Idea | Qué esperar | Costo |
|---|---|---|---|
| 1 | **Calibrar con el leaderboard:** 1 envío de `sub_ensamble_v2.csv` (5 envíos por día, se eligen 2 al final) | Saber si nuestro ~0.48 estimado es real y si el cambio de temporada nos afecta | Nulo |
| 2 | **Pesar la pérdida como Kaggle:** `--vw 0.68 0.98 0.85 0.93` (= ref local / ref Kaggle). En el test pesan más T y el viento | Pequeño, pero alineado con lo que se califica | Bajo |
| 3 | **Más semillas** de los mejores (`b_b1_suave`, `e_xattn_post`, `e_tattn_causal`, `b_b1_w128`) | +0.002 a +0.005 en el ensamble | ~1 h de GPU |
| 4 | **Cabeza probabilística** (P10/P50/P90 con pérdida por cuantiles) | = en Kaggle; **rangos y probabilidad de helada o viento fuerte para el producto** | Bajo |
| 5 | **Gradient boosting por píxel** (LightGBM) | Diversidad para el ensamble | Bajo, en CPU |
| 6 | Humedad vía **punto de rocío** (predecir Td y calcular RH con la T predicha) | +0.001 a +0.003 | Medio |
| 7 | Ensamble con pesos **por hora o por banda de altura** | +0.001 a +0.003 (riesgo de sobreajuste) | Bajo |
| 8 | **Robustez a la temporada:** ruido en las entradas, pseudo-etiquetas en el test, entradas como anomalías de 30 días | Depende de lo que diga el leaderboard (de 0 a +0.01) | Medio |
| 9 | **U-Net con bloques Swin** (los híbridos Swin + CNN le ganan al U-Net en temperatura según la literatura) | Diversidad, quizás mejora T | Medio |
| 10 | Difusión residual tipo **CorrDiff** (NVIDIA) | ≈ 0 en RMSE; escenarios realistas | Alto. No recomendado para el plazo |

---

## 11. Antes de entregar (recordatorio)

- **Notebook reproducible:** `train/submission_notebook.ipynb` debe entrenar el ensamble desde cero solo con la data oficial y generar el `submission.csv` (seguir `final.sh` + `final_v2.sh`).
- **Pesos:** los 18 modelos pesan 272 MB, cada uno < 100 MB. Se pueden subir con `git add -f`, pero conviene **Hugging Face o Drive** con el link en el README.
- **Repo:** hacerlo **público** antes del 8 dic, 23:59.
- **Licencia:** agregar `LICENSE` MIT.

---

## 12. Glosario rápido

- **GFS:** pronóstico global (27 km). Es la entrada.
- **WRF:** modelo físico regional (3 km). Es lo que imitamos.
- **Downscaling:** pasar de baja a alta resolución.
- **Climatología:** promedio histórico por píxel y hora. Es la referencia de la métrica.
- **Fold:** periodo de validación.
- **SWA:** promedio de los pesos de las últimas épocas.
- **Greedy (Caruana):** selección de modelos para el ensamble.
- **Permutación:** romper una entrada para medir cuánto la usa el modelo.
- **Atención cruzada:** la grilla fina consulta a la gruesa.
- **Causal:** solo ve el pasado.
- **Post-norm / pre-norm:** dónde va la normalización en una capa de atención.
- **TPI:** índice de posición topográfica (altura menos la altura media alrededor).

---

## 13. Anexo: todas las tablas (generadas)

<!-- generado por analysis/tablas_resultados.py -->

### Referencia: RMSE de la climatología por fold (denominadores locales)

| Fold | Periodo | RH (%) | T (K) | U (m/s) | V (m/s) |
|---|---|---|---|---|---|
| f0 | 20240616 → 20240701 (excl.) | 17.253 | 2.020 | 2.264 | 1.948 |
| f1 | 20240601 → 20240616 (excl.) | 27.774 | 1.791 | 2.229 | 1.761 |
| f2 | 20240516 → 20240601 (excl.) | 15.898 | 1.374 | 1.897 | 1.644 |
| Kaggle (test, jul–ago) | 2 jul → 4 ago | 25.3600 | 2.0652 | 2.6776 | 2.0876 |

#### Redes neuronales (U-Net y variantes), 3 folds

| # | Configuración | Qué es | f0 | f1 | f2 | **Media** | RH | T | U | V | LB estimado | Media sin SWA (mejor época) | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `e_xattn_post` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · atención cruzada · post-norm | 0.3775 | 0.3707 | 0.3308 | **0.3597** | 0.446 | 0.517 | 0.764 | 0.835 | 0.463 | 0.3589 | 4.8 |
| 2 | `b_b1_w128` | interp · t-3…+3 · ancho 128 · sesgo 30d · +vap+daily | 0.3802 | 0.3682 | 0.3285 | **0.3590** | 0.441 | 0.520 | 0.761 | 0.841 | 0.464 | 0.3622 | 7.3 |
| 3 | `d_xattn_w128` | interp · t-3…+3 · ancho 128 · sesgo 30d · +vap+daily · atención cruzada | 0.3788 | 0.3681 | 0.3286 | **0.3585** | 0.443 | 0.518 | 0.763 | 0.843 | 0.463 | 0.3585 | 7.2 |
| 4 | `a_w96` | interp · t-2…+2 · ancho 96 | 0.3603 | 0.3799 | 0.3318 | **0.3573** | 0.463 | 0.512 | 0.760 | 0.835 | 0.450 | 0.3559 | 4.5 |
| 5 | `c_w128_oh_init` | interp · t-3…+3 · ancho 128 · sesgo 30d · +vap+daily · hora one-hot · GFS 00 UTC | 0.3715 | 0.3668 | 0.3320 | **0.3568** | 0.448 | 0.515 | 0.767 | 0.842 | 0.457 | 0.3554 | 6.9 |
| 6 | `b_b1_viento` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · pesos 0.5/0.5/1.5/1.5 | 0.3641 | 0.3719 | 0.3332 | **0.3564** | 0.456 | 0.523 | 0.758 | 0.838 | 0.452 | 0.3559 | 4.5 |
| 7 | `b_b1` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily | 0.3754 | 0.3706 | 0.3228 | **0.3563** | 0.453 | 0.516 | 0.767 | 0.840 | 0.461 | 0.3559 | 4.5 |
| 8 | `c_b1_oh` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · hora one-hot | 0.3741 | 0.3674 | 0.3270 | **0.3562** | 0.451 | 0.518 | 0.766 | 0.840 | 0.460 | 0.3572 | 4.5 |
| 9 | `d_xattn` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · atención cruzada | 0.3737 | 0.3678 | 0.3264 | **0.3560** | 0.450 | 0.521 | 0.762 | 0.842 | 0.459 | 0.3559 | 4.8 |
| 10 | `e_tattn_causal` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · atención temporal causal | 0.3680 | 0.3797 | 0.3196 | **0.3558** | 0.452 | 0.520 | 0.763 | 0.842 | 0.454 | 0.3542 | 5.4 |
| 11 | `a_bias30` | interp · t-2…+2 · sesgo 30d | 0.3600 | 0.3809 | 0.3251 | **0.3553** | 0.464 | 0.516 | 0.764 | 0.835 | 0.449 | 0.3553 | 2.7 |
| 12 | `a_vap` | interp · t-2…+2 · +vap | 0.3611 | 0.3742 | 0.3303 | **0.3552** | 0.464 | 0.518 | 0.765 | 0.832 | 0.450 | 0.3550 | 2.7 |
| 13 | `e_tattn_xattn` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · atención cruzada · atención temporal full | 0.3613 | 0.3805 | 0.3238 | **0.3552** | 0.453 | 0.519 | 0.766 | 0.841 | 0.449 | 0.3520 | 5.7 |
| 14 | `a_lags3` | interp · t-3…+3 | 0.3642 | 0.3725 | 0.3286 | **0.3551** | 0.460 | 0.516 | 0.766 | 0.838 | 0.453 | 0.3522 | 2.7 |
| 15 | `b_b1_suave` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · lr 0.001 · 25 ép | 0.3712 | 0.3631 | 0.3298 | **0.3547** | 0.448 | 0.519 | 0.768 | 0.846 | 0.457 | 0.3564 | 7.5 |
| 16 | `b_b1_s1` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · semilla 1 | 0.3672 | 0.3775 | 0.3193 | **0.3547** | 0.455 | 0.524 | 0.763 | 0.839 | 0.454 | 0.3573 | 4.5 |
| 17 | `c_b1_oh_init` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · hora one-hot · GFS 00 UTC | 0.3712 | 0.3661 | 0.3252 | **0.3542** | 0.449 | 0.519 | 0.769 | 0.845 | 0.457 | 0.3535 | 4.5 |
| 18 | `b_b0_s2` | interp · t-2…+2 · semilla 2 | 0.3602 | 0.3735 | 0.3287 | **0.3541** | 0.464 | 0.518 | 0.767 | 0.834 | 0.450 | 0.3515 | 2.7 |
| 19 | `a_daily` | interp · t-2…+2 · +daily | 0.3617 | 0.3714 | 0.3268 | **0.3533** | 0.451 | 0.526 | 0.766 | 0.843 | 0.449 | 0.3492 | 2.7 |
| 20 | `c_b1_init` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · GFS 00 UTC | 0.3674 | 0.3671 | 0.3254 | **0.3533** | 0.453 | 0.521 | 0.771 | 0.842 | 0.454 | 0.3519 | 4.5 |
| 21 | `a_static` | interp · t-2…+2 · +relieve | 0.3555 | 0.3747 | 0.3270 | **0.3524** | 0.469 | 0.515 | 0.766 | 0.841 | 0.446 | 0.3502 | 2.7 |
| 22 | `a_enc2` | interp · t-2…+2 · enc×2 | 0.3604 | 0.3707 | 0.3256 | **0.3522** | 0.464 | 0.521 | 0.769 | 0.837 | 0.449 | 0.3501 | 2.7 |
| 23 | `b_b0_s1` | interp · t-2…+2 · semilla 1 | 0.3588 | 0.3753 | 0.3225 | **0.3522** | 0.469 | 0.521 | 0.767 | 0.834 | 0.448 | 0.3542 | 2.7 |
| 24 | `a_b0` | interp · t-2…+2 | 0.3546 | 0.3684 | 0.3333 | **0.3521** | 0.470 | 0.517 | 0.766 | 0.838 | 0.445 | 0.3517 | 2.7 |
| 25 | `a_drop10` | interp · t-2…+2 · dropout 0.1 | 0.3515 | 0.3800 | 0.3248 | **0.3521** | 0.470 | 0.521 | 0.768 | 0.833 | 0.442 | 0.3559 | 2.7 |
| 26 | `a_b0_fp32` | interp · t-2…+2 · fp32 | 0.3546 | 0.3685 | 0.3326 | **0.3519** | 0.469 | 0.519 | 0.766 | 0.839 | 0.445 | 0.3520 | 3.3 |
| 27 | `a_attn` | interp · t-2…+2 · atención bottleneck | 0.3524 | 0.3713 | 0.3292 | **0.3510** | 0.472 | 0.516 | 0.766 | 0.841 | 0.443 | 0.3478 | 3.0 |
| 28 | `a_wind` | interp · t-2…+2 · +wind | 0.3575 | 0.3663 | 0.3257 | **0.3498** | 0.468 | 0.516 | 0.774 | 0.843 | 0.447 | 0.3489 | 2.7 |
| 29 | `e_tattn` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · atención temporal full | 0.3639 | 0.3664 | 0.3154 | **0.3486** | 0.457 | 0.525 | 0.771 | 0.852 | 0.452 | 0.3479 | 5.4 |
| 30 | `e_tattn_post` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · atención temporal full · post-norm | 0.3632 | 0.3689 | 0.3120 | **0.3480** | 0.458 | 0.529 | 0.771 | 0.851 | 0.450 | 0.3467 | 5.4 |
| 31 | `a_recent30x3` | interp · t-2…+2 · recientes 30d×3 | 0.3490 | 0.3649 | 0.3195 | **0.3445** | 0.471 | 0.521 | 0.778 | 0.852 | 0.440 | 0.3464 | 4.1 |
| 32 | `d_xattn_dias` | interp · t-48,-24,-3,-2,-1,+0,+1,+2,+3,+24,+48 · ancho 96 · sesgo 30d · +vap+daily · atención cruzada | 0.3528 | 0.3428 | 0.3129 | **0.3362** | 0.463 | 0.539 | 0.785 | 0.868 | 0.442 | 0.3407 | 4.8 |
| 33 | `d_dias` | interp · t-48,-24,-3,-2,-1,+0,+1,+2,+3,+24,+48 · ancho 96 · sesgo 30d · +vap+daily | 0.3499 | 0.3352 | 0.3178 | **0.3343** | 0.457 | 0.535 | 0.791 | 0.880 | 0.439 | 0.3407 | 4.5 |
| 34 | `d_semana` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily+semana | 0.3397 | 0.3643 | 0.2881 | **0.3307** | 0.471 | 0.543 | 0.791 | 0.872 | 0.430 | 0.3321 | 4.5 |

#### Análogos (k vecinos, sin red), 3 folds

| # | Configuración | Qué es | f0 | f1 | f2 | **Media** | RH | T | U | V | LB estimado | Media sin SWA (mejor época) | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `knn_k50_tdia` | k=50 · situación: t+dia · ±1 h · 30 comp. PCA | 0.3083 | 0.3435 | 0.2504 | **0.3007** | 0.502 | 0.591 | 0.828 | 0.876 | 0.401 | — | 0.0 |
| 2 | `knn_k50_t` | k=50 · situación: t · ±1 h · 30 comp. PCA | 0.3001 | 0.3466 | 0.2467 | **0.2978** | 0.514 | 0.595 | 0.826 | 0.875 | 0.394 | — | 0.0 |
| 3 | `knn_k50_h3` | k=50 · situación: t+dia+semana+tendencia · ±3 h · 30 comp. PCA | 0.2687 | 0.3272 | 0.1744 | **0.2568** | 0.562 | 0.628 | 0.870 | 0.912 | 0.368 | — | 0.3 |
| 4 | `knn_k20` | k=20 · situación: t+dia+semana+tendencia · ±1 h · 30 comp. PCA | 0.2654 | 0.3225 | 0.1758 | **0.2546** | 0.562 | 0.627 | 0.874 | 0.918 | 0.366 | — | 0.3 |
| 5 | `knn_k50_c10` | k=50 · situación: t+dia+semana+tendencia · ±1 h · 10 comp. PCA | 0.2662 | 0.3232 | 0.1701 | **0.2532** | 0.570 | 0.638 | 0.867 | 0.912 | 0.366 | — | 0.3 |
| 6 | `knn_k50` | k=50 · situación: t+dia+semana+tendencia · ±1 h · 30 comp. PCA | 0.2638 | 0.3276 | 0.1598 | **0.2504** | 0.575 | 0.642 | 0.868 | 0.913 | 0.364 | — | 0.3 |
| 7 | `knn_k100` | k=100 · situación: t+dia+semana+tendencia · ±1 h · 30 comp. PCA | 0.2501 | 0.3150 | 0.1607 | **0.2419** | 0.588 | 0.656 | 0.870 | 0.918 | 0.351 | — | 0.3 |
| 8 | `knn_k50_sem` | k=50 · situación: t+semana+tendencia · ±1 h · 30 comp. PCA | 0.2444 | 0.3189 | 0.1449 | **0.2361** | 0.595 | 0.656 | 0.879 | 0.925 | 0.347 | — | 0.0 |

#### Corridas de un solo fold (explicabilidad y benchmark de GPU; no comparables con la media de 3 folds)

| Corrida | Qué es | Score f0 | RMSE RH | T | U | V | LB estimado |
|---|---|---|---|---|---|---|---|
| `cuda_s0` | interp · t-2…+2 · fp32 | 0.3592 | 9.745 | 0.904 | 1.670 | 1.570 | 0.451 |
| `b_b1_explica` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily | 0.3746 | 8.964 | 0.902 | 1.670 | 1.554 | 0.460 |
| `c_explica` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · hora one-hot · GFS 00 UTC | 0.3701 | 8.874 | 0.902 | 1.686 | 1.585 | 0.456 |
| `d_xattn_explica` | interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · atención cruzada | 0.3759 | 8.936 | 0.903 | 1.654 | 1.560 | 0.461 |

### RMSE absoluto por fold y variable (todas las corridas)

| Configuración | Fold | Score | RMSE RH (%) | RMSE T (K) | RMSE U (m/s) | RMSE V (m/s) |
|---|---|---|---|---|---|---|
| `a_attn` | f0 | 0.3524 | 9.86 | 0.910 | 1.699 | 1.593 |
| `a_attn` | f1 | 0.3713 | 9.48 | 0.933 | 1.711 | 1.558 |
| `a_attn` | f2 | 0.3292 | 8.01 | 0.793 | 1.481 | 1.350 |
| `a_b0` | f0 | 0.3546 | 9.86 | 0.910 | 1.689 | 1.585 |
| `a_b0` | f1 | 0.3684 | 9.40 | 0.946 | 1.720 | 1.564 |
| `a_b0` | f2 | 0.3333 | 7.93 | 0.788 | 1.480 | 1.337 |
| `a_b0_fp32` | f0 | 0.3546 | 9.86 | 0.910 | 1.690 | 1.584 |
| `a_b0_fp32` | f1 | 0.3685 | 9.40 | 0.954 | 1.716 | 1.559 |
| `a_b0_fp32` | f2 | 0.3326 | 7.90 | 0.788 | 1.481 | 1.347 |
| `a_bias30` | f0 | 0.3600 | 9.58 | 0.910 | 1.687 | 1.577 |
| `a_bias30` | f1 | 0.3809 | 9.19 | 0.923 | 1.701 | 1.527 |
| `a_bias30` | f2 | 0.3251 | 8.05 | 0.798 | 1.489 | 1.361 |
| `a_daily` | f0 | 0.3617 | 9.07 | 0.932 | 1.687 | 1.598 |
| `a_daily` | f1 | 0.3714 | 9.28 | 0.963 | 1.709 | 1.542 |
| `a_daily` | f2 | 0.3268 | 7.84 | 0.795 | 1.493 | 1.371 |
| `a_drop10` | f0 | 0.3515 | 9.82 | 0.922 | 1.697 | 1.594 |
| `a_drop10` | f1 | 0.3800 | 9.17 | 0.936 | 1.710 | 1.516 |
| `a_drop10` | f2 | 0.3248 | 8.10 | 0.802 | 1.495 | 1.348 |
| `a_enc2` | f0 | 0.3604 | 9.55 | 0.909 | 1.683 | 1.582 |
| `a_enc2` | f1 | 0.3707 | 9.38 | 0.953 | 1.734 | 1.531 |
| `a_enc2` | f2 | 0.3256 | 7.98 | 0.799 | 1.488 | 1.363 |
| `a_lags3` | f0 | 0.3642 | 9.42 | 0.906 | 1.682 | 1.569 |
| `a_lags3` | f1 | 0.3725 | 9.35 | 0.922 | 1.723 | 1.559 |
| `a_lags3` | f2 | 0.3286 | 7.91 | 0.802 | 1.483 | 1.352 |
| `a_recent30x3` | f0 | 0.3490 | 9.74 | 0.917 | 1.712 | 1.615 |
| `a_recent30x3` | f1 | 0.3649 | 9.48 | 0.946 | 1.754 | 1.556 |
| `a_recent30x3` | f2 | 0.3195 | 8.06 | 0.798 | 1.502 | 1.386 |
| `a_static` | f0 | 0.3555 | 9.79 | 0.911 | 1.692 | 1.583 |
| `a_static` | f1 | 0.3747 | 9.32 | 0.932 | 1.703 | 1.552 |
| `a_static` | f2 | 0.3270 | 8.01 | 0.789 | 1.491 | 1.361 |
| `a_vap` | f0 | 0.3611 | 9.67 | 0.905 | 1.680 | 1.569 |
| `a_vap` | f1 | 0.3742 | 9.23 | 0.944 | 1.720 | 1.536 |
| `a_vap` | f2 | 0.3303 | 7.95 | 0.795 | 1.483 | 1.346 |
| `a_w96` | f0 | 0.3603 | 9.72 | 0.906 | 1.681 | 1.567 |
| `a_w96` | f1 | 0.3799 | 9.20 | 0.916 | 1.695 | 1.544 |
| `a_w96` | f2 | 0.3318 | 7.86 | 0.793 | 1.475 | 1.354 |
| `a_wind` | f0 | 0.3575 | 9.63 | 0.909 | 1.700 | 1.581 |
| `a_wind` | f1 | 0.3663 | 9.40 | 0.936 | 1.739 | 1.573 |
| `a_wind` | f2 | 0.3257 | 8.05 | 0.792 | 1.500 | 1.355 |
| `b_b0_s1` | f0 | 0.3588 | 9.67 | 0.913 | 1.690 | 1.571 |
| `b_b0_s1` | f1 | 0.3753 | 9.23 | 0.949 | 1.713 | 1.528 |
| `b_b0_s1` | f2 | 0.3225 | 8.17 | 0.797 | 1.492 | 1.363 |
| `b_b0_s2` | f0 | 0.3602 | 9.68 | 0.900 | 1.678 | 1.580 |
| `b_b0_s2` | f1 | 0.3735 | 9.24 | 0.944 | 1.729 | 1.533 |
| `b_b0_s2` | f2 | 0.3287 | 7.91 | 0.800 | 1.487 | 1.351 |
| `b_b1` | f0 | 0.3754 | 8.98 | 0.905 | 1.663 | 1.549 |
| `b_b1` | f1 | 0.3706 | 9.14 | 0.933 | 1.734 | 1.567 |
| `b_b1` | f2 | 0.3228 | 8.08 | 0.797 | 1.493 | 1.371 |
| `b_b1_s1` | f0 | 0.3672 | 9.19 | 0.916 | 1.673 | 1.570 |
| `b_b1_s1` | f1 | 0.3775 | 9.06 | 0.947 | 1.692 | 1.542 |
| `b_b1_s1` | f2 | 0.3193 | 8.05 | 0.809 | 1.500 | 1.375 |
| `b_b1_suave` | f0 | 0.3712 | 8.84 | 0.902 | 1.678 | 1.588 |
| `b_b1_suave` | f1 | 0.3631 | 9.15 | 0.958 | 1.742 | 1.588 |
| `b_b1_suave` | f2 | 0.3298 | 7.99 | 0.792 | 1.479 | 1.352 |
| `b_b1_viento` | f0 | 0.3641 | 9.29 | 0.923 | 1.673 | 1.577 |
| `b_b1_viento` | f1 | 0.3719 | 9.43 | 0.946 | 1.688 | 1.562 |
| `b_b1_viento` | f2 | 0.3332 | 7.80 | 0.802 | 1.473 | 1.342 |
| `b_b1_w128` | f0 | 0.3802 | 8.64 | 0.904 | 1.651 | 1.561 |
| `b_b1_w128` | f1 | 0.3682 | 9.04 | 0.959 | 1.725 | 1.571 |
| `b_b1_w128` | f2 | 0.3285 | 7.91 | 0.794 | 1.481 | 1.364 |
| `c_b1_init` | f0 | 0.3674 | 9.04 | 0.917 | 1.681 | 1.578 |
| `c_b1_init` | f1 | 0.3671 | 9.19 | 0.961 | 1.748 | 1.550 |
| `c_b1_init` | f2 | 0.3254 | 8.02 | 0.788 | 1.491 | 1.372 |
| `c_b1_oh` | f0 | 0.3741 | 9.08 | 0.901 | 1.663 | 1.552 |
| `c_b1_oh` | f1 | 0.3674 | 9.19 | 0.953 | 1.744 | 1.559 |
| `c_b1_oh` | f2 | 0.3270 | 7.90 | 0.790 | 1.484 | 1.378 |
| `c_b1_oh_init` | f0 | 0.3712 | 8.92 | 0.901 | 1.677 | 1.581 |
| `c_b1_oh_init` | f1 | 0.3661 | 9.28 | 0.955 | 1.730 | 1.572 |
| `c_b1_oh_init` | f2 | 0.3252 | 7.91 | 0.795 | 1.501 | 1.367 |
| `c_w128_oh_init` | f0 | 0.3715 | 8.77 | 0.900 | 1.693 | 1.583 |
| `c_w128_oh_init` | f1 | 0.3668 | 9.47 | 0.947 | 1.723 | 1.567 |
| `c_w128_oh_init` | f2 | 0.3320 | 7.89 | 0.786 | 1.479 | 1.355 |
| `d_dias` | f0 | 0.3499 | 9.16 | 0.933 | 1.740 | 1.634 |
| `d_dias` | f1 | 0.3352 | 9.30 | 1.001 | 1.809 | 1.680 |
| `d_dias` | f2 | 0.3178 | 8.03 | 0.803 | 1.502 | 1.393 |
| `d_semana` | f0 | 0.3397 | 9.36 | 0.958 | 1.757 | 1.652 |
| `d_semana` | f1 | 0.3643 | 9.61 | 0.969 | 1.725 | 1.553 |
| `d_semana` | f2 | 0.2881 | 8.31 | 0.845 | 1.562 | 1.457 |
| `d_xattn` | f0 | 0.3737 | 8.94 | 0.909 | 1.657 | 1.569 |
| `d_xattn` | f1 | 0.3678 | 9.17 | 0.955 | 1.721 | 1.573 |
| `d_xattn` | f2 | 0.3264 | 7.99 | 0.798 | 1.485 | 1.362 |
| `d_xattn_dias` | f0 | 0.3528 | 9.32 | 0.935 | 1.722 | 1.607 |
| `d_xattn_dias` | f1 | 0.3428 | 9.43 | 1.002 | 1.784 | 1.637 |
| `d_xattn_dias` | f2 | 0.3129 | 8.10 | 0.816 | 1.506 | 1.399 |
| `d_xattn_w128` | f0 | 0.3788 | 8.65 | 0.902 | 1.660 | 1.566 |
| `d_xattn_w128` | f1 | 0.3681 | 9.27 | 0.948 | 1.730 | 1.565 |
| `d_xattn_w128` | f2 | 0.3286 | 7.83 | 0.794 | 1.480 | 1.373 |
| `e_tattn` | f0 | 0.3639 | 9.27 | 0.911 | 1.673 | 1.591 |
| `e_tattn` | f1 | 0.3664 | 9.29 | 0.943 | 1.741 | 1.570 |
| `e_tattn` | f2 | 0.3154 | 7.94 | 0.820 | 1.506 | 1.395 |
| `e_tattn_causal` | f0 | 0.3680 | 8.97 | 0.914 | 1.667 | 1.597 |
| `e_tattn_causal` | f1 | 0.3797 | 9.22 | 0.932 | 1.700 | 1.526 |
| `e_tattn_causal` | f2 | 0.3196 | 8.02 | 0.808 | 1.497 | 1.380 |
| `e_tattn_post` | f0 | 0.3632 | 9.15 | 0.917 | 1.675 | 1.604 |
| `e_tattn_post` | f1 | 0.3689 | 9.26 | 0.962 | 1.733 | 1.543 |
| `e_tattn_post` | f2 | 0.3120 | 8.09 | 0.819 | 1.508 | 1.401 |
| `e_tattn_xattn` | f0 | 0.3613 | 9.12 | 0.928 | 1.694 | 1.594 |
| `e_tattn_xattn` | f1 | 0.3805 | 9.16 | 0.920 | 1.708 | 1.529 |
| `e_tattn_xattn` | f2 | 0.3238 | 7.95 | 0.803 | 1.488 | 1.375 |
| `e_xattn_post` | f0 | 0.3775 | 8.91 | 0.901 | 1.662 | 1.545 |
| `e_xattn_post` | f1 | 0.3707 | 9.12 | 0.948 | 1.723 | 1.561 |
| `e_xattn_post` | f2 | 0.3308 | 7.82 | 0.792 | 1.488 | 1.356 |
| `knn_k100` | f0 | 0.2501 | 10.57 | 1.201 | 1.984 | 1.784 |
| `knn_k100` | f1 | 0.3150 | 12.68 | 1.062 | 1.790 | 1.562 |
| `knn_k100` | f2 | 0.1607 | 11.05 | 1.073 | 1.763 | 1.564 |
| `knn_k20` | f0 | 0.2654 | 10.56 | 1.136 | 1.956 | 1.753 |
| `knn_k20` | f1 | 0.3225 | 11.42 | 1.026 | 1.839 | 1.586 |
| `knn_k20` | f2 | 0.1758 | 10.55 | 1.026 | 1.770 | 1.567 |
| `knn_k50` | f0 | 0.2638 | 10.41 | 1.156 | 1.964 | 1.757 |
| `knn_k50` | f1 | 0.3276 | 11.94 | 1.032 | 1.789 | 1.552 |
| `knn_k50` | f2 | 0.1598 | 11.01 | 1.068 | 1.771 | 1.574 |
| `knn_k50_c10` | f0 | 0.2662 | 10.31 | 1.147 | 1.967 | 1.755 |
| `knn_k50_c10` | f1 | 0.3232 | 11.96 | 1.035 | 1.808 | 1.562 |
| `knn_k50_c10` | f2 | 0.1701 | 10.82 | 1.056 | 1.749 | 1.559 |
| `knn_k50_h3` | f0 | 0.2687 | 10.46 | 1.134 | 1.952 | 1.745 |
| `knn_k50_h3` | f1 | 0.3272 | 11.47 | 1.025 | 1.822 | 1.564 |
| `knn_k50_h3` | f2 | 0.1744 | 10.62 | 1.032 | 1.767 | 1.565 |
| `knn_k50_sem` | f0 | 0.2444 | 10.84 | 1.195 | 2.002 | 1.788 |
| `knn_k50_sem` | f1 | 0.3189 | 12.46 | 1.045 | 1.797 | 1.561 |
| `knn_k50_sem` | f2 | 0.1449 | 11.27 | 1.091 | 1.797 | 1.595 |
| `knn_k50_t` | f0 | 0.3001 | 9.62 | 1.094 | 1.869 | 1.705 |
| `knn_k50_t` | f1 | 0.3466 | 10.98 | 1.014 | 1.750 | 1.527 |
| `knn_k50_t` | f2 | 0.2467 | 9.34 | 0.930 | 1.643 | 1.451 |
| `knn_k50_tdia` | f0 | 0.3083 | 9.39 | 1.075 | 1.859 | 1.694 |
| `knn_k50_tdia` | f1 | 0.3435 | 10.74 | 1.017 | 1.769 | 1.545 |
| `knn_k50_tdia` | f2 | 0.2504 | 9.14 | 0.924 | 1.647 | 1.451 |

### Primeras pruebas en la Mac (protocolo anterior: solo fold 0, sin purga de 24 h)

| Corrida | Qué es | Score (mejor época) | Score honesto (SWA) | RMSE RH | T | U | V | LB estimado |
|---|---|---|---|---|---|---|---|---|
| `unet_clim` | clim · t-2…+0 · 30 ép · pérdida mse | 0.3384 | — | 9.161 | 1.051 | 1.719 | 1.615 | 0.429 |
| `unet_interp` | interp · t-2…+0 · 30 ép · pérdida mse | 0.3603 | — | 9.406 | 0.914 | 1.684 | 1.579 | 0.450 |
| `unet_interp_fut` | interp · t-2…+2 · 30 ép · pérdida mse | 0.3627 | — | 8.816 | 0.932 | 1.720 | 1.579 | 0.451 |
| `unet_clim_fut` | clim · t-2…+2 · 30 ép · pérdida mse | 0.3383 | — | 9.530 | 1.037 | 1.714 | 1.591 | 0.430 |
| `if_s1` | interp · t-2…+2 · 30 ép · pérdida mse · semilla 1 | 0.3604 | — | 8.936 | 0.916 | 1.699 | 1.616 | 0.449 |
| `if_s2` | interp · t-2…+2 · 30 ép · pérdida mse · semilla 2 | 0.3558 | — | 9.472 | 0.906 | 1.698 | 1.603 | 0.446 |
| `if_wd` | interp · t-2…+2 · 30 ép · wd 0.01 · pérdida mse | 0.3637 | — | 8.798 | 0.931 | 1.719 | 1.575 | 0.451 |
| `if_mloss` | interp · t-2…+2 · 30 ép | 0.3665 | — | 8.690 | 0.923 | 1.722 | 1.570 | 0.454 |
| `if_ep15` | interp · t-2…+2 · pérdida mse | 0.3561 | — | 9.521 | 0.915 | 1.700 | 1.584 | 0.447 |
| `swa_s0` | interp · t-2…+2 | 0.3572 | 0.3592 | 9.746 | 0.904 | 1.670 | 1.570 | 0.451 |
| `swa_s1` | interp · t-2…+2 · semilla 1 | 0.3515 | 0.3547 | 9.707 | 0.913 | 1.689 | 1.585 | 0.446 |
| `swa_s2` | interp · t-2…+2 · semilla 2 | 0.3509 | 0.3527 | 9.783 | 0.903 | 1.702 | 1.590 | 0.445 |
| `swa_s3` | interp · t-2…+2 · semilla 3 | 0.3592 | 0.3626 | 9.508 | 0.894 | 1.680 | 1.572 | 0.453 |
| `swa_s4` | interp · t-2…+2 · semilla 4 | 0.3579 | 0.3630 | 9.179 | 0.909 | 1.688 | 1.585 | 0.452 |

### Explicabilidad · `b_b1_explica_f0` (fold 0, score 0.3746)

Modelo: interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily

| Entrada | Caída del score al permutarla |
|---|---|
| hora del día | +0.0689 |
| base RH | +0.0344 |
| GFS RH | +0.0332 |
| GFS U | +0.0306 |
| GFS RH_24h | +0.0302 |
| base U | +0.0249 |
| GFS U_24h | +0.0218 |
| relieve altura | +0.0199 |
| base V | +0.0184 |
| GFS hora t-3 | +0.0130 |
| GFS hora t-2 | +0.0068 |
| GFS hora t+0 | +0.0066 |
| GFS V | +0.0057 |
| GFS hora t-1 | +0.0057 |
| relieve TPI9 | +0.0044 |
| GFS T | +0.0040 |
| relieve dx | +0.0038 |
| GFS hora t+3 | +0.0037 |
| GFS hora t+2 | +0.0028 |
| GFS hora t+1 | +0.0026 |
| base T | +0.0019 |
| GFS Td | +0.0018 |
| relieve dy | +0.0015 |
| GFS V_24h | +0.0014 |
| GFS e | +0.0005 |
| GFS T_24h | +0.0000 |

| Altura | RH | T | U | V |
|---|---|---|---|---|
| 0-1000 m | 0.322 | 0.371 | 0.542 | 0.492 |
| 1000-2000 m | 0.478 | 0.565 | 0.678 | 0.882 |
| 2000-3000 m | 0.462 | 0.468 | 0.878 | 0.902 |
| 3000-4000 m | 0.501 | 0.448 | 0.743 | 0.782 |
| 4000-6000 m | 0.578 | 0.424 | 0.708 | 0.785 |

| Hora UTC (Lima) | RH | T | U | V |
|---|---|---|---|---|
| 00 (19) | 0.286 | 0.126 | 0.118 | 0.104 |
| 01 (20) | 0.368 | 0.317 | 0.496 | 0.439 |
| 02 (21) | 0.430 | 0.365 | 0.640 | 0.577 |
| 03 (22) | 0.467 | 0.426 | 0.750 | 0.719 |
| 04 (23) | 0.479 | 0.451 | 0.779 | 0.754 |
| 05 (00) | 0.493 | 0.473 | 0.797 | 0.791 |
| 06 (01) | 0.547 | 0.476 | 0.780 | 0.817 |
| 07 (02) | 0.563 | 0.490 | 0.774 | 0.820 |
| 08 (03) | 0.588 | 0.509 | 0.792 | 0.826 |
| 09 (04) | 0.615 | 0.533 | 0.795 | 0.861 |
| 10 (05) | 0.605 | 0.544 | 0.766 | 0.859 |
| 11 (06) | 0.615 | 0.547 | 0.751 | 0.848 |
| 12 (07) | 0.627 | 0.461 | 0.773 | 0.851 |
| 13 (08) | 0.581 | 0.449 | 0.788 | 0.860 |
| 14 (09) | 0.547 | 0.434 | 0.751 | 0.789 |
| 15 (10) | 0.516 | 0.420 | 0.719 | 0.762 |
| 16 (11) | 0.496 | 0.407 | 0.689 | 0.758 |
| 17 (12) | 0.479 | 0.404 | 0.666 | 0.779 |
| 18 (13) | 0.469 | 0.416 | 0.716 | 0.823 |
| 19 (14) | 0.486 | 0.443 | 0.762 | 0.852 |
| 20 (15) | 0.502 | 0.472 | 0.809 | 0.900 |
| 21 (16) | 0.519 | 0.468 | 0.837 | 0.973 |
| 22 (17) | 0.525 | 0.454 | 0.831 | 0.963 |
| 23 (18) | 0.530 | 0.454 | 0.792 | 0.931 |

### Explicabilidad · `c_explica_f0` (fold 0, score 0.3701)

Modelo: interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · hora one-hot · GFS 00 UTC

| Entrada | Caída del score al permutarla |
|---|---|
| hora del día | +0.0961 |
| base RH | +0.0457 |
| GFS RH | +0.0317 |
| base U | +0.0246 |
| GFS U | +0.0234 |
| GFS RH_24h | +0.0227 |
| GFS U_24h | +0.0215 |
| relieve altura | +0.0210 |
| base V | +0.0177 |
| GFS hora 00 UTC del día | +0.0100 |
| GFS hora t-3 | +0.0090 |
| GFS hora t+0 | +0.0085 |
| GFS hora t+1 | +0.0047 |
| GFS hora t-2 | +0.0046 |
| GFS hora t+2 | +0.0039 |
| GFS hora t+3 | +0.0039 |
| GFS T | +0.0038 |
| relieve TPI9 | +0.0032 |
| GFS hora t-1 | +0.0031 |
| GFS V | +0.0029 |
| base T | +0.0023 |
| relieve dx | +0.0022 |
| relieve dy | +0.0019 |
| GFS Td | +0.0011 |
| GFS T_24h | +0.0004 |
| GFS e | +0.0002 |
| GFS V_24h | -0.0012 |

| Altura | RH | T | U | V |
|---|---|---|---|---|
| 0-1000 m | 0.330 | 0.369 | 0.544 | 0.500 |
| 1000-2000 m | 0.465 | 0.565 | 0.704 | 0.885 |
| 2000-3000 m | 0.448 | 0.466 | 0.898 | 0.919 |
| 3000-4000 m | 0.490 | 0.442 | 0.748 | 0.802 |
| 4000-6000 m | 0.579 | 0.429 | 0.708 | 0.802 |

| Hora UTC (Lima) | RH | T | U | V |
|---|---|---|---|---|
| 00 (19) | 0.274 | 0.128 | 0.108 | 0.097 |
| 01 (20) | 0.333 | 0.304 | 0.490 | 0.425 |
| 02 (21) | 0.408 | 0.360 | 0.657 | 0.585 |
| 03 (22) | 0.460 | 0.415 | 0.768 | 0.727 |
| 04 (23) | 0.473 | 0.444 | 0.795 | 0.774 |
| 05 (00) | 0.484 | 0.471 | 0.812 | 0.817 |
| 06 (01) | 0.538 | 0.476 | 0.787 | 0.836 |
| 07 (02) | 0.556 | 0.487 | 0.783 | 0.825 |
| 08 (03) | 0.574 | 0.505 | 0.797 | 0.830 |
| 09 (04) | 0.601 | 0.522 | 0.797 | 0.858 |
| 10 (05) | 0.600 | 0.540 | 0.770 | 0.858 |
| 11 (06) | 0.609 | 0.551 | 0.763 | 0.855 |
| 12 (07) | 0.625 | 0.451 | 0.779 | 0.856 |
| 13 (08) | 0.592 | 0.442 | 0.801 | 0.870 |
| 14 (09) | 0.558 | 0.443 | 0.764 | 0.806 |
| 15 (10) | 0.525 | 0.432 | 0.733 | 0.783 |
| 16 (11) | 0.499 | 0.417 | 0.702 | 0.778 |
| 17 (12) | 0.480 | 0.410 | 0.664 | 0.791 |
| 18 (13) | 0.459 | 0.422 | 0.712 | 0.842 |
| 19 (14) | 0.479 | 0.444 | 0.769 | 0.882 |
| 20 (15) | 0.503 | 0.480 | 0.815 | 0.936 |
| 21 (16) | 0.519 | 0.480 | 0.844 | 1.014 |
| 22 (17) | 0.520 | 0.464 | 0.826 | 1.007 |
| 23 (18) | 0.516 | 0.447 | 0.790 | 0.954 |

### Explicabilidad · `d_xattn_explica_f0` (fold 0, score 0.3759)

Modelo: interp · t-3…+3 · ancho 96 · sesgo 30d · +vap+daily · atención cruzada

| Entrada | Caída del score al permutarla |
|---|---|
| hora del día | +0.0707 |
| base RH | +0.0364 |
| GFS RH_24h | +0.0322 |
| GFS U | +0.0315 |
| GFS RH | +0.0299 |
| GFS U_24h | +0.0242 |
| base U | +0.0230 |
| relieve altura | +0.0192 |
| base V | +0.0171 |
| GFS hora t-3 | +0.0114 |
| GFS hora t+0 | +0.0083 |
| GFS hora t-2 | +0.0077 |
| GFS hora t-1 | +0.0073 |
| GFS V | +0.0064 |
| GFS hora t+1 | +0.0051 |
| GFS T | +0.0047 |
| GFS hora t+3 | +0.0047 |
| relieve TPI9 | +0.0045 |
| GFS hora t+2 | +0.0037 |
| base T | +0.0032 |
| relieve dx | +0.0025 |
| GFS V_24h | +0.0021 |
| relieve dy | +0.0020 |
| GFS Td | +0.0012 |
| GFS T_24h | +0.0008 |
| GFS e | +0.0004 |

| Altura | RH | T | U | V |
|---|---|---|---|---|
| 0-1000 m | 0.334 | 0.363 | 0.542 | 0.493 |
| 1000-2000 m | 0.492 | 0.554 | 0.667 | 0.868 |
| 2000-3000 m | 0.466 | 0.465 | 0.868 | 0.887 |
| 3000-4000 m | 0.498 | 0.447 | 0.737 | 0.779 |
| 4000-6000 m | 0.572 | 0.430 | 0.700 | 0.804 |

| Hora UTC (Lima) | RH | T | U | V |
|---|---|---|---|---|
| 00 (19) | 0.288 | 0.130 | 0.124 | 0.113 |
| 01 (20) | 0.357 | 0.314 | 0.501 | 0.439 |
| 02 (21) | 0.427 | 0.368 | 0.646 | 0.583 |
| 03 (22) | 0.466 | 0.421 | 0.759 | 0.725 |
| 04 (23) | 0.481 | 0.443 | 0.788 | 0.758 |
| 05 (00) | 0.495 | 0.468 | 0.799 | 0.796 |
| 06 (01) | 0.545 | 0.479 | 0.778 | 0.819 |
| 07 (02) | 0.560 | 0.491 | 0.766 | 0.815 |
| 08 (03) | 0.584 | 0.512 | 0.782 | 0.817 |
| 09 (04) | 0.609 | 0.530 | 0.782 | 0.854 |
| 10 (05) | 0.601 | 0.538 | 0.749 | 0.849 |
| 11 (06) | 0.612 | 0.542 | 0.727 | 0.842 |
| 12 (07) | 0.626 | 0.455 | 0.755 | 0.849 |
| 13 (08) | 0.584 | 0.444 | 0.774 | 0.864 |
| 14 (09) | 0.550 | 0.442 | 0.732 | 0.801 |
| 15 (10) | 0.519 | 0.432 | 0.706 | 0.777 |
| 16 (11) | 0.493 | 0.417 | 0.676 | 0.774 |
| 17 (12) | 0.469 | 0.406 | 0.655 | 0.794 |
| 18 (13) | 0.456 | 0.416 | 0.712 | 0.831 |
| 19 (14) | 0.476 | 0.448 | 0.760 | 0.857 |
| 20 (15) | 0.503 | 0.479 | 0.803 | 0.908 |
| 21 (16) | 0.525 | 0.469 | 0.833 | 0.978 |
| 22 (17) | 0.529 | 0.458 | 0.823 | 0.967 |
| 23 (18) | 0.534 | 0.450 | 0.784 | 0.934 |

### Ensamble final: pesos por variable (`models/ensamble.json`)

Cada número es cuántas veces la selección greedy eligió ese modelo para esa variable; el peso efectivo es ese número dividido por el total de la columna.

| Modelo (entrenado con todo el train) | RH | T | U | V |
|---|---|---|---|---|
| `full_b_b1_suave` | 1 | 1 | 2 | 2 |
| `full_a_recent30x3` | · | 1 | 2 | 2 |
| `full_d_semana` | 1 | 1 | 1 | 1 |
| `full_a_w96` | · | 1 | 1 | 1 |
| `full_e_tattn_causal` | · | 1 | 1 | 1 |
| `full_a_wind` | · | 1 | · | 1 |
| `full_a_vap` | · | · | 1 | 1 |
| `full_a_daily` | 1 | · | 1 | · |
| `full_a_static` | · | 1 | 1 | · |
| `full_a_enc2` | · | · | 1 | 1 |
| `full_b_b1_w128` | 1 | · | 1 | · |
| `full_c_w128_oh_init` | 1 | 1 | · | · |
| `full_a_drop10` | · | · | · | 1 |
| `full_b_b0_s1` | · | · | · | 1 |
| `full_b_b1_viento` | · | · | 1 | · |
| `full_e_tattn_post` | · | · | 1 | · |
| `full_e_xattn_post` | · | · | · | 1 |
| `full_e_tattn_xattn` | · | · | · | 1 |
| **Total** | **5** | **8** | **14** | **14** |

### Datos: correlación de anomalías GFS(t+k) ↔ WRF(t), media del dominio

| k (horas) | RH | T | U | V |
|---|---|---|---|---|
| -6 | 0.601 | 0.448 | 0.391 | 0.149 |
| -5 | 0.623 | 0.496 | 0.403 | 0.164 |
| -4 | 0.647 | 0.545 | 0.419 | 0.183 |
| -3 | 0.672 | 0.593 | 0.437 | 0.207 |
| -2 | 0.695 | 0.639 | 0.458 | 0.232 |
| -1 | 0.715 | 0.676 | 0.479 | 0.256 |
| +0 | 0.728 | 0.691 | 0.495 | 0.269 |
| +1 | 0.730 | 0.671 | 0.495 | 0.259 |
| +2 | 0.722 | 0.630 | 0.485 | 0.238 |
| +3 | 0.707 | 0.579 | 0.469 | 0.215 |
| +4 | 0.689 | 0.523 | 0.450 | 0.193 |
| +5 | 0.668 | 0.469 | 0.432 | 0.175 |
| +6 | 0.649 | 0.418 | 0.415 | 0.158 |

### Datos: medias mensuales en el dominio WRF

| Fuente | Variable | feb | mar | abr | may | jun | jul | ago |
|---|---|---|---|---|---|---|---|---|
| GFS (train) | RH | 74.57 | 78.26 | 78.82 | 73.54 | 66.06 |  |  |
| GFS (train) | T | 283.03 | 283.30 | 282.99 | 282.90 | 281.78 |  |  |
| GFS (train) | U | 0.98 | -0.04 | -0.09 | -0.30 | -0.37 |  |  |
| GFS (train) | V | 0.27 | -0.04 | -0.05 | -0.07 | -0.12 |  |  |
| WRF (train) | RH | 77.07 | 81.16 | 80.88 | 71.72 | 61.10 |  |  |
| WRF (train) | T | 282.96 | 283.37 | 283.11 | 283.16 | 282.05 |  |  |
| WRF (train) | U | 0.98 | -0.35 | -0.36 | -0.51 | -0.54 |  |  |
| WRF (train) | V | 0.02 | -0.40 | -0.38 | -0.38 | -0.42 |  |  |
| GFS (test) | RH |  |  |  |  |  | 59.85 | 61.51 |
| GFS (test) | T |  |  |  |  |  | 281.77 | 282.14 |
| GFS (test) | U |  |  |  |  |  | -1.00 | 0.02 |
| GFS (test) | V |  |  |  |  |  | -0.39 | -0.21 |

### Datos: otras métricas

| Métrica | RH | T | U | V |
|---|---|---|---|---|
| Correlación media por píxel GFS↔WRF (anomalías) | 0.720 | 0.676 | 0.432 | 0.260 |
| Correlación de esa correlación con la altura | 0.247 | 0.091 | 0.421 | 0.250 |

Relaciones en el WRF: T vs altura -0.929 · RH vs altura 0.412 · RH vs T -0.361 · velocidad del viento vs altura 0.182. Sesgo de T (WRF − GFS) vs altura: -0.325 (-0.93 K/km).

Limpieza: valores faltantes {'X': 0, 'Y': 0, 'Xte': 0} · horas duplicadas 0 · frames del WRF repetidos 0 · horas atípicas (|z|>5) ['2024-02-21 17:00:00', '2024-02-21 18:00:00', '2024-02-21 19:00:00'].

Huecos en el train (inicio → horas): 2024-03-04 20 → 4 h, 2024-03-05 20 → 4 h, 2024-03-20 22 → 122 h, 2024-03-31 21 → 3 h, 2024-04-15 04 → 2 h, 2024-04-17 22 → 2 h, 2024-04-23 23 → 97 h, 2024-04-28 12 → 2 h, 2024-04-28 14 → 3 h, 2024-04-29 21 → 3 h, 2024-05-17 05 → 2 h, 2024-05-30 23 → 25 h, 2024-06-21 23 → 25 h · en el test: 2024-07-30 23 → 25 h
