# Análisis, experimentos y explicabilidad del modelo · BurrosMX

Este documento resume qué aprendimos de la data, qué probamos para el modelo de downscaling (GFS 27 km → WRF 3 km), qué funcionó, qué no, y cómo se explica lo que hace el modelo. Todo se reproduce con los scripts de `analysis/` y `train/`.

**Resultado:** un ensamble de 18 redes con pesos por variable llega a **0.388** de score en validación cruzada temporal. El starter de la organización saca 0.338 y nuestro mejor modelo individual, 0.360.

---

## 1. La data (`analysis/datos.py`)

**Limpieza.** No hay valores faltantes, horas duplicadas ni frames del WRF repetidos. Hay huecos de tiempo documentados (del 21 al 25 de marzo, del 24 al 27 de abril, el 31 de mayo y el 22 de junio en el train; el 31 de julio en el test). Solo aparecen 3 horas "atípicas" (21 feb, 17–19 UTC, con viento a z≈5) y son consistentes entre sí. Parecen un evento real, no un error, así que se conservaron.

**El test es otra temporada** (`docs/analisis/temporada.png`). El train va de febrero a junio (de la temporada de lluvias a la seca) y el test es julio–agosto (temporada seca). La humedad del WRF baja de ~81 % (marzo–abril) a 61 % en junio, y el GFS del test está en ~60 %: el test se parece a **junio**, no al promedio del train. Además, el sesgo de humedad WRF − GFS **cambia de signo** con la temporada (+2.9 en marzo, −5.0 en junio). En julio, el viento U del GFS (−1.0 m/s) es más del este que en cualquier mes del train.
→ Por eso la base del modelo usa el sesgo de los **últimos 30 días**, y por eso los folds de validación son de mayo y junio.

**El WRF se reinicia cada día a las 00 UTC.** A esa hora el WRF salta el doble de lo normal (cambio medio de RH: 12.1 a las 00 UTC contra ~5.6 en el resto; T: 2.2 contra ~1.0) y el GFS no. Cada día del WRF es entonces un pronóstico que arranca a las 00 UTC desde el GFS: a esa hora el error del modelo es casi nulo (~0.1 del error de la climatología) y después crece con las horas.

**Desfase GFS ↔ WRF** (`desfase.png`). La hora t del GFS es la que mejor explica el WRF en t, y t+1 aporta casi lo mismo que t−1 (correlación de anomalías en RH: 0.728 en t, 0.730 en t+1, 0.715 en t−1). Conviene mirar horas a ambos lados.

**Qué tan predecible es cada variable** (`correlacion_pixel.png`). Correlación media por píxel de anomalías GFS ↔ WRF: RH 0.72, T 0.68, **U 0.43, V 0.26**. El viento es el cuello de botella: depende del relieve local y del ciclo diurno más que del flujo de gran escala. El GFS sigue mejor el viento en las cumbres (correlación con la altura: +0.42 para U) que en los valles. En la **ladera costera** (oeste, de ~200 a 1,500 m), la temperatura del GFS no sigue al WRF (correlación ≤ 0) por una inversión térmica que el GFS a 27 km no resuelve.

**Relaciones físicas** (WRF): T vs altura −0.93 (la temperatura baja con la altura), RH vs T −0.36, RH vs altura +0.41.

---

## 2. Cómo validamos

- **3 folds temporales:** 16–30 jun, 1–15 jun y 16–31 may. Se excluyen 24 h antes y después de cada uno para evitar fuga por autocorrelación. Nunca usamos un split aleatorio.
- **Misma métrica que Kaggle:** `1 − promedio(RMSE_v / RMSE_climatología_v)`. Se decide con la **media de los 3 folds**.
- **Ruido entre semillas:** el mismo modelo con 3 semillas da 0.3521, 0.3522 y 0.3541 (±0.001 en la media). Diferencias menores a ~0.003 no son concluyentes.
- **Honestidad:** los modelos se guardan con promedio de pesos de las últimas épocas (SWA), sin elegir la "mejor época" mirando la validación. Así se entrena después con todo el train.

---

## 3. Experimentos (media de 3 folds, `train/rondas/`)

Referencia **B0**: U-Net con encoder a 27 km → `grid_sample` a 3 km. Recibe las horas t−2…t+2 del GFS y predice la corrección sobre una **base = GFS interpolado + sesgo medio por (hora, píxel)**, con pérdida igual a la métrica, 15 épocas y SWA desde la 8. **B1** = B0 + ancho 96 + sesgo de 30 días + presión de vapor y punto de rocío + horas t±3 + media del GFS de ±12 h.

| Ronda | Cambio | Score | Efecto |
|---|---|---|---|
| — | Starter de la organización (fold 0) | 0.338 | |
| A | B0 (referencia) | 0.352 | |
| A | Precisión mixta fp16 | 0.352 | = (20 % más rápido) |
| A | Ancho 96 | 0.357 | +0.005 |
| A | Sesgo con los últimos 30 días | 0.355 | +0.003 |
| A | Presión de vapor / punto de rocío | 0.355 | +0.003 |
| A | Horas t±3 | 0.355 | +0.003 |
| A | Media de ±12 h, más relieve, encoder más profundo | ~0.353 | ≈ 0 |
| A | Atención en el bottleneck, velocidad/dirección del viento, dropout | ~0.350–0.352 | ≈ 0 |
| A | Sobremuestrear los últimos 30 días | 0.345 | −0.008 |
| B | B1 (combinación) | 0.356 | |
| B | B1 ancho 128 | 0.359 | |
| B | Especialista en viento (más peso a U, V) / entrenamiento suave | ~0.355 | ≈ 0 |
| C | Hora en one-hot (tiempo desde el reinicio del WRF) | 0.356 | ≈ 0 |
| C | GFS de las 00 UTC del día (condición inicial del WRF) | 0.353 | ≈ 0 |
| D | Atención cruzada (cada zona de 3 km mira a todo el GFS) | 0.356 | ≈ 0 |
| D | Misma hora ±1 y ±2 días | 0.334 | **−0.02** |
| D | Media de ±3.5 días y tendencia | 0.331 | **−0.02** |
| E | Atención en el tiempo, completa | 0.349 | −0.007 |
| E | **Atención en el tiempo, causal** (solo horas ≤ t) | 0.356 | ≈ 0 (y +0.007 vs la completa) |
| E | **Atención cruzada con post-norm** | **0.360** | +0.004 (mejor modelo individual) |
| — | Análogos (k vecinos, sin red), el mejor | 0.301 | −0.05 |

**Lecciones.**
1. **Los modelos individuales se estancan en ~0.355–0.360.** Ningún cambio de arquitectura o variable mueve más de ~0.005.
2. **Las ventanas de varios días empeoran** (redes y análogos): con el cambio de temporada, "días parecidos" del pasado pertenecen a otro régimen y el modelo se sobreajusta.
3. **La atención causal le gana a la completa** dentro de la atención temporal. Coincide con la física: el WRF integra hacia adelante desde las 00 UTC, así que su estado en t solo depende del GFS hasta t.
4. El reinicio a las 00 UTC es real, pero la base ya lo absorbe (el sesgo se calcula por hora).

---

## 4. Ensamble (`train/ensemble.py`)

La varianza entre modelos es grande, sobre todo en humedad, así que **promediar modelos distintos** es lo que más suma:

| | Score CV | RH | T | U | V |
|---|---|---|---|---|---|
| Un modelo (B0) | 0.352 | 0.57 | 0.50 | 0.75 | 0.82 |
| Promedio simple de 35 configuraciones (incluye análogos) | 0.384 | 0.431 | 0.494 | 0.734 | 0.805 |
| **Greedy por variable, 18 modelos (final)** | **0.388** | **0.426** | **0.490** | **0.731** | **0.801** |
| + análogos | 0.389 | | | | |

(Las columnas de variables son RMSE / RMSE de la climatología; menos es mejor.)

- La selección greedy (Caruana) agrega con reemplazo el modelo que más mejora cada variable. Es algo optimista porque elige sobre los mismos folds.
- El ensamble baja mucho el error de **humedad** (0.57 → 0.43) y casi nada el del **viento**: el error del viento no es varianza, es información que el GFS no trae.
- Los análogos aportan +0.0007. **Se dejan fuera** porque en el producto necesitarían los datos de entrenamiento, y las reglas prohíben redistribuirlos.
- Modelos finales: cada configuración elegida, reentrenada con **todo** el train (`train/rondas/final.sh`, `final_v2.sh`). Pesos por variable en `train/runs/ensamble.json`.

---

## 5. Explicabilidad (`exp.py --explain`, `analysis/explicar.py`, `analysis/atencion.py`)

**¿Qué usa el modelo?** (`importancia.png`). Importancia por permutación en el fold 0, medida como cuánto cae el score al romper cada entrada:

1. **Hora del día** (0.069): el ciclo diurno y el tiempo desde el reinicio del WRF.
2. **Humedad**, tanto la base a 3 km (0.034) como el GFS (0.033) y su media de 24 h (0.030).
3. **Viento U del GFS** (0.031) y su media de 24 h (0.022).
4. **Altura** (0.020).

La temperatura del GFS casi no se usa (0.004): la temperatura a 3 km la explican la hora y el relieve.

**¿Dónde se equivoca?** (`error_pixel.png`, `error_altura_hora.png`)
- **Por altura:** el viento falla más entre **1,000 y 3,000 m** (valles del Callejón de Huaylas y laderas de la Cordillera Negra: U 0.88, V 0.90 del error de la climatología). Falla menos en la costa (0.49–0.54). La temperatura falla más en la franja de 1,000–2,000 m (la inversión de la ladera costera).
- **Por hora:** casi perfecto a las 00 UTC (19:00 en Lima, el reinicio del WRF). Después el error crece, y el viento V es peor por la tarde (15–18 h en Lima), cuando dominan las brisas de valle.
- **Borde oeste del dominio:** franja de error alto en U, probablemente la zona de relajación de bordes del WRF.

**¿A dónde mira la atención cruzada?** (`atencion.png`)
El modelo con atención cruzada (el mejor individual, `e_xattn_post`) aprende a qué celdas del GFS mirar para cada zona de 3 km. Promediando sobre las horas del test:

- **Todas las zonas miran sobre todo las mismas celdas:** las del **sureste**, fuera del dominio (lat −10.5 a −11, lon −77 a −76.2), hacia los Andes centrales y la Amazonía. El modelo las usa como **detector del régimen de gran escala**: en temporada seca domina el flujo del este que cruza la cordillera, y esas celdas lo describen "río arriba".
- **Cada zona mira además su propia celda o celdas vecinas.** En Huaraz, la celda del valle tiene un peso claro (0.07). Es la parte local del viento (encajonamiento en el Callejón de Huaylas).
- La entropía media de la atención es 0.66 (1 = mira todo por igual): está moderadamente concentrada.

Ojo: los pesos de atención muestran *dónde mira* el modelo, no prueban causalidad. Coinciden con la importancia por permutación: el viento U del GFS y sus medias pesan más que su valor local.

---

## 6. Reproducibilidad y producto

- `modelo.py` es la **única fuente** del modelo: lo usan `train/exp.py` (entrenar) y el producto (`Corrida`, `Ensamble`). Cada corrida guarda `model.pt`, `stats.npz` (normalización y sesgo de la base) y `args.json`.
- `modelo.Ensamble.desde_json('models/ensamble.json')` reproduce la submission. Diferencia máxima: 0.03 % de RH, 0.003 K y 0.008 m/s, por la precisión fp16 (entrenamiento) contra fp32 (producto). Es despreciable frente al error del modelo.
- Los 18 modelos pesan **272 MB** en total (cada uno < 100 MB). Para entregar, conviene subirlos a Hugging Face o Drive y poner el link en el README (lo permite la guía).
- `train/test_modelo.py` verifica que la atención causal no vea el futuro.
- **Ideas no exploradas:** más semillas de los mejores modelos (barato y suele sumar en ensambles), pérdida específica para extremos (heladas, humedad muy alta o baja), y validar contra el leaderboard público para calibrar la relación entre el score local y el de Kaggle.
