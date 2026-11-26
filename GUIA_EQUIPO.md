# Guía del equipo BurrosMX · Hackathon IA PUCP 2026

Todo lo que necesitas para arrancar: de dónde sale cada cosa, cómo dejar tu compu lista y qué reglas nos pueden descalificar. Si algo de aquí contradice las **bases oficiales**, mandan las bases.

> **El reto en una línea:** entrenar un modelo que convierta el pronóstico GFS (~27 km) en mapas de 3 km de humedad, temperatura y viento para la Cordillera Blanca, competir con él en Kaggle (35 %) y llevarlo a un producto que le sirva a un usuario real (65 %).

---

## 1. Links oficiales

| Qué | Dónde |
|---|---|
| Evento (Luma) | https://luma.com/rdg0awcq |
| Bases, diapositivas del reto y grabación del kickoff | [Drive 1](https://drive.google.com/drive/folders/1WNuAFp6mZzV-w9eJRvvzfuoo2CUi37pR) |
| Guía del participante | [Drive 2](https://drive.google.com/drive/folders/1cp2fxh7AhPTgWOSoNxBYWUxbndgwkxQR) |
| Competencia en Kaggle | https://www.kaggle.com/competitions/hackathon-ia-pucp-2026 |
| Invitación a Kaggle (para entrar) | https://www.kaggle.com/t/5f175aa8a8b54c70b6b7b03d5d27070c |
| Formulario de registro de equipos | https://forms.gle/Km8GFezhHfAJZvpEA |
| Plantilla oficial del repo | https://github.com/D4rkPatata/Hackathon-IA-PUCP-2026 |
| **Nuestro repo** | https://github.com/Jimbotz/BurrosMX-Hackathon-IA-PUCP-2026 |
| Paper base | Carhuancho et al. (2025), https://doi.org/10.1016/j.aiig.2025.100143 (la data fue reprocesada: no compares números con el paper) |

---

## 2. Lo que cada integrante tiene que hacer (una sola vez)

1. **Luma:** inscribirte con tu cuenta en https://luma.com/rdg0awcq.
2. **Kaggle:**
   1. Entra con el link de invitación de arriba y haz clic en **Join Competition** (acepta las reglas).
   2. Únete al equipo **`BurrosMX`** (pestaña *Team*). El nombre debe ser **exactamente** ese, con las mismas mayúsculas: si no coincide con el formulario, nuestras submissions no cuentan.
   3. **No envíes nada a Kaggle antes de estar dentro del equipo.**
3. **GitHub:** pásale tu usuario a Jimbotz para que te agregue como colaborador del repo.
4. **Solo el líder:** registrar el equipo `BurrosMX` en el formulario (cierra **lun 7 dic, 23:59**), cuando los 4 ya estén en Luma.

---

## 3. Setup local (paso a paso)

Necesitas **Python 3.10–3.12** (la versión 3.9 que trae macOS no sirve). Lo más fácil es usar [uv](https://docs.astral.sh/uv/), que baja Python solo.

### 3.1 Clonar e instalar

```bash
git clone https://github.com/Jimbotz/BurrosMX-Hackathon-IA-PUCP-2026.git
cd BurrosMX-Hackathon-IA-PUCP-2026

# Opción A · con uv (recomendado)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt kaggle

# Opción B · con pip (si ya tienes Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt kaggle
```

### 3.2 Token de Kaggle (para bajar la data por terminal)

1. Ve a **Kaggle → tu foto → Settings → API Tokens** (https://www.kaggle.com/settings/api).
2. Clic en **Generate New Token** y usa el botón de copiar. **Nunca** lo pegues en el chat, en el código ni en GitHub.
3. Guárdalo en el archivo `~/.kaggle/access_token`:

   ```bash
   # macOS: con el token YA copiado en el portapapeles
   mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```

   ⚠️ **Ojo con el portapapeles:** si copias este comando *después* de copiar el token, el portapapeles ya no tiene el token. Escribe o pega primero el comando, luego copia el token y recién ahí presiona Enter.

   En Windows/Linux: crea la carpeta `.kaggle` en tu carpeta de usuario y guarda el token en un archivo llamado `access_token` (sin extensión). También sirve la variable de entorno `KAGGLE_API_TOKEN`.

4. Prueba que funciona:

   ```bash
   .venv/bin/kaggle competitions files hackathon-ia-pucp-2026
   ```

   Si ves 4 archivos, todo bien. Si dice *Authentication required*, el archivo no tiene el token correcto: repite el paso 3.

> Sin token también puedes bajar la data a mano desde la pestaña **Data** de la competencia (*Download All*) y descomprimirla en `data/`.

### 3.3 Bajar la data (~137 MB)

```bash
.venv/bin/kaggle competitions download -c hackathon-ia-pucp-2026 -p data
cd data && unzip -o hackathon-ia-pucp-2026.zip && rm hackathon-ia-pucp-2026.zip && cd ..
```

Debe quedar `data/train.npz`, `data/test.npz`, `data/static.npz` y `data/sample_submission.csv`.

**Qué NO se sube a GitHub** (ya está en el `.gitignore`): la data (`data/`, `*.npz`, `*.csv`), los pesos (`models/*.pt`), las salidas de experimentos (`train/runs/`), el caché del GFS y `.venv/`. Cada quien genera eso en su compu. **Excepción al entregar:** los pesos del modelo final sí van al repo; se agregan a la fuerza con `git add -f models/<modelo_final>.pt`.

### 3.4 Correr el starter (baseline ≈ 0.40)

```bash
cd train
../.venv/bin/jupyter notebook starter_notebook.ipynb     # o ábrelo en VS Code con el kernel de .venv
```

Genera `train/submission.csv` y los pesos en `models/modelo.pt`.

- En una Mac M4 (CPU) tarda **~5.5 min** (25 épocas de ~13 s). Score de validación local (últimas 2 semanas de junio): **0.338**. En el leaderboard público ese mismo baseline da **0.428**: la validación local y el leaderboard no son comparables 1 a 1 (otro periodo, otra climatología), úsala para comparar *nuestros* experimentos entre sí.
- También corre en Kaggle (pestaña *Code*, con *GPU T4*, la data ya está adjunta) o en Google Colab.

### 3.5 Probar el producto de ejemplo

```bash
.venv/bin/python app/ejemplo_producto.py
```

Baja el GFS actual de Open-Meteo (sin API key), lo pasa por un modelo **de relleno** y guarda `app/salidas/alerta_heladas.png`. Hay que reemplazar `predecir()` por nuestro modelo real.

- `app/gfs_forecast.py` baja el GFS en el mismo formato del input de Kaggle, hasta **16 días** hacia adelante.
- Cada descarga completa gasta ~221 de las 5,000 llamadas por hora del plan gratis: **siempre con caché** (`cache_path=...`).

---

## 4. La data y la métrica

| | Input GFS (`X`) | Target WRF (`Y`) y relieve (`dem`) |
|---|---|---|
| Tamaño | 17 × 13 celdas (~27 km) | 70 × 35 celdas (3 km) |
| Fila 0 | **Norte** | **Sur** |
| Columna 0 | Oeste | Oeste |

- **Variables**, siempre en este orden: `RH` (%), `T` (**Kelvin**), `U` (m/s, + = hacia el este), `V` (m/s, + = hacia el norte).
- **Train:** 2,935 horas, del 18 feb al 30 jun 2024. **Test:** 792 horas, del 2 jul al 4 ago 2024.
- **Entrada sugerida:** las horas t−2, t−1 y t del GFS para predecir el WRF en t.
- **Horas en UTC.** Hora de Perú = UTC − 5.
- **Hay huecos** (21–25 mar, 24–27 abr, 31 may, 22 jun, 1 jul, 31 jul y horas sueltas): dos filas seguidas no siempre están a 1 hora.
- **Valida con un corte temporal** (por ejemplo, las últimas 2 semanas de junio). Nunca con un split aleatorio.

**Métrica (mayor es mejor; 0 = climatología, 1 = perfecto):**

```
score = 1 − ( RMSE_RH/25.360 + RMSE_T/2.0652 + RMSE_U/2.6776 + RMSE_V/2.0876 ) / 4
```

- Las 4 variables pesan igual después de dividir. Conviene que la pérdida de entrenamiento use esos mismos pesos.
- Según los organizadores, donde más se puede ganar es en el **viento** (U, V).
- La pestaña *Data* dice que la climatología da "score 1.0": es una errata, la fórmula y el *Overview* dicen 0.

**Submission:** CSV con `id,RH,T,U,V`, donde `id = YYYYMMDDHH_fila_columna` (fila 0 = sur). Solo se evalúan las horas 00, 03, …, 21 UTC: 646,800 filas (~30 MB, se puede subir en .zip). El starter ya trae el código para armarlo.

**Leaderboard:** el público usa ~30 % de los días del test y el privado el ~70 % restante. **El que cuenta es el privado**: no sobreajustes al público. Cuenta con **5 envíos por día** (la regla de Kaggle dice 10, el overview y la guía dicen 5). Al cierre se eligen 2 submissions finales.

---

## 5. Reglas que nos pueden descalificar

- **Para entrenar y predecir en Kaggle, solo la data oficial.** Nada de datasets externos (sí se permiten modelos preentrenados públicos y crear variables nuevas a partir de la data).
- **El producto usa el mismo modelo** (mismos pesos) que nuestra submission final, alimentado con el GFS de Open-Meteo. Si mejoramos el modelo, se reenvía a Kaggle y se usan esos pesos.
- **El producto no puede funcionar igual sin el modelo**, ni ser solo una interfaz que muestra predicciones sobre la data de Kaggle.
- **No compartir código ni predicciones con otros equipos.**
- **Nombre exacto `BurrosMX`** en el formulario y en Kaggle.
- **Ninguna clave de API en el código** (usar variables de entorno).

---

## 6. Cómo nos califican (preselección, 100 puntos)

| Criterio | Puntos | Cómo se mide |
|---|---|---|
| Modelo en Kaggle | 35 | `35 × nuestro score privado ÷ el mejor score` |
| Valor para el usuario | 20 | Un usuario concreto y una decisión real (incluye UI/UX) |
| Uso del modelo y punta a punta | 15 | El valor depende de nuestro modelo y corre con el GFS actual |
| Originalidad | 10 | Qué tan distinta es la idea frente a lo obvio (un mapa con el pronóstico) |
| Reproducibilidad | 10 | README que se instala en pocos pasos y submission reproducible |
| Video pitch | 10 | Se entiende el problema y se ve el producto funcionando |

**Filtro previo (si fallas uno, no te evalúan):** los 4 entregables a tiempo · score privado > 0 · el producto usa nuestro modelo · repo y video accesibles.

**Estrategia:** el 26 nov el leaderboard público iba de 0.43 (starter) a 0.52 (primer lugar), con 28 equipos. Llegar a ~0.50 vale ~34 de 35 puntos; subir de 0.50 a 0.52 da solo 1 o 2 más. **La diferencia está en el producto (65 puntos).**

**Originalidad:** las 4 ideas de ejemplo de los organizadores (heladas para agricultores, viento y frío para andinistas, reportes de riesgo para municipios, mejores horas para turismo) son "lo obvio". La alerta de heladas además ya viene hecha en la plantilla.

Pasan a la final los 6 a 8 equipos con más puntaje. La final tiene su propia rúbrica.

---

## 7. Entregables y fechas (hora de Lima)

| Fecha | Qué |
|---|---|
| lun 7 dic, 23:59 | Cierre de inscripciones y del registro de equipos |
| **mar 8 dic, 23:59** | **Cierra Kaggle y el formulario de entregables. No se acepta nada después.** |
| jue 10 dic | Anuncio de finalistas por correo |
| sáb 12 dic, 2–5 p. m. | Final presencial en el auditorio de FACI (PUCP). Basta con que vaya uno |

**Los 4 entregables (todos obligatorios):**

- [ ] **A · Kaggle:** 2 submissions finales elegidas antes del cierre.
- [ ] **B · Repo público** con:
  - [ ] `README.md` reescrito: qué hace el producto y para quién, cómo instalarlo y correrlo, link al video e integrantes.
  - [ ] `train/submission_notebook.ipynb`: corre de principio a fin solo con la data oficial y genera el `submission.csv` final (semillas fijas, rutas relativas, salidas guardadas).
  - [ ] Pesos en `models/`, agregados con `git add -f` porque el `.gitignore` los excluye (si pasan de 100 MB, en Drive/Hugging Face con link en el README).
  - [ ] Código del producto en `app/` que carga **esos mismos pesos**.
  - [ ] `requirements.txt` con versiones.
  - [ ] `REPORTE_PRODUCTO.md` con las 6 preguntas respondidas.
- [ ] **C · Producto:** link al app desplegado, o instrucciones en el README para correrlo en local.
- [ ] **D · Video pitch:** link público, **máximo 3 minutos** (el último aviso dice 3; las bases decían 5). Menos diapositivas, más demo.

**Antes de enviar:**

- [ ] Alguien que no es del equipo clonó el repo y corrió el producto siguiendo solo el README.
- [ ] **Repo en público** (hoy está privado para que otros equipos no vean nuestro código):
  `gh repo edit Jimbotz/BurrosMX-Hackathon-IA-PUCP-2026 --visibility public --accept-visibility-change-consequences`
- [ ] Archivo `LICENSE` MIT agregado (lo exigen las reglas de Kaggle).
- [ ] Quitar de esta guía el link de invitación a Kaggle (es solo para participantes confirmados).
- [ ] El video se abre sin pedir permiso.
- [ ] Todo registrado en el formulario de entregables antes de las 23:59 del 8 dic.

---

## 8. Dudas

- **Sobre la data, Kaggle o las reglas:** al grupo oficial de WhatsApp de participantes.
- **Mentorías** (opcionales, 24 nov – 8 dic): la agenda está en el grupo; citas de 20–30 min, reservadas con 6 h de anticipación. Hay que llevar el nombre del equipo y una duda concreta.
- **Organizador:** Daniel Malca · dsc.pucp@gmail.com

---

## 9. Experimentos del modelo (bitácora)

Dos scripts en `train/` (requieren la data en `data/`; corren en ~3–7 min por modelo en una Mac M4 con `mps`, o en GPU):

```bash
# una corrida = un modelo; valida con las últimas 2 semanas de junio y anota en train/runs/log.jsonl
.venv/bin/python train/exp.py --name prueba --base interp --lags -2 -1 0 1 2 --loss metric --epochs 15 --swa_from 8

# score local del promedio de varias corridas, y submission de Kaggle
.venv/bin/python train/ensemble.py val prueba otra_corrida
.venv/bin/python train/ensemble.py train/runs/sub.csv corrida1 corrida2
```

Opciones de `exp.py`: `--base clim|interp` (sobre qué predice la red), `--lags` (horas del GFS relativas a t), `--loss mse|metric`, `--wd`, `--epochs`, `--seed`, `--swa_from N` (promedia los pesos desde la época N en vez de elegir la mejor época con val) y `--full` (entrena con todo el train, sin validación). Cada corrida guarda en `train/runs/<nombre>/` los pesos (`model.pt`, ~8 MB), `val_pred.npy` y `test_pred.npy`.

> ⚠️ En zsh (Mac) no pases las opciones dentro de una variable (`$OPCIONES`): zsh no las separa y argparse falla. Escríbelas directo.

### Resultados en validación local (mayor es mejor)

| Modelo | Val | RMSE RH | T | U | V |
|---|---|---|---|---|---|
| Starter (CNN chica, base climatología) | 0.338 | 9.98 | 0.97 | 1.72 | 1.61 |
| Regresión lineal por píxel (ridge) | 0.268 | 10.55 | 1.19 | 1.87 | 1.75 |
| U-Net, base climatología | 0.338 | 9.16 | 1.05 | 1.72 | 1.62 |
| U-Net, base **GFS interpolado + sesgo por hora** | 0.360 | 9.41 | 0.91 | 1.68 | 1.58 |
| + horas futuras del GFS (t−2…t+2) | 0.363 | 8.82 | 0.93 | 1.72 | 1.58 |
| 1 modelo honesto (pérdida = métrica, 15 épocas, SWA desde la 8) | 0.353–0.363 | ~9.6 | 0.90 | 1.69 | 1.58 |
| **Ensamble de 5 semillas (honesto)** | **0.377** | 9.27 | 0.88 | 1.64 | 1.54 |

"Honesto" = sin elegir la época mirando la validación; es el mismo procedimiento que usamos al entrenar con todo (`--full`). Los scores que eligen la mejor época salen ~0.01–0.02 más altos de lo real.

### Lo que aprendimos

- **Lo que más suma es promediar modelos** (semillas distintas): +0.02. El modelo tiene mucha varianza, sobre todo en el viento.
- **Base = GFS interpolado + sesgo medio por (hora, píxel)** le gana a la climatología: el test es julio–agosto (otra temporada) y la climatología del train (feb–jun) se queda corta, sobre todo en humedad.
- **El cuello de botella es el viento:** T y RH bajan a ~45–55 % del error de la climatología; U y V se quedan en ~73–79 %.
- La U-Net sola (sin cambiar la base) no mejora al starter. La regresión lineal por píxel es peor en viento.
- Entrenar 30 épocas sobreajusta (la mejor época suele ser la 12–15).
- La validación local **no** es comparable 1 a 1 con el leaderboard: el starter da 0.338 local y 0.428 en el público.

### Submission actual

`train/runs/sub_full_swa5.csv` = promedio de `full_s0` … `full_s4` (5 semillas entrenadas con todo el train, misma configuración que el ensamble honesto). Pendiente: subirla a Kaggle cuando los 4 estemos en el equipo `BurrosMX`.

### Ideas siguientes

- Más contexto para el viento: encoder del GFS con más alcance o un vector global del campo de 17×13.
- Validación cruzada temporal (varios cortes, no solo fines de junio) para decidir con menos ruido.
- Más semillas / arquitecturas distintas en el ensamble (de 3 a 5 semillas ganó +0.006).
- Ver cómo se correlaciona la validación local con el leaderboard público.
