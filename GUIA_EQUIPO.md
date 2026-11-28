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

## 9. Experimentos del modelo

**Resultados de todos los modelos con todas las métricas: [`docs/RESULTADOS.md`](docs/RESULTADOS.md).** El reporte narrativo está en [`docs/ANALISIS.md`](docs/ANALISIS.md): análisis de datos, todos los experimentos, el ensamble y la explicabilidad, con figuras en `docs/analisis/`.

**Estado actual:** ensamble de 18 redes con pesos por variable → **0.388** en validación cruzada temporal (starter: 0.338). La submission está generada (`train/runs/sub_ensamble_v2.csv`) y **todavía no se envió a Kaggle**.

### Cómo correr experimentos

```bash
# una configuración en un fold (0: 16–30 jun · 1: 1–15 jun · 2: 16–31 may); resultados en train/runs/log.jsonl
.venv/bin/python train/exp.py --name prueba_f0 --fold 0 --base interp --lags -3 -2 -1 0 1 2 3 --loss metric \
    --epochs 15 --swa_from 8 --amp --width 96 --bias_days 30 --feats vap daily
.venv/bin/python train/resumen.py prueba            # tabla por configuración (media de folds)
.venv/bin/python train/ensemble.py val cfg1 cfg2    # score CV de un ensamble
.venv/bin/python train/ensemble.py greedy cfg1 ...  # pesos por variable -> train/runs/ensamble.json
bash train/rondas/final.sh                          # modelos finales con todo el train + submission
.venv/bin/python train/test_modelo.py               # chequeo rápido del modelo
```

Las rondas de experimentos están versionadas en `train/rondas/` (a.sh … e.sh). Con una GPU (en nuestro caso, una RTX 2060 por SSH) cada modelo tarda ~1–2 min; en la Mac M4, ~3–7 min.

> ⚠️ En zsh (Mac) no pases las opciones dentro de una variable (`$OPCIONES`): zsh no las separa. Los scripts de `train/rondas/` son de **bash**.

### Usar el modelo en el producto

```python
from modelo import Ensamble
ens = Ensamble.desde_json('models/ensamble.json')   # los pesos deben estar en models/full_*/
Y = ens.predecir(X, times)    # X (H,17,13,4) del GFS de Open-Meteo, times 'YYYYMMDDHH' UTC -> (H,70,35,4)
```

Ojo: algunas configuraciones miran hasta ±3.5 días (`semana`) y ±3 h del GFS. Al pedir el pronóstico, incluye horas pasadas (`past_hours`) para que las ventanas tengan datos.
