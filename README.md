# Visión Artificial · 2026B

Código del curso de **Visión Artificial** (CUTLAJO).

Aquí está lo que vemos en clase y el código base para las actividades. Cada
carpeta es un tema con su propio README, y **todas comparten un solo entorno
de Python**: se instala una vez en la raíz y sirve para todo el semestre.

## 📌 Última clase

**[`morfologia_bgsub/formas_ml.ipynb`](morfologia_bgsub/formas_ml.ipynb)** ·
Parte 2 de descriptores de forma: tabla de características de muchas figuras,
reglas a mano y machine learning clásico (árbol de decisión, k-NN y Random
Forest). Viene después de
[`formas_paso_a_paso.ipynb`](morfologia_bgsub/formas_paso_a_paso.ipynb) (parte 1).

```bash
git pull
uv sync
uv run jupyter lab   # y abre morfologia_bgsub/formas_ml.ipynb
```

## Contenido

| Carpeta | Tema | Qué hay | Cámara | GPU |
|---|---|---|---|---|
| [`morfologia_bgsub/`](morfologia_bgsub/) | Morfología, descriptores de forma, machine learning clásico y background subtraction | Notebooks paso a paso y scripts con webcam | Sí (scripts de bgsub) | No |
| [`media_pipe/`](media_pipe/) | Landmarks de cuerpo, manos y cara con MediaPipe | Scripts de pose y detector configurable con Holistic | Sí | No |
| [`sam3-dataset/`](sam3-dataset/) | Generar datasets con SAM 3 | Scripts y notebook de Colab para recortar objetos por prompt de texto | No | Recomendada |
| [`entrenar_yolo/`](entrenar_yolo/) | Entrenar y comparar YOLO con tu propio dataset | Un notebook que corre igual en Colab, en tu PC o en la DGX Spark | No | Recomendada |

Si no sabes por dónde empezar, empieza por `morfologia_bgsub/`.

## Antes de empezar

Necesitas:

- **Python 3.12**
- **Git**
- **[uv](https://docs.astral.sh/uv/)**, el gestor de paquetes que usamos en clase.
  Crea el entorno virtual e instala todo con un solo comando (y si no tienes
  Python 3.12, lo descarga solo).

Instalar uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Clonar el repositorio

```bash
git clone https://github.com/strahlenguy/Vision-Artificial-26B.git
cd Vision-Artificial-26B
```

Para traer el material nuevo que se vaya subiendo durante el semestre:

```bash
git pull
```

> **Tip:** no edites los archivos del repo directamente; copia el script a tu
> propia carpeta o crea un archivo nuevo. Así `git pull` nunca choca con tus
> cambios.

## Instalar el entorno (una sola vez)

Desde la raíz del repo:

```bash
uv sync
```

Esto crea **un solo** `.venv/` en la raíz con todo lo del curso (OpenCV,
scikit-image, scikit-learn, MediaPipe, Jupyter…). Las versiones exactas
quedan fijas en `uv.lock`, así que a todos nos instala lo mismo.

SAM 3 necesita PyTorch y pesa varios GB, por eso **no** se instala por
default. Solo si vas a usar `sam3-dataset/`:

```bash
uv sync --group sam3
```

Lo mismo para entrenar YOLO con `entrenar_yolo/` (Ultralytics + PyTorch):

```bash
uv sync --group yolo
```

## Correr un tema

Entra a la carpeta del tema y usa `uv run`. uv busca el `pyproject.toml`
hacia arriba, así que usa el entorno de la raíz desde cualquier subcarpeta.

```bash
cd morfologia_bgsub
uv run bgsub_webcam.py
```

Para los notebooks, abre Jupyter desde la raíz y navega a la carpeta:

```bash
uv run jupyter lab
```

O abre la carpeta del repo en VS Code y elige el kernel de `.venv` (el de la
raíz). Es el mismo para todos los notebooks.

## Google Colab

Si no tienes Python instalado o tu computadora no tiene GPU, estos notebooks
corren directo en Colab:

| Notebook | Abrir |
|---|---|
| Entrenar YOLO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/strahlenguy/Vision-Artificial-26B/blob/main/entrenar_yolo/entrenar_yolo.ipynb) |
| Dataset con SAM 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/strahlenguy/Vision-Artificial-26B/blob/main/sam3-dataset/notebooks/sam3_dataset_colab.ipynb) |

Los scripts con webcam (`cv2.imshow`) **no** funcionan en Colab: córrelos en
tu computadora.

## Problemas comunes

- **La cámara no abre o sale negra.** Prueba otro índice (`--camara 1` o
  pasando `1` como argumento, según el script). En macOS, da permiso de cámara
  a tu terminal o a VS Code en *Configuración del sistema → Privacidad y
  seguridad → Cámara*.
- **La ventana no responde.** Las ventanas de OpenCV se cierran con `q` (o
  `ESC`) con la ventana enfocada, no con la ❌.
- **`uv: command not found`.** Cierra y abre la terminal después de instalar uv.
- **Error al instalar `mediapipe`.** MediaPipe todavía no tiene versión para
  todos los Python; el repo ya está fijo en 3.12 (`.python-version`). Si uv no
  lo encuentra, corre `uv python install 3.12`.
- **`ModuleNotFoundError` en un notebook.** Revisa que el kernel sea el
  `.venv` de la raíz del repo y que hayas corrido `uv sync`.

## Estructura

```
Vision-Artificial-26B/
├── pyproject.toml      # dependencias de todo el curso
├── uv.lock             # versiones exactas (no se edita a mano)
├── .python-version     # Python 3.12
├── morfologia_bgsub/   # morfología, regionprops, ML clásico, background subtraction
├── media_pipe/         # landmarks con MediaPipe (API clásica)
├── sam3-dataset/       # generación de datasets con SAM 3
└── entrenar_yolo/      # entrenar y comparar YOLO (Colab, PC o DGX Spark)
```

Cada carpeta nueva que se agregue durante el semestre seguirá el mismo
formato: su propio README y código listo para correr. Si necesita una
librería nueva, se agrega a la raíz con `uv add <paquete>`.
