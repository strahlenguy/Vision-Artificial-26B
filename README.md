# Visión Artificial · 2026B

Código del curso de **Visión Artificial** (CUTLAJO).

Aquí está lo que vemos en clase y el código base para las actividades. Cada
carpeta es un tema independiente: tiene su propio README, sus dependencias y
se instala por separado.

## Contenido

| Carpeta | Tema | Qué hay | Cámara | GPU |
|---|---|---|---|---|
| [`morfologia_bgsub/`](morfologia_bgsub/) | Morfología, descriptores de forma y background subtraction | Notebooks paso a paso (OpenCV y NumPy puro) y scripts con webcam | Sí (scripts de bgsub) | No |
| [`media_pipe/`](media_pipe/) | Landmarks de cuerpo, manos y cara con MediaPipe | Scripts de pose y detector configurable con Holistic | Sí | No |
| [`ejemplos/`](ejemplos/) | Demos en tiempo real | MediaPipe Tasks (manos/pose/cara) y segmentación con SAM 3 | Sí | Recomendada para SAM 3 |
| [`sam3-dataset/`](sam3-dataset/) | Generar datasets con SAM 3 | Scripts y notebook de Colab para recortar objetos por prompt de texto | No | Recomendada |

Si no sabes por dónde empezar, empieza por `morfologia_bgsub/`.

## Antes de empezar

Necesitas:

- **Python 3.12**
- **Git**
- **[uv](https://docs.astral.sh/uv/)**, el gestor de paquetes que usamos en clase.
  Crea el entorno virtual e instala todo con un solo comando.

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

## Correr un tema

Entra a la carpeta y usa `uv run`. La primera vez crea el entorno `.venv` e
instala las dependencias de esa carpeta; las siguientes es inmediato.

```bash
cd morfologia_bgsub
uv run bgsub_webcam.py
```

Para los notebooks:

```bash
cd morfologia_bgsub
uv run --with jupyter jupyter lab
```

(o abre la carpeta en VS Code y elige el kernel de `.venv`).

`sam3-dataset/` usa `requirements.txt` en lugar de uv; sigue su README.

## Google Colab

Si no tienes Python instalado o tu computadora no tiene GPU, estos notebooks
corren directo en Colab:

| Notebook | Abrir |
|---|---|
| Descriptores de forma (OpenCV) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/strahlenguy/Vision-Artificial-26B/blob/main/morfologia_bgsub/formas_paso_a_paso_colab.ipynb) |
| Descriptores de forma (NumPy desde cero) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/strahlenguy/Vision-Artificial-26B/blob/main/morfologia_bgsub/formas_paso_a_paso_numpy.ipynb) |
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
  todos los Python; usa 3.12 (`uv python install 3.12`).

## Estructura

```
Vision-Artificial-26B/
├── morfologia_bgsub/   # morfología, regionprops, background subtraction
├── media_pipe/         # landmarks con MediaPipe (API clásica)
├── ejemplos/           # demos en vivo: MediaPipe Tasks y SAM 3
└── sam3-dataset/       # generación de datasets con SAM 3
```

Cada carpeta nueva que se agregue durante el semestre seguirá el mismo
formato: su propio README, `pyproject.toml` y código listo para correr.
