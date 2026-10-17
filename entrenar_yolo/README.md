# Entrenar YOLO con tu propio dataset

Un solo notebook, `entrenar_yolo.ipynb`, para entrenar un YOLO con tu dataset, ver sus
métricas y compararlo con otras versiones y tamaños. Funciona igual en **Google Colab**,
en **tu computadora** y en la **DGX Spark**: detecta dónde está corriendo y se ajusta solo.

| | Colab | Tu computadora | DGX Spark |
|---|---|---|---|
| GPU | T4 gratis (hay que prenderla) | la tuya: NVIDIA o Mac M | GB10, compartida |
| Resultados | tu Google Drive | `runs/`, junto al notebook | `runs/` común del grupo |
| Batch automático | según la T4 | según tu GPU | conservador (memoria unificada) |
| Turnos de GPU | no | no | sí, uno entrena a la vez |
| Si se cae la sesión | checkpoint cada 5 épocas y reanudas | — | — |

## Qué hace el notebook

| Paso | Qué pasa |
|---|---|
| 0 | Instala lo que falte |
| 1 | **Configuración**: la única celda que editas (tu nombre, dataset, versión, tamaño…) |
| 2 | Funciones de apoyo (oculta; córrela y sigue) |
| 3 | Detecta el equipo: GPU, memoria, rutas |
| 4 | Prepara y revisa el dataset, y te muestra muestras con sus etiquetas |
| 5 | Tabla de versiones y tamaños disponibles |
| 6 | Entrena, con gráficas que se actualizan cada época |
| 7 | Reanuda una corrida que se cortó |
| 8 | Curvas, matriz de confusión y métricas por clase |
| 9 | Prueba el modelo con imágenes nuevas |
| 10 | **Compara todas las corridas**: mAP contra minutos de entrenamiento |
| 11 | Exporta el modelo (ONNX) |

Versiones disponibles: YOLOv8, v9, v10, 11, 12 y 26, en detección y segmentación (el
Paso 5 dice qué tamaños hay de cada una).

## Datasets que acepta

En la variable `DATASET` del Paso 1:

| Tienes | Pones |
|---|---|
| Nada todavía | `"demo"`: genera figuras para probar el flujo en minutos |
| Export de **Roboflow** (formato YOLOv8/YOLO11) | la ruta del `.zip` |
| Salida de [`sam3-dataset/`](../sam3-dataset/) | la carpeta `yolo/` que generó |
| Tu propio `data.yaml` | su ruta |
| Carpeta con `images/` y `labels/` sin yaml | la carpeta, y los nombres en `CLASES` |

Lo que el notebook arregla solo:

- **`train` y `val` iguales** (así sale el export de SAM 3): divide 80/20. Si no, el modelo
  se califica con imágenes que ya vio y el mAP sale inflado.
- **Rutas relativas** del `data.yaml`, incluidas las de Roboflow con `../`.
- **Polígonos** (SAM 3): sirven tanto para `segment` como para `detect`.

Tus archivos originales no se modifican: lo que el notebook genera va a `datos/`.

## Cómo abrirlo

### En tu computadora

Ultralytics y PyTorch pesan varios GB, por eso van en un grupo aparte del entorno del
curso. Desde la **raíz del repo**, una sola vez:

```bash
uv sync --group yolo
```

y luego:

```bash
uv run jupyter lab
```

y abre `entrenar_yolo/entrenar_yolo.ipynb`. El `pyproject.toml` de la raíz elige solo el
PyTorch correcto: CUDA 12.8 en Windows y Linux, CUDA 13 en ARM64 (la DGX Spark) y el de
Apple (MPS) en Mac.

Pon tus datasets en `entrenar_yolo/datos/` (se crea sola la primera vez).

### En Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/strahlenguy/Vision-Artificial-26B/blob/main/entrenar_yolo/entrenar_yolo.ipynb)

1. Ábrelo con el botón y guarda una copia en tu Drive (*Archivo → Guardar una copia en Drive*).
2. *Entorno de ejecución → Cambiar tipo de entorno de ejecución → **T4 GPU***.
3. Sube tu dataset a tu Drive en `VisionArtificial/entrenar_yolo/datos/`. En `DATASET`
   basta con el nombre: `"humo.zip"`.

El notebook monta tu Drive y guarda ahí las corridas, así no se pierden si Colab se
desconecta. Antes de entrenar copia el dataset al disco de Colab, porque leer miles de
imágenes directo de Drive es lentísimo.

**Comparar con tu equipo en Colab:** una persona comparte su carpeta
`VisionArtificial` (*Compartir → Editor*) y las demás le agregan un acceso directo en su
propio Drive (clic derecho → *Organizar* → *Añadir acceso directo*). Sin el acceso
directo, la ruta no existe para ellas.

### En la DGX Spark (para el profesor)

```bash
cd entrenar_yolo
./servir_dgx.sh 8888 alumnos.txt
```

`alumnos.txt` lleva un nombre por línea. El script:

- crea `alumnos/<nombre>/` con una copia del notebook para cada quien,
- hace que todas las corridas vayan a `runs/`, así la tabla del Paso 10 muestra al grupo entero,
- descarga una sola vez los pesos base más usados a `pesos/`,
- activa los **turnos**: si alguien está entrenando, la celda de los demás dice
  *Esperando turno* y arranca sola cuando se libera,
- imprime la dirección con token que los alumnos abren en el navegador.

**Por qué turnos:** la Spark tiene memoria unificada (la RAM y la memoria de la GPU son
la misma). Un batch de más no da un error limpio de CUDA: puede congelar la máquina de
todos. Por eso el batch automático usa solo una cuarta parte de la memoria libre.

Si PyTorch avisa que la capacidad 12.1 del GB10 está fuera de rango, se puede ignorar.

Alternativa sin instalar nada, con el contenedor oficial de Ultralytics:

```bash
sudo docker run -it --ipc=host --device nvidia.com/gpu=all -p 8888:8888 \
    -v "$PWD":/work -w /work ultralytics/ultralytics:latest-nvidia-arm64
```

## Lo que se genera

```
entrenar_yolo/
├── entrenar_yolo.ipynb
├── datos/          tus datasets, y las versiones divididas que crea el notebook
├── runs/           una carpeta por corrida: pesos, curvas, results.csv, meta_clase.json
└── pesos/          pesos base descargados (yolo11n.pt, etc.)
```

Nada de eso se sube a git (está en el `.gitignore`).

`meta_clase.json` guarda quién entrenó, con qué modelo, cuánto tardó y si terminó.
Es lo que usa la tabla comparativa.

## Problemas comunes

| Síntoma | Causa y arreglo |
|---|---|
| `[!] ¡Colab está SIN GPU!` | Prende la T4 (Entorno de ejecución) y corre de nuevo desde el Paso 0 |
| `No encuentro 'algo.zip'` | La ruta está mal. En Colab, ¿está en `VisionArtificial/entrenar_yolo/datos/` de tu Drive? |
| `[X] Hay etiquetas con clase N` | Tus `.txt` usan una clase que el `data.yaml` no declara. Revisa `names` |
| `ModuleNotFoundError: ultralytics` | Te faltó `uv sync --group yolo` en la raíz del repo |
| `CUDA out of memory` | Pon un `BATCH` más chico (16, 8…) o baja `IMGSZ` |
| En la Spark, `GPU NVIDIA: False` | Falta el PyTorch de CUDA 13: `uv sync --group yolo` o `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` |
| En Windows se traba al empezar a entrenar | Pasa `workers=0` al final de `entrenar(...)` en el Paso 6 |
| Se cortó el entrenamiento | Paso 7: `reanudar(E)` |
| El mAP sale casi perfecto con muy pocas épocas | Revisa que `val` no tenga copias de `train` (fotos casi iguales en los dos) |
