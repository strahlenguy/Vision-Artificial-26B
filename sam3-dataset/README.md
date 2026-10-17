# sam3-dataset

Herramientas para generar datasets de recortes de objetos usando **SAM 3**.
Le das un directorio de imágenes y un prompt de texto, y te regresa los objetos
recortados, organizados por clase y listos para entrenar.

Material del curso de **Visión Artificial** (CUTLAJO).

## Estructura

```
sam3-dataset/
├── scripts/
│   ├── sam3_recortar_dataset_transformers.py  # versión transformers (Mac, Linux, Windows, Colab)
│   ├── sam3_recortar_dataset_oficial.py       # versión oficial con sam3.pt (Linux + CUDA)
│   └── descargar_pesos_sam3.py                # baja y verifica sam3.pt
├── notebooks/
│   └── sam3_dataset_colab.ipynb               # recorrido paso a paso, listo para Colab
├── pesos/                                     # sam3.pt va aquí (ignorado por git)
├── datos/
│   ├── raw/                                   # imágenes de entrada (ignorado)
│   ├── referencias/                           # fotos del objeto buscado (ignorado, opcional)
│   └── salida/                                # el dataset generado (ignorado)
├── requirements.txt                           # para pip/Colab (con uv: uv sync --group sam3)
└── README.md
```

Las cuatro carpetas de datos y pesos están en `.gitignore`. El repo lleva solo
código: los pesos pesan GB y tienen su propia licencia, y las imágenes son de
cada quien. Los `.gitkeep` existen para que la estructura sí se clone.

## Dos versiones del mismo pipeline

Los dos scripts hacen exactamente lo mismo (mismos argumentos, filtros y
salida). Solo cambia cómo se carga y se llama a SAM 3:

| | `..._transformers.py` | `..._oficial.py` |
|---|---|---|
| Librería | Hugging Face `transformers` | repo oficial de Meta (`sam3`) |
| Pesos | `facebook/sam3` (se bajan solos a la caché de HF) | archivo `sam3.pt` local |
| Corre en | GPU NVIDIA, Apple Silicon (`mps`), CPU | solo Linux + GPU NVIDIA |
| `--checkpoint` | id de HF o carpeta con `model.safetensors` | ruta a `sam3.pt` |

El oficial depende de `triton`, que solo existe para Linux con CUDA; por eso no
corre en Mac. Si no sabes cuál usar, usa el de `transformers`.

## Instalación

**Con uv (recomendado en tu compu).** Usa el entorno compartido de la raíz del
repo, agregando el grupo `sam3` (torch, transformers y compañía):

```bash
git clone https://github.com/strahlenguy/Vision-Artificial-26B.git
cd Vision-Artificial-26B
uv sync --group sam3
cd sam3-dataset
```

Con eso ya puedes correr la versión transformers cambiando `python` por
`uv run` en los comandos de abajo (por ejemplo,
`uv run scripts/sam3_recortar_dataset_transformers.py --help`).

**Con pip** (Colab o si no usas uv). `requirements.txt` tiene lo mismo que el
grupo `sam3`:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Versión oficial** (solo Linux + CUDA), además de lo anterior:

```bash
pip install git+https://github.com/facebookresearch/sam3.git
pip install einops pycocotools psutil "setuptools<81"   # deps que el repo no declara
```

Requiere Python >= 3.12.

## Pesos del modelo

1. Pide acceso en https://huggingface.co/facebook/sam3 y espera la aprobación.
2. Autentícate: `hf auth login`

**Versión transformers:** no hay que hacer nada más; la primera corrida baja
los pesos (~3.4 GB) a `~/.cache/huggingface`. El `sam3.pt` no le sirve.

**Versión oficial:** baja el `sam3.pt` a `pesos/`:

```bash
python scripts/descargar_pesos_sam3.py -d pesos/
```

Si alguien te compartió los pesos por un espejo interno (sin necesidad de
cuenta ni token):

```bash
python scripts/descargar_pesos_sam3.py -d pesos/ \
    --url http://<ip-del-servidor>:8000/sam3.pt --sha256 <hash>
```

El uso de los pesos está sujeto a la SAM License. Si los redistribuyes, tienes
que incluir una copia de esa licencia; el script la deja junto al `.pt`.

## Uso

```bash
# 1. Pon tus imágenes en datos/raw/

# 2. Calibra umbrales con pocas imágenes
python scripts/sam3_recortar_dataset_transformers.py \
    -i datos/raw -o datos/salida/prueba \
    -p "casco=hard hat" \
    --limite 20

# 3. Revisa datos/salida/prueba/mosaico_*.jpg y metadatos.csv,
#    ajusta --umbral y --area-min, y corre todo

python scripts/sam3_recortar_dataset_transformers.py \
    -i datos/raw -o datos/salida/v1 \
    -p "casco=hard hat" -p "chaleco=safety vest" \
    --fondo transparente --tamano 224 --exportar-yolo
```

Con la versión oficial es igual, cambiando el script y agregando el checkpoint:

```bash
python scripts/sam3_recortar_dataset_oficial.py \
    -i datos/raw -o datos/salida/v1 -p "casco=hard hat" \
    --checkpoint pesos/sam3.pt
```

**Ejemplo con las fotos de óvulos de `datos/raw/`:** SAM 3 no reconoce
"oocyte" ni "egg", pero sí "round cell". `--area-min 0.01` quita los restos:

```bash
python scripts/sam3_recortar_dataset_transformers.py \
    -i datos/raw -o datos/salida/demo -p "ovocito=round cell" --area-min 0.01
```

## Colab

Abre `notebooks/sam3_dataset_colab.ipynb` en https://colab.research.google.com
(Archivo → Subir notebook, o desde Google Drive con "Abrir con → Google
Colaboratory"). Selecciona una GPU en Entorno de ejecución → Cambiar tipo de
entorno de ejecución → T4 GPU. El notebook explica cada paso del pipeline y
al final descarga el dataset como `.zip`.

Los prompts van en inglés, como frases nominales cortas. El formato
`clase=frase` deja la carpeta en español y el prompt en el idioma del modelo.

Ayuda completa de cada script:

```bash
python scripts/sam3_recortar_dataset_transformers.py --help
python scripts/sam3_recortar_dataset_oficial.py --help
python scripts/descargar_pesos_sam3.py --help
```

## Salida

```
datos/salida/v1/
├── recortes/
│   ├── casco/            # un archivo por objeto
│   └── chaleco/
├── metadatos.csv         # score, caja, área y similitud de cada recorte
├── mosaico_casco.jpg     # hoja de contacto para revisar de un vistazo
├── resumen.json          # parámetros usados en la corrida
└── yolo/                 # solo con --exportar-yolo
    ├── images/
    ├── labels/
    └── data.yaml
```

Revisa siempre los mosaicos antes de entrenar. SAM 3 etiqueta muy bien, pero no
distingue variantes específicas de un objeto, así que el filtro humano sigue
siendo parte del proceso.

## Licencia

El código de este repo: elige la que quieras (MIT, Apache 2.0) y agrega el
archivo `LICENSE`. Los pesos de SAM 3 **no** están cubiertos por ella; se rigen
por la SAM License de Meta.
