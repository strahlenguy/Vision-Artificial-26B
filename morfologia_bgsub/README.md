# Morfología, descriptores de forma y background subtraction

Temas que van juntos: primero aprendemos a **describir formas con números**
(área, circularidad, solidez…), luego a **clasificarlas** con reglas y con
machine learning clásico, y después a **separar lo que se mueve del fondo**
en video. En todos, la morfología (apertura y cierre) es la que limpia la
máscara binaria.

## Archivos

| Archivo | Qué hace |
|---|---|
| `formas_paso_a_paso.ipynb` | **Parte 1.** Recorrido completo con OpenCV y scikit-image: gris → binaria → morfología → etiquetas → `regionprops` → descriptores → clasificación |
| `formas_ml.ipynb` | **Parte 2.** Tabla de características de muchas figuras, reglas a mano y machine learning: árbol de decisión, k-NN y Random Forest |
| `formas_regionprops.py` | Versión en script del pipeline de descriptores, con ventanas paso a paso |
| `generar_formas.py` | Dibuja imágenes sintéticas de figuras (una clase por columna) para hacer datasets |
| `bgsub_webcam.py` | Background subtraction mínimo con MOG2 (~30 líneas) |
| `bgsub_webcam_full.py` | Compara 4 métodos (diferencia de cuadros, promedio móvil, MOG2, KNN) con morfología y sliders |
| `imagenes/formas.jpg` | Imagen de la parte 1 |
| `imagenes/formas_anotada.jpg` | Resultado esperado del clasificador de la parte 1 |
| `imagenes/formas_dataset.png` | 3 figuras de cada clase (entrenamiento de la parte 2) |
| `imagenes/formas_prueba.png` | 2 figuras nuevas de cada clase (prueba de la parte 2) |

Orden sugerido: `formas_paso_a_paso.ipynb` → `formas_ml.ipynb` → `bgsub_webcam.py` → `bgsub_webcam_full.py`.

## Uso

Primero instala el entorno una vez desde la raíz del repo (`uv sync`, ver el
README principal). Luego:

```bash
cd morfologia_bgsub

# Notebooks (o ábrelos en VS Code con el kernel .venv de la raíz)
uv run jupyter lab

# Descriptores de forma
uv run formas_regionprops.py                 # usa imagenes/formas.jpg
uv run formas_regionprops.py mi_imagen.png

# Más figuras para el dataset
uv run generar_formas.py                     # regenera las dos imágenes de la parte 2
uv run generar_formas.py mas_formas.png 10 5 # 10 por clase, semilla 5

# Background subtraction
uv run bgsub_webcam.py
uv run bgsub_webcam_full.py                  # cámara 0
uv run bgsub_webcam_full.py 1                # otra cámara
uv run bgsub_webcam_full.py video.mp4        # un video
```

### Teclas de `bgsub_webcam_full.py`

| Tecla | Acción |
|---|---|
| `1`–`4` | Cambiar de método |
| `m` | Morfología on/off |
| `r` | Reiniciar el fondo |
| `q` | Salir |

Tip: apunta a una pared, presiona `r`, y luego entra al cuadro.

## Descriptores

| Descriptor | Fórmula | Interpretación |
|---|---|---|
| Circularidad | 4π·área / perímetro² | 1 = círculo, baja con esquinas |
| Solidez | área / área del casco convexo | 1 = sin entrantes |
| Convexidad | perímetro del casco / perímetro | 1 = contorno sin entrantes |
| Extensión | área / área de la caja | cuadrado 1, círculo π/4, triángulo 1/2 |
| Excentricidad | — | 0 = círculo, → 1 mientras más alargada |
| Relación de aspecto | ancho / alto de la caja | — |

## Ideas para practicar

- Toma una foto de objetos sobre una hoja blanca y corre `formas_regionprops.py`
  con ella. ¿Qué descriptores separan mejor tus clases?
- En `formas_paso_a_paso.ipynb`, cambia el elemento estructurante (cruz vs.
  cuadrado) y compara el resultado de la apertura.
- En `formas_ml.ipynb`, genera 10 figuras por clase y vuelve a probar k-NN con
  distintos valores de k.
- En `bgsub_webcam_full.py`, ¿qué método aguanta mejor un cambio de luz? ¿Y una
  persona que se queda quieta?
