# Morfología, descriptores de forma y background subtraction

Dos temas que van juntos: primero aprendemos a **describir formas con
números** (área, circularidad, solidez…) y después a **separar lo que se
mueve del fondo** en video. En los dos, la morfología (apertura y cierre) es
la que limpia la máscara binaria.

## Archivos

| Archivo | Qué hace |
|---|---|
| `formas_paso_a_paso.ipynb` | Recorrido completo con OpenCV y scikit-image: gris → binaria → morfología → etiquetas → `regionprops` → descriptores → clasificación |
| `formas_paso_a_paso_colab.ipynb` | El mismo notebook, listo para Colab (te pide subir la imagen o genera una) |
| `formas_paso_a_paso_numpy.ipynb` | El mismo recorrido **sin OpenCV ni scikit-image**: cada operación escrita a mano con NumPy |
| `formas_regionprops.py` | Versión en script del pipeline de descriptores, con ventanas paso a paso |
| `bgsub_webcam.py` | Background subtraction mínimo con MOG2 (~30 líneas) |
| `bgsub_webcam_full.py` | Compara 4 métodos (diferencia de cuadros, promedio móvil, MOG2, KNN) con morfología y sliders |
| `formas.jpg` | Imagen de prueba |
| `formas_anotada.jpg` | Resultado esperado del clasificador |

Orden sugerido: `formas_paso_a_paso.ipynb` → `formas_paso_a_paso_numpy.ipynb`
→ `bgsub_webcam.py` → `bgsub_webcam_full.py`.

## Uso

```bash
cd morfologia_bgsub

# Notebooks
uv run --with jupyter jupyter lab

# Descriptores de forma
uv run formas_regionprops.py                 # usa formas.jpg
uv run formas_regionprops.py mi_imagen.png

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
- En el notebook de NumPy, cambia el elemento estructurante (cruz vs. cuadrado)
  y compara el resultado de la apertura.
- En `bgsub_webcam_full.py`, ¿qué método aguanta mejor un cambio de luz? ¿Y una
  persona que se queda quieta?
