"""Genera imágenes sintéticas de figuras para practicar descriptores y ML.

Cada COLUMNA de la imagen es una clase (en el orden de CLASES) y cada renglón
es un ejemplo distinto: cambia el tamaño, la rotación y el color. Así etiquetar
es fácil: la clase de una figura depende de en qué columna cae su centro.

    uv run generar_formas.py                      # las dos imágenes del notebook
    uv run generar_formas.py mis_formas.png 5 7   # 5 por clase, semilla 7
"""

import sys

import cv2
import numpy as np

CLASES = ["círculo", "cuadrado", "rectángulo", "triángulo", "hexágono", "estrella"]
CELDA = 200          # tamaño de cada celda en píxeles
FONDO = (230, 230, 230)
COLORES = [          # BGR, todos más oscuros que el fondo
    (35, 35, 220), (200, 60, 30), (60, 160, 40), (150, 50, 140),
    (20, 120, 230), (140, 120, 20), (80, 80, 80), (40, 90, 160),
]


def poligono_regular(n, radio, angulo):
    t = angulo + np.arange(n) * 2 * np.pi / n
    return np.stack([radio * np.cos(t), radio * np.sin(t)], 1)


def puntos_figura(clase, rng):
    """Vértices de la figura centrados en (0, 0). None para el círculo."""
    giro = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(45, 80)
    if clase == "círculo":
        return None, r
    if clase == "cuadrado":
        p = poligono_regular(4, r, giro)
    elif clase == "rectángulo":
        ancho, alto = r * 1.9, r * rng.uniform(0.75, 1.05)
        p = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [ancho / 2, alto / 2]
    elif clase == "triángulo":
        # equilátero estirado a lo ancho o a lo alto: no todos los triángulos son iguales
        r = min(r, 65)
        p = poligono_regular(3, r, -np.pi / 2) * [rng.uniform(0.75, 1.45), 1]
    elif clase == "hexágono":
        p = poligono_regular(6, r, giro)
    elif clase == "estrella":
        externos = poligono_regular(5, r, -np.pi / 2)
        internos = poligono_regular(5, r * rng.uniform(0.4, 0.5), -np.pi / 2 + np.pi / 5)
        p = np.empty((10, 2))
        p[0::2], p[1::2] = externos, internos
    if clase in ("rectángulo", "triángulo", "estrella"):
        c, s = np.cos(giro), np.sin(giro)
        p = p @ np.array([[c, s], [-s, c]])
    return p, r


def generar(por_clase=3, semilla=0):
    rng = np.random.default_rng(semilla)
    img = np.full((CELDA * por_clase, CELDA * len(CLASES), 3), FONDO, np.uint8)
    for col, clase in enumerate(CLASES):
        for fila in range(por_clase):
            cx = col * CELDA + CELDA // 2 + rng.integers(-8, 9)
            cy = fila * CELDA + CELDA // 2 + rng.integers(-8, 9)
            color = COLORES[rng.integers(len(COLORES))]
            p, r = puntos_figura(clase, rng)
            if p is None:
                cv2.circle(img, (int(cx), int(cy)), int(r), color, -1, cv2.LINE_AA)
            else:
                p = np.round(p + [cx, cy]).astype(np.int32)
                cv2.fillPoly(img, [p], color, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        semilla = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        cv2.imwrite(ruta, generar(n, semilla))
        print(f"Guardé {ruta}")
    else:
        cv2.imwrite("imagenes/formas_dataset.png", generar(3, semilla=1))
        cv2.imwrite("imagenes/formas_prueba.png", generar(2, semilla=2))
        print("Guardé imagenes/formas_dataset.png e imagenes/formas_prueba.png")
