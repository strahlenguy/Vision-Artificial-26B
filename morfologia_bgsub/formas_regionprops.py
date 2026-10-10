"""
Descriptores de forma con regionprops
=====================================

Pipeline:
    imagen -> gris -> umbral (Otsu) -> limpieza morfológica -> etiquetado
           -> regionprops -> descriptores -> clasificación simple

Descriptores que calculamos para cada figura:

    área          píxeles que ocupa la región
    perímetro     longitud del contorno
    circularidad  4·π·área / perímetro²     (1 = círculo perfecto, baja con esquinas)
    solidez       área / área del casco convexo      (1 = no tiene "huecos" hacia adentro)
    convexidad    perímetro del casco / perímetro    (1 = contorno sin entrantes)
    extensión     área / área de la caja (bbox)      (cuadrado 1, círculo π/4, triángulo 1/2)
    excentricidad 0 = círculo, se acerca a 1 mientras más alargada
    relación de aspecto   ancho / alto de la caja

Uso:
    uv run formas_regionprops.py                 # usa formas.jpg
    uv run formas_regionprops.py otra_imagen.png

Teclas: cualquier tecla para pasar a la siguiente ventana, q para salir.
"""

import sys

import cv2
import numpy as np
from skimage import measure

RUTA = sys.argv[1] if len(sys.argv) > 1 else "formas.jpg"
AREA_MIN = 100  # px; regiones más chicas se consideran ruido


# 1. Leer y binarizar ----------------------------------------------------------
imagen = cv2.imread(RUTA)
if imagen is None:
    sys.exit(f"No pude abrir {RUTA}")
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Fondo claro y figuras oscuras -> THRESH_BINARY_INV para que las figuras queden en blanco.
# Otsu elige el umbral solo, viendo el histograma.
umbral, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(f"Umbral de Otsu: {umbral:.0f}")

# 2. Limpieza morfológica --------------------------------------------------------
# Apertura (erosión -> dilatación): quita puntitos sueltos.
# Cierre (dilatación -> erosión): tapa huequitos dentro de las figuras.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
limpia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, kernel)

# 3. Etiquetado: cada región conectada recibe un número 1, 2, 3...
etiquetas = measure.label(limpia > 0, connectivity=2)
regiones = [r for r in measure.regionprops(etiquetas) if r.area >= AREA_MIN]
print(f"Regiones encontradas: {len(regiones)}\n")


# 4. Descriptores ----------------------------------------------------------------
def perimetro_casco(region):
    """Perímetro del casco convexo, medido igual que el de la región."""
    casco = np.pad(region.image_convex, 1)
    return measure.perimeter(casco)


def clasificar(d):
    """Reglas simples basadas en los descriptores. Ajusta los umbrales y observa."""
    if d["circularidad"] > 0.85 and d["excentricidad"] < 0.3:
        return "circulo"
    if d["extension"] > 0.9:
        return "cuadrado" if 0.9 < d["aspecto"] < 1.1 else "rectangulo"
    if 0.4 < d["extension"] < 0.6 and d["solidez"] > 0.95:
        return "triangulo"
    return "?"


descriptores = []
for r in regiones:
    y0, x0, y1, x1 = r.bbox
    d = {
        "id": r.label,
        "area": r.area,
        "perimetro": r.perimeter,
        "circularidad": 4 * np.pi * r.area / r.perimeter**2,
        "solidez": r.solidity,
        "convexidad": perimetro_casco(r) / r.perimeter,
        "extension": r.extent,
        "excentricidad": r.eccentricity,
        "aspecto": (x1 - x0) / (y1 - y0),
        "centroide": (int(r.centroid[1]), int(r.centroid[0])),
        "bbox": (x0, y0, x1, y1),
        "region": r,
    }
    d["forma"] = clasificar(d)
    descriptores.append(d)

# Tabla en consola
columnas = ["id", "forma", "area", "perimetro", "circularidad", "solidez",
            "convexidad", "extension", "excentricidad", "aspecto"]
print("  ".join(f"{c:>13}" for c in columnas))
for d in descriptores:
    fila = []
    for c in columnas:
        v = d[c]
        fila.append(f"{v:>13.3f}" if isinstance(v, float) else f"{v!s:>13}")
    print("  ".join(fila))


# 5. Visualización ---------------------------------------------------------------
anotada = imagen.copy()
for d in descriptores:
    r = d["region"]
    x0, y0, x1, y1 = d["bbox"]

    # Contorno real (verde) vs. casco convexo (magenta)
    mascara = (etiquetas == r.label).astype(np.uint8)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    casco = cv2.convexHull(contornos[0])
    cv2.drawContours(anotada, [casco], -1, (255, 0, 255), 2)
    cv2.drawContours(anotada, contornos, -1, (0, 255, 0), 2)

    # Caja (amarillo) y centroide
    cv2.rectangle(anotada, (x0, y0), (x1, y1), (0, 220, 255), 1)
    cv2.circle(anotada, d["centroide"], 4, (0, 0, 0), -1)

    # Etiqueta con los descriptores principales
    texto = [
        f"#{d['id']} {d['forma']}",
        f"A={d['area']:.0f}",
        f"circ={d['circularidad']:.2f}",
        f"sol={d['solidez']:.2f}",
        f"ext={d['extension']:.2f}",
    ]
    cx, cy = d["centroide"]
    for i, t in enumerate(texto):
        pos = (cx - 45, cy - 30 + i * 16)
        cv2.putText(anotada, t, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(anotada, t, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

# Etiquetas en color falso, para ver qué región es cuál
color_falso = np.zeros_like(imagen)
rng = np.random.default_rng(1)
for d in descriptores:
    color_falso[etiquetas == d["id"]] = rng.integers(60, 255, 3)

cv2.imwrite("formas_anotada.jpg", anotada)
print("\nGuardé formas_anotada.jpg")

ventanas = [
    ("1. Original", imagen),
    ("2. Binaria (Otsu)", binaria),
    ("3. Limpieza morfologica", limpia),
    ("4. Etiquetas", color_falso),
    ("5. Descriptores (verde=contorno, magenta=casco convexo)", anotada),
]
for nombre, im in ventanas:
    cv2.imshow(nombre, im)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
