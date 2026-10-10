# MediaPipe: landmarks de cuerpo, manos y cara

[MediaPipe](https://ai.google.dev/edge/mediapipe) detecta puntos clave
(*landmarks*) del cuerpo en tiempo real y en CPU. Con esos puntos se pueden
medir ángulos, contar repeticiones, reconocer gestos, etc.

Esta carpeta usa la **API clásica** (`mp.solutions`), por eso está fija en
`mediapipe==0.10.21`. La versión con la API nueva (Tasks) está en
[`../ejemplos/`](../ejemplos/).

## Archivos

| Archivo | Qué hace |
|---|---|
| `run.py` | Lo mínimo: detecta la pose y resalta los dos brazos (hombro → codo → muñeca) |
| `run2.py` | Detector configurable con **Holistic** (pose + manos + cara en un solo modelo). Trae la referencia de índices de cada punto |

## Uso

```bash
cd media_pipe
uv run run.py
uv run run2.py
```

### Teclas de `run2.py`

| Tecla | Acción |
|---|---|
| `1`–`9`, `0` | Activar/desactivar el grupo con ese número (la lista sale en pantalla) |
| `a` | Activar todos |
| `n` | Desactivar todos |
| `m` | Mostrar/ocultar la malla completa |
| `q` | Salir |

## Índices útiles

```
Pose (33 puntos)
  0 nariz | 11/12 hombros izq/der | 13/14 codos | 15/16 muñecas
  23/24 caderas | 25/26 rodillas | 27/28 tobillos

Mano (21 puntos)
  0 muñeca | pulgar 1-4 | índice 5-8 | medio 9-12 | anular 13-16 | meñique 17-20
  puntas: 4, 8, 12, 16, 20

Cara (478 puntos)
  1 punta de la nariz | 152 mentón | 468 y 473 centro de cada iris
```

Las coordenadas vienen **normalizadas** (0 a 1). Para pasarlas a píxeles:
`x_px = lm.x * ancho`, `y_px = lm.y * alto`.

## Ideas para practicar

- Calcula el ángulo del codo con los puntos 12-14-16 y muéstralo en pantalla.
- Cuenta sentadillas usando el ángulo de la rodilla (24-26-28).
- Detecta si la mano está abierta o cerrada comparando cada punta con su
  articulación media.
