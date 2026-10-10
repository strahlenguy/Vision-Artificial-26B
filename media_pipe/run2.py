"""
Detector configurable con MediaPipe Holistic (pose + manos + cara en un solo modelo).

Requiere: Python 3.12 y mediapipe==0.10.21 (trae OpenCV incluido)
    uv run --python 3.12 --with "mediapipe==0.10.21" detector.py

Teclas:
    1-9, 0  activar/desactivar el grupo con ese número (ver lista en pantalla)
    a       activar todos
    n       desactivar todos
    m       mostrar/ocultar la malla completa de MediaPipe
    q       salir
"""
import math

import cv2
import mediapipe as mp

holistic_mod = mp.solutions.holistic
face_mesh_mod = mp.solutions.face_mesh
dibujo = mp.solutions.drawing_utils

# =====================================================================
# REFERENCIA DE ÍNDICES
# ---------------------------------------------------------------------
# Pose (33 puntos):
#   0 nariz | 11/12 hombros izq/der | 13/14 codos | 15/16 muñecas
#   23/24 caderas | 25/26 rodillas | 27/28 tobillos | 31/32 punta del pie
#
# Mano (21 puntos, igual para ambas):
#   0 muñeca | pulgar 1-4 | índice 5-8 | medio 9-12 | anular 13-16 | meñique 17-20
#   (el último de cada dedo es la punta: 4, 8, 12, 16, 20)
#
# Cara (478 puntos con refine_face_landmarks=True):
#   1 punta de la nariz | 152 mentón | 468 y 473 centro de cada iris
#   Los contornos de ojos, cejas, labios y cara están en GRUPOS abajo.
# =====================================================================

DEDOS = {
    "pulgar":  [0, 1, 2, 3, 4],
    "indice":  [0, 5, 6, 7, 8],
    "medio":   [0, 9, 10, 11, 12],
    "anular":  [0, 13, 14, 15, 16],
    "menique": [0, 17, 18, 19, 20],
}
PALMA = [5, 9, 13, 17]

CARA = {
    "ojo_der":  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
    "ojo_izq":  [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263],
    "ceja_der": [70, 63, 105, 66, 107],
    "ceja_izq": [300, 293, 334, 296, 336],
    "labios":   [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 409, 270, 269, 267, 0, 37, 39, 40, 185, 61],
    "contorno": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                 234, 127, 162, 21, 54, 103, 67, 109, 10],
}

# =====================================================================
# EL DICCIONARIO: aquí defines qué quieres detectar
# ---------------------------------------------------------------------
#   fuente   : "pose" | "mano_izq" | "mano_der" | "cara"
#   cadenas  : listas de índices que se unen con líneas, en orden
#   puntos   : índices sueltos que solo se marcan con un círculo
#   angulos  : tripletas (a, b, c) -> muestra el ángulo en el punto b
#   color    : BGR
#   grosor   : grosor de línea (opcional, 2 por defecto)
# =====================================================================
GRUPOS = {
    "brazos": {
        "fuente": "pose",
        "cadenas": [[12, 14, 16], [11, 13, 15]],
        "angulos": [(12, 14, 16), (11, 13, 15)],      # ángulo de cada codo
        "color": (0, 255, 0),
    },
    "piernas": {
        "fuente": "pose",
        "cadenas": [[24, 26, 28, 32], [23, 25, 27, 31]],
        "angulos": [(24, 26, 28), (23, 25, 27)],      # ángulo de cada rodilla
        "color": (0, 200, 255),
    },
    "torso": {
        "fuente": "pose",
        "cadenas": [[11, 12, 24, 23, 11]],
        "color": (255, 200, 0),
    },
    "mano_der": {
        "fuente": "mano_der",
        "cadenas": list(DEDOS.values()) + [PALMA],
        "color": (0, 0, 255),
    },
    "mano_izq": {
        "fuente": "mano_izq",
        "cadenas": list(DEDOS.values()) + [PALMA],
        "color": (255, 0, 0),
    },
    "indice_der": {
        "fuente": "mano_der",
        "cadenas": [DEDOS["indice"]],
        "angulos": [(5, 6, 7), (6, 7, 8)],            # flexión de las falanges
        "color": (255, 0, 255),
    },
    "puntas_dedos": {
        "fuente": "mano_der",
        "puntos": [4, 8, 12, 16, 20],
        "color": (0, 255, 255),
    },
    "ojos_cejas": {
        "fuente": "cara",
        "cadenas": [CARA["ojo_der"], CARA["ojo_izq"], CARA["ceja_der"], CARA["ceja_izq"]],
        "color": (255, 255, 0),
        "grosor": 1,
    },
    "labios": {
        "fuente": "cara",
        "cadenas": [CARA["labios"]],
        "color": (80, 80, 255),
        "grosor": 1,
    },
    "cara_contorno": {
        "fuente": "cara",
        "cadenas": [CARA["contorno"]],
        "puntos": [1, 468, 473],                      # nariz + iris
        "color": (200, 200, 200),
        "grosor": 1,
    },
}

# Grupos encendidos al arrancar
ACTIVOS = {"brazos", "mano_der", "labios"}

UMBRAL_VISIBILIDAD = 0.5   # solo aplica a pose: ignora puntos fuera de cuadro
TECLAS = "1234567890"      # los primeros 10 grupos se pueden alternar con estas teclas
FUENTE_TXT = cv2.FONT_HERSHEY_SIMPLEX


def obtener_landmarks(res, fuente):
    # Si izquierda/derecha te salen cruzadas en las manos, intercambia estas dos líneas.
    return {
        "pose": res.pose_landmarks,
        "mano_izq": res.left_hand_landmarks,
        "mano_der": res.right_hand_landmarks,
        "cara": res.face_landmarks,
    }[fuente]


def angulo(a, b, c):
    """Ángulo en b (grados) formado por a-b-c, en el plano de la imagen."""
    ang = abs(math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    ))
    return 360 - ang if ang > 180 else ang


def dibujar_grupo(frame, res, grupo):
    lms = obtener_landmarks(res, grupo["fuente"])
    if lms is None:
        return
    h, w = frame.shape[:2]
    color = grupo.get("color", (0, 255, 0))
    grosor = grupo.get("grosor", 2)
    cache = {}

    def px(i):
        if i not in cache:
            lm = lms.landmark[i]
            if grupo["fuente"] == "pose" and lm.visibility < UMBRAL_VISIBILIDAD:
                cache[i] = None
            else:
                cache[i] = (int(lm.x * w), int(lm.y * h))
        return cache[i]

    for cadena in grupo.get("cadenas", []):
        for i, j in zip(cadena, cadena[1:]):
            a, b = px(i), px(j)
            if a and b:
                cv2.line(frame, a, b, color, grosor)
        if grosor > 1:
            for i in cadena:
                if px(i):
                    cv2.circle(frame, px(i), 4, color, -1)

    for i in grupo.get("puntos", []):
        if px(i):
            cv2.circle(frame, px(i), 5, color, -1)

    for a, b, c in grupo.get("angulos", []):
        A, B, C = px(a), px(b), px(c)
        if A and B and C:
            cv2.putText(frame, f"{angulo(A, B, C):.0f}", (B[0] + 8, B[1] - 8),
                        FUENTE_TXT, 0.6, color, 2)


def dibujar_menu(frame, teclas_grupo, activos, malla):
    y = 22
    for tecla, nombre in teclas_grupo.items():
        on = nombre in activos
        cv2.putText(frame, f"[{tecla}] {nombre}", (10, y), FUENTE_TXT, 0.5,
                    (0, 255, 0) if on else (120, 120, 120), 1 if not on else 2)
        y += 20
    cv2.putText(frame, f"[m] malla: {'on' if malla else 'off'}   [a] todos  [n] ninguno  [q] salir",
                (10, y + 6), FUENTE_TXT, 0.45, (255, 255, 255), 1)


def dibujar_malla(frame, res):
    gris = dibujo.DrawingSpec(color=(90, 90, 90), thickness=1, circle_radius=1)
    dibujo.draw_landmarks(frame, res.pose_landmarks, holistic_mod.POSE_CONNECTIONS, gris, gris)
    dibujo.draw_landmarks(frame, res.left_hand_landmarks, holistic_mod.HAND_CONNECTIONS, gris, gris)
    dibujo.draw_landmarks(frame, res.right_hand_landmarks, holistic_mod.HAND_CONNECTIONS, gris, gris)
    dibujo.draw_landmarks(frame, res.face_landmarks, face_mesh_mod.FACEMESH_TESSELATION, None, gris)


def main():
    activos = set(ACTIVOS)
    teclas_grupo = dict(zip(TECLAS, GRUPOS))
    malla = False
    cap = cv2.VideoCapture(0)

    with holistic_mod.Holistic(
        refine_face_landmarks=True,      # necesario para los iris (468-477)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            if malla:
                dibujar_malla(frame, res)
            for nombre in GRUPOS:
                if nombre in activos:
                    dibujar_grupo(frame, res, GRUPOS[nombre])
            dibujar_menu(frame, teclas_grupo, activos, malla)

            cv2.imshow("MediaPipe Holistic", frame)

            k = cv2.waitKey(1) & 0xFF
            if k == 255:
                continue
            ch = chr(k)
            if ch == "q":
                break
            elif ch == "a":
                activos = set(GRUPOS)
            elif ch == "n":
                activos.clear()
            elif ch == "m":
                malla = not malla
            elif ch in teclas_grupo:
                activos ^= {teclas_grupo[ch]}

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
