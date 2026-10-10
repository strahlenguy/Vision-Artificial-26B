"""Landmarks en tiempo real con MediaPipe (manos, pose o cara) desde la webcam.

MediaPipe corre en CPU y es muy ligero, así que procesa cada frame en vivo.
Usa la API de "Tasks": cada tarea carga un modelo .task que se descarga la
primera vez.

Uso:
    uv run mediapipe-webcam --modo manos
    uv run mediapipe-webcam --modo pose
    uv run mediapipe-webcam --modo cara

Teclas:
    q / ESC  salir
    m        cambiar de modo (manos -> pose -> cara)
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

URL_MODELOS = "https://storage.googleapis.com/mediapipe-models"
MODELOS = {
    "manos": f"{URL_MODELOS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose": f"{URL_MODELOS}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "cara": f"{URL_MODELOS}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}
CARPETA_MODELOS = Path.home() / ".cache" / "mediapipe_modelos"

CONEXIONES = {
    "manos": vision.HandLandmarksConnections.HAND_CONNECTIONS,
    "pose": vision.PoseLandmarksConnections.POSE_LANDMARKS,
    "cara": vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
}

# Puntas y articulaciones medias de los dedos (índices de los 21 puntos de la mano).
PUNTAS = [8, 12, 16, 20]
ARTICULACIONES = [6, 10, 14, 18]


def descargar_modelo(modo: str) -> Path:
    ruta = CARPETA_MODELOS / Path(MODELOS[modo]).name
    if not ruta.exists():
        CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
        print(f"Descargando modelo de {modo}...")
        urllib.request.urlretrieve(MODELOS[modo], ruta)
    return ruta


def crear_detector(modo: str):
    base = BaseOptions(
        model_asset_path=str(descargar_modelo(modo)), delegate=BaseOptions.Delegate.CPU
    )
    video = vision.RunningMode.VIDEO
    if modo == "manos":
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base, running_mode=video, num_hands=2)
        )
    if modo == "pose":
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base, running_mode=video, num_poses=2)
        )
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base, running_mode=video, num_faces=2, output_face_blendshapes=True
        )
    )


def detectar(detector, modo: str, imagen, timestamp_ms: int):
    """Devuelve (lista de personas/manos/caras con sus puntos, resultado crudo)."""
    resultado = detector.detect_for_video(imagen, timestamp_ms)
    if modo == "manos":
        return resultado.hand_landmarks, resultado
    if modo == "pose":
        return resultado.pose_landmarks, resultado
    return resultado.face_landmarks, resultado


def dibujar_puntos(frame, puntos, conexiones, color, radio=3):
    alto, ancho = frame.shape[:2]
    pixeles = [(int(p.x * ancho), int(p.y * alto)) for p in puntos]
    for c in conexiones:
        cv2.line(frame, pixeles[c.start], pixeles[c.end], color, 2)
    for x, y in pixeles:
        cv2.circle(frame, (x, y), radio, (255, 255, 255), -1)


def contar_dedos(mano) -> int:
    """Un dedo está levantado si su punta queda más lejos de la muñeca que su articulación.

    Comparar distancias (y no sólo alturas) funciona aunque la mano esté girada.
    """

    def distancia(i, j):
        return math.dist((mano[i].x, mano[i].y), (mano[j].x, mano[j].y))

    dedos = sum(distancia(p, 0) > distancia(a, 0) for p, a in zip(PUNTAS, ARTICULACIONES))
    # El pulgar se abre hacia un lado: lo comparamos contra la base del meñique (17).
    dedos += distancia(4, 17) > distancia(3, 17)
    return dedos


def anotar(frame, modo: str, detecciones, resultado) -> None:
    alto, ancho = frame.shape[:2]
    colores = [(0, 200, 255), (255, 120, 0)]
    for i, puntos in enumerate(detecciones):
        color = colores[i % len(colores)]
        radio = 1 if modo == "cara" else 3
        dibujar_puntos(frame, puntos, CONEXIONES[modo], color, radio)
        x, y = int(puntos[0].x * ancho), int(puntos[0].y * alto)

        if modo == "manos":
            lado = resultado.handedness[i][0].category_name
            texto = f"{'izq' if lado == 'Right' else 'der'}: {contar_dedos(puntos)} dedos"
        elif modo == "pose":
            texto = f"persona {i + 1}"
        else:
            # Las "blendshapes" son ~50 gestos de la cara con un valor de 0 a 1.
            gestos = {b.category_name: b.score for b in resultado.face_blendshapes[i]}
            sonrisa = (gestos["mouthSmileLeft"] + gestos["mouthSmileRight"]) / 2
            texto = f"sonrisa {sonrisa:.0%}  boca {gestos['jawOpen']:.0%}"
            x, y = int(min(p.x for p in puntos) * ancho), int(min(p.y for p in puntos) * alto)
        cv2.putText(frame, texto, (x, max(y - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe en tiempo real con la webcam")
    parser.add_argument("--modo", choices=list(MODELOS), default="manos")
    parser.add_argument("--camara", type=int, default=0, help="índice de la webcam")
    args = parser.parse_args()

    modo = args.modo
    detector = crear_detector(modo)

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camara}")

    print("q=salir  m=cambiar modo (manos / pose / cara)")
    inicio = time.perf_counter()
    fps, t_prev = 0.0, inicio
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # efecto espejo

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # En modo VIDEO el timestamp debe crecer en cada llamada.
        timestamp_ms = int((time.perf_counter() - inicio) * 1000)
        detecciones, resultado = detectar(detector, modo, imagen, timestamp_ms)
        anotar(frame, modo, detecciones, resultado)

        ahora = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (ahora - t_prev)
        t_prev = ahora
        cv2.putText(frame, f"{fps:.0f} FPS | modo: {modo}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("MediaPipe - tiempo real", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("m"):
            detector.close()
            modos = list(MODELOS)
            modo = modos[(modos.index(modo) + 1) % len(modos)]
            detector = crear_detector(modo)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
