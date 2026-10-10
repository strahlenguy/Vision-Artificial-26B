"""
Background subtraction con la webcam (versión mínima)
=====================================================

Izquierda: la cámara. Derecha: la máscara de MOG2
(blanco = primer plano, gris = sombra, negro = fondo).

Quédate quieto unos segundos para que aprenda el fondo y luego muévete.
La versión con más métodos, morfología y sliders está en bgsub_webcam_full.py.

Uso:  uv run bgsub_webcam.py        q para salir
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
sustractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

    mascara = sustractor.apply(frame)

    cv2.imshow("Camara | Background subtraction", np.hstack([frame, cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)]))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
