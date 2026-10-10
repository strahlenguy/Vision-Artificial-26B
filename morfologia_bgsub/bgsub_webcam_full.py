"""
Background subtraction con la webcam
====================================

Idea: modelar cómo se ve el fondo y marcar como "primer plano" lo que no
coincide con ese modelo. Comparamos cuatro métodos:

    1  Diferencia de cuadros   |frame_t - frame_t-1| > umbral
                               Solo ve lo que se MUEVE; un objeto quieto desaparece.
    2  Promedio móvil          fondo = (1-α)·fondo + α·frame
                               Fondo que se adapta lento; α = tasa de aprendizaje.
    3  MOG2                    Cada píxel es una mezcla de gaussianas. Detecta sombras (gris).
    4  KNN                     Cada píxel compara contra sus K muestras más parecidas.

Después de la resta, la máscara sale con ruido. Ahí entra la morfología:
apertura para quitar puntitos y cierre para rellenar huecos. Al final,
contornos con área mínima para encerrar los objetos en movimiento.

Uso:
    uv run bgsub_webcam_full.py            # cámara 0
    uv run bgsub_webcam_full.py 1          # otra cámara
    uv run bgsub_webcam_full.py video.mp4  # un video

Teclas:
    1-4  cambiar de método        m  morfología on/off
    r    reiniciar el fondo       q  salir
Sliders: umbral, tamaño del kernel, área mínima y α (promedio móvil).

Tip para la demo: apunta a una pared, presiona r, y luego entra al cuadro.
"""

import sys

import cv2
import numpy as np

METODOS = {1: "Diferencia de cuadros", 2: "Promedio movil", 3: "MOG2", 4: "KNN"}
VENTANA = "Background subtraction"


def nada(_):
    pass


def crear_sustractor(metodo):
    if metodo == 3:
        return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    if metodo == 4:
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
    return None


class Estado:
    def __init__(self, metodo=3):
        self.metodo = metodo
        self.morfologia = True
        self.reiniciar()

    def reiniciar(self):
        self.anterior = None   # para diferencia de cuadros
        self.fondo = None      # para promedio móvil (float32)
        self.sustractor = crear_sustractor(self.metodo)


def mascara_cruda(gris, estado, umbral, alfa):
    """Primer plano antes de limpiar: 255 = movimiento, 127 = sombra (MOG2/KNN), 0 = fondo."""
    if estado.metodo == 1:
        if estado.anterior is None:
            estado.anterior = gris
        diff = cv2.absdiff(gris, estado.anterior)
        estado.anterior = gris
        return cv2.threshold(diff, umbral, 255, cv2.THRESH_BINARY)[1]

    if estado.metodo == 2:
        if estado.fondo is None:
            estado.fondo = gris.astype(np.float32)
        cv2.accumulateWeighted(gris, estado.fondo, alfa)
        diff = cv2.absdiff(gris, cv2.convertScaleAbs(estado.fondo))
        return cv2.threshold(diff, umbral, 255, cv2.THRESH_BINARY)[1]

    return estado.sustractor.apply(gris)


def limpiar(mascara, k):
    # Las sombras (127) no cuentan como objeto
    binaria = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)       # quita ruido
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=2)  # rellena
    return binaria


def procesar(frame, estado, umbral=25, k=5, area_min=1500, alfa=0.02):
    """Regresa (frame_anotado, mascara_cruda, mascara_final)."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)  # suaviza ruido del sensor

    cruda = mascara_cruda(gris, estado, umbral, alfa)
    if estado.morfologia:
        final = limpiar(cruda, k)
    else:
        final = cv2.threshold(cruda, 200, 255, cv2.THRESH_BINARY)[1]

    anotado = frame.copy()
    contornos, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objetos = 0
    for c in contornos:
        if cv2.contourArea(c) < area_min:
            continue
        objetos += 1
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(anotado, (x, y), (x + w, y + h), (0, 255, 0), 2)

    info = [
        f"[{estado.metodo}] {METODOS[estado.metodo]}",
        f"morfologia: {'ON' if estado.morfologia else 'OFF'}   objetos: {objetos}",
    ]
    for i, t in enumerate(info):
        pos = (10, 25 + i * 25)
        cv2.putText(anotado, t, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(anotado, t, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return anotado, cruda, final


def main():
    fuente = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = cv2.VideoCapture(int(fuente) if fuente.isdigit() else fuente)
    if not cap.isOpened():
        sys.exit(f"No pude abrir la fuente de video: {fuente}")

    cv2.namedWindow(VENTANA)
    cv2.createTrackbar("umbral", VENTANA, 25, 100, nada)
    cv2.createTrackbar("kernel", VENTANA, 5, 21, nada)
    cv2.createTrackbar("area min", VENTANA, 1500, 20000, nada)
    cv2.createTrackbar("alfa x1000", VENTANA, 20, 200, nada)

    estado = Estado()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # efecto espejo
        frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

        k = max(1, cv2.getTrackbarPos("kernel", VENTANA)) | 1  # impar
        anotado, cruda, final = procesar(
            frame, estado,
            umbral=max(1, cv2.getTrackbarPos("umbral", VENTANA)),
            k=k,
            area_min=cv2.getTrackbarPos("area min", VENTANA),
            alfa=max(1, cv2.getTrackbarPos("alfa x1000", VENTANA)) / 1000,
        )

        # Mosaico: frame | máscara cruda | máscara final
        etiquetar = lambda im, t: cv2.putText(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), t, (10, 25),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow(VENTANA, np.hstack([anotado, etiquetar(cruda, "mascara cruda"),
                                       etiquetar(final, "mascara final")]))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla in (ord("1"), ord("2"), ord("3"), ord("4")):
            estado.metodo = tecla - ord("0")
            estado.reiniciar()
        elif tecla == ord("m"):
            estado.morfologia = not estado.morfologia
        elif tecla == ord("r"):
            estado.reiniciar()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
