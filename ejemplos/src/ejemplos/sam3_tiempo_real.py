"""Segmentación en tiempo real con SAM3 (Segment Anything 3) desde la webcam.

SAM3 recibe un texto ("person", "cup", "phone"...) y segmenta *todas* las
instancias de ese concepto en cada frame, asignándoles un ID que se mantiene
mientras el objeto es rastreado.

Uso:
    uv run sam3-webcam --prompt person cup

Teclas:
    q / ESC  salir
    p        cambiar los prompts (se escriben en la terminal)
    r        reiniciar el rastreo
"""

import argparse
import threading
import time

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

MODELO = "facebook/sam3"
# Cuántos frames de historia guardamos; SAM3 sólo usa los más recientes como memoria.
FRAMES_EN_MEMORIA = 16


def elegir_dispositivo() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def color_para(obj_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(obj_id * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def nueva_sesion(processor, dispositivo, dtype, prompts):
    sesion = processor.init_video_session(inference_device=dispositivo, dtype=dtype)
    processor.add_text_prompt(sesion, prompts)
    return sesion


def podar_historia(sesion, frame_idx: int) -> None:
    """Libera frames y salidas viejas para que la memoria no crezca sin límite."""
    limite = frame_idx - FRAMES_EN_MEMORIA
    for idx in [i for i in sesion.processed_frames if i < limite]:
        del sesion.processed_frames[idx]
    for salidas in sesion.output_dict_per_obj.values():
        for idx in [i for i in salidas["non_cond_frame_outputs"] if i < limite]:
            del salidas["non_cond_frame_outputs"][idx]
        # Conservamos sólo los últimos frames de condicionamiento (detecciones).
        cond = salidas["cond_frame_outputs"]
        for idx in sorted(cond)[:-4]:
            del cond[idx]


def dibujar(frame, resultados, prompts_por_id):
    overlay = frame.copy()
    for obj_id, score, caja, mascara in zip(
        resultados["object_ids"].tolist(),
        resultados["scores"].tolist(),
        resultados["boxes"].tolist(),
        resultados["masks"],
    ):
        color = color_para(obj_id)
        overlay[mascara] = color
        x1, y1, x2, y2 = map(int, caja)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        etiqueta = f"{prompts_por_id.get(obj_id, '?')} #{obj_id} {score:.2f}"
        cv2.putText(frame, etiqueta, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)


class Segmentador(threading.Thread):
    """Corre SAM3 en segundo plano sobre el frame más reciente de la cámara.

    SAM3 es pesado (en un Mac M4 tarda ~2.5 s por frame), así que la cámara se
    muestra a velocidad normal y las máscaras se actualizan cuando el modelo termina.
    """

    def __init__(self, processor, model, dispositivo, dtype, prompts):
        super().__init__(daemon=True)
        self.processor, self.model = processor, model
        self.dispositivo, self.dtype = dispositivo, dtype
        self.prompts = prompts
        self.reiniciar = True
        self.frame = None
        self.resultado = None
        self.segundos_por_frame = 0.0
        self.candado = threading.Lock()
        self.activo = True

    def enviar(self, frame) -> None:
        with self.candado:
            self.frame = frame

    def cambiar_prompts(self, prompts) -> None:
        self.prompts = prompts
        self.reiniciar = True

    def run(self) -> None:
        while self.activo:
            with self.candado:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            if self.reiniciar:
                self.reiniciar = False
                sesion = nueva_sesion(self.processor, self.dispositivo, self.dtype, self.prompts)
                frame_idx = 0
                self.resultado = None

            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            entradas = self.processor(images=rgb, device=self.dispositivo, return_tensors="pt")
            with torch.inference_mode():
                salidas = self.model(
                    inference_session=sesion,
                    frame=entradas.pixel_values[0],
                    frame_idx=frame_idx,
                )
            resultados = self.processor.postprocess_outputs(
                sesion, salidas, original_sizes=entradas.original_sizes
            )
            podar_historia(sesion, frame_idx)
            frame_idx += 1

            # Pasamos todo a CPU aquí: el hilo principal no debe tocar la GPU
            # mientras el modelo corre (en MPS eso truena con un error de Metal).
            resultados = {
                "object_ids": resultados["object_ids"].cpu(),
                "scores": resultados["scores"].cpu(),
                "boxes": resultados["boxes"].cpu(),
                "masks": resultados["masks"].cpu().numpy(),
                "prompt_to_obj_ids": resultados["prompt_to_obj_ids"],
            }
            resultados["etiquetas"] = {
                obj_id: texto
                for texto, ids in resultados["prompt_to_obj_ids"].items()
                for obj_id in ids
            }
            if not self.reiniciar:
                self.resultado = resultados
            self.segundos_por_frame = time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 en tiempo real con la webcam")
    parser.add_argument("--prompt", nargs="+", default=["person"], help="conceptos a segmentar")
    parser.add_argument("--camara", type=int, default=0, help="índice de la webcam")
    parser.add_argument("--ancho", type=int, default=640, help="ancho de captura")
    args = parser.parse_args()

    dispositivo, dtype = elegir_dispositivo()
    print(f"Cargando {MODELO} en {dispositivo} ({dtype})...")
    processor = Sam3VideoProcessor.from_pretrained(MODELO)
    model = Sam3VideoModel.from_pretrained(MODELO).to(dispositivo, dtype=dtype).eval()

    cap = cv2.VideoCapture(args.camara)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.ancho)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camara}")

    segmentador = Segmentador(processor, model, dispositivo, dtype, args.prompt)
    segmentador.start()

    print(f"Segmentando: {args.prompt}  |  q=salir  p=cambiar prompts  r=reiniciar")
    fps, t_prev = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        segmentador.enviar(frame.copy())

        resultado = segmentador.resultado
        vista = dibujar(frame, resultado, resultado["etiquetas"]) if resultado else frame

        ahora = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (ahora - t_prev)
        t_prev = ahora
        estado = "cargando..." if resultado is None else f"SAM3 {segmentador.segundos_por_frame:.2f}s/frame"
        cv2.putText(vista, f"{fps:.0f} FPS | {estado} | {', '.join(segmentador.prompts)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("SAM3 - tiempo real", vista)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("p"):
            texto = input("Nuevos prompts separados por coma: ").strip()
            if texto:
                segmentador.cambiar_prompts([t.strip() for t in texto.split(",") if t.strip()])
        if tecla == ord("r"):
            segmentador.reiniciar = True

    segmentador.activo = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
