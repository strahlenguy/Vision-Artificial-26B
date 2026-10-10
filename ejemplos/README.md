# Ejemplos en tiempo real

Demos con webcam de dos modelos actuales:

| Comando | Modelo | Qué hace | Hardware |
|---|---|---|---|
| `mediapipe-webcam` | MediaPipe Tasks | Landmarks de manos, pose o cara | CPU, cualquier laptop |
| `sam3-webcam` | SAM 3 (Meta) | Segmenta y rastrea todo lo que coincide con un texto ("person", "cup"…) | GPU NVIDIA o Apple Silicon; en CPU va muy lento |

El código está en `src/ejemplos/`.

## Instalación

```bash
cd ejemplos
uv sync
```

Instala PyTorch y transformers, así que tarda un poco la primera vez.

## MediaPipe

```bash
uv run mediapipe-webcam --modo manos
uv run mediapipe-webcam --modo pose
uv run mediapipe-webcam --modo cara
uv run mediapipe-webcam --camara 1      # otra cámara
```

Los modelos `.task` se descargan solos la primera vez a
`~/.cache/mediapipe_modelos`.

| Tecla | Acción |
|---|---|
| `m` | Cambiar de modo (manos → pose → cara) |
| `q` / `ESC` | Salir |

## SAM 3

SAM 3 recibe una frase en **inglés** y segmenta *todas* las instancias de ese
concepto, con un ID que se mantiene mientras el objeto es rastreado.

Antes de la primera corrida:

1. Pide acceso al modelo en https://huggingface.co/facebook/sam3 (tarda un
   rato en aprobarse).
2. Inicia sesión: `uv run hf auth login`

Los pesos (~3.4 GB) se descargan solos la primera vez.

```bash
uv run sam3-webcam --prompt person
uv run sam3-webcam --prompt person cup phone
uv run sam3-webcam --prompt hand --ancho 480     # más rápido
```

| Tecla | Acción |
|---|---|
| `p` | Cambiar los prompts (se escriben en la terminal) |
| `r` | Reiniciar el rastreo |
| `q` / `ESC` | Salir |
