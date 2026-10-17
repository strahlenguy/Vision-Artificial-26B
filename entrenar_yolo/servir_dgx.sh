#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sirve el notebook en la DGX Spark para todo el grupo.
#
#   ./servir_dgx.sh                    # puerto 8888
#   ./servir_dgx.sh 8888 alumnos.txt   # además crea una carpeta por alumno
#
# alumnos.txt: un nombre por línea (sin espacios: ana_lopez, equipo3...).
#
# Qué hace:
#   - Cada alumno trabaja en alumnos/<nombre>/ con su propia copia del
#     notebook y sus propios datos, así nadie pisa el trabajo de otro.
#   - Todas las corridas van a runs/ (común): la tabla comparativa del
#     Paso 10 muestra al grupo entero.
#   - Los pesos base se bajan UNA vez a pesos/ (si 30 personas descargan
#     al mismo tiempo, el internet del aula se cae).
#   - Activa los turnos: solo entrena una persona a la vez.
# ---------------------------------------------------------------------------
set -euo pipefail

PUERTO="${1:-8888}"
LISTA="${2:-}"
AQUI="$(cd "$(dirname "$0")" && pwd)"
cd "$AQUI"

export VA_RUNS="$AQUI/runs"
export VA_PESOS="$AQUI/pesos"
export VA_TURNOS=1
export VA_CANDADO="$AQUI/.turno_gpu.lock"
mkdir -p "$VA_RUNS" "$VA_PESOS" alumnos

# --- Una carpeta por alumno (no sobrescribe copias existentes) ------------- #
if [[ -n "$LISTA" ]]; then
    while IFS= read -r nombre || [[ -n "$nombre" ]]; do
        nombre="$(echo "$nombre" | tr -d '[:space:]')"
        [[ -z "$nombre" || "$nombre" == \#* ]] && continue
        mkdir -p "alumnos/$nombre"
        if [[ ! -f "alumnos/$nombre/entrenar_yolo.ipynb" ]]; then
            cp entrenar_yolo.ipynb "alumnos/$nombre/"
            echo "  + alumnos/$nombre"
        fi
    done < "$LISTA"
fi

# --- Entorno e instalación (la primera vez tarda) -------------------------- #
# uv usa el pyproject.toml de la raíz del repo; el grupo "yolo" trae
# Ultralytics y el PyTorch de CUDA 13 para ARM64.
uv sync --group yolo --quiet
uv run --group yolo python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch no ve la GPU. Revisa el driver y el índice cu130 del pyproject.toml de la raíz."
print(f"GPU: {torch.cuda.get_device_name(0)}  ·  torch {torch.__version__}")
PY

# --- Pesos base más usados, descargados una sola vez ----------------------- #
uv run --group yolo python - <<'PY'
import os
from pathlib import Path
from ultralytics import YOLO
pesos = Path(os.environ["VA_PESOS"])
for v in ("yolov8", "yolo11", "yolo26"):
    for s in ("n", "s"):
        for suf in ("", "-seg"):
            f = pesos / f"{v}{s}{suf}.pt"
            if not f.exists():
                YOLO(str(f))
for f in ("yolov10n.pt", "yolov10s.pt", "yolo12n.pt", "yolo12s.pt", "yolov9t.pt", "yolov9s.pt"):
    if not (pesos / f).exists():
        YOLO(str(pesos / f))
print(f"Pesos base listos en {pesos}")
PY

# --- Servidor ------------------------------------------------------------- #
TOKEN="${VA_TOKEN:-$(uv run --group yolo python -c 'import secrets; print(secrets.token_hex(4))')}"
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "=================================================================="
echo "  Que los alumnos abran en su navegador:"
echo "    http://${IP:-<ip-de-la-spark>}:${PUERTO}/lab?token=${TOKEN}"
echo "  y entren a alumnos/<su_nombre>/entrenar_yolo.ipynb"
echo "  Para detener: Ctrl+C"
echo "=================================================================="
echo
exec uv run --group yolo jupyter lab --ip 0.0.0.0 --port "$PUERTO" --no-browser \
    --ServerApp.root_dir="$AQUI" --IdentityProvider.token="$TOKEN"
