# =============================================================================
# Dockerfile — ComfyUI Serverless para RunPod (FLUX.2 Klein 4B)
# =============================================================================
#
# CÓMO CONSTRUIR Y PUBLICAR:
#
#   # Opción A: modelos descargados AL CONSTRUIR la imagen (imagen ~12 GB)
#   # Ventaja: cold start rápido (~30s). Desventaja: imagen grande, build lento.
#   docker build \
#     --build-arg HF_TOKEN=hf_xxxx \
#     --build-arg BAKE_MODELS=true \
#     -t tuusuario/comfyui-flux2-klein:latest .
#
#   # Opción B: sin modelos en la imagen (imagen ~6 GB)
#   # Ventaja: imagen pequeña. Desventaja: primer arranque descarga ~10 GB.
#   # → Recomendado usar RunPod Network Volume para persistir modelos.
#   docker build -t tuusuario/comfyui-flux2-klein:latest .
#
#   # Publicar en Docker Hub (RunPod necesita la imagen en un registry público)
#   docker push tuusuario/comfyui-flux2-klein:latest
#
# CÓMO CREAR EL ENDPOINT EN RUNPOD:
#   1. Ir a https://www.runpod.io/console/serverless
#   2. "New Endpoint" → "Custom Source"
#   3. Docker Image: tuusuario/comfyui-flux2-klein:latest
#   4. Container Disk: 20 GB (o más si horneas modelos)
#   5. GPU: RTX 4090 / A100 (mínimo 24 GB VRAM para FLUX)
#   6. Environment Variables:
#        HF_TOKEN = hf_xxxx   (si usas Opción B)
#   7. Si usas Network Volume: montarlo en /runpod-volume/models
#      y el entrypoint enlazará simbólicamente los modelos.
# =============================================================================

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# ── Variables ─────────────────────────────────────────────────────────────────
ENV COMFYUI_DIR=/app/ComfyUI \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ARG disponible solo durante el build (no queda en la imagen final)
ARG HF_TOKEN=""
# Si BAKE_MODELS=true, los modelos se descargan dentro del build
ARG BAKE_MODELS="false"

# ── Dependencias de sistema ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── Clonar ComfyUI ────────────────────────────────────────────────────────────
# --depth 1 para no descargar todo el historial de git (imagen más pequeña)
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_DIR}

WORKDIR ${COMFYUI_DIR}

# ── Dependencias Python ───────────────────────────────────────────────────────
# requirements.txt incluye: torch (ya instalado en la base), xformers, etc.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install runpod requests

# ── Estructura de directorios ─────────────────────────────────────────────────
RUN mkdir -p \
    models/text_encoders \
    models/diffusion_models \
    models/vae \
    input \
    output \
    user/default/workflows

# ── Copiar archivos del proyecto ──────────────────────────────────────────────
COPY Workflows/                         ${COMFYUI_DIR}/user/default/workflows/
COPY runpod/handler.py                  /app/handler.py
COPY runpod/download_models.sh          /app/download_models.sh
RUN chmod +x /app/download_models.sh

# ── Descarga de modelos durante el build (solo si BAKE_MODELS=true) ───────────
#
# Si prefieres RunPod Network Volume (recomendado para producción):
#   - No hornees los modelos aquí (deja BAKE_MODELS=false)
#   - Monta el volumen en /runpod-volume/models desde el panel de RunPod
#   - El handler.py crea symlinks automáticamente al arrancar
#
RUN if [ "$BAKE_MODELS" = "true" ]; then \
    echo "[build] Descargando modelos FLUX.2 Klein 4B..." && \
    HF_TOKEN="${HF_TOKEN}" COMFYUI_DIR="${COMFYUI_DIR}" /app/download_models.sh; \
    else \
    echo "[build] BAKE_MODELS=false — modelos se descargarán en arranque o desde Network Volume"; \
    fi

# ── Comando de inicio ─────────────────────────────────────────────────────────
# handler.py levanta ComfyUI en background y luego inicia el worker de RunPod
CMD ["python", "-u", "/app/handler.py"]
