# =============================================================================
# Dockerfile -- ComfyUI Serverless para RunPod (FLUX.2 Klein 4B)
# =============================================================================
#
# BASE IMAGE: ghcr.io/ai-dock/comfyui
#   Ya incluye: ComfyUI en /opt/ComfyUI, PyTorch 2.2, xformers, CUDA 12.1,
#               todas las dependencias del sistema.
#   Solo agregamos: RunPod SDK, handler y scripts de descarga de modelos.
#   Build: ~2-3 min en lugar de ~15-20 min con la imagen anterior.
#
# IMPORTANTE: contexto de build = carpeta runpod/ (no la raiz del proyecto)
#
#   # Opcion A: modelos bakeados en la imagen (~12 GB, cold start rapido ~30s)
#   docker build -f runpod/Dockerfile \
#     --build-arg HF_TOKEN=hf_xxxx \
#     --build-arg BAKE_MODELS=true \
#     -t tuusuario/comfyui-flux2-klein:latest \
#     runpod/
#
#   # Opcion B: sin modelos, usar Network Volume (recomendado para produccion)
#   docker build -f runpod/Dockerfile \
#     -t tuusuario/comfyui-flux2-klein:latest \
#     runpod/
#
#   docker push tuusuario/comfyui-flux2-klein:latest
#
# CREAR ENDPOINT EN RUNPOD:
#   1. Serverless -> New Endpoint -> Custom Source
#   2. Docker Image: tuusuario/comfyui-flux2-klein:latest
#   3. Container Disk: 20 GB | GPU: RTX 4090 / A100 (min 24 GB VRAM)
#   4. Env vars: HF_TOKEN=hf_xxxx
#   5. Network Volume montado en /runpod-volume (para persistir modelos)
#
# TAGS DISPONIBLES de ai-dock/comfyui:
#   https://github.com/ai-dock/comfyui/pkgs/container/comfyui
#   Ejemplos: latest-cuda-12.1.0-base-22.04 | latest-cuda-12.3.2-base-22.04
# =============================================================================

# -- Imagen base: AI Dock ComfyUI ---------------------------------------------
# Incluye: ComfyUI clonado en /opt/ComfyUI, PyTorch, xformers, CUDA, wget, git
FROM ghcr.io/ai-dock/comfyui:latest-cuda-12.1.0-base-22.04

# -- Variables ----------------------------------------------------------------
# AI Dock instala ComfyUI en /opt/ComfyUI
ENV COMFYUI_DIR=/opt/ComfyUI \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG HF_TOKEN=""
ARG BAKE_MODELS="false"

# -- Instalar RunPod SDK y requests -------------------------------------------
# torch, xformers y los requisitos de ComfyUI ya estan en la imagen base.
# pip3 apunta al entorno Python de la imagen (micromamba/venv de AI Dock).
RUN pip3 install --upgrade pip && \
    pip3 install runpod requests

# -- Estructura de directorios de modelos -------------------------------------
RUN mkdir -p \
    ${COMFYUI_DIR}/models/text_encoders \
    ${COMFYUI_DIR}/models/diffusion_models \
    ${COMFYUI_DIR}/models/vae \
    ${COMFYUI_DIR}/input \
    ${COMFYUI_DIR}/output

# -- Copiar handler y script de descarga --------------------------------------
# Contexto de build = runpod/ -> rutas relativas a esa carpeta.
# El workflow NO va en la imagen: llega en cada request via input.workflow.
COPY handler.py          /app/handler.py
COPY download_models.sh  /app/download_models.sh
RUN chmod +x /app/download_models.sh

# -- Descarga de modelos durante el build (solo Opcion A: BAKE_MODELS=true) ---
# Con Network Volume (Opcion B): dejar BAKE_MODELS=false.
# El handler descargara al volumen en el primer arranque y los persistira.
RUN if [ "$BAKE_MODELS" = "true" ]; then \
        echo "[build] Descargando modelos FLUX.2 Klein 4B..." && \
        HF_TOKEN="${HF_TOKEN}" COMFYUI_DIR="${COMFYUI_DIR}" /app/download_models.sh; \
    else \
        echo "[build] BAKE_MODELS=false -- modelos desde Network Volume o primer arranque"; \
    fi

# -- Sobreescribir el entrypoint de AI Dock -----------------------------------
# AI Dock usa supervisord como ENTRYPOINT por defecto.
# Lo reemplazamos con nuestro handler de RunPod.
# handler.py levanta ComfyUI como subproceso internamente.
ENTRYPOINT []
CMD ["python3", "-u", "/app/handler.py"]
