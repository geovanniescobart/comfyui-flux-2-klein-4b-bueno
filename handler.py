#!/bin/bash
# =============================================================================
# download_models.sh — Descarga los modelos de FLUX.2 Klein 4B desde HuggingFace
# =============================================================================
#
# USO:
#   # Desde el Dockerfile (durante el build):
#   HF_TOKEN=hf_xxxx COMFYUI_DIR=/app/ComfyUI ./download_models.sh
#
#   # Desde el contenedor ya corriendo (manual):
#   docker exec -it comfyui-runpod bash /app/download_models.sh
#
# VARIABLES DE ENTORNO:
#   HF_TOKEN     — Token de HuggingFace (necesario para flux-2-klein, repo gated).
#                  Crear en: https://huggingface.co/settings/tokens
#                  Aceptar términos en: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8
#   COMFYUI_DIR  — Directorio raíz de ComfyUI (default: /app/ComfyUI)
#
# MODELOS DESCARGADOS:
#   ~500 MB  — qwen_3_4b.safetensors          (text encoder)
#   ~4.5 GB  — flux-2-klein-base-4b-fp8.safetensors  (diffusion model, FP8)
#   ~335 MB  — flux2-vae.safetensors           (VAE)
#   Total: ~5.3 GB
# =============================================================================
set -euo pipefail

# MODELS_DIR puede venir como variable de entorno (p.ej. /runpod-volume/models)
# Si no se pasa, usa la ruta local de ComfyUI (comportamiento por defecto)
MODELS_DIR="${MODELS_DIR:-${COMFYUI_DIR:-/app/ComfyUI}/models}"
HF_TOKEN="${HF_TOKEN:-}"

# ── Helper de descarga ────────────────────────────────────────────────────────
# Usa wget con reintentos. Si el archivo ya existe, lo salta.
download() {
    local url="$1"
    local dest="$2"
    local name
    name=$(basename "$dest")

    if [ -f "$dest" ]; then
        echo "[✓] $name ya existe — saltando"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    echo "[↓] Descargando $name ..."

    local wget_args=(
        "--tries=3"
        "--timeout=60"
        "--waitretry=5"
        "--quiet"
        "--show-progress"
        "--progress=bar:force"
        "-O" "$dest"
    )

    # El modelo de Black Forest Labs está en un repo gated — requiere HF_TOKEN
    if [ -n "$HF_TOKEN" ]; then
        wget_args+=("--header=Authorization: Bearer ${HF_TOKEN}")
    fi

    if wget "${wget_args[@]}" "$url"; then
        echo "[✓] $name descargado"
    else
        echo "[✗] Error descargando $name — eliminando archivo parcial"
        rm -f "$dest"
        exit 1
    fi
}

echo ""
echo "════════════════════════════════════════════════"
echo "  Descarga de modelos FLUX.2 Klein 4B"
echo "  Destino: $MODELS_DIR"
echo "════════════════════════════════════════════════"

if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "  ⚠  HF_TOKEN no definido."
    echo "     flux-2-klein-base-4b-fp8 es un modelo GATED."
    echo "     Sin token la descarga fallará con HTTP 401."
    echo ""
fi

# ── Text encoder: Qwen 3 4B ───────────────────────────────────────────────────
download \
    "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
    "${MODELS_DIR}/text_encoders/qwen_3_4b.safetensors"

# ── Diffusion model: FLUX.2 Klein 4B FP8 (GATED — requiere HF_TOKEN) ─────────
download \
    "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors" \
    "${MODELS_DIR}/diffusion_models/flux-2-klein-base-4b-fp8.safetensors"

# ── VAE: FLUX2 VAE ────────────────────────────────────────────────────────────
download \
    "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors" \
    "${MODELS_DIR}/vae/flux2-vae.safetensors"

echo ""
echo "[✓] Todos los modelos están listos en $MODELS_DIR"
echo "════════════════════════════════════════════════"
echo ""
