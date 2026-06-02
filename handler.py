#!/usr/bin/env python3
"""
handler.py — RunPod Serverless Worker para ComfyUI FLUX.2 Klein 4B
====================================================================

FLUJO DE EJECUCIÓN:
  1. Al arrancar el contenedor (startup):
     a. Si existe /runpod-volume/models → crea symlinks hacia ComfyUI/models
        (permite usar RunPod Network Volume sin copiar archivos)
     b. Si los modelos no están y HF_TOKEN existe → los descarga con download_models.sh
     c. Levanta ComfyUI en segundo plano (puerto 8188, solo localhost)
     d. Espera que ComfyUI esté listo (~20-60s)
     e. Inicia el worker de RunPod (empieza a aceptar jobs)

  2. Por cada job recibido (handler):
     a. Decodifica y sube las imágenes de entrada a ComfyUI (/upload/image)
     b. Encola el workflow (/prompt)
     c. Hace polling hasta que termine (/history/{prompt_id})
     d. Descarga las imágenes de salida y las retorna como base64

REQUEST esperado (compatible con RunPodService.cs):
  POST https://api.runpod.ai/v2/{endpoint_id}/runsync
  {
    "input": {
      "workflow": { ...ComfyUI workflow en formato API... },
      "images": [
        { "name": "selfie.png",     "image": "<base64>" },
        { "name": "instagram.png",  "image": "<base64>" }
      ]
    }
  }

RESPONSE:
  {
    "output": {
      "images": [
        { "data": "<base64>" }
      ]
    }
  }
"""

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import requests
import runpod

# ── Configuración ─────────────────────────────────────────────────────────────
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", "/app/ComfyUI")
COMFYUI_URL = "http://127.0.0.1:8188"
# Ruta del RunPod Network Volume (si existe)
NETWORK_VOLUME = "/runpod-volume/models"
MODELS_DIR = os.path.join(COMFYUI_DIR, "models")


# =============================================================================
# Startup helpers
# =============================================================================

def setup_network_volume_symlinks() -> None:
    """
    Si el usuario montó un RunPod Network Volume en /runpod-volume/models,
    crea symlinks para que ComfyUI encuentre los modelos sin copiarlos.
    Estructura esperada en el volumen:
        /runpod-volume/models/text_encoders/qwen_3_4b.safetensors
        /runpod-volume/models/diffusion_models/flux-2-klein-base-4b-fp8.safetensors
        /runpod-volume/models/vae/flux2-vae.safetensors
    """
    if not os.path.isdir(NETWORK_VOLUME):
        return

    print(
        f"[startup] Network Volume detectado en {NETWORK_VOLUME} — creando symlinks...")
    for subdir in os.listdir(NETWORK_VOLUME):
        src = os.path.join(NETWORK_VOLUME, subdir)
        dst = os.path.join(MODELS_DIR, subdir)
        if os.path.isdir(src) and not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"[startup]   symlink: {dst} → {src}")


def download_models_if_needed() -> None:
    """
    Descarga los modelos si no existen (solo cuando BAKE_MODELS no se usó en build).
    Requiere HF_TOKEN para el modelo gated de Black Forest Labs.
    """
    marker = os.path.join(MODELS_DIR, "diffusion_models",
                          "flux-2-klein-base-4b-fp8.safetensors")
    if os.path.isfile(marker):
        print("[startup] Modelos ya presentes — saltando descarga.")
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("[startup] ADVERTENCIA: HF_TOKEN no definido. "
              "El modelo FLUX.2 Klein está en un repo gated y puede fallar la descarga.")

    print("[startup] Iniciando descarga de modelos (esto puede tardar varios minutos)...")
    env = {**os.environ, "HF_TOKEN": hf_token, "COMFYUI_DIR": COMFYUI_DIR}
    result = subprocess.run(["/bin/bash", "/app/download_models.sh"], env=env)
    if result.returncode != 0:
        print("[startup] ERROR al descargar modelos — abortando.", file=sys.stderr)
        sys.exit(1)


def start_comfyui() -> subprocess.Popen:
    """Levanta ComfyUI en background escuchando solo en localhost."""
    print("[startup] Levantando ComfyUI en segundo plano...")
    process = subprocess.Popen(
        [sys.executable, "main.py",
         "--listen", "127.0.0.1",
         "--port", "8188",
         "--disable-auto-launch"],
        cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return process


def wait_for_comfyui(timeout: int = 180) -> bool:
    """Hace polling a /system_stats hasta que ComfyUI responda o se agote el timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
            if r.status_code == 200:
                print("[startup] ComfyUI listo ✓")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# =============================================================================
# ComfyUI API helpers
# =============================================================================

def upload_image(name: str, image_b64: str) -> str:
    """
    Sube una imagen (base64) a ComfyUI vía POST /upload/image.
    Retorna el nombre con que quedó guardada en ComfyUI.
    """
    image_bytes = base64.b64decode(image_b64)
    response = requests.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (name, io.BytesIO(image_bytes), "image/png")},
        data={"overwrite": "true"},
        timeout=30,
    )
    response.raise_for_status()
    saved_name = response.json().get("name", name)
    print(f"[handler] Imagen subida: {name!r} → {saved_name!r}")
    return saved_name


def queue_workflow(workflow: dict) -> str:
    """
    Encola un workflow en ComfyUI.
    Retorna el prompt_id para hacer polling del resultado.
    """
    response = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    prompt_id: str = response.json()["prompt_id"]
    print(f"[handler] Workflow encolado — prompt_id={prompt_id}")
    return prompt_id


def poll_history(prompt_id: str, timeout: int = 300) -> dict:
    """
    Hace polling a /history/{prompt_id} hasta que el job termine.
    Lanza TimeoutError si supera el timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            history = r.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed"):
                    print(f"[handler] Job completado — prompt_id={prompt_id}")
                    return history[prompt_id]
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI reportó error: {messages}")
        time.sleep(2)

    raise TimeoutError(f"Timeout ({timeout}s) esperando prompt_id={prompt_id}")


def collect_output_images(history: dict) -> list[dict]:
    """
    Extrae todas las imágenes de salida de la historia del job.
    Las descarga de /view y las retorna como lista de {"data": "<base64>"}.
    """
    images_out = []
    for _node_id, node_output in history.get("outputs", {}).items():
        for img_info in node_output.get("images", []):
            params = {
                "filename": img_info["filename"],
                "type":     img_info.get("type", "output"),
            }
            if img_info.get("subfolder"):
                params["subfolder"] = img_info["subfolder"]

            url = f"{COMFYUI_URL}/view?{urllib.parse.urlencode(params)}"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            images_out.append(
                {"data": base64.b64encode(r.content).decode("utf-8")})
            print(
                f"[handler] Imagen de salida recopilada: {img_info['filename']}")

    return images_out


# =============================================================================
# RunPod handler
# =============================================================================

def handler(job: dict) -> dict:
    """
    Handler principal invocado por RunPod por cada job.

    job["input"] debe contener:
      - workflow (dict)  : el workflow en formato API de ComfyUI
      - images   (list)  : lista de {"name": str, "image": "<base64>"}

    Retorna:
      {"images": [{"data": "<base64>"}, ...]}
    o en caso de error:
      {"error": "<mensaje>"}
    """
    job_input = job.get("input", {})
    workflow = job_input.get("workflow")
    images = job_input.get("images", [])

    if not workflow:
        return {"error": "Campo 'workflow' ausente en el input."}

    try:
        # 1. Subir imágenes de entrada
        for img_entry in images:
            name = img_entry.get("name", "image.png")
            b64data = img_entry.get("image", "")
            if b64data:
                upload_image(name, b64data)

        # 2. Encolar workflow
        prompt_id = queue_workflow(workflow)

        # 3. Esperar resultado
        history = poll_history(prompt_id)

        # 4. Recopilar y retornar imágenes de salida
        output_images = collect_output_images(history)
        return {"images": output_images}

    except TimeoutError as exc:
        print(f"[handler] Timeout: {exc}", file=sys.stderr)
        return {"error": str(exc)}
    except Exception as exc:
        print(f"[handler] Error inesperado: {exc}", file=sys.stderr)
        return {"error": str(exc)}


# =============================================================================
# Entry point — startup + RunPod worker
# =============================================================================

if __name__ == "__main__":
    # Paso 1: enlazar Network Volume si está montado
    setup_network_volume_symlinks()

    # Paso 2: descargar modelos si no existen
    download_models_if_needed()

    # Paso 3: levantar ComfyUI en background
    _comfyui_proc = start_comfyui()

    # Paso 4: esperar que ComfyUI esté listo
    if not wait_for_comfyui(timeout=180):
        print("[startup] ERROR: ComfyUI no respondió en 180 segundos.",
              file=sys.stderr)
        _comfyui_proc.terminate()
        sys.exit(1)

    # Paso 5: iniciar RunPod Serverless worker
    print("[startup] Iniciando RunPod Serverless worker...")
    runpod.serverless.start({"handler": handler})
