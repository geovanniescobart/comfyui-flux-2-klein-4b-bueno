#!/usr/bin/env python3
"""
handler.py — RunPod Serverless Worker para ComfyUI FLUX.2 Klein 4B
====================================================================

FLUJO DE EJECUCIÓN:
  1. Al arrancar el contenedor (startup):
     a. Si existe /runpod-volume/models → crea symlinks hacia ComfyUI/models
     b. Si los modelos no están y HF_TOKEN existe → los descarga con download_models.sh
     c. Levanta ComfyUI en segundo plano (puerto 8188, solo localhost)
     d. Espera que ComfyUI esté listo (~20-60s)
     e. Inicia el worker de RunPod (empieza a aceptar jobs)

  2. Por cada job recibido (handler):
     a. Sube la imagen al ComfyUI local
     b. Inyecta el prompt y el nombre de imagen en el workflow embebido
     c. Encola el workflow en ComfyUI (/prompt)
     d. Hace polling hasta que termine (/history/{prompt_id})
     e. Retorna las imágenes de salida como base64

REQUEST (simplificado — el workflow vive en el servidor):
  POST https://api.runpod.ai/v2/{endpoint_id}/runsync
  {
    "input": {
      "prompt": "change her t-shirt to pink pastel",
      "image":  "<base64>",
      "seed":   42,        ← opcional (default: aleatorio)
      "steps":  20,        ← opcional (default: 20)
      "cfg":    5.0        ← opcional (default: 5.0)
    }
  }

RESPONSE:
  {
    "output": {
      "images": [ { "data": "<base64>" } ]
    }
  }
"""

import base64
import io
import json
import os
import shutil
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
COMFYUI_MODELS_DIR = os.path.join(
    COMFYUI_DIR, "models")   # ruta que ve ComfyUI
# punto de montaje del volumen
NETWORK_VOLUME_MOUNT = "/runpod-volume"
# donde se guardan/leen los modelos
NETWORK_VOLUME_MODELS = "/runpod-volume/models"

MODEL_SUBDIRS = ["text_encoders", "diffusion_models", "vae"]


# =============================================================================
# Startup helpers
# =============================================================================

def effective_models_dir() -> str:
    """
    Devuelve el directorio real donde deben vivir los modelos:
      - Si el Network Volume está montado → /runpod-volume/models  (persistente)
      - Si no                             → /app/ComfyUI/models     (efímero)
    """
    if os.path.isdir(NETWORK_VOLUME_MOUNT):
        os.makedirs(NETWORK_VOLUME_MODELS, exist_ok=True)
        return NETWORK_VOLUME_MODELS
    return COMFYUI_MODELS_DIR


def setup_symlinks(models_dir: str) -> None:
    """
    Crea symlinks desde ComfyUI/models/* → models_dir/* para que ComfyUI
    lea los modelos del volumen sin necesidad de copiarlos.
    Solo actúa cuando models_dir != ComfyUI/models (es decir, con volumen).
    """
    if models_dir == COMFYUI_MODELS_DIR:
        return  # los modelos ya están en la ruta nativa de ComfyUI

    print(f"[startup] Configurando symlinks: ComfyUI/models → {models_dir}")
    for subdir in MODEL_SUBDIRS:
        src = os.path.join(models_dir, subdir)          # en el volumen
        dst = os.path.join(COMFYUI_MODELS_DIR, subdir)  # donde ComfyUI espera
        os.makedirs(src, exist_ok=True)
        if os.path.islink(dst) or os.path.exists(dst):
            continue  # ya existe (symlink o directorio real)
        os.symlink(src, dst)
        print(f"[startup]   symlink: {dst} → {src}")


def download_models_if_needed(models_dir: str) -> None:
    """
    Descarga los modelos al directorio indicado si no existen.
    Cuando se usa Network Volume, los archivos quedan persistidos entre arranques.
    Requiere HF_TOKEN para el modelo gated de Black Forest Labs.
    """
    marker = os.path.join(models_dir, "diffusion_models",
                          "flux-2-klein-base-4b-fp8.safetensors")
    if os.path.isfile(marker):
        location = "volumen" if models_dir == NETWORK_VOLUME_MODELS else "contenedor"
        print(
            f"[startup] Modelos ya presentes en {location} — saltando descarga.")
        return

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("[startup] ADVERTENCIA: HF_TOKEN no definido. "
              "El modelo FLUX.2 Klein está en un repo gated y puede fallar la descarga.")

    print(f"[startup] Descargando modelos en: {models_dir} ...")
    # MODELS_DIR sobreescribe la ruta destino en download_models.sh
    env = {**os.environ, "HF_TOKEN": hf_token, "MODELS_DIR": models_dir}
    result = subprocess.run(["/bin/bash", "/app/download_models.sh"], env=env)
    if result.returncode != 0:
        print("[startup] ERROR al descargar modelos — abortando.", file=sys.stderr)
        sys.exit(1)


def _find_python() -> str:
    """
    Encuentra el intérprete Python correcto para lanzar ComfyUI.
    En imágenes AI Dock el Python vive en el entorno micromamba/conda;
    se prueban las rutas conocidas antes de caer en sys.executable.
    """
    candidates = [
        # AI Dock micromamba (nombre de env puede variar)
        "/opt/micromamba/envs/python_env/bin/python",
        "/opt/micromamba/envs/comfyui/bin/python",
        "/opt/conda/envs/comfyui/bin/python",
        # python3/python en PATH (funciona si Docker ENV PATH ya lo incluye)
        shutil.which("python3") or "",
        shutil.which("python") or "",
        sys.executable,
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return sys.executable


def start_comfyui() -> subprocess.Popen:
    """Levanta ComfyUI en background escuchando solo en localhost."""
    python = _find_python()
    print(f"[startup] Python: {python}")
    print(f"[startup] COMFYUI_DIR: {COMFYUI_DIR}")
    print("[startup] Levantando ComfyUI en segundo plano...")
    # IMPORTANTE: NO usar stdout=PIPE sin leerlo.
    # El buffer de 64 KB se llena con la salida de ComfyUI y el proceso se congela.
    # Dejamos que stdout/stderr se hereden para que aparezcan en los logs del contenedor.
    process = subprocess.Popen(
        [python, "main.py",
         "--listen", "127.0.0.1",
         "--port", "8188",
         "--disable-auto-launch"],
        cwd=COMFYUI_DIR,
    )
    return process


def wait_for_comfyui(process: subprocess.Popen, timeout: int = 180) -> bool:
    """Hace polling a /system_stats hasta que ComfyUI responda o el proceso muera."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Detectar muerte prematura del proceso
        ret = process.poll()
        if ret is not None:
            print(f"[startup] ERROR: ComfyUI terminó inesperadamente (exit code {ret})",
                  file=sys.stderr)
            return False
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
            if r.status_code == 200:
                elapsed = timeout - (deadline - time.time())
                print(f"[startup] ComfyUI listo ({elapsed:.0f}s) ✓")
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
# Workflow embebido — FLUX.2 Klein 4B Image Edit
# El cliente solo necesita enviar: prompt + image (base64)
# =============================================================================

IMAGE_INPUT_NODE = "76"   # LoadImage
PROMPT_NODE = "74"   # CLIPTextEncode positivo
SEED_NODE = "73"   # RandomNoise
STEPS_NODE = "62"   # Flux2Scheduler
CFG_NODE = "63"   # CFGGuider
IMAGE_FILENAME = "input_image.jpeg"

BASE_WORKFLOW: dict = {
    "76": {
        "class_type": "LoadImage",
        "inputs": {"image": IMAGE_FILENAME, "upload": "image"}
    },
    "80": {
        "class_type": "ImageScaleToTotalPixels",
        "inputs": {"image": ["76", 0], "upscale_method": "nearest-exact", "megapixels": 1.0}
    },
    "100": {
        "class_type": "GetImageSize",
        "inputs": {"image": ["80", 0]}
    },
    "70": {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": "FLUX.2-klein/flux-2-klein-base-4b-fp8.safetensors", "weight_dtype": "default"}
    },
    "71": {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}
    },
    "72": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "flux2-dev/flux2-vae.safetensors"}
    },
    "73": {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": 42}
    },
    "74": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["71", 0], "text": ""}
    },
    "67": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["71", 0], "text": ""}
    },
    "78": {
        "class_type": "VAEEncode",
        "inputs": {"pixels": ["80", 0], "vae": ["72", 0]}
    },
    "77": {
        "class_type": "ReferenceLatent",
        "inputs": {"conditioning": ["74", 0], "latent": ["78", 0]}
    },
    "101": {
        "class_type": "ReferenceLatent",
        "inputs": {"conditioning": ["67", 0], "latent": ["78", 0]}
    },
    "61": {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"}
    },
    "62": {
        "class_type": "Flux2Scheduler",
        "inputs": {"steps": 20, "width": ["100", 0], "height": ["100", 1]}
    },
    "66": {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {"width": ["100", 0], "height": ["100", 1], "batch_size": 1}
    },
    "63": {
        "class_type": "CFGGuider",
        "inputs": {"model": ["70", 0], "positive": ["77", 0], "negative": ["101", 0], "cfg": 5.0}
    },
    "64": {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {"noise": ["73", 0], "guider": ["63", 0], "sampler": ["61", 0], "sigmas": ["62", 0], "latent_image": ["66", 0]}
    },
    "65": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["64", 0], "vae": ["72", 0]}
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "Flux2-Klein-4b-base", "images": ["65", 0]}
    }
}


def build_workflow(prompt: str, image_name: str, seed: int, steps: int, cfg: float) -> dict:
    """Clona el workflow base e inyecta los parámetros del request."""
    import copy
    wf = copy.deepcopy(BASE_WORKFLOW)
    wf[IMAGE_INPUT_NODE]["inputs"]["image"] = image_name
    wf[PROMPT_NODE]["inputs"]["text"] = prompt
    wf[SEED_NODE]["inputs"]["noise_seed"] = seed
    wf[STEPS_NODE]["inputs"]["steps"] = steps
    wf[CFG_NODE]["inputs"]["cfg"] = cfg
    return wf


# =============================================================================
# RunPod handler
# =============================================================================

def handler(job: dict) -> dict:
    """
    Handler principal invocado por RunPod por cada job.

    job["input"] debe contener:
      - prompt (str)         : instrucción de edición
      - image  (str)         : imagen de entrada en base64
      - seed   (int)         : opcional, default aleatorio
      - steps  (int)         : opcional, default 20
      - cfg    (float)       : opcional, default 5.0

    Retorna:
      {"images": [{"data": "<base64>"}]}
    o en caso de error:
      {"error": "<mensaje>"}
    """
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "")
    image_b64 = job_input.get("image", "")
    seed = int(job_input.get("seed",  __import__('random').randint(0, 2**32)))
    steps = int(job_input.get("steps", 20))
    cfg = float(job_input.get("cfg",  5.0))

    if not prompt:
        return {"error": "Campo 'prompt' ausente en el input."}
    if not image_b64:
        return {"error": "Campo 'image' (base64) ausente en el input."}

    try:
        # 1. Subir imagen de entrada
        saved_name = upload_image(IMAGE_FILENAME, image_b64)

        # 2. Construir workflow con los parámetros del request
        workflow = build_workflow(prompt, saved_name, seed, steps, cfg)

        # 3. Encolar workflow
        prompt_id = queue_workflow(workflow)

        # 4. Esperar resultado
        history = poll_history(prompt_id)

        # 5. Recopilar y retornar imágenes de salida
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
    # Paso 1: determinar dónde vivirán los modelos (volumen o local)
    _models_dir = effective_models_dir()
    print(f"[startup] Directorio de modelos: {_models_dir}")

    # Paso 2: descargar modelos al destino correcto si no existen
    download_models_if_needed(_models_dir)

    # Paso 3: crear symlinks para que ComfyUI los encuentre
    setup_symlinks(_models_dir)

    # Paso 4: levantar ComfyUI en background
    _comfyui_proc = start_comfyui()

    # Paso 5: esperar que ComfyUI esté listo
    if not wait_for_comfyui(_comfyui_proc, timeout=180):
        print("[startup] ERROR: ComfyUI no respondió en 180 segundos.",
              file=sys.stderr)
        _comfyui_proc.terminate()
        sys.exit(1)

    # Paso 5: iniciar RunPod Serverless worker
    print("[startup] Iniciando RunPod Serverless worker...")
    runpod.serverless.start({"handler": handler})
