# Self-hosted OpenAI-compatible endpoint on Modal (https://modal.com): vLLM serving an
# open-weights model on rented GPUs. For models Together offers only as "dedicated"
# (non-serverless) endpoints — or any HF model. The engine needs nothing new: point a
# provider block at the URL this app prints (see docs/configuration.md, "Self-hosted
# models on Modal").
#
# One-time setup (outside the project venv — Modal is infra, not a project dependency):
#   pip install modal && modal setup
#   modal secret create huggingface-secret HF_TOKEN=hf_...        # gated/private weights
#   modal secret create vllm-api-key VLLM_API_KEY=<any long random string>
#
# Deploy / run (settings via env vars, read at deploy time):
#   MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" GPU="H100:4" TP=4 modal deploy modal_vllm.py
#   MODEL="Qwen/Qwen2.5-72B-Instruct" GPU="H100:2" TP=2 modal serve modal_vllm.py   # dev: live logs, stops on Ctrl+C
#
# The deploy prints the endpoint URL, e.g. https://<workspace>--llm-reputation-vllm-serve.modal.run
# -> base_url: <that URL>/v1, api_key_env: VLLM_API_KEY (put the same key into .env).
# Cost = GPU-seconds while a container is up. MIN_CONTAINERS=1 keeps one warm during a
# sweep (no cold start per burst); MIN_CONTAINERS=0 (default) scales to zero when idle —
# the next request then waits for a container + weight load (minutes for big models).

from __future__ import annotations

import os

import modal

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("GPU", "A10G")                 # e.g. "H100", "H100:4", "A100-80GB:2"
TP = int(os.environ.get("TP", GPU.split(":")[1] if ":" in GPU else "1"))   # tensor parallel = #GPUs
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
MIN_CONTAINERS = int(os.environ.get("MIN_CONTAINERS", "0"))
SCALEDOWN_MINUTES = int(os.environ.get("SCALEDOWN_MINUTES", "15"))
VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.10.1")
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Weights and vLLM's compile cache persist across containers -> only the first start downloads.
hf_cache = modal.Volume.from_name("llm-reputation-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("llm-reputation-vllm-cache", create_if_missing=True)

app = modal.App("llm-reputation-vllm")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,
    scaledown_window=60 * SCALEDOWN_MINUTES,
    min_containers=MIN_CONTAINERS,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("huggingface-secret"),
             modal.Secret.from_name("vllm-api-key")],
)
@modal.concurrent(max_inputs=32)                    # requests batched by vLLM inside one container
@modal.web_server(port=PORT, startup_timeout=30 * 60)
def serve() -> None:
    import subprocess

    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--api-key", os.environ["VLLM_API_KEY"],   # the engine sends it as Authorization: Bearer
        "--served-model-name", MODEL,              # so `model:` in the config is the HF id
        "--tensor-parallel-size", str(TP),
        "--max-model-len", str(MAX_MODEL_LEN),
    ]
    subprocess.Popen(cmd)                          # web_server waits for the port to open
