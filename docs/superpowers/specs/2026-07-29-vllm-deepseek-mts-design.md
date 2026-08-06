# vLLM + DeepSeek-R1-Distill-Qwen-14B on MTS — design

Date: 2026-07-29
Status: approved

## Goal

Serve `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` (Ollama tag `deepseek-r1:14b`) as an
OpenAI-compatible endpoint on the MTS server (A100 80GB, `ssh MTS`), to be used as the
**agent provider** for research runs executed on MTS itself.

## Deployment

- `docker-compose.yml` placed at `~/LLM_reputation/LLM_reputation/` on MTS (server-side
  file in the repo checkout; not managed from the local machine).
- Image: `vllm/vllm-openai:v0.20.1` — pinned to the image already present on MTS (the same
  one the Qwen judge container uses), avoiding a second ~30GB pull onto the 91%-full disk.
- The image entrypoint is `["vllm", "serve"]`, so the compose `command` holds only the
  model (positional — `--model` is deprecated in 0.20) plus flags:
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
  - `--gpu-memory-utilization 0.5` — cap at ~40GB so the dreamerv3 training (~36GB) keeps running
  - `--max-model-len 32768`
  - `--reasoning-parser deepseek_r1` — `<think>` goes to `reasoning_content`; `message.content`
    stays clean JSON for `src/core/jsonextract.py`
- Ports: `127.0.0.1:8000 -> 8000` (localhost-only; 8100 stays reserved for the Qwen judge).
- Volumes: `~/.cache/huggingface:/root/.cache/huggingface` (persist ~28GB weights).
- `ipc: host`, `restart: unless-stopped`, healthcheck on `GET /health`.

## Episode-config provider block (MTS side)

```yaml
provider: &provider
  base_url: http://localhost:8000/v1
  model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
  temperature: 0.7
  max_tokens: 8000        # must cover <think> reasoning + final JSON
  timeout_s: 300
```

No `api_key_env` (vLLM ignores the dummy token). No `chat_template_kwargs` — R1-distill
cannot disable thinking.

## Tokenizer patch (required)

The image ships transformers 5.7.0, whose unified `LlamaTokenizer` class rebuilds the
decoder as a sentencepiece-style `Sequence`, breaking decode for this byte-level BPE
model (spaces lost / raw `Ġ`/`Ċ` tokens in output). Fix applied on MTS: in the HF cache
snapshot (`~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/
snapshots/*/tokenizer_config.json`), `tokenizer_class` is patched
`LlamaTokenizerFast` → `PreTrainedTokenizerFast`, which loads `tokenizer.json` verbatim
(correct `ByteLevel` decoder, chat template intact). **If the model cache is ever wiped
and re-downloaded, the patch must be re-applied.**

## Constraints / risks

- Disk on MTS is 91% full (74GB free): image + weights ≈ 45GB, leaving ~28GB.
- GPU is shared: if the training's usage grows past ~40GB the vLLM startup or KV cache
  allocation can fail; utilization cap chosen to avoid contention.

## Verification

1. `docker compose up -d`; wait for model load (watch `docker compose logs -f`).
2. `curl localhost:8000/health` returns 200.
3. A chat-completions probe with a DECIDE-style prompt returns clean `{"number": N}` JSON
   in `message.content` (reasoning in `reasoning_content`).
