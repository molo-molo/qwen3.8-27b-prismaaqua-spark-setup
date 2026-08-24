# Qwen3.8-27B on DGX Spark — PrismaAQUA + SGLang DFlash2 + froggeric template

**Status: RECOMMENDED / validated.** tool-eval-bench **95/100 quality, 40/100
responsiveness (median turn 3.9s)**, zero safety failures, ~21.7 tok/s
single-stream. Hardmode: **93/100**, median turn 3.8s.

This is a from-scratch, one-box guide for DGX Spark (GB10, sm_121a) using
`sparkrun`. It serves `rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm` under
SGLang with a DFlash2 block-diffusion draft and the `froggeric` fixed chat
template.

---

## Results (tool-eval-bench)

| Run | Score | Quality | Responsiveness | Median turn | Duration |
|---|---|---|---|---|---|
| Standard (69 scenarios) | **95** | **95** | **40** | **3.9s** | ~17 min |
| Hardmode (84 scenarios) | **93** | **93** | **41** | **3.8s** | ~21 min |

- Standard: 131/138 pts, ★★★★★, **0 failures**, 0 safety-critical.
- Hardmode: 156/168 pts, ★★★★★, 2 failures (both 5★ long-horizon tasks:
  TC-80 preconditioned update safety, TC-84 long-horizon recovery).

Why it's fast: DFlash2 drafts an 8-token block in one pass (acceptance ~3.15
tokens/step), and SGLang's parallel verify + radix cache keep latency low.
Why it's accurate: the froggeric chat template fixes empty-think poisoning,
prefix-KV invalidation and reasoning-effort aliasing, and keeps the model's
native XML tool format (`qwen3_coder` parser).

---

## Prerequisites

- DGX Spark (GB10, 121-128 GB unified memory), DGX OS, Docker + NVIDIA
  Container Toolkit.
- `sparkrun` installed (validated on 0.2.40).
- ~70 GB free disk (SGLang image ~39 GB + weights ~21 GB + draft ~3.6 GB +
  template + caches). First boot JIT-compiles sm_121a kernels (10-20 min).
- HF access (public repos; a `HF_TOKEN` speeds downloads).

---

## 1. Get the support files

Create a working directory, e.g. `~/qwen38-prismaaqua-sglang/`, and inside it:

```
qwen3.8-27b-paqua5.5-dflash2-sglang-m0l0.yaml   (recipe)
mods/fix-qwen35-prismaaqua-sglang/run.sh                       (mod launcher)
mods/fix-qwen35-prismaaqua-sglang/utils.py                     (patched SGLang file)
templates/fixed_chat_template.jinja                            (froggeric template)
```

Clone the MiaAI repo for the DFlash2 image overlay:

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark.git
```

### 1.1 Build the SGLang + DFlash2 image

SGLang's released tags don't ship DFlash2; the MiaAI repo overlays the five
DFlash2 modules onto the pinned cookbook image:

```bash
cd Qwen3.8-27B-SGLang-DGX-Spark
bash patch/build-dflash2-image.sh --minimal
# -> lmsysorg/sglang:qwen38-27b-dflash2-minoverlay
```

The base image `lmsysorg/sglang:qwen38-27b` is pulled automatically (arm64,
CUDA 13.0.3, flashinfer 0.6.18 with sm_121a JIT support). You can delete the
un-tagged base image afterwards to reclaim ~39 GB:
`docker rmi lmsysorg/sglang:qwen38-27b` (the overlay image carries all layers).

### 1.2 Download models (HF cache)

```bash
export HF_HOME=/opt/llm-models/huggingface   # or ~/.cache/huggingface
hf download rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm
hf download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4
```

(Note: `hf download` writes to `<HF_HOME>/models--...`; if your SGLang
container mounts `<HF_HOME>` at `/root/.cache/huggingface`, keep the repo IDs
as-is and sparkrun/SGLang resolve them through the cache. The `z-lab` draft
is the same weights as `incoai/Qwen3.8-27B-DFlash2`, pinned for reproducibility.)

### 1.3 Fetch the froggeric chat template

```bash
hf download froggeric/Qwen-Fixed-Chat-Templates
cp <cache>/models--froggeric--Qwen-Fixed-Chat-Templates/snapshots/*/chat_template.jinja \
   templates/fixed_chat_template.jinja
```

Copy it to the HF cache root too, so the container sees it mounted:

```bash
cp templates/fixed_chat_template.jinja "$HF_HOME/fixed_chat_template.jinja"
```

### 1.4 The mod (makes PrismaAQUA load under SGLang)

**Why this is needed.** SGLang names Qwen3.5 language-model modules
`model.language_model.layers.N.*`, but PrismaAQUA's compressed-tensors config
writes its group targets / ignore list in HF style `language_model.model.layers.N.*`,
and `Qwen3_5ForConditionalGeneration.hf_to_sglang_mapper` is `None` — so SGLang
never translates the config and every layer lookup fails with *"Unable to find
matching target for model.language_model.layers.0.linear_attn.in_proj_qkvz"*.
vLLM loads the same checkpoint because it applies the mapper.

**The fix.** A sparkrun `mod` patches
`sglang/srt/layers/quantization/compressed_tensors/utils.py` so the matcher
tries both spellings of the Qwen-VL prefix. It adds `_layer_name_candidates()`
and makes `check_equal_or_regex_match`, `should_ignore_layer` and
`find_matched_target` accept every candidate.

`mods/fix-qwen35-prismaaqua-sglang/run.sh`:

```bash
#!/bin/bash
set -e
TARGET=/sgl-workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/utils.py
cp utils.py "${TARGET}"
echo "=======> patched ${TARGET} with Qwen3.5-VL prefix-tolerant compressed-tensors matching"
```

`mods/fix-qwen35-prismaaqua-sglang/utils.py` is the patched SGLang file (diff
against the pristine file = ~25 lines: the `_QWEN35_PREFIX_SWAP` table +
`_layer_name_candidates()` + three call sites). It ships in this repo.

---

## 2. The recipe

`qwen3.8-27b-paqua5.5-dflash2-sglang-m0l0.yaml`:

```yaml
recipe_version: '2'
model: rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm
runtime: sglang
container: lmsysorg/sglang:qwen38-27b-dflash2-minoverlay

mods:
  - fix-qwen35-prismaaqua-sglang

defaults:
  host: 0.0.0.0
  port: 2000
  served_model_name: qwen
  tensor_parallel: 1
  max_num_seqs: 8
  max_model_len: 262144
  mem_fraction_static: 0.90
  speculative_draft_model_path: z-lab/Qwen3.8-27B-DFlash2
  speculative_draft_model_revision: 50307d4c4cde6860d4eee73e2547cd786fe8e8a4

env:
  HF_HUB_OFFLINE: '0'
  SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK: '0'

command: |
  python3 -m sglang.launch_server \
    --model-path rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm \
    --served-model-name {served_model_name} \
    --trust-remote-code \
    --mem-fraction-static {mem_fraction_static} \
    --attention-backend flashinfer \
    --chunked-prefill-size 8192 \
    --disable-prefill-cuda-graph \
    --kv-cache-dtype fp8_e4m3 \
    --mamba-ssm-dtype bfloat16 \
    --mamba-full-memory-ratio 4.21 \
    --mamba-radix-cache-strategy extra_buffer \
    --max-running-requests {max_num_seqs} \
    --context-length {max_model_len} \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path {speculative_draft_model_path} \
    --speculative-draft-model-revision {speculative_draft_model_revision} \
    --speculative-num-draft-tokens 8 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --chat-template /cache/huggingface/fixed_chat_template.jinja \
    --default-chat-template-kwargs '{"reasoning_effort":"medium","preserve_thinking":true}' \
    --sampling-defaults model \
    --enable-metrics \
    --enable-cache-report \
    --host {host} \
    --port {port}
```

Notes on the flags:
- `--chat-template` points at the froggeric template (`.jinja` file path —
  SGLang loads it via `_load_explicit_jinja_template`). The container sees it
  because `<HF_HOME>` is mounted at `/cache/huggingface`.
- `--default-chat-template-kwargs` applies `reasoning_effort=medium` (zero
  injected tokens, avoids token-budget burn) and `preserve_thinking=true`
  (100% prefix-KV cache reuse).
- `--mamba-radix-cache-strategy extra_buffer` is **required** for DFLASH
  (`extra_buffer_lazy` asserts out).
- DFLASH needs `--mamba-full-memory-ratio 4.21` + `--max-running-requests` to
  size the GDN state pool correctly.
- `--mem-fraction-static 0.90` is the validated NVFP4 value (0.95 has caused
  GB10 hard reboots at draft-graph capture).
- Your `--chat-template` path must match the container mount; if you keep
  models under `~/.cache/huggingface` and mount it, use that path instead.

---

## 3. Run it (sparkrun)

```bash
sparkrun run qwen3.8-27b-paqua5.5-dflash2-sglang-m0l0.yaml
```

- First boot downloads/verifies the weights, JIT-compiles sm_121a kernels and
  captures draft/target CUDA graphs: **10-20 min** to `healthy` (warm restarts
  ~2-3 min).
- Check: `sparkrun status`, `curl http://localhost:2000/v1/models`
- Stop: `sparkrun stop <job>`

Sanity check (thinking works, froggeric template active):

```bash
curl http://localhost:2000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen",
  "messages": [{"role": "user", "content": "What is the capital of France? One sentence."}],
  "max_tokens": 60, "reasoning_effort": "medium"}'
```

Expect `reasoning_tokens > 0` (thinking) and a clean answer.

---

## 4. Benchmark (tool-eval-bench)

```bash
cd <workdir>
tool-eval-bench --base-url=http://localhost:2000/v1 --perf        # throughput + 69 scenarios
tool-eval-bench --base-url=http://localhost:2000/v1 --hardmode    # 84 scenarios (incl. Category P)
```

Reports land in `./runs/`.

Known caveat: llama-benchy throughput inside `--perf` may fail on the
unpatched image (the known streaming + `return_token_ids` → 400 issue on
this SGLang build); the tool-call scenarios still run. Single-stream decode
measured separately ≈ **21.7 tok/s** (300 tok completion), TTFT ≈ 1-2s.

---

## 5. Gotchas / reproducibility

- **Image is Spark-only**: `lmsysorg/sglang:qwen38-27b` is arm64/SBSA with
  sm_121a JIT — built for DGX Spark GB10, not x86 GB200/GB300.
- **PrismaAQUA needs the mod** to load under SGLang (naming-convention shim).
  RadixArk's NVFP4 checkpoint doesn't need it, but PrismaAQUA scores equal
  quality at lower latency here once patched.
- The draft `z-lab/Qwen3.8-27B-DFlash2@50307d4` is pinned; the base
  `lmsysorg/sglang:qwen38-27b` tag is the official cookbook image.
- YaRN/context >262144 is NOT compatible with DFLASH on this build — keep
  `context-length` ≤ 262144.
- If your `hf download` puts models in a legacy cache path, move them into
  `<HF_HOME>/hub/models--...` (the layout SGLang's cache resolver reads), or
  point the container mount at the right directory.