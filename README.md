# Qwen3.8-27B on DGX Spark (sparkrun)

Recommended, validated setup for serving **rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm**
on a DGX Spark (GB10) with **SGLang + DFlash2** speculative decoding and the
**froggeric fixed chat template** — all via `sparkrun`.

tool-eval-bench: **95/100** (quality 95, responsiveness 40, median turn 3.9s,
zero failures) on the standard suite, **93/100** on hardmode (median turn 3.8s).
Single-stream decode ≈ 21.7 tok/s.

## Quick start

Read **[SETUP-GUIDE.md](SETUP-GUIDE.md)** for the complete from-scratch walkthrough:
SGLang + DFlash2 image build, model weights, the froggeric chat template, the
PrismaAQUA load fix (mod), the recipe, and benchmarking.

```bash
sparkrun run qwen3.8-27b-paqua5.5-dflash2-sglang-m0l0.yaml
tool-eval-bench --base-url=http://localhost:2000/v1 --perf
tool-eval-bench --base-url=http://localhost:2000/v1 --hardmode
```

## Repo contents

| File | Purpose |
|---|---|
| `SETUP-GUIDE.md` | From-scratch setup, run and benchmark walkthrough |
| `qwen3.8-27b-paqua5.5-dflash2-sglang-m0l0.yaml` | **Recommended sparkrun recipe** (SGLang + DFlash2 + froggeric template) |
| `mods/fix-qwen35-prismaaqua-sglang/` | sparkrun mod — makes PrismaAQUA load under SGLang (prefix-tolerant compressed-tensors matcher) |
| `templates/fixed_chat_template.jinja` | froggeric chat template |

## Why this setup

- **Fast:** DFlash2 drafts an 8-token block in one pass (acceptance ≈ 3.15
  tokens/step); SGLang's parallel verify + radix cache keep latency low.
- **Accurate:** the froggeric chat template fixes empty-think poisoning,
  prefix-KV invalidation and reasoning-effort aliasing, and keeps the model's
  native XML tool format (`qwen3_coder` parser).

## Prerequisites

- DGX Spark (GB10, 121-128 GB unified memory), DGX OS, Docker + NVIDIA
  Container Toolkit, `sparkrun` (validated on 0.2.40).
- ~70 GB free disk (SGLang image ~39 GB + weights ~21 GB + draft ~3.6 GB +
  template + caches). First boot JIT-compiles sm_121a kernels (10-20 min).
- HF access to the public model/draft repos.

## Credits

This setup builds on the work of others:

- **Rob Tand (PrismaQuant)** — the
  [PrismaAQUA-5.5bit checkpoint](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm)
  this recipe serves, quantized with the [PrismaQuant](https://prismaquant.org)
  AQUA-AURA method.
- **[Qwen](https://qwenlm.github.io/)** — the base
  [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) model family.
- **[sparkrun](https://github.com/spark-arena/sparkrun)** — launch/manage/stop
  workloads on DGX Spark; recipes use the sparkrun recipe format.
- **[SGLang](https://github.com/sgl-project/sglang)** — the serving runtime and
  the [image](https://hub.docker.com/r/lmsysorg/sglang) this repo overlays with
  DFlash2.
- **[z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)** — the DFlash2
  draft model used for speculative decoding (pinned revision in the recipe).

Also in the stack:

- **[froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)** —
  the fixed chat template shipped in `templates/`.
- **[MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)** —
  the SGLang + DFlash2 image overlay and build script.
- **[incoai](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)** — original
  DFlash2 draft weights (`z-lab` is a pinned mirror).