#!/bin/bash
set -e

# Install the compressed-tensors matching shim for PrismaAQUA / Qwen3.5-VL.
#
# SGLang names Qwen3.5 language-model modules as
#   model.language_model.layers.N.*
# while llm-compressor checkpoints (e.g. rdtand/Qwen3.8-27B-PrismaAQUA) write
# config-group targets/ignore in HF style:
#   language_model.model.layers.N.*
# Because Qwen3_5ForConditionalGeneration.hf_to_sglang_mapper is None, the
# config targets are never translated, so every layer lookup fails with
# "Unable to find matching target for ... in the compressed-tensors config."
#
# The patched utils.py makes check_equal_or_regex_match / should_ignore_layer /
# find_matched_target try both spellings of the Qwen-VL prefix.
TARGET=/sgl-workspace/sglang/python/sglang/srt/layers/quantization/compressed_tensors/utils.py

cp utils.py "${TARGET}"

echo "=======> patched ${TARGET} with Qwen3.5-VL prefix-tolerant compressed-tensors matching"