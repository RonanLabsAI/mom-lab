#!/usr/bin/env bash
# Restore every GPU service stopped for the MoM study (claim mom-study-20261029).
set -u
ssh station-a  'docker start glmf-daily nemotron-super-rtx'
ssh station-b 'docker start dsv41-daily-sidecar dsv41-daily'
ssh contact@example.invalid 'sudo -n systemctl start ollama; systemctl --user start comfyui generate-api kokoro'
ssh contact@example.invalid 'docker start qwen38-spark2'
ssh contact@example.invalid 'sudo -n systemctl start ollama'
echo "Restored. Pinned Ollama models (gpt-oss:120b, bge-m3, nomic-embed-text, nemotron-3-nano:30b) reload on first request or via keep_alive=-1 warmup."
