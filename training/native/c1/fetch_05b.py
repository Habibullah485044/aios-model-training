#!/usr/bin/env python3
"""fetch_05b.py - download Qwen2.5-0.5B-Instruct from HuggingFace (Windows side).

C1 (D16) develops the config/manifest loader against the SMALL model first:
the 0.5B export is ~1 GB of BF16 weights, small enough to produce on the dev
PC (Windows Python has torch + transformers + numpy; WSL has no torch).

Only the safetensors shards + config.json are needed; the tokenizer is
downloaded too because export_weights.py writes ref_tokens.txt with it.
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_05b")


def main():
    allow = ["config.json", "model*.safetensors", "tokenizer*", "*.json"]
    path = snapshot_download(repo_id=REPO, local_dir=OUT, allow_patterns=allow)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    print("downloaded to %s (%.1f MB)" % (path, total / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
