#!/usr/bin/env python3
"""export_small.py - produce the C1 model files on the dev PC (Windows side).

C1 (D16) develops the config/manifest loader against the SMALL model first.
This writes exactly the three files the engine reads at runtime:

    config.txt     key/value model config (same contract as the Kaggle export)
    manifest.txt   "idx name dtype rows cols offset_bytes nbytes" per tensor
    weights.bin    raw BF16 bytes, tensors concatenated in manifest order

It reuses kernel/export_weights.py's tensor ORDER (the contract with the C
oracle) but skips the reference logits: C1 only needs the layout, and the
float32 reference model does not fit in this PC's RAM.

Usage:
    python export_small.py --model <hf_dir> --out <model_dir>
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kernel"))

import export_weights  # noqa: E402  (the Kaggle-era exporter: order + config)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local HF model directory")
    ap.add_argument("--out", required=True, help="output model directory")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    print("loading %s ..." % a.model, flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    model.eval()

    os.makedirs(a.out, exist_ok=True)
    n = export_weights.export(model, a.out)
    print("exported %d params -> %s" % (n, a.out))
    with open(os.path.join(a.out, "manifest.txt")) as f:
        lines = f.read().splitlines()
    total = sum(int(l.split()[6]) for l in lines if l.strip())
    size = os.path.getsize(os.path.join(a.out, "weights.bin"))
    print("tensors: %d   manifest bytes: %d   weights.bin bytes: %d   match: %s"
          % (len(lines), total, size, total == size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
