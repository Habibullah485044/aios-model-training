#!/usr/bin/env python3
"""compile_check - builds the oracle and verifies it on tiny random Qwen2 models.

Checks (thresholds set before running; never loosened):
  a) gcc -std=c99 -Wall -Wextra -Werror -O2 compile is clean.
  b) Tiny random Qwen2 (tied AND untied), weights rounded to BF16 first
     (model.to(torch.bfloat16).float()): oracle logits vs PyTorch,
     max abs diff < 1e-5 and argmax matches at every position.
  c) Every exported tensor: BF16 -> FP32 bit conversion matches torch's
     own .float() bit-for-bit (0 mismatched bits).

Paths are resolved relative to this file; generated artifacts go to
$AIOS_TMP (default /tmp/aios_native).
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NATIVE, "kernel"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import export_weights  # noqa: E402

CSRC = os.path.join(NATIVE, "runtime", "aios_infer.c")
TMP = os.environ.get("AIOS_TMP", "/tmp/aios_native")
EXE = os.path.join(TMP, "aios_infer")

TINY_DIFF_MAX = 1e-5  # set before running (historical FP32 result was ~3e-7)
SEED = 1234
VOCAB = 256


def compile_oracle():
    os.makedirs(TMP, exist_ok=True)
    cmd = ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", CSRC, "-o", EXE, "-lm"]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("compile OK ->", EXE, flush=True)


def build_tiny(tie):
    torch.manual_seed(SEED)
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
    )
    # round weights to BF16 exactly like the real export does
    return Qwen2ForCausalLM(cfg).to(torch.bfloat16).float()


def bit_exact_check(model, exported_dir):
    """(c) exported BF16 bytes -> FP32 must equal torch's .float() bits."""
    tensors = export_weights.ordered_tensors(model)
    manifest = {}
    with open(os.path.join(exported_dir, "manifest.txt")) as f:
        for line in f:
            idx, name, dtype, rows, cols, off, nbytes = line.split()
            manifest[name] = (int(off), int(nbytes))
    raw = np.fromfile(os.path.join(exported_dir, "weights.bin"), dtype=np.uint16)
    total_bad = 0
    for name, ten in tensors:
        off, nbytes = manifest[name]
        chunk = raw[off // 2: off // 2 + nbytes // 2]
        got32 = chunk.astype(np.uint32) << 16
        ref32 = ten.detach().float().numpy().reshape(-1).view(np.uint32)  # torch's own conversion
        bad = int((got32 != ref32).sum())
        if bad:
            print("  BIT MISMATCH in %s: %d/%d bits" % (name, bad, ref32.size))
        total_bad += bad
    return total_bad


def run_case(tag, tie):
    d = os.path.join(TMP, "tiny_%s" % tag)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    model = build_tiny(tie)
    n = export_weights.export(model, d)
    export_weights.write_reference(model, list(range(VOCAB))[:8], d)
    print("[%s] exported %d params" % (tag, n), flush=True)

    bad = bit_exact_check(model, d)
    print("[%s] bit-exact mismatches: %d" % (tag, bad), flush=True)

    out = os.path.join(d, "oracle_logits.bin")
    subprocess.run([EXE, d, os.path.join(d, "ref_tokens.txt"), out], check=True)
    ref = np.fromfile(os.path.join(d, "ref_logits.bin"), dtype=np.float32)
    got = np.fromfile(out, dtype=np.float32)
    assert ref.shape == got.shape, (ref.shape, got.shape)
    diff = float(np.abs(ref - got).max())
    arg_ok = bool((ref.reshape(-1, VOCAB).argmax(1) == got.reshape(-1, VOCAB).argmax(1)).all())
    print("[%s] max abs diff = %.3e (threshold %.0e) argmax match = %s"
          % (tag, diff, TINY_DIFF_MAX, arg_ok), flush=True)

    assert bad == 0, "BF16->FP32 bit mismatch"
    assert diff < TINY_DIFF_MAX, "logit diff above threshold"
    assert arg_ok, "argmax mismatch"
    del model


def main():
    compile_oracle()
    run_case("tied", True)
    run_case("untied", False)
    print("ALL COMPILE_CHECKS PASSED")


if __name__ == "__main__":
    main()
