#!/usr/bin/env python3
"""real_check - Phase 0 verification on Qwen2.5-3B-Instruct.

Checks (thresholds set before running; never loosened):
  1) weights.bin is BF16 and exactly 6,171,877,376 bytes
     (3,085,938,688 params x 2); manifest sums to the same param count.
  2) Every tensor: BF16 -> FP32 conversion equals torch's own .float()
     bits bit-for-bit (0 mismatched bits across all params).
  3) Oracle logits from the BF16 file vs PyTorch fp32 forward on the
     reference prompt: max abs diff < 1e-3, argmax matches at all 8
     positions.

PyTorch export and the oracle run in separate subprocess steps so each
stays within Kaggle RAM. Artifacts go to $AIOS_TMP (default /tmp/aios_native).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)

MODEL = os.environ.get("AIOS_MODEL", "Qwen/Qwen2.5-3B-Instruct")
# results land next to this file -> captured in Kaggle kernel output artifacts
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt")
CSRC = os.path.join(NATIVE, "runtime", "aios_infer.c")
TMP = os.environ.get("AIOS_TMP", "/tmp/aios_native")
EXE = os.path.join(TMP, "aios_infer")
EXPORT_DIR = os.path.join(TMP, "qwen2_5_3b")
PROMPT = "The capital of France is Paris, and it is also the largest city in the country."
N_TOKENS = 8

EXPECTED_BYTES = 6_171_877_376
EXPECTED_PARAMS = 3_085_938_688
LOGIT_DIFF_MAX = 1e-3  # set before running


def step(tag, code):
    """Run a python snippet in a fresh subprocess to keep RAM bounded."""
    env = dict(os.environ, AIOS_TMP=TMP, AIOS_MODEL=MODEL, AIOS_RESULTS=RESULTS)
    print("=== %s ===" % tag, flush=True)
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       cwd=os.path.dirname(HERE))
    if r.returncode != 0:
        print("STEP FAILED: %s (exit %d)" % (tag, r.returncode), file=sys.stderr)
        sys.exit(r.returncode)


EXPORT_CODE = r'''
import os, sys
sys.path.insert(0, os.environ.get("AIOS_SRC", "kernel"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import export_weights

model_id = os.environ["AIOS_MODEL"]
out = os.path.join(os.environ["AIOS_TMP"], "qwen2_5_3b")
tok = AutoTokenizer.from_pretrained(model_id)
ids = tok("The capital of France is Paris, and it is also the largest city in the country.")["input_ids"][:8]
assert len(ids) == 8
print("reference tokens:", ids)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
n = export_weights.export(model, out)
print("exported params:", n)
ref = export_weights.write_reference(model.float(), ids, out)
print("ref logits:", ref.shape)
with open(os.environ["AIOS_RESULTS"], "a") as f:
    f.write("tokens: %s\n" % ids)
    f.write("exported params: %d\n" % n)
    f.write("ref logits shape: %s\n" % (ref.shape,))
'''

VERIFY_CODE = r'''
import os, sys
sys.path.insert(0, os.environ.get("AIOS_SRC", "kernel"))
import numpy as np

tmp = os.environ["AIOS_TMP"]
d = os.path.join(tmp, "qwen2_5_3b")

# manifest param count
total = 0
with open(os.path.join(d, "manifest.txt")) as f:
    for line in f:
        idx, name, dtype, rows, cols, off, nbytes = line.split()
        assert dtype == "bf16", dtype
        total += int(rows) * int(cols)
print("manifest params:", total)
assert total == 3_085_938_688, total

sz = os.path.getsize(os.path.join(d, "weights.bin"))
print("weights.bin bytes:", sz)
assert sz == 6_171_877_376, sz

ref = np.fromfile(os.path.join(d, "ref_logits.bin"), dtype=np.float32)
out = os.path.join(d, "oracle_logits.bin")
got = np.fromfile(out, dtype=np.float32)
assert ref.shape == got.shape, (ref.shape, got.shape)
n = len(open(os.path.join(d, "ref_tokens.txt")).read().split())
assert n == 8, n
ref = ref.reshape(n, -1); got = got.reshape(n, -1)
diff = float(np.abs(ref - got).max())
arg_ok = bool((ref.argmax(1) == got.argmax(1)).all())
print("max abs diff = %.3e (threshold %.0e)" % (diff, 1e-3))
print("argmax ref:", ref.argmax(1).tolist())
print("argmax got:", got.argmax(1).tolist())
print("argmax match at all %d positions: %s" % (n, arg_ok))
with open(os.environ["AIOS_RESULTS"], "a") as f:
    f.write("max abs diff: %.6e (threshold 1e-3)\n" % diff)
    f.write("argmax ref: %s\n" % ref.argmax(1).tolist())
    f.write("argmax got: %s\n" % got.argmax(1).tolist())
    f.write("argmax match at all %d positions: %s\n" % (n, arg_ok))
assert diff < 1e-3, "logit diff above threshold"
assert arg_ok, "argmax mismatch"
print("REAL_CHECK PASSED")
with open(os.environ["AIOS_RESULTS"], "a") as f:
    f.write("REAL_CHECK PASSED\n")
'''

# per-tensor bit-exact check runs inside the export process (model still in RAM)
BITEXACT_CODE = r'''
import os, sys
sys.path.insert(0, os.environ.get("AIOS_SRC", "kernel"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import export_weights

model_id = os.environ["AIOS_MODEL"]
d = os.path.join(os.environ["AIOS_TMP"], "qwen2_5_3b")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tensors = export_weights.ordered_tensors(model)
manifest = {}
with open(os.path.join(d, "manifest.txt")) as f:
    for line in f:
        idx, name, dtype, rows, cols, off, nbytes = line.split()
        manifest[name] = (int(off), int(nbytes))
raw = np.memmap(os.path.join(d, "weights.bin"), dtype=np.uint16, mode="r")
total_bad = 0
total_params = 0
for name, ten in tensors:
    off, nbytes = manifest[name]
    chunk = raw[off // 2: off // 2 + nbytes // 2]
    got32 = chunk.astype(np.uint32) << 16
    ref32 = ten.detach().float().numpy().reshape(-1).view(np.uint32)
    bad = int((got32 != ref32).sum())
    if bad:
        print("  BIT MISMATCH %s: %d/%d" % (name, bad, ref32.size))
    total_bad += bad
    total_params += ref32.size
    if total_params % 200_000_000 < 50_000_000:
        print("  checked %d params, %d bad so far" % (total_params, total_bad), flush=True)
print("bit-exact: params=%d mismatched_bits=%d" % (total_params, total_bad))
with open(os.environ["AIOS_RESULTS"], "a") as f:
    f.write("bit-exact params: %d\n" % total_params)
    f.write("bit-exact mismatched bits: %d\n" % total_bad)
assert total_bad == 0, "BF16->FP32 bit mismatch"
'''


def main():
    os.makedirs(TMP, exist_ok=True)
    if os.path.exists(RESULTS):
        os.remove(RESULTS)

    # (0) compile oracle (same flags as compile_check)
    cmd = ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", CSRC, "-o", EXE, "-lm"]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # (1) export weights.bin (BF16) + reference logits
    step("export", EXPORT_CODE)

    # (2) bit-exact BF16->FP32 for every tensor (fresh process, model reloaded)
    step("bit-exact", BITEXACT_CODE)

    # (3) oracle forward on the BF16 file (separate step: only the C binary + file)
    subprocess.run([EXE, EXPORT_DIR, os.path.join(EXPORT_DIR, "ref_tokens.txt"),
                    os.path.join(EXPORT_DIR, "oracle_logits.bin")], check=True)

    # (4) compare
    step("compare", VERIFY_CODE)

    # (5) Phase 1: exponent statistics over the already-exported weights.bin
    #     (no model download; memmap only). Writes phase1_results.txt.
    phase1 = os.path.join(NATIVE, "phase1", "measure.py")
    if os.path.exists(phase1):
        print("=== phase1 (exponent statistics) ===", flush=True)
        env = dict(os.environ, AIOS_TMP=TMP,
                   AIOS_RESULTS=os.path.join(NATIVE, "phase1", "phase1_results.txt"))
        r = subprocess.run([sys.executable, phase1], env=env,
                           cwd=os.path.dirname(HERE))
        if r.returncode != 0:
            print("STEP FAILED: phase1 (exit %d)" % r.returncode, file=sys.stderr)
            sys.exit(r.returncode)
    else:
        print("phase1/measure.py not bundled; skipping Phase 1", flush=True)

    # (6) Phase 2: pack the already-exported weights.bin, then verify the
    #     scalar reference unpacker reproduces every BF16 bit exactly.
    phase2 = os.path.join(NATIVE, "phase2", "phase2_check.py")
    if os.path.exists(phase2):
        print("=== phase2 (pack + bit-exact unpack) ===", flush=True)
        env = dict(os.environ, AIOS_TMP=TMP,
                   AIOS_RESULTS=os.path.join(NATIVE, "phase2", "phase2_results.txt"))
        r = subprocess.run([sys.executable, phase2], env=env,
                           cwd=os.path.dirname(HERE))
        if r.returncode != 0:
            print("STEP FAILED: phase2 (exit %d)" % r.returncode, file=sys.stderr)
            sys.exit(r.returncode)
    else:
        print("phase2/phase2_check.py not bundled; skipping Phase 2", flush=True)

    # (7) Phase 3 (A2): the kernel race on the packed + BF16 data.
    phase3 = os.path.join(NATIVE, "phase3", "race_check.py")
    if os.path.exists(phase3):
        print("=== phase3 (A2 kernel race) ===", flush=True)
        env = dict(os.environ, AIOS_TMP=TMP,
                   AIOS_RESULTS=os.path.join(NATIVE, "phase3", "phase3_results.txt"))
        r = subprocess.run([sys.executable, phase3], env=env,
                           cwd=os.path.dirname(HERE))
        if r.returncode != 0:
            print("STEP FAILED: phase3 (exit %d)" % r.returncode, file=sys.stderr)
            sys.exit(r.returncode)
    else:
        print("phase3/race_check.py not bundled; skipping Phase 3", flush=True)

    # (8) Phase 4 (A4): packed oracle vs BF16 oracle, bit-identical logits.
    phase4 = os.path.join(NATIVE, "phase4", "phase4_check.py")
    if os.path.exists(phase4):
        print("=== phase4 (A4 packed oracle) ===", flush=True)
        env = dict(os.environ, AIOS_TMP=TMP,
                   AIOS_RESULTS=os.path.join(NATIVE, "phase4", "phase4_results.txt"))
        r = subprocess.run([sys.executable, phase4], env=env,
                           cwd=os.path.dirname(HERE))
        if r.returncode != 0:
            print("STEP FAILED: phase4 (exit %d)" % r.returncode, file=sys.stderr)
            sys.exit(r.returncode)
    else:
        print("phase4/phase4_check.py not bundled; skipping Phase 4", flush=True)

    # (9) A3: build the old-PC benchmark .exe files with zig, statically
    #     verify them (PE32+/PE32, KERNEL32-only, SSSE3 present, no AVX),
    #     slice real matrices from the already-exported/packed files, and
    #     package aios-oldpc-bench.zip for the user's old PC.
    a3 = os.path.join(NATIVE, "a3", "a3_pack_check.py")
    if os.path.exists(a3):
        print("=== a3 (old-PC benchmark package) ===", flush=True)
        env = dict(os.environ, AIOS_TMP=TMP,
                   AIOS_RESULTS=os.path.join(NATIVE, "a3", "a3_results.txt"))
        r = subprocess.run([sys.executable, a3, "--with-real"], env=env,
                           cwd=os.path.dirname(HERE))
        if r.returncode != 0:
            print("STEP FAILED: a3 (exit %d)" % r.returncode, file=sys.stderr)
            sys.exit(r.returncode)
    else:
        print("a3/a3_pack_check.py not bundled; skipping A3", flush=True)


if __name__ == "__main__":
    main()
