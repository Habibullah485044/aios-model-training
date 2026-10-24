#!/usr/bin/env python3
"""Phase 4 (A4) - packed oracle vs BF16 oracle.

Pass criteria (threshold fixed before running):
  1) unpack_ref reproduces weights.bin from packed.bin bit-exactly
     (already proven in Phase 2; re-asserted here via file sizes and
     the unpacker's own mismatch count).
  2) The C oracle (double accumulation) run on the UNPACKED weights
     produces BIT-IDENTICAL logits to the Phase 0 run on weights.bin:
     0 differing bits out of 8 x 151936 floats. Any difference fails.

Also probes for Windows cross-compilation tools (A3 packaging plan).
"""
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)

TMP = os.environ.get("AIOS_TMP", "/tmp/aios_native")
EXPORT_DIR = os.path.join(TMP, "qwen2_5_3b")
PACK_DIR = os.path.join(TMP, "packed")
UNPACK_REF = os.path.join(TMP, "unpack_ref")
ORACLE = os.path.join(TMP, "aios_infer")
UNPACKED = os.path.join(PACK_DIR, "unpacked.bin")
ALT_DIR = os.path.join(TMP, "qwen2_5_3b_unpacked")
RESULTS = os.environ.get("AIOS_RESULTS",
                         os.path.join(HERE, "phase4_results.txt"))

REPORT = []


def say(line):
    print(line, flush=True)
    REPORT.append(line)


def probe_cross_tools():
    """A3: can this machine build a Windows .exe of our C code?"""
    for tool in ("x86_64-w64-mingw32-gcc", "i686-w64-mingw32-gcc", "wine",
                 "wine64"):
        r = subprocess.run(["bash", "-lc", "command -v %s" % tool],
                           capture_output=True, text=True)
        say("cross-probe %s: %s" % (tool, r.stdout.strip() or "not found"))


def run_unpack():
    say("=== unpack packed.bin -> unpacked.bin (reference unpacker) ===")
    r = subprocess.run([UNPACK_REF, PACK_DIR,
                        os.path.join(EXPORT_DIR, "weights.bin"), UNPACKED],
                       capture_output=True, text=True)
    if r.stderr:
        for line in r.stderr.splitlines():
            say("unpack_ref: " + line)
    mism = None
    for line in r.stdout.splitlines():
        if line.startswith("mismatched_bits"):
            mism = int(line.split()[1])
    if r.returncode != 0 or mism != 0:
        say("UNPACK FAILED (exit %d, mismatched_bits=%s)" % (r.returncode, mism))
        sys.exit(1)
    ub = os.path.getsize(UNPACKED)
    ob = os.path.getsize(os.path.join(EXPORT_DIR, "weights.bin"))
    say("unpacked bytes=%d original bytes=%d equal=%s" % (ub, ob, ub == ob))
    if ub != ob:
        sys.exit(1)


def run_oracles():
    """Oracle on unpacked weights; compare bit-for-bit with Phase 0 output."""
    say("=== oracle on unpacked weights ===")
    os.makedirs(ALT_DIR, exist_ok=True)
    # the oracle reads manifest.txt + weights.bin (+ tokens) from one dir
    for rel in ("manifest.txt", "config.txt", "ref_tokens.txt"):
        src = os.path.join(EXPORT_DIR, rel)
        dst = os.path.join(ALT_DIR, rel)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        if os.path.exists(src):
            os.symlink(src, dst)
    dst_w = os.path.join(ALT_DIR, "weights.bin")
    if os.path.exists(dst_w) or os.path.islink(dst_w):
        os.remove(dst_w)
    os.symlink(UNPACKED, dst_w)

    out_bf16 = os.path.join(EXPORT_DIR, "oracle_logits.bin")
    out_pack = os.path.join(ALT_DIR, "oracle_logits.bin")
    if not os.path.exists(out_bf16):
        say("MISSING Phase 0 oracle logits: %s" % out_bf16)
        sys.exit(1)
    r = subprocess.run([ORACLE, ALT_DIR,
                        os.path.join(ALT_DIR, "ref_tokens.txt"), out_pack],
                       capture_output=True, text=True)
    if r.stderr:
        for line in r.stderr.splitlines():
            say("oracle: " + line)
    if r.returncode != 0:
        say("ORACLE FAILED (exit %d)" % r.returncode)
        sys.exit(1)

    say("=== compare logits (bitwise) ===")
    import numpy as np
    a = np.fromfile(out_bf16, dtype=np.uint32)
    b = np.fromfile(out_pack, dtype=np.uint32)
    if a.shape != b.shape:
        say("SHAPE MISMATCH %s vs %s" % (a.shape, b.shape))
        sys.exit(1)
    nd = int((a != b).sum())
    say("logits compared: %d  differing bits: %d (threshold: 0)" % (a.size, nd))
    if nd != 0:
        f_a = a.view(np.float32)
        f_b = b.view(np.float32)
        say("max abs diff: %.6e" % float(np.abs(f_a - f_b).max()))
        sys.exit(1)
    say("A4 VERDICT: packed oracle == BF16 oracle, bit-identical")


def main():
    say("=== A4 packed-oracle check ===")
    probe_cross_tools()
    for p in (UNPACK_REF, ORACLE, os.path.join(EXPORT_DIR, "weights.bin"),
              os.path.join(PACK_DIR, "packed.bin")):
        if not os.path.exists(p):
            say("MISSING %s" % p)
            sys.exit(1)
    run_unpack()
    run_oracles()
    say("PHASE4 PASSED")


def guarded_main():
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        say("PHASE4 FAILED (exception):")
        for line in traceback.format_exc().splitlines():
            say("  " + line)
        code = 1
    try:
        with open(RESULTS, "w") as f:
            f.write("\n".join(REPORT) + "\n")
    except Exception:
        pass
    sys.exit(code)


if __name__ == "__main__":
    guarded_main()
