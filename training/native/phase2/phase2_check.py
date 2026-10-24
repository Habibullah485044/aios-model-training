#!/usr/bin/env python3
"""Phase 2 - packer + scalar reference unpacker, correctness only.

Pass criteria (thresholds set before running, never loosened):
  1) gcc -std=c99 -Wall -Wextra -Werror -O2 compile of unpack_ref.c is clean.
  2) Every tensor packs and unpacks bit-exactly: 0 mismatched bits across all
     3,085,938,688 weights (verified against the original BF16 weights.bin).
  3) packed.bin size matches the format's arithmetic prediction
     (12 bits/weight + 5 bytes/exception + 19 bytes/tensor), and is reported
     against the expected ~4.638 GB.

The C unpacker is the answer key for Phase 3: the fast SIMD kernel must
reproduce its output bit-for-bit.
"""
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
CSRC = os.path.join(HERE, "unpack_ref.c")
PACK_PY = os.path.join(HERE, "pack.py")

TMP = os.environ.get("AIOS_TMP", "/tmp/aios_native")
EXPORT_DIR = os.path.join(TMP, "qwen2_5_3b")
PACK_DIR = os.path.join(TMP, "packed")
EXE = os.path.join(TMP, "unpack_ref")

EXPECTED_PARAMS = 3_085_938_688
# ~4.638 GB from the brief, derived as 12 bits/weight + 5 bytes/exception
# x 1,897,535 exceptions (Phase 1's GLOBAL-table count) + 19 bytes/tensor.
EXPECTED_PACKED_BYTES = 4_638_000_000
PHASE1_GLOBAL_EXC = 1_897_535
SIZE_TOL = 0.005  # allow 0.5% slack on the size estimate
WRITE_UNPACKED = os.environ.get("AIOS_WRITE_UNPACKED") == "1"
RESULTS = os.environ.get("AIOS_RESULTS",
                         os.path.join(HERE, "phase2_results.txt"))

REPORT = []


def say(line):
    print(line, flush=True)
    REPORT.append(line)


def compile_unpacker():
    os.makedirs(TMP, exist_ok=True)
    cmd = ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O2", CSRC, "-o", EXE]
    say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        say("gcc stdout: " + r.stdout.strip())
    if r.stderr:
        for line in r.stderr.splitlines():
            say("gcc: " + line)
    if r.returncode != 0:
        say("COMPILE FAILED (exit %d)" % r.returncode)
        sys.exit(1)
    say("compile OK -> " + EXE)


def run_packer():
    say("=== pack ===")
    env = dict(os.environ, AIOS_TMP=TMP, AIOS_PACK_DIR=PACK_DIR)
    subprocess.run([sys.executable, PACK_PY], env=env, check=True)


def disk_report():
    try:
        out = subprocess.run(["df", "-h", TMP], capture_output=True, text=True)
        for line in out.stdout.splitlines():
            say("df: " + line)
    except Exception as e:
        say("df failed: %r" % (e,))


def parse_stats():
    stats = {}
    with open(os.path.join(PACK_DIR, "pack_stats.txt")) as f:
        for line in f:
            k, v = line.split()
            stats[k] = v
    return stats


def run_verify():
    say("=== unpack + verify (bit-exact) ===")
    cmd = [EXE, PACK_DIR, os.path.join(EXPORT_DIR, "weights.bin")]
    if WRITE_UNPACKED:
        cmd.append(os.path.join(PACK_DIR, "unpacked.bin"))
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(r.stderr)
    if r.stdout:
        sys.stdout.write(r.stdout)
    mismatches = None
    for line in r.stdout.splitlines():
        if line.startswith("mismatched_bits"):
            mismatches = int(line.split()[1])
    if r.returncode != 0 or mismatches != 0:
        say("UNPACK VERIFY FAILED (exit %d, mismatched_bits=%s)" % (r.returncode, mismatches))
        sys.exit(1)
    say("unpack verify: mismatched_bits=0")
    return mismatches


def main():
    say("=== disk before pack ===")
    disk_report()
    compile_unpacker()
    run_packer()
    stats = parse_stats()
    n = int(stats["weights"])
    n_exc = int(stats["exceptions"])
    actual = int(stats["bytes"])

    # arithmetic prediction: stream A (n) + stream B (n/2) + 15-byte table +
    # 4-byte exc count per tensor, per-tensor overhead estimated from the
    # observed exception count and tensor count in the index.
    with open(os.path.join(PACK_DIR, "packed_index.txt")) as f:
        n_tensors = sum(1 for line in f if line.strip())
    predicted = n + n // 2 + 15 * n_tensors + 4 * n_tensors + 5 * n_exc
    say("tensors=%d weights=%d exceptions=%d" % (n_tensors, n, n_exc))
    say("per-tensor tables give %d exceptions vs Phase 1 global-table %d (%+d)"
        % (n_exc, PHASE1_GLOBAL_EXC, n_exc - PHASE1_GLOBAL_EXC))
    say("predicted bytes (arithmetic): %d (%.4f GB)" % (predicted, predicted / 1e9))
    say("actual     bytes            : %d (%.4f GB)" % (actual, actual / 1e9))
    say("vs expected ~4.638 GB       : %+.4f GB" % ((actual - EXPECTED_PACKED_BYTES) / 1e9))
    assert actual == predicted, "packed size != arithmetic prediction"

    if n != EXPECTED_PARAMS:
        say("WEIGHT COUNT MISMATCH: %d != %d" % (n, EXPECTED_PARAMS))
        sys.exit(1)
    if abs(actual - EXPECTED_PACKED_BYTES) / EXPECTED_PACKED_BYTES > SIZE_TOL:
        say("packed size off expected by > %.1f%%" % (100 * SIZE_TOL))
        sys.exit(1)

    run_verify()

    if WRITE_UNPACKED:
        unpacked = os.path.join(PACK_DIR, "unpacked.bin")
        ub = os.path.getsize(unpacked)
        ob = os.path.getsize(os.path.join(EXPORT_DIR, "weights.bin"))
        say("unpacked bytes=%d original bytes=%d" % (ub, ob))
        assert ub == ob, "unpacked size != original"
    say("PHASE2 PASSED")


def guarded_main():
    """Always leave a results file, even on a hard failure (empty Kaggle logs)."""
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        say("PHASE2 FAILED (exception):")
        for line in traceback.format_exc().splitlines():
            say("  " + line)
        disk_report()
        code = 1
    try:
        with open(RESULTS, "w") as f:
            f.write("\n".join(REPORT) + "\n")
    except Exception:
        pass
    sys.exit(code)


if __name__ == "__main__":
    guarded_main()
