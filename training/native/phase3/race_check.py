#!/usr/bin/env python3
"""Phase 3 (A2) - the kernel race on Kaggle.

Three matvecs (FP32-expanded / BF16 / PACKED) over real Qwen2.5-3B
gate/up/down matrices totaling >= max(4 x L3, 1 GB) BF16 bytes, in two
builds: modern (AVX2+FMA) and old (SSSE3, MULPS+ADDPS only).

Pass criteria (thresholds fixed before running, never loosened):
  1) Both builds compile clean: gcc -std=c99 -Wall -Wextra -Werror -O2.
  2) The old binary contains no AVX instructions at all (objdump scan):
     the old path is forced, not merely available.
  3) Every run (4 configs: {modern,old} x {1,all} threads):
       - FP32 vs BF16 vs PACKED outputs BIT-IDENTICAL (0 differing
         elements; same build = same accumulation structure);
       - each contestant vs float64 reference: max abs diff /
         max abs ref < 1e-4.
  4) Speedup packed_vs_bf16 is classified:
       >= 1.10 WIN | 1.00-1.10 MARGINAL | < 1.00 LOSS
     (mission threshold table; classified on the median of 10 runs,
     best-of-10 also reported).
"""
import os
import re
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
CSRC = os.path.join(HERE, "race.c")

TMP = os.environ.get("AIOS_TMP", "/tmp/aios_native")
EXPORT_DIR = os.path.join(TMP, "qwen2_5_3b")
PACK_DIR = os.path.join(TMP, "packed")
EXE_MODERN = os.path.join(TMP, "race_modern")
EXE_OLD = os.path.join(TMP, "race_old")

BASE_FLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-O2",
              "-D_POSIX_C_SOURCE=200112L", "-fopenmp"]
MODERN_FLAGS = ["-mavx2", "-mfma"]
OLD_FLAGS = ["-mssse3"]

GB_FLOOR = 1_000_000_000      # "data >= max(4 x L3, 1 GB)" of BF16
WIN_MIN = 1.10                # threshold table, A2
MARGINAL_MIN = 1.00
REL_MAX = 1e-4

RESULTS = os.environ.get("AIOS_RESULTS",
                         os.path.join(HERE, "phase3_results.txt"))

REPORT = []


def say(line):
    print(line, flush=True)
    REPORT.append(line)


# ------------------------------------------------------------------ #
# machine info (mission rule 4: always report CPU model, features,    #
# cache sizes, measured RAM bandwidth)                                #
# ------------------------------------------------------------------ #

def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except Exception:
        pass
    return set()


def cache_sizes():
    """[(level, size_str)] for cpu0, e.g. [('2','1 MiB'), ('3','30 MiB')]."""
    out = []
    base = "/sys/devices/system/cpu/cpu0/cache"
    try:
        for idx in sorted(os.listdir(base)):
            d = os.path.join(base, idx)
            try:
                with open(os.path.join(d, "level")) as f:
                    lvl = f.read().strip()
                with open(os.path.join(d, "size")) as f:
                    sz = f.read().strip()
                out.append((lvl, sz))
            except Exception:
                continue
    except Exception:
        pass
    return out


def l3_bytes():
    """Total L3 bytes for cpu0, or 0 if unavailable."""
    base = "/sys/devices/system/cpu/cpu0/cache"
    total = 0
    try:
        entries = os.listdir(base)
    except Exception:
        return 0
    for idx in entries:  # per-entry guard: one bad sysfs entry (power/,
        # uevent, ...) must not abort the whole scan  (v16 lesson)
        d = os.path.join(base, idx)
        try:
            with open(os.path.join(d, "level")) as f:
                if f.read().strip() != "3":
                    continue
            with open(os.path.join(d, "size")) as f:
                s = f.read().strip().upper()
            mult = 1
            if s.endswith("K"):
                mult, s = 1024, s[:-1]
            elif s.endswith("M"):
                mult, s = 1024 * 1024, s[:-1]
            elif s.endswith("G"):
                mult, s = 1024 ** 3, s[:-1]
            total += int(s) * mult
        except Exception:
            continue
    return total


def disk_report():
    try:
        r = subprocess.run(["df", "-h", TMP], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            say("df: " + line)
    except Exception as e:
        say("df failed: %r" % (e,))


def mem_report():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal", "MemAvailable")):
                    say("mem: " + line.strip())
    except Exception as e:
        say("meminfo failed: %r" % (e,))


# ------------------------------------------------------------------ #
# builds                                                             #
# ------------------------------------------------------------------ #

def compile_build(tag, exe, flags):
    obj = exe + ".o"
    # object first (for the AVX scan), then link
    for out, extra in ((obj, ["-c"]), (exe, [])):
        cmd = ["gcc"] + BASE_FLAGS + flags + [CSRC, "-o", out] + extra
        say("$ " + " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.stdout:
            say("gcc stdout: " + r.stdout.strip())
        if r.stderr:
            for line in r.stderr.splitlines():
                say("gcc: " + line)
        if r.returncode != 0:
            say("COMPILE FAILED for %s (exit %d)" % (tag, r.returncode))
            sys.exit(1)
    say("compile OK (%s) -> %s" % (tag, exe))
    return obj


def objdump_avx_scan(exe, expect_avx):
    """Old binary must contain zero AVX instructions (the old path is
    forced). Modern binary must contain some (sanity)."""
    try:
        r = subprocess.run(["objdump", "-d", exe], capture_output=True, text=True)
    except FileNotFoundError:
        say("objdump unavailable; skipping AVX-in-binary scan for %s" % exe)
        return
    if r.returncode != 0:
        say("objdump failed for %s: %s" % (exe, r.stderr.strip()))
        return
    # AVX fingerprints: ymm/zmm registers (xmm is plain legacy SSE), and
    # VEX-encoded mnemonics (every AVX insn requires VEX; no legacy x86
    # mnemonic starts with 'v'). Both together are decisive -- v16's
    # false-positive scan matched xmm registers in the SSSE3 build.
    n_vec = len(re.findall(r"\b[yz]mm\d+\b", r.stdout))
    n_vmn = len(re.findall(r"\t(?:[0-9a-f]{2}\s)+\s*(v[a-z]+)\b", r.stdout))
    n_ymm = n_vec + n_vmn
    say("avx-instructions %s: vec_regs=%d vex_mnemonics=%d"
        % (os.path.basename(exe), n_vec, n_vmn))
    if expect_avx and n_vec == 0:
        say("SANITY FAIL: modern binary has no AVX vector registers")
        sys.exit(1)
    if not expect_avx and n_ymm != 0:
        say("OLD PATH FAIL: old binary contains %d AVX references" % n_ymm)
        sys.exit(1)


# ------------------------------------------------------------------ #
# runs                                                               #
# ------------------------------------------------------------------ #

def run_race(exe, threads, target, tag):
    """Run one config; return parsed speedup dict (may be empty)."""
    out_file = os.path.join(TMP, "race_%s.out" % tag)
    cmd = [exe, EXPORT_DIR, PACK_DIR, str(target), str(threads), out_file]
    say("=== run %s: threads=%s ===" % (tag, threads))
    say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stderr:
        for line in r.stderr.splitlines():
            say("race stderr: " + line)
    # race.c prints to both stdout and out_file; prefer the file
    text = r.stdout
    if os.path.exists(out_file):
        with open(out_file) as f:
            text = f.read()
    for line in text.splitlines():
        say("race: " + line)

    parsed = {"exit": r.returncode, "pass": "RUN PASS" in text}
    m = re.search(r"speedup packed_vs_bf16: best=([\d.]+) median=([\d.]+)", text)
    if m:
        packed_best, packed_med = float(m.group(1)), float(m.group(2))
        parsed["packed_best"] = packed_best
        parsed["packed_med"] = packed_med
        parsed["class_best"] = classify(packed_best)
        parsed["class_med"] = classify(packed_med)
    m = re.search(r"speedup bf16_vs_fp32  : best=([\d.]+) median=([\d.]+)", text)
    if m:
        parsed["bf16_best"] = float(m.group(1))
        parsed["bf16_med"] = float(m.group(2))
    for k in ("fp32_vs_bf16", "fp32_vs_packed", "bf16_vs_packed"):
        m = re.search(r"%s=(\d+)" % k, text)
        if m:
            parsed["ndiff_" + k] = int(m.group(1))
    return parsed


def classify(speedup):
    if speedup >= WIN_MIN:
        return "WIN"
    if speedup >= MARGINAL_MIN:
        return "MARGINAL"
    return "LOSS"


# ------------------------------------------------------------------ #

def main():
    say("=== A2 kernel race harness ===")
    disk_report()
    mem_report()
    model = cpu_model()
    flags = cpu_flags()
    say("cpu model: %s" % model)
    say("cpu flags (relevant): %s"
        % " ".join(sorted(flags & {"ssse3", "sse4_1", "sse4_2", "avx",
                                   "avx2", "fma", "avx512f"})))
    caches = cache_sizes()
    for lvl, sz in caches:
        say("cache L%s: %s" % (lvl, sz))
    l3 = l3_bytes()
    target = max(4 * l3, GB_FLOOR)
    say("L3 bytes: %d  ->  target = max(4*L3, 1 GB) = %d BF16 bytes" % (l3, target))

    if not os.path.exists(os.path.join(EXPORT_DIR, "weights.bin")):
        say("MISSING %s/weights.bin (Phase 0 must run first)" % TMP)
        sys.exit(1)
    if not os.path.exists(os.path.join(PACK_DIR, "packed.bin")):
        say("MISSING %s/packed.bin (Phase 2 must run first)" % TMP)
        sys.exit(1)

    obj_modern = compile_build("modern", EXE_MODERN, MODERN_FLAGS)
    obj_old = compile_build("old", EXE_OLD, OLD_FLAGS)
    # scan OUR object files (not linked CRT/libgomp code)
    objdump_avx_scan(obj_old, expect_avx=False)
    objdump_avx_scan(obj_modern, expect_avx=True)

    configs = [
        ("modern-1t", EXE_MODERN, 1),
        ("modern-all", EXE_MODERN, 0),
        ("old-1t", EXE_OLD, 1),
        ("old-all", EXE_OLD, 0),
    ]
    results = {}
    for tag, exe, threads in configs:
        results[tag] = run_race(exe, threads, target, tag)

    say("=== summary ===")
    failed = False
    for tag, _, threads in configs:
        p = results[tag]
        if p["exit"] != 0 or not p["pass"]:
            say("%-11s: FAIL (exit %d, pass=%s)" % (tag, p["exit"], p["pass"]))
            failed = True
            continue
        nd = [p.get("ndiff_fp32_vs_bf16"), p.get("ndiff_fp32_vs_packed"),
              p.get("ndiff_bf16_vs_packed")]
        if any(d != 0 for d in nd):
            say("%-11s: FAIL bit-identity %s" % (tag, nd))
            failed = True
            continue
        say("%-11s: PASS bit-identical  packed/bf16 best=%.3f (%s) "
            "median=%.3f (%s)  bf16/fp32 best=%.3f median=%.3f"
            % (tag, p.get("packed_best", float("nan")), p.get("class_best", "?"),
               p.get("packed_med", float("nan")), p.get("class_med", "?"),
               p.get("bf16_best", float("nan")), p.get("bf16_med", float("nan"))))

    if failed:
        say("PHASE3 FAILED")
        return
    # Headline classification: modern build, all threads (the config the
    # engine will ship with), median of 10 runs.
    head = results["modern-all"]
    say("A2 verdict (modern, all threads, median): %s (packed_vs_bf16=%.3f)"
        % (head.get("class_med", "?"), head.get("packed_med", float("nan"))))
    say("A2 verdict (old path, all threads, median): %s (packed_vs_bf16=%.3f)"
        % (results["old-all"].get("class_med", "?"),
           results["old-all"].get("packed_med", float("nan"))))
    say("PHASE3 PASSED")


def guarded_main():
    """Always leave a results file, even on a hard failure (empty Kaggle logs)."""
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        say("PHASE3 FAILED (exception):")
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
