#!/usr/bin/env python3
"""a3_pack_check - A3 Kaggle-side harness: build the old-PC benchmark .exe
files with zig, verify them statically, produce a3bench.bin, package the zip.

Run standalone (build kernel, synthetic data) or as real_check.py step (9)
with --with-real (slices real Qwen2.5-3B matrices from the exported/packed
files in $AIOS_TMP - no second model download).

Checks (thresholds fixed before running, never loosened):
  1) zig cc builds bench64.exe (PE32+) and bench32.exe (PE32) with
     -Wall -Wextra -Werror; both link against KERNEL32.dll only.
  2) file(1): PE32+ / PE32 console executables.
  3) objdump -d: pshufb present (SSSE3 decode path), ZERO ymm/zmm registers,
     ZERO AVX (v-prefixed) mnemonics, ZERO SSE4.1+ denylist mnemonics -
     i.e. the old-PC path is forced, same standard as A2's race_old.o scan.
  4) objdump -p: imported DLLs == exactly {KERNEL32.dll}.
  5) a3bench.bin parses to matrices whose packed streams decode back to the
     original BF16 bits with 0 mismatched bits (numpy answer key), before it
     is shipped in the zip.

Exit codes: 0 all checks passed, 1 a check failed, 2 environment problem.
Results always land in a3/a3_results.txt (Kaggle empty-log quirk, D-003).
"""
import os
import re
import subprocess
import sys
import traceback
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "a3_results.txt")

REPORT = []


def say(s):
    print(s, flush=True)
    REPORT.append(s)


def run(cmd, **kw):
    say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, **kw)
    return r.returncode, r.stdout or ""


SSE41_DENY = [
    "pmovzxbw", "pmovzxbd", "pmovzxbq", "pmovzxwd", "pmovzxwq",
    "pmovsxbw", "pmovsxbd", "pmovsxbq", "pmovsxwd", "pmovsxwq",
    "pblendvb", "blendvps", "blendvpd", "pblendw",
    "roundss", "roundps", "roundpd", "roundsd",
    "insertps", "extractps", "ptest", "phminposuw", "crc32",
    "pcmpeqq", "pcmpgtq", "pminsb", "pmaxsb", "pminuw", "pmaxuw",
    "pminud", "pmaxud", "pminsd", "pmaxsd", "pmulld", "pmuldq",
    "pcmpistrm", "pcmpestri",
]


def scan_exe(path):
    """Static instruction/import scan of a PE built by zig cc."""
    bad = []
    rc, out = run(["file", path])
    say(out.rstrip())
    if rc != 0:
        bad.append("file(1) failed on %s" % path)
    want = "PE32+" if "64" in os.path.basename(path) else "PE32 "
    if want.strip() not in out:
        bad.append("%s: file(1) does not report %s" % (path, want.strip()))
    rc, dis = run(["objdump", "-d", path])
    if rc != 0:
        bad.append("objdump -d failed on %s" % path)
        return bad
    lines = dis.splitlines()
    vec_lines = [l for l in lines if "ymm" in l or "zmm" in l]
    avx_lines = [l for l in lines
                 if re.search(r"^\s*[0-9a-f]+:(?:\s+[0-9a-f]{2})+\s+(v[a-z0-9]+)\s", l)]
    pshufb = sum(1 for l in lines if "pshufb" in l)
    deny = {}
    for l in lines:
        m = re.search(r"^\s*[0-9a-f]+:(?:\s+[0-9a-f]{2})+\s+([a-z0-9]+)\s", l)
        if m and m.group(1) in SSE41_DENY:
            deny[m.group(1)] = deny.get(m.group(1), 0) + 1
    say("scan %s: pshufb=%d ymm_zmm_lines=%d avx_mnemonics=%d sse41_deny=%s"
        % (os.path.basename(path), pshufb, len(vec_lines), len(avx_lines),
           deny if deny else 0))
    if pshufb < 1:
        bad.append("%s: no pshufb (SSSE3 decode path missing)" % path)
    if vec_lines:
        bad.append("%s: %d ymm/zmm references" % (path, len(vec_lines)))
    if avx_lines:
        bad.append("%s: %d AVX mnemonics e.g. %s" % (path, len(avx_lines), avx_lines[:3]))
    if deny:
        bad.append("%s: SSE4.1+ denylist mnemonics present: %s" % (path, deny))
    rc, imp = run(["objdump", "-p", path])
    dlls = re.findall(r"^\s*DLL Name:\s*(\S+)", imp, re.M)
    say("imports %s: %s" % (os.path.basename(path), dlls))
    if [d.upper() for d in dlls] != ["KERNEL32.DLL"]:
        bad.append("%s: imports %s (must be exactly KERNEL32.dll)" % (path, dlls))
    return bad


ZIG = [sys.executable, "-m", "ziglang", "cc"]


def build(bench_src):
    cfgs = [
        ("x86_64-windows-gnu", "bench64"),
        ("x86-windows-gnu", "bench32"),
    ]
    for target, base in cfgs:
        for args, outp in (
            (["-target", target, "-mcpu=core2", "-O2", "-fno-stack-protector",
              "-fno-builtin", "-Wall", "-Wextra", "-Werror", "-c", bench_src,
              "-o", base + ".o"], base + ".o"),
            (["-target", target, "-nostdlib", "-Wl,--entry=start",
              "-Wl,--subsystem,console", "-o", base + ".exe", base + ".o",
              "-lkernel32"], base + ".exe"),
        ):
            rc, out = run(ZIG + args, cwd=HERE)
            say(out)
            if rc != 0:
                raise AssertionError("zig cc failed for %s (exit %d)" % (outp, rc))
            if not os.path.exists(os.path.join(HERE, outp)):
                raise AssertionError("zig cc produced no %s" % outp)
            say("built %s (%d bytes)" % (outp, os.path.getsize(os.path.join(HERE, outp))))


RUN_BAT = (
    "@echo off\r\n"
    "cd /d \"%~dp0\"\r\n"
    "echo === A3 old-PC benchmark (AIOS engine mission) ===\r\n"
    "echo Close other programs first; the test needs about 500 MB of free RAM\r\n"
    "echo and a few minutes per benchmark.\r\n"
    "echo.\r\n"
    "echo [1/2] 64-bit benchmark...\r\n"
    "bench64.exe\r\n"
    "echo.\r\n"
    "echo [2/2] 32-bit benchmark...\r\n"
    "bench32.exe\r\n"
    "echo.\r\n"
    "echo All done. Please send back these two files from this folder:\r\n"
    "echo   results_64bit.txt   and   results_32bit.txt\r\n"
    "pause\r\n"
)

README = """A3 old-PC benchmark - AIOS engine mission
=========================================

What it does
------------
Measures three matrix-vector kernels (FP32 vs BF16 vs 4-bit-packed) over two
real Qwen2.5-3B weight matrices, on THIS computer, so we get real numbers
from a DDR2-era CPU. It only reads its own data file; it needs no network,
no install, and writes nothing outside this folder.

Requirements
------------
- Windows (32-bit or 64-bit).
- CPU with SSSE3: Intel Core 2 (2006+) or AMD with SSSE3 (Barcelona 2007+).
  The benchmark prints flag_ssse3=0 and will not work on older CPUs.
- About 500 MB free RAM. Close other programs first.
- A few minutes: it runs two benchmarks (64-bit and 32-bit).

Steps
-----
1. Unzip this folder anywhere (keep all files together).
2. Double-click run.bat.
3. Wait for "All done".
4. Send back results_64bit.txt and results_32bit.txt (or paste their
   contents into chat).

Notes
-----
- The .exe files have no C runtime (they are tiny and self-contained);
  some antivirus scanners flag such files falsely. The checksums below
  let you verify what you received.
- a3bench.bin holds the real weight matrices (~158 MB); do not rename it.
"""


def package(checksums):
    zpath = os.path.join(HERE, "aios-oldpc-bench.zip")
    readme = README + "\nChecksums (sha256):\n" + "".join(
        "  %s  %s\n" % (h, n) for n, h in checksums.items())
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("bench64.exe", "bench32.exe", "a3bench.bin"):
            z.write(os.path.join(HERE, name), name)
        z.writestr("run.bat", RUN_BAT)
        z.writestr("README.txt", readme)
    say("zip: %s (%.1f MB)" % (zpath, os.path.getsize(zpath) / 1e6))


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with_real = "--with-real" in sys.argv
    say("=== A3 pack check (%s data) ===" % ("real" if with_real else "synthetic"))
    rc, out = run([sys.executable, "-m", "ziglang", "version"])
    say(out)
    if rc != 0:
        say("installing ziglang...")
        rc, out = run([sys.executable, "-m", "pip", "install", "--quiet", "ziglang"])
        say(out)
        rc, out = run([sys.executable, "-m", "ziglang", "version"])
        say(out)
        if rc != 0:
            raise AssertionError("ziglang unavailable (pip install failed)")

    build(os.path.join(HERE, "bench.c"))

    fails = []
    for exe in ("bench64.exe", "bench32.exe"):
        fails += scan_exe(os.path.join(HERE, exe))

    # a3bench.bin: real slices (step 9) or synthetic (fast build kernel)
    cmd = [sys.executable, os.path.join(HERE, "make_a3bench_data.py")]
    cmd += ["--mode", "real"] if with_real else ["--mode", "synthetic"]
    rc, out = run(cmd)
    say(out)
    if rc != 0:
        raise AssertionError("make_a3bench_data.py failed (exit %d)" % rc)
    if os.path.getsize(os.path.join(HERE, "a3bench.bin")) < 64:
        raise AssertionError("a3bench.bin too small")

    sums = {n: sha256(os.path.join(HERE, n))
            for n in ("bench64.exe", "bench32.exe", "a3bench.bin")}
    for n, h in sums.items():
        say("sha256 %s = %s" % (n, h))
    package(sums)

    if fails:
        for f in fails:
            say("FAIL: " + f)
        raise AssertionError("%d static checks failed" % len(fails))
    say("A3 PACK CHECK PASSED")


def guarded_main():
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        say("A3 PACK CHECK FAILED (exception):")
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
