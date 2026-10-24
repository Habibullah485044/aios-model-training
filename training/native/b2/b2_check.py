#!/usr/bin/env python3
"""b2_check.py - B2 pass check: our x86-64 assembler vs GNU as, byte for byte.

Runs on Kaggle (Linux: GNU as/ld/objcopy/readelf/objdump, executing ELF
files), never on the build machine. Python is used only to build and test.

Method per corpus file (b2/corpus/*.s, each a complete `_start` program):
  - `as --64` assembles it and `ld -e _start` links it, so every relocation
    is resolved; the linked .text is then extracted with objcopy (a
    relocatable .o would leave zeros behind its relocations and could skew
    the comparison).
  - our asm.py assembles the SAME source with the load address that ld gave
    .text, so absolute fixups (.long/.quad label, the absolute-memory form
    of a bare-label operand) agree.
  - the two byte strings must be identical. Where GNU as pads its section
    out to an alignment the nops ride along in the linked .text, so the
    comparison is: our bytes equal the linked .text truncated to our
    length, and everything after that is nop padding (verified by
    disassembling it) at most 15 bytes long. ELF objects need no such
    tolerance - .text there has alignment 1 and ends exactly at our last
    byte; the padding only appears with the PE/COFF backend (align 16).

Pass criteria (thresholds fixed before running, never loosened):
  1) GNU as assembles every corpus file with no error on stderr, ld links it
     (ld's own warnings are reported, not treated as errors); our assembler
     accepts the same source. The corpus must match EXPECT_EXIT exactly, so
     no file can be silently skipped.
  2) The linked .text of the GNU side and our code are byte-identical over
     the whole corpus: 0 differing bytes, equal lengths, and both sides
     place _start at the same offset.
  3) Both linked programs run and exit with the file's marker status (the
     bytes are not only equal, they work). The AVX2/FMA file (06_avx2.s)
     is byte-compared on any CPU, but it only EXECUTES where
     /proc/cpuinfo reports avx2+fma; elsewhere its run is skipped with
     "SKIPPED: no AVX2" and B2 only counts as PASSED once it has
     executed on AVX2 hardware (Kaggle's EPYC or a later run).
  4) Evidence per file: GAS version, byte count, statement count, entry
     point, exit codes, plus the source line and disassembly around the
     first mismatch when one occurs.

Exit codes: 0 pass, 1 check failed, 2 environment problem.
Results always land in b2/b2_results.txt (Kaggle empty-log quirk, D-003).
"""
import glob
import os
import re
import shutil
import stat
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
CORPUS = os.path.join(HERE, "corpus")
B1 = os.path.join(NATIVE, "b1")
TMP = os.path.join(os.environ.get("AIOS_TMP", "/tmp/aios_native"), "b2")

RESULTS = os.environ.get("AIOS_RESULTS",
                         os.path.join(HERE, "b2_results.txt"))

# Exit status every corpus file's _start must produce (its epilogue loads the
# marker into edi). The check requires this set exactly.
EXPECT_EXIT = {
    "01_gp_mov.s": 1,
    "02_gp_arith.s": 2,
    "03_gp_flow.s": 3,
    "04_sse_move.s": 4,
    "05_sse_arith.s": 5,
    "06_avx2.s": 6,
}

REPORT = []

AVX2_FILE = "06_avx2.s"


def cpu_has_avx2():
    """True when this CPU can EXECUTE the AVX2/FMA corpus file (ymm and
    FMA3 instructions need avx2 and fma). Reads /proc/cpuinfo; anywhere
    else (or without the flags) the answer is False."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    have = set(line.split(":", 1)[1].split())
                    return "avx2" in have and "fma" in have
    except OSError:
        pass
    return False


HAVE_AVX2 = cpu_has_avx2()


def say(line):
    print(line, flush=True)
    REPORT.append(line)


def run(cmd, **kw):
    say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, **kw)
    for stream in (r.stdout or "", r.stderr or ""):
        for line in stream.splitlines():
            if line.strip():
                say("  " + line)
    return r.returncode, r.stdout or "", r.stderr or ""


def tool(name):
    path = shutil.which(name)
    if not path:
        return None
    return name


def text_section(elf):
    """(address, size) of .text in a linked ELF, from readelf -SW."""
    rc, out, err = run(["readelf", "-SW", elf])
    if rc != 0:
        raise AssertionError("readelf -SW failed (exit %d)" % rc)
    for line in out.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+\.text\s+\S+\s+([0-9a-f]+)\s+"
                     r"([0-9a-f]+)\s+([0-9a-f]+)", line)
        if m:
            return int(m.group(1), 16), int(m.group(3), 16)
    raise AssertionError("no .text section in %s" % elf)


def entry_point(elf):
    rc, out, err = run(["readelf", "-hW", elf])
    if rc != 0:
        raise AssertionError("readelf -hW failed (exit %d)" % rc)
    m = re.search(r"Entry point address:\s*(0x[0-9a-f]+)", out)
    if not m:
        raise AssertionError("no entry point in readelf -hW output")
    return int(m.group(1), 16)


def objcopy_text(elf, dest):
    rc, out, err = run(["objcopy", "-O", "binary", "--only-section=.text",
                        elf, dest])
    if rc != 0:
        raise AssertionError("objcopy of .text failed (exit %d)" % rc)
    with open(dest, "rb") as f:
        return f.read()


def disassemble(path):
    """[(hex bytes, intel text)] for a raw x86-64 byte stream."""
    rc, out, err = run(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
                        "-M", "intel", path])
    if rc != 0:
        raise AssertionError("objdump failed on %s (exit %d)" % (path, rc))
    insns = []
    for line in out.splitlines():
        m = re.match(r"^\s*[0-9a-f]+:\s+((?:[0-9a-f]{2}\s)+)\s*(.*?)\s*$",
                     line)
        if m:
            insns.append((re.sub(r"\s+", "", m.group(1)),
                          re.sub(r"\s+", " ", m.group(2))))
    return insns


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def window(data, off, span=16):
    lo = max(0, off - span)
    return "%06x: %s" % (lo, data[lo:off + span].hex(" "))


PAD_MAX = 15  # .text is 16-byte aligned, so at most 15 bytes of padding


def padding_is_nops(pad, path):
    """True when `pad` disassembles to nothing but nop/xchg ax,ax."""
    with open(path, "wb") as f:
        f.write(pad)
    for raw, text in disassemble(path):
        if not text.startswith("nop") and not text.startswith("xchg"):
            return False
    return True


def report_mismatch(name, ours, gnu, smap, ours_path, gnu_path):
    off = first_diff(ours, gnu)
    say("MISMATCH %s: ours=%d bytes, gnu=%d bytes, first difference at %s"
        % (name, len(ours), len(gnu),
           "end of the shorter stream" if off is not None and off >= min(
               len(ours), len(gnu)) else str(off)))
    if off is not None:
        for lineno, text, start, size in smap:
            if start <= off < start + size:
                say("  source line %d: %s" % (lineno, text))
                break
        say("  ours %s" % window(ours, off))
        say("  gnu  %s" % window(gnu, off))
    try:
        di = disassemble(ours_path)
        dg = disassemble(gnu_path)
        shown = 0
        for i in range(min(len(di), len(dg))):
            if di[i] != dg[i]:
                say("  insn %d: ours %s %s | gnu %s %s"
                    % (i, di[i][0], di[i][1], dg[i][0], dg[i][1]))
                shown += 1
                if shown == 5:
                    break
        if shown == 0:
            say("  instruction streams agree; difference is in data/length "
                "(ours %d insns, gnu %d)" % (len(di), len(dg)))
    except AssertionError as e:
        say("  (disassembly unavailable: %s)" % e)


def run_elf(path, expect):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    r = subprocess.run([path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return r.returncode


def check_file(name, asm_mod, elf_mod, tools):
    """Returns (ok, bytes_compared, note)."""
    src_path = os.path.join(CORPUS, name)
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()

    stem = name[:-2]
    as_path = os.path.join(TMP, name)
    with open(as_path, "w", encoding="utf-8") as f:
        f.write(text)

    obj = os.path.join(TMP, stem + ".o")
    gnu_elf = os.path.join(TMP, stem + ".gnu.elf")
    gnu_txt = os.path.join(TMP, stem + ".gnu.text")
    ours_txt = os.path.join(TMP, stem + ".ours.bin")
    ours_elf = os.path.join(TMP, stem + ".ours.elf")

    expect = EXPECT_EXIT[name]

    # ---- GNU side ----
    rc, out, err = run(["as", "--64", "-o", obj, as_path])
    if rc != 0:
        return False, 0, "GNU as failed (exit %d)" % rc
    bad = [l for l in err.splitlines() if "error" in l.lower()]
    if bad:
        return False, 0, "GNU as reported errors: %s" % bad[:2]
    ld_cmd = ["ld", "-z", "noexecstack", "-e", "_start", "-o", gnu_elf, obj]
    rc, out, err = run(ld_cmd)
    if rc != 0:
        say("ld with -z noexecstack failed; retrying without the flag")
        rc, out, err = run(["ld", "-e", "_start", "-o", gnu_elf, obj])
        if rc != 0:
            return False, 0, "ld failed (exit %d)" % rc

    addr, size = text_section(gnu_elf)
    gnu = objcopy_text(gnu_elf, gnu_txt)
    if size != len(gnu):
        return False, 0, "readelf says .text size %d, objcopy gave %d" % (
            size, len(gnu))

    # ---- our side ----
    ours, entry, smap = asm_mod.assemble_map(text, base=addr)
    with open(ours_txt, "wb") as f:
        f.write(ours)
    if entry != 0:
        return False, 0, "_start is not at offset 0 (entry 0x%x)" % entry
    if entry_point(gnu_elf) != addr:
        return False, 0, "gnu entry 0x%x != .text address 0x%x" % (
            entry_point(gnu_elf), addr)

    # ---- byte comparison ----
    if len(gnu) < len(ours) or gnu[:len(ours)] != ours:
        report_mismatch(name, ours, gnu, smap, ours_txt, gnu_txt)
        return False, min(len(ours), len(gnu)), "bytes differ"
    pad = gnu[len(ours):]
    # On ELF objects GNU as emits no trailing padding at all (.text ends
    # at our last byte; the nop padding only appears in the PE/COFF
    # backend, where .text is aligned to 16), and objdump cannot
    # disassemble an empty file - so only a non-empty pad is checked.
    if len(pad) > PAD_MAX or (pad and not padding_is_nops(pad, os.path.join(
            TMP, stem + ".pad.bin"))):
        say("%s: %d trailing bytes after our code are not section-alignment "
            "nop padding: %s" % (name, len(pad), pad.hex(" ")))
        return False, len(ours), "unexpected trailing bytes"

    # ---- both must also run: our ELF is built at the address the B1 writer
    # loads code at, so absolute data references stay truthful ----
    base = elf_mod.BASE + elf_mod.CODE_OFF
    code_run, entry_run, _ = asm_mod.assemble_map(text, base=base)
    if entry_run != 0:
        return False, len(ours), "runtime build: _start not at offset 0"
    with open(ours_elf, "wb") as f:
        f.write(elf_mod.emit_elf64(code_run))
    if name == AVX2_FILE and not HAVE_AVX2:
        # bytes are already proven identical; running it would SIGILL
        say("%s: %d bytes == %d (gnu: %d code + %d nop padding) - "
            "execution SKIPPED: no AVX2 on this CPU"
            % (name, len(ours), len(gnu), len(ours), len(pad)))
        return True, len(ours), "identical; SKIPPED: no AVX2"
    ours_exit = run_elf(ours_elf, expect)
    gnu_exit = run_elf(gnu_elf, expect)
    say("%s: %d bytes == %d (gnu: %d code + %d nop padding), %d statements, "
        "_start at 0x%x, exit ours=%d gnu=%d expected=%d"
        % (name, len(ours), len(gnu), len(ours), len(pad), len(smap), addr,
           ours_exit, gnu_exit, expect))
    if ours_exit != expect or gnu_exit != expect:
        return False, len(ours), "exit status mismatch (ours %d, gnu %d, "\
            "expected %d)" % (ours_exit, gnu_exit, expect)
    return True, len(ours), "identical"


def main():
    tools = {}
    missing = []
    for t in ("as", "ld", "objcopy", "readelf", "objdump"):
        tools[t] = tool(t)
        if tools[t] is None:
            missing.append(t)
    if missing:
        say("FATAL: missing tools: %s" % ", ".join(missing))
        sys.exit(2)

    files = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(CORPUS, "*.s")))
    if files != sorted(EXPECT_EXIT):
        say("FATAL: corpus is %s but EXPECT_EXIT lists %s"
            % (files, sorted(EXPECT_EXIT)))
        sys.exit(1)
    if not files:
        say("FATAL: no corpus files in %s" % CORPUS)
        sys.exit(2)

    os.makedirs(TMP, exist_ok=True)
    sys.path.insert(0, HERE)
    sys.path.insert(0, B1)
    import asm as asm_mod
    import elf64_writer as elf_mod

    rc, out, err = run(["as", "--version"])
    if rc != 0:
        say("FATAL: GNU as is not usable")
        sys.exit(2)
    say("=== B2 pass check: %d corpus files vs GNU as ===" % len(files))
    say("GNU as: %s" % (out.splitlines()[0] if out else "?"))

    failures = []
    compared = 0
    for name in files:
        say("--- %s ---" % name)
        try:
            ok, n, note = check_file(name, asm_mod, elf_mod, tools)
        except Exception:
            say("%s: FAILED (exception)" % name)
            for line in traceback.format_exc().splitlines():
                say("  " + line)
            ok, n, note = False, 0, "exception"
        compared += n
        say("%s: %s (%s)" % (name, "OK" if ok else "FAIL", note))
        if not ok:
            failures.append(name)

    say("corpus: %d files, %d bytes compared, %d differing, %d failed"
        % (len(files), compared, len(failures), len(failures)))
    if failures:
        say("B2 FAILED: %s" % ", ".join(failures))
        sys.exit(1)
    if not HAVE_AVX2:
        say("B2 PASSED (bytes identical for all %d files; %s execution "
            "SKIPPED: no AVX2 on this CPU)"
            % (len(files), AVX2_FILE))
        say("NOTE: %s must still EXECUTE on AVX2 hardware (Kaggle) "
            "before B2 counts as PASSED" % AVX2_FILE)
    else:
        say("B2 PASSED")


def guarded_main():
    """Always leave a results file, even on a hard failure (empty Kaggle logs)."""
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        say("B2 FAILED (exception):")
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
