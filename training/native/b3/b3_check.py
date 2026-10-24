#!/usr/bin/env python3
"""b3_check.py - B3 pass check: our linker vs GNU as + GNU ld, byte for
byte, on multi-module programs.

Runs on Kaggle (Linux: GNU as/ld/objcopy/readelf/nm, executing ELF
files), or locally under WSL (USER_ACTIONS.md) - never on bare Windows.
Python builds and tests only; the engine is assembly + our own tools.

Method per corpus program (b3/corpus/<name>/m*.s, modules linked in
sorted file-name order):
  - `as --64` assembles every module and `ld -e _start` links them in
    the given order; the linked .text is extracted with objcopy.
  - our link.py links the SAME module list at the address ld gave .text,
    so absolute fixups (offset label, movabs, [label] disp32, .long/
    .quad label) agree.
  - the two .text images must be byte identical (GNU ld adds no trailing
    padding to .text - verified; at most 15 trailing nop bytes are
    tolerated and disassembled like in B2).
  - our symbol table must agree with `nm -n` of the GNU ELF for every
    label defined exactly once.
  - both linked programs run and exit with the program's marker status.
  - the whole program is linked AGAIN in reverse module order on both
    sides and gets the same treatment: placement and every cross-module
    relocation must follow the modules wherever they land.

Then a generated grid sweeps module 2's section alignment and module 1's
size: 64 cases at .align 64 must produce every inter-module gap size
0..63 exactly once (so asm.gap_fill is compared against ld's fill for
every gap), plus aligns 2..128 at the edge sizes.

Pass criteria (fixed before running, never loosened):
  1) GNU as assembles every module with no error on stderr, ld links it,
     our linker accepts the same sources. The corpus must match
     EXPECT_EXIT exactly - no program can be silently skipped.
  2) Byte-identical .text for every program, both link orders, and every
     grid case: 0 differing bytes.
  3) Equal entry points and matching nm symbols per program.
  4) Both ELFs run and exit with the marker status (forward order; the
     reverse-order image must at least be byte identical and is executed
     too, expecting the same status).
  5) The gap grid must cover gap sizes 0..63 (coverage asserted).

Exit codes: 0 pass, 1 check failed, 2 environment problem.
Results always land in b3/b3_results.txt (Kaggle empty-log quirk, D-003).
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
TMP = os.path.join(os.environ.get("AIOS_TMP", "/tmp/aios_native"), "b3")

RESULTS = os.environ.get("AIOS_RESULTS",
                         os.path.join(HERE, "b3_results.txt"))

# Exit status each corpus program's _start must produce. The check
# requires this set exactly.
EXPECT_EXIT = {
    "01_xmod": 1,
    "02_align": 2,
    "03_order": 3,
}

PAD_MAX = 15  # tolerated trailing section-alignment nop padding

REPORT = []


def say(line):
    print(line, flush=True)
    REPORT.append(line)


def run(cmd, quiet=False):
    if not quiet:
        say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    if not quiet:
        for stream in (r.stdout or "", r.stderr or ""):
            for line in stream.splitlines():
                if line.strip():
                    say("  " + line)
    return r.returncode, r.stdout or "", r.stderr or ""


def tool(name):
    return name if shutil.which(name) else None


def text_section(elf):
    """(address, size) of .text in a linked ELF, from readelf -SW."""
    rc, out, err = run(["readelf", "-SW", elf], quiet=True)
    if rc != 0:
        raise AssertionError("readelf -SW failed (exit %d)" % rc)
    for line in out.splitlines():
        m = re.match(r"\s*\[\s*\d+\]\s+\.text\s+\S+\s+([0-9a-f]+)\s+"
                     r"([0-9a-f]+)\s+([0-9a-f]+)", line)
        if m:
            return int(m.group(1), 16), int(m.group(3), 16)
    raise AssertionError("no .text section in %s" % elf)


def entry_point(elf):
    rc, out, err = run(["readelf", "-hW", elf], quiet=True)
    if rc != 0:
        raise AssertionError("readelf -hW failed (exit %d)" % rc)
    m = re.search(r"Entry point address:\s*(0x[0-9a-f]+)", out)
    if not m:
        raise AssertionError("no entry point in readelf -hW output")
    return int(m.group(1), 16)


def objcopy_text(elf, dest):
    rc, out, err = run(["objcopy", "-O", "binary", "--only-section=.text",
                        elf, dest], quiet=True)
    if rc != 0:
        raise AssertionError("objcopy of .text failed (exit %d)" % rc)
    with open(dest, "rb") as f:
        return f.read()


def nm_symbols(elf):
    """{name: [(address, type), ...]} from nm -n."""
    rc, out, err = run(["nm", "-n", elf], quiet=True)
    if rc != 0:
        raise AssertionError("nm failed (exit %d)" % rc)
    syms = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            addr = int(parts[0], 16)
        except ValueError:
            continue
        syms.setdefault(parts[2], []).append((addr, parts[1]))
    return syms


def disassemble(path):
    """[(hex bytes, intel text)] for a raw x86-64 byte stream."""
    rc, out, err = run(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
                        "-M", "intel", path], quiet=True)
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


def padding_is_nops(pad, path):
    with open(path, "wb") as f:
        f.write(pad)
    for raw, text in disassemble(path):
        if not text.startswith("nop") and not text.startswith("xchg"):
            return False
    return True


def locate(mods, off):
    """Source line of our linked image covering byte offset `off`."""
    for m in mods:
        if m.start <= off < m.start + len(m.code):
            rel = off - m.start
            for lineno, text, start, size in m.stmt_map:
                if start <= rel < start + size:
                    return "%s:%d: %s" % (m.name, lineno, text)
            return "%s at +%d" % (m.name, rel)
    return "gap fill at %d" % off


def report_mismatch(tag, ours, gnu, mods, ours_path, gnu_path):
    off = first_diff(ours, gnu)
    say("MISMATCH %s: ours=%d bytes, gnu=%d bytes, first difference at %s"
        % (tag, len(ours), len(gnu),
           "end of the shorter stream" if off is not None and off >= min(
               len(ours), len(gnu)) else str(off)))
    if off is not None:
        say("  " + locate(mods, off))
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
            say("  instruction streams agree; difference is in data/length")
    except AssertionError as e:
        say("  (disassembly unavailable: %s)" % e)


def run_elf(path):
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    r = subprocess.run([path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return r.returncode


def gnu_link(tag, sources, order):
    """Assemble and link the modules in `order` (list of indices) with
    GNU as + ld. Returns (elf, objs)."""
    objs = []
    for i in range(len(sources)):
        objs.append(None)
    for slot, idx in enumerate(order):
        name, text = sources[idx]
        path = os.path.join(TMP, "%s_%02d_%s" % (tag, slot, name))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        obj = path[:-2] + ".o"
        rc, out, err = run(["as", "--64", "-o", obj, path])
        if rc != 0:
            raise AssertionError("GNU as failed on %s (exit %d)" % (name, rc))
        bad = [l for l in err.splitlines() if "error" in l.lower()]
        if bad:
            raise AssertionError("GNU as reported errors on %s: %s"
                                 % (name, bad[:2]))
        objs[slot] = obj
    elf = os.path.join(TMP, tag + ".gnu.elf")
    rc, out, err = run(["ld", "-z", "noexecstack", "-e", "_start",
                        "-o", elf] + objs)
    if rc != 0:
        say("ld with -z noexecstack failed; retrying without the flag")
        rc, out, err = run(["ld", "-e", "_start", "-o", elf] + objs)
        if rc != 0:
            raise AssertionError("ld failed (exit %d)" % rc)
    return elf


def check_program(name, order, expect, tag, link_mod, elf_mod):
    """(ok, bytes_compared, note) for one program in one link order."""
    d = os.path.join(CORPUS, name)
    files = sorted(glob.glob(os.path.join(d, "*.s")))
    all_sources = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            all_sources.append((os.path.basename(p), f.read()))
    sources = [all_sources[i] for i in order]

    gnu_elf = gnu_link(tag, sources, list(range(len(sources))))
    addr, size = text_section(gnu_elf)
    gnu_txt = os.path.join(TMP, tag + ".gnu.text")
    gnu = objcopy_text(gnu_elf, gnu_txt)
    if size != len(gnu):
        return False, 0, "readelf says .text size %d, objcopy gave %d" % (
            size, len(gnu))

    # ---- our side, same base address ld gave .text ----
    ours, entry, symbols, mods = link_mod.link(sources, base=addr)
    ours_txt = os.path.join(TMP, tag + ".ours.text")
    with open(ours_txt, "wb") as f:
        f.write(ours)
    if entry_point(gnu_elf) != entry:
        return False, 0, "entry mismatch: gnu 0x%x, ours 0x%x" % (
            entry_point(gnu_elf), entry)

    # ---- byte comparison ----
    if len(gnu) < len(ours) or gnu[:len(ours)] != ours:
        report_mismatch(tag, ours, gnu, mods, ours_txt, gnu_txt)
        return False, min(len(ours), len(gnu)), "bytes differ"
    pad = gnu[len(ours):]
    if len(pad) > PAD_MAX or (pad and not padding_is_nops(
            pad, os.path.join(TMP, tag + ".pad.bin"))):
        say("%s: %d trailing bytes after our linked .text are not "
            "section-alignment nop padding: %s"
            % (tag, len(pad), pad.hex(" ")))
        return False, len(ours), "unexpected trailing bytes"

    # ---- symbol table vs nm -n of the GNU link ----
    nm = nm_symbols(gnu_elf)
    matched = 0
    for sym, addr_ours in sorted(symbols.items()):
        if sym not in nm:
            say("%s: symbol %r missing from nm output" % (tag, sym))
            return False, len(ours), "symbol %r missing" % sym
        addrs = [a for a, t in nm[sym]]
        if addr_ours not in addrs:
            say("%s: symbol %r at 0x%x for us, nm says %s"
                % (tag, sym, addr_ours,
                   ", ".join("0x%x" % a for a in addrs)))
            return False, len(ours), "symbol %r address mismatch" % sym
        matched += 1

    # ---- both must run and exit with the marker ----
    base_run = elf_mod.BASE + elf_mod.CODE_OFF
    rt_code, rt_entry, _, _ = link_mod.link(sources, base=base_run)
    ours_elf = os.path.join(TMP, tag + ".ours.elf")
    with open(ours_elf, "wb") as f:
        f.write(elf_mod.emit_elf64(rt_code,
                                   entry_off=rt_entry - elf_mod.BASE))
    ours_exit = run_elf(ours_elf)
    gnu_exit = run_elf(gnu_elf)
    say("%s [%s]: %d bytes == %d (gnu: %d code + %d nop padding), "
        "%d symbols matched, entry 0x%x, exit ours=%d gnu=%d expected=%d"
        % (name, "+".join(sources[i][0] for i in range(len(sources))),
           len(ours), len(gnu), len(ours), len(pad), matched, entry,
           ours_exit, gnu_exit, expect))
    if ours_exit != expect or gnu_exit != expect:
        return False, len(ours), "exit status mismatch (ours %d, gnu %d, " \
            "expected %d)" % (ours_exit, gnu_exit, expect)
    return True, len(ours), "identical"


def grid_case(align, pad, link_mod):
    """Two generated modules: m1 sized by .space pad, m2 on an .align
    boundary. Returns (ours, gnu, mods, gap)."""
    m1 = ("# generated gap/align grid case (align %d, pad %d)\n"
          "\t.intel_syntax noprefix\n"
          "\t.text\n"
          "\t.global _start\n"
          "_start:\n"
          "\tcall fn\n"
          "\tcmp eax, 7\n"
          "\tjne bad\n"
          "\tmov rax, qword ptr [rip+data]\n"
          "\tmov edi, 0\n"
          "\tmov eax, 60\n"
          "\tsyscall\n"
          "bad:\n"
          "\tmov edi, 1\n"
          "\tmov eax, 60\n"
          "\tsyscall\n"
          "\t.space %d\n" % (align, pad, pad))
    m2 = ("# generated gap/align grid case (align %d, pad %d)\n"
          "\t.intel_syntax noprefix\n"
          "\t.text\n"
          "\t.global fn\n"
          "\t.global data\n"
          "\t.align %d\n"
          "fn:\n"
          "\tmov eax, 7\n"
          "\tret\n"
          "data:\n"
          "\t.quad 0x0123456789abcdef\n" % (align, pad, align))
    sources = [("m1.s", m1), ("m2.s", m2)]
    gnu_elf = gnu_link("grid_a%d_p%d" % (align, pad), sources, [0, 1])
    addr, size = text_section(gnu_elf)
    gnu = objcopy_text(gnu_elf, os.path.join(
        TMP, "grid_a%d_p%d.gnu.text" % (align, pad)))
    ours, entry, symbols, mods = link_mod.link(sources, base=addr)
    gap = mods[1].start - len(mods[0].code)
    return ours, gnu, mods, gap


def check_grid(link_mod):
    """Byte-identical linking over the generated gap/align sweep."""
    cases = [((64, p) for p in range(64))]
    cases.append(((a, p) for a in (2, 4, 8, 16, 32, 128)
                  for p in (0, 1, a - 1)))
    total = 0
    diffs = 0
    gaps = set()
    n = 0
    for group in cases:
        for align, pad in group:
            ours, gnu, mods, gap = grid_case(align, pad, link_mod)
            n += 1
            if align == 64:
                gaps.add(gap)
            total += len(ours)
            if gnu[:len(ours)] != ours or len(gnu) != len(ours):
                diffs += 1
                off = first_diff(ours, gnu)
                say("GRID MISMATCH align=%d pad=%d: ours=%d gnu=%d, first "
                    "difference at %s" % (align, pad, len(ours), len(gnu),
                                          off))
                if off is not None:
                    say("  " + locate(mods, off))
                if diffs == 5:
                    say("  (further grid mismatches suppressed)")
                    break
    if diffs:
        return False, total, "%d grid cases differ" % diffs
    if gaps != set(range(64)):
        missing = sorted(set(range(64)) - gaps)
        return False, total, "gap sweep missed sizes %s" % missing[:8]
    say("grid: %d cases, %d bytes compared, 0 differing; gap sizes 0..63 "
        "all covered at .align 64" % (n, total))
    return True, total, "identical"


def main():
    missing = [t for t in ("as", "ld", "objcopy", "readelf", "objdump", "nm")
               if tool(t) is None]
    if missing:
        say("FATAL: missing tools: %s" % ", ".join(missing))
        sys.exit(2)

    programs = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(CORPUS, "*")) if os.path.isdir(p))
    if programs != sorted(EXPECT_EXIT):
        say("FATAL: corpus is %s but EXPECT_EXIT lists %s"
            % (programs, sorted(EXPECT_EXIT)))
        sys.exit(1)
    if not programs:
        say("FATAL: no corpus programs in %s" % CORPUS)
        sys.exit(2)

    os.makedirs(TMP, exist_ok=True)
    sys.path.insert(0, HERE)
    sys.path.insert(0, B1)
    import link as link_mod
    import elf64_writer as elf_mod

    rc, out, err = run(["as", "--version"], quiet=True)
    if rc != 0:
        say("FATAL: GNU as is not usable")
        sys.exit(2)
    say("=== B3 pass check: %d programs vs GNU as + GNU ld ==="
        % len(programs))
    say("GNU as: %s" % (out.splitlines()[0] if out else "?"))
    rc, out, err = run(["ld", "--version"], quiet=True)
    say("GNU ld: %s" % (out.splitlines()[0] if out else "?"))

    failures = []
    compared = 0
    for name in programs:
        d = os.path.join(CORPUS, name)
        count = len(glob.glob(os.path.join(d, "*.s")))
        expect = EXPECT_EXIT[name]
        for rev in (False, True):
            order = list(range(count))
            if rev:
                order.reverse()
            tag = "%s_%s" % (name, "rev" if rev else "fwd")
            say("--- %s (%s order) ---" % (name, "reverse" if rev
                                           else "forward"))
            try:
                ok, n, note = check_program(name, order, expect, tag,
                                            link_mod, elf_mod)
            except Exception:
                say("%s: FAILED (exception)" % tag)
                for line in traceback.format_exc().splitlines():
                    say("  " + line)
                ok, n, note = False, 0, "exception"
            compared += n
            say("%s: %s (%s)" % (tag, "OK" if ok else "FAIL", note))
            if not ok:
                failures.append(tag)

    say("--- gap/align grid ---")
    try:
        ok, n, note = check_grid(link_mod)
    except Exception:
        say("grid: FAILED (exception)")
        for line in traceback.format_exc().splitlines():
            say("  " + line)
        ok, n, note = False, 0, "exception"
    compared += n
    say("grid: %s (%s)" % ("OK" if ok else "FAIL", note))
    if not ok:
        failures.append("grid")

    say("total: %d bytes compared, %d differing, %d failed"
        % (compared, len(failures), len(failures)))
    if failures:
        say("B3 FAILED: %s" % ", ".join(failures))
        sys.exit(1)
    say("B3 PASSED (%d programs x2 link orders + %d grid cases)"
        % (len(programs), 82))


def guarded_main():
    """Always leave a results file, even on a hard failure (empty Kaggle
    logs, D-003)."""
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        say("B3 FAILED (exception):")
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
