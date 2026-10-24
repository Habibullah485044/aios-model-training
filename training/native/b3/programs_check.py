#!/usr/bin/env python3
"""programs_check.py - B3 programs pass check: the four mission programs
(exit42, hello, sumloop, read_weights) written as `.asm` sources, built
ONLY by our assembler + our ELF writer (no GNU as/ld in the build),
then executed against fixed targets fixed before running.

Per program:
  - GNU as assembles the same source and ld links it; the linked .text
    must equal OUR assembler's .text byte for byte (the B2 criterion,
    reused here per the mission's "extra check for every B3/B4
    program"), with the same tolerance for trailing nop padding.
  - Our ELF is emitted by elf64_writer with the W^X split: an R-X PT_LOAD
    over [0, data label) and an RW- PT_LOAD over the data (the data
    label is `.align 4096` in the source so the two segments share no
    page). exit42 has no data and must stay byte identical to B1's
    132-byte exit42 (one RWX PT_LOAD). The split is verified with an
    independent struct parse AND readelf (clean stderr).
  - Both ELFs run and agree with the fixed targets.

Fixed targets (never loosened):
  exit42       exit 42, stdout ""
  hello        exit 0, stdout "Hello, world!\\n"  (14 bytes)
  sumloop      exit 0, stdout "500500\\n"         (7 bytes)
  read_weights exit 0 with exactly 3 args; out.bin = the three 64-byte
               blocks at offsets 0, 622,333,952 and 6,171,873,280 of a
               SPARSE 6,171,877,376-byte file (same size as the real
               weights.bin; the third offset is above 4 GB); a wrong
               path exits 2 (ENOENT) and any wrong argc exits 2, with
               no crash (plain status, never a signal).

Runs on Kaggle (Linux) or locally under WSL (USER_ACTIONS.md) - never on
bare Windows. Exit codes: 0 pass, 1 check failed, 2 environment problem.
Results always land in b3/programs_results.txt (D-003).
"""
import glob
import os
import random
import re
import stat
import struct
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
B1 = os.path.join(NATIVE, "b1")
B2 = os.path.join(NATIVE, "b2")
PROGRAMS = os.path.join(HERE, "programs")
TMP = os.path.join(os.environ.get("AIOS_TMP", "/tmp/aios_native"),
                   "b3_programs")
RESULTS = os.environ.get("AIOS_RESULTS",
                         os.path.join(HERE, "programs_results.txt"))

WEIGHTS_SIZE = 6171877376      # same size as the real weights.bin
WEIGHTS_OFFS = (0, 622333952, 6171873280)   # embed_tokens, q_proj, norm

# name -> (data label or None, expected exit, expected stdout)
CHECK = {
    "exit42":      (None, 42, b""),
    "hello":       ("msg", 0, b"Hello, world!\n"),
    "sumloop":     ("places", 0, b"500500\n"),
}
READ_WEIGHTS = "read_weights"

PAD_MAX = 15  # tolerated trailing section-alignment nop padding

REPORT = []


def say(line):
    print(line, flush=True)
    REPORT.append(line)


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return r.returncode, r.stdout or "", r.stderr or ""


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


def objcopy_text(elf, dest):
    rc, out, err = run(["objcopy", "-O", "binary", "--only-section=.text",
                        elf, dest])
    if rc != 0:
        raise AssertionError("objcopy of .text failed (exit %d)" % rc)
    with open(dest, "rb") as f:
        return f.read()


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def window(data, off, span=16):
    lo = max(0, off - span)
    return "%06x: %s" % (lo, data[lo:off + span].hex(" "))


def run_elf(path, argv=()):
    os.chmod(path, os.stat(path).st_mode
             | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    r = subprocess.run([path] + list(argv), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=120)
    return r.returncode, r.stdout


def parse_phdrs(blob):
    """Independent struct-level parse: [(type, flags, offset, vaddr, size)]."""
    assert blob[:4] == b"\x7fELF", "bad magic"
    (e_type, e_machine, _v, _entry, e_phoff, _so, _f, _h, e_phentsize,
     e_phnum) = struct.unpack_from("<HHIQQQIHHH", blob, 16)
    assert e_type == 2 and e_machine == 62
    phdrs = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        (p_type, p_flags, p_offset, p_vaddr, _p, p_filesz, p_memsz,
         p_align) = struct.unpack_from("<IIQQQQQQ", blob, off)
        assert p_align == 0x1000 and p_offset % 0x1000 == p_vaddr % 0x1000
        phdrs.append((p_type, p_flags, p_offset, p_vaddr, p_filesz, p_memsz))
    return e_entry_of(blob), phdrs


def e_entry_of(blob):
    return struct.unpack_from("<Q", blob, 24)[0]


def check_phdrs(tag, blob, elf_path, data_off, expected_len):
    """W^X layout assertions (independent struct parse + readelf).
    Returns nothing - raises on any lie."""
    assert len(blob) == expected_len, "%s: %d bytes, expected %d" % (
        tag, len(blob), expected_len)
    entry, phdrs = parse_phdrs(blob)
    if data_off is None:
        assert len(phdrs) == 1, "%s: %d PT_LOADs" % (tag, len(phdrs))
        t, flags, offset, vaddr, filesz, memsz = phdrs[0]
        assert t == 1 and flags == 7, "%s: flags %d != RWX" % (tag, flags)
        assert offset == 0 and vaddr == 0x400000
        assert filesz == memsz == len(blob)
        assert entry == 0x400078, "%s: entry 0x%x" % (tag, entry)
        say("%s: 1 PT_LOAD RWX @0x400000, entry 0x%x, %d bytes (B1 layout)"
            % (tag, entry, len(blob)))
        return
    assert len(phdrs) == 2, "%s: %d PT_LOADs" % (tag, len(phdrs))
    t, flags, offset, vaddr, filesz, memsz = phdrs[0]
    assert t == 1 and flags == 5, "%s: code segment flags %d != R-X" % (
        tag, flags)
    code_end = 4096 + data_off
    assert offset == 0 and vaddr == 0x400000
    assert filesz == memsz == code_end
    t, flags, offset, vaddr, filesz, memsz = phdrs[1]
    assert t == 1 and flags == 6, "%s: data segment flags %d != RW-" % (
        tag, flags)
    assert offset == code_end and vaddr == 0x400000 + code_end
    assert filesz == memsz == len(blob) - code_end
    assert entry == 0x401000, "%s: entry 0x%x != 0x401000" % (tag, entry)
    say("%s: R-X PT_LOAD [0x400000,+%d) + RW- PT_LOAD [0x%08x,+%d), "
        "entry 0x%x, %d bytes (W^X, no shared page)"
        % (tag, code_end, 0x400000 + code_end, filesz, entry, len(blob)))
    # readelf must agree and not warn
    rc, out, err = run(["readelf", "-lW", elf_path])
    assert rc == 0 and err == "", "readelf -lW warnings: %r" % err


def gnu_build(tag, src_path):
    """GNU as + ld of the same source: (elf, text bytes)."""
    obj = os.path.join(TMP, tag + ".gnu.o")
    elf = os.path.join(TMP, tag + ".gnu.elf")
    rc, out, err = run(["as", "--64", "-o", obj, src_path])
    if rc != 0:
        raise AssertionError("GNU as failed (exit %d): %s" % (rc, err))
    bad = [l for l in err.splitlines() if "error" in l.lower()]
    if bad:
        raise AssertionError("GNU as reported errors: %s" % bad[:2])
    rc, out, err = run(["ld", "-z", "noexecstack", "-e", "_start",
                        "-o", elf, obj])
    if rc != 0:
        say("ld with -z noexecstack failed; retrying without the flag")
        rc, out, err = run(["ld", "-e", "_start", "-o", elf, obj])
        if rc != 0:
            raise AssertionError("ld failed (exit %d): %s" % (rc, err))
    addr, size = text_section(elf)
    gnu = objcopy_text(elf, os.path.join(TMP, tag + ".gnu.text"))
    if size != len(gnu):
        raise AssertionError("readelf says .text size %d, objcopy gave %d"
                             % (size, len(gnu)))
    return elf, gnu, addr


def assemble_ours(asm_mod, text, base):
    a = asm_mod.Assembler(base=base)
    code, entry = a.assemble(text)
    return code, entry, a.labels


def compare_text(tag, ours, gnu, ours_path, gnu_path):
    off = first_diff(ours, gnu)
    if off is not None and (len(gnu) < len(ours) or gnu[:len(ours)] != ours):
        say("MISMATCH %s: ours=%d bytes, gnu=%d bytes, first difference "
            "at %s" % (tag, len(ours), len(gnu), off))
        say("  ours %s" % window(ours, off))
        say("  gnu  %s" % window(gnu, off))
        return False
    pad = gnu[len(ours):]
    if len(pad) > PAD_MAX:
        say("%s: %d trailing bytes after our code exceed the %d byte "
            "nop-padding tolerance" % (tag, len(pad), PAD_MAX))
        return False
    return True


def make_sparse_weights(path):
    """SPARSE file, exactly WEIGHTS_SIZE bytes, with a known random
    64-byte block at each of the three tensor offsets. Returns the
    blocks so the outputs can be checked against them."""
    rnd = random.Random(0xB3B3)   # fixed seed: the check is reproducible
    blocks = [bytes(rnd.getrandbits(8) for _ in range(64))
              for _ in WEIGHTS_OFFS]
    with open(path, "wb") as f:
        f.truncate(WEIGHTS_SIZE)
        for off, block in zip(WEIGHTS_OFFS, blocks):
            f.seek(off)
            f.write(block)
    size = os.path.getsize(path)
    assert size == WEIGHTS_SIZE, "sparse file is %d bytes" % size
    # independent re-read of the three blocks from the input file
    with open(path, "rb") as f:
        for off, block in zip(WEIGHTS_OFFS, blocks):
            f.seek(off)
            assert f.read(64) == block, "input block at %d corrupt" % off
    return blocks


def check_program(name, data_label, asm_mod, elf_mod):
    """(ok, bytes_compared) for exit42/hello/sumloop."""
    src = os.path.join(PROGRAMS, name + ".asm")
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()

    gnu_elf, gnu, addr = gnu_build(name, src)

    # ---- our side at ld's .text address: byte identity ----
    code, entry, labels = assemble_ours(asm_mod, text, addr)
    if entry != 0:
        return False, 0, "_start is not at offset 0 (entry 0x%x)" % entry
    ours_txt = os.path.join(TMP, name + ".ours.bin")
    with open(ours_txt, "wb") as f:
        f.write(code)
    gnu_txt = os.path.join(TMP, name + ".gnu.text")
    if not compare_text(name, code, gnu, ours_txt, gnu_txt):
        return False, min(len(code), len(gnu)), "bytes differ"

    # ---- our runtime ELF with the W^X split ----
    data_off = labels.get(data_label) if data_label else None
    if data_label:
        assert data_off is not None and 0 < data_off < len(code), \
            "%s: data label %r at %r" % (name, data_label, data_off)
        base_run = elf_mod.BASE + elf_mod.SPLIT_CODE_OFF
        code_run, entry_run, _ = assemble_ours(asm_mod, text, base_run)
        blob = elf_mod.emit_elf64(code_run, entry_off=entry_run,
                                  data_off=data_off)
    else:
        base_run = elf_mod.BASE + elf_mod.CODE_OFF
        code_run, entry_run, _ = assemble_ours(asm_mod, text, base_run)
        blob = elf_mod.emit_elf64(code_run,
                                  entry_off=elf_mod.CODE_OFF + entry_run)
    if name == "exit42":
        # the .asm build must reproduce B1's deliverable exactly
        b1_blob = elf_mod.emit_elf64(elf_mod.exit_program(42))
        if blob != b1_blob:
            off = first_diff(blob, b1_blob)
            say("exit42: our emitted image differs from B1's 132-byte "
                "exit42 at %s" % off)
            return False, len(code), "not byte identical to B1"
        say("exit42: emitted image byte identical to B1's 132-byte "
            "exit42 ELF")
    our_elf = os.path.join(TMP, name + ".ours.elf")
    with open(our_elf, "wb") as f:
        f.write(blob)
    check_phdrs(name, blob, our_elf, data_off,
                (elf_mod.SPLIT_CODE_OFF + len(code)
                 if data_label else elf_mod.CODE_OFF + len(code)))

    # ---- both must run and agree with the fixed targets ----
    for label, elf_path in (("ours", our_elf), ("gnu ", gnu_elf)):
        rc, out = run_elf(elf_path)
        if out != CHECK_EXPECT[name][1] or rc != CHECK_EXPECT[name][0]:
            return False, len(code), "%s ELF: exit %d stdout %r != %r/%r" % (
                label, rc, out, CHECK_EXPECT[name][0], CHECK_EXPECT[name][1])
    say("%s: %d bytes == GNU as, W^X ELF, exit %d, stdout %r (ours and "
        "GNU)" % (name, len(code), CHECK_EXPECT[name][0],
                  CHECK_EXPECT[name][1]))
    return True, len(code), "identical"


CHECK_EXPECT = {k: (v[1], v[2]) for k, v in CHECK.items()}


def check_read_weights(asm_mod, elf_mod):
    """(ok, bytes_compared) for read_weights + the sparse-file test."""
    name = READ_WEIGHTS
    src = os.path.join(PROGRAMS, name + ".asm")
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()

    gnu_elf, gnu, addr = gnu_build(name, src)
    code, entry, labels = assemble_ours(asm_mod, text, addr)
    if entry != 0:
        return False, 0, "_start is not at offset 0 (entry 0x%x)" % entry
    ours_txt = os.path.join(TMP, name + ".ours.bin")
    with open(ours_txt, "wb") as f:
        f.write(code)
    if not compare_text(name, code, gnu, ours_txt,
                        os.path.join(TMP, name + ".gnu.text")):
        return False, min(len(code), len(gnu)), "bytes differ"

    data_off = labels["offs"]
    base_run = elf_mod.BASE + elf_mod.SPLIT_CODE_OFF
    code_run, entry_run, _ = assemble_ours(asm_mod, text, base_run)
    blob = elf_mod.emit_elf64(code_run, entry_off=entry_run, data_off=data_off)
    our_elf = os.path.join(TMP, name + ".ours.elf")
    with open(our_elf, "wb") as f:
        f.write(blob)
    check_phdrs(name, blob, our_elf, data_off,
                elf_mod.SPLIT_CODE_OFF + len(code))

    # ---- the sparse 6.17 GB weights file ----
    weights = os.path.join(TMP, "weights_sparse.bin")
    say("creating sparse %d-byte weights file with 64-byte blocks at %s"
        % (WEIGHTS_SIZE, WEIGHTS_OFFS))
    blocks = make_sparse_weights(weights)
    expected_out = b"".join(blocks)

    fails = []
    for label, elf_path in (("ours", our_elf), ("gnu ", gnu_elf)):
        out_bin = os.path.join(TMP, "out.%s.bin" % label.strip())
        rc, out = run_elf(elf_path, [weights, out_bin])
        if rc != 0 or out != b"":
            fails.append("%s: exit %d stdout %r on the sparse file"
                         % (label, rc, out))
            continue
        with open(out_bin, "rb") as f:
            got = f.read()
        if got != expected_out or len(got) != 192:
            fails.append("%s: out.bin is %d bytes, blocks %s"
                         % (label, len(got),
                            [first_diff(got, expected_out)]))
            continue
        # out.bin must equal the input file at the three offsets
        with open(weights, "rb") as f:
            for i, off in enumerate(WEIGHTS_OFFS):
                f.seek(off)
                if f.read(64) != got[i * 64:(i + 1) * 64]:
                    fails.append("%s: out.bin block %d != input at %d"
                                 % (label, i, off))
        say("%s ELF: sparse run ok - out.bin 192 bytes == blocks at %s"
            % (label, WEIGHTS_OFFS))

        # wrong path -> exit 2 (ENOENT), no crash
        rc, out = run_elf(elf_path, ["/nonexistent/weights.bin", out_bin])
        if rc != 2:
            fails.append("%s: wrong path exit %d != 2 (signal?)" % (
                label, rc))
        else:
            say("%s ELF: wrong path exits 2 (ENOENT), no crash" % label)
        # wrong argc -> exit 2, no crash
        for argv in ([], [weights], [weights, out_bin, "extra"]):
            rc, out = run_elf(elf_path, argv)
            if rc != 2:
                fails.append("%s: argc=%d exit %d != 2"
                             % (label, len(argv) + 1, rc))
        if not any("argc" in f for f in fails):
            say("%s ELF: wrong argc (1, 2, 4) all exit 2" % label)

    if fails:
        for f in fails:
            say("FAIL: " + f)
        return False, len(code), "; ".join(fails)[:200]
    say("%s: %d bytes == GNU as, W^X ELF, sparse-file behavior targets "
        "all met (ours and GNU)" % (name, len(code)))
    return True, len(code), "identical"


def main():
    missing = [t for t in ("as", "ld", "objcopy", "readelf")
               if not any(os.access(os.path.join(d, t), os.X_OK)
                          for d in os.environ.get("PATH", "").split(os.pathsep))]
    if not sys.platform.startswith("linux"):
        say("FATAL: this check must run on Linux (Kaggle or WSL), "
            "never on bare Windows (D14)")
        sys.exit(2)
    if missing:
        say("FATAL: missing GNU tools: %s" % ", ".join(missing))
        sys.exit(2)

    sources = sorted(os.path.basename(p)[:-4] for p in glob.glob(
        os.path.join(PROGRAMS, "*.asm")))
    expect = sorted(list(CHECK) + [READ_WEIGHTS])
    if sources != expect:
        say("FATAL: programs/ holds %s but the plan lists %s"
            % (sources, expect))
        sys.exit(1)

    os.makedirs(TMP, exist_ok=True)
    sys.path.insert(0, B1)
    sys.path.insert(0, B2)
    import asm as asm_mod
    import elf64_writer as elf_mod

    rc, out, err = run(["as", "--version"])
    say("=== B3 programs check: %d programs, built ONLY by our assembler "
        "+ ELF writer ===" % len(sources))
    say("GNU as: %s (byte-identity oracle only)" % (
        out.splitlines()[0] if out else "?"))

    failures = []
    compared = 0
    for name in sorted(CHECK):
        say("--- %s ---" % name)
        try:
            ok, n, note = check_program(name, CHECK[name][0], asm_mod,
                                        elf_mod)
        except Exception:
            say("%s: FAILED (exception)" % name)
            for line in traceback.format_exc().splitlines():
                say("  " + line)
            ok, n, note = False, 0, "exception"
        compared += n
        say("%s: %s (%s)" % (name, "OK" if ok else "FAIL", note))
        if not ok:
            failures.append(name)

    say("--- %s (sparse %d-byte file) ---" % (READ_WEIGHTS, WEIGHTS_SIZE))
    try:
        ok, n, note = check_read_weights(asm_mod, elf_mod)
    except Exception:
        say("%s: FAILED (exception)" % READ_WEIGHTS)
        for line in traceback.format_exc().splitlines():
            say("  " + line)
        ok, n, note = False, 0, "exception"
    compared += n
    say("%s: %s (%s)" % (READ_WEIGHTS, "OK" if ok else "FAIL", note))
    if not ok:
        failures.append(READ_WEIGHTS)

    say("total: %d bytes compared against GNU as, %d programs failed"
        % (compared, len(failures)))
    if failures:
        say("B3 programs FAILED: %s" % ", ".join(failures))
        sys.exit(1)
    say("B3 programs PASSED (4 programs byte-identical to GNU as, W^X "
        "ELFs, fixed behavior targets incl. the sparse 6.17 GB file)")


def guarded_main():
    """Always leave a results file, even on a hard failure (D-003)."""
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        say("B3 programs FAILED (exception):")
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
