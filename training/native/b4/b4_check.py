#!/usr/bin/env python3
"""b4_check.py - B4 pass check: the runtime (raw syscalls, threads,
timers) hand-written in asm, built two ways, compared byte for byte,
then executed.

Pass criteria (fixed before running):
  1. The atomics unit: our assembler's bytes for lock xadd/cmpxchg/add/or,
     xchg (accumulator short form, generic, memory forms) and pause must
     equal GNU as's bytes on the fixed source (the forms the runtime
     uses - measured first, then encoded).
  2. For each corpus program, our assembler + B3 linker produce a .text
     byte-identical to `GNU as` + `GNU ld` on the same two modules
     (runtime.s + the program), and identical global symbols.
  3. Our ELF is W^X (an R-X PT_LOAD for the code and a separate RW-
     PT_LOAD for the data page after `.align 4096`; never a writable+
     executable segment), asserted by an independent header parse and
     a warning-free `readelf -lW`.
  4. Both ELFs execute on Linux and agree with the fixed targets:
       01_syscalls: stdout "B4 syscalls ok\n"  exit 41
       02_threads:  stdout "B4 threads ok\n"   exit 42
                    (02 spawns 3 clone/futex tasks and joins them)
       03_timers:   stdout "B4 timers ok\n"    exit 43
                    (03 measures a 50 ms nanosleep on CLOCK_MONOTONIC
                    with a fixed 40 ms floor)
       04_features: stdout == /proc/cpuinfo flags restricted to
                    (sse2, ssse3, sse4_1, avx, avx2, fma) in the
                    printed order  exit 44
       05_sum:      stdout "B4 sum ok\n"       exit 45
                    (sum 1..10^8 = 5,000,000,050,000,000 verified by
                    the program itself for EVERY thread count 1..N,
                    N = sched_getaffinity CPU count)
       06_barrier:  stdout "B4 barrier ok\n"   exit 46
                    (10,000 barrier rounds over all N workers; a
                    deadlock trips the run timeout instead)

The check runs on Kaggle (Linux) or under WSL once a distro exists
(USER_ACTIONS.md) - never on bare Windows. Python is used only to build
and test, never inside the engine. Exit 0 only if every case passes.
"""
import os
import re
import struct
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NATIVE, "b1"))
sys.path.insert(0, os.path.join(NATIVE, "b2"))
sys.path.insert(0, os.path.join(NATIVE, "b3"))
from link import link
import asm
import elf64_writer

REL32 = {"call", "jmp", "jo", "jno", "jb", "jnb", "je", "jne", "jna",
         "ja", "js", "jns", "jp", "jnp", "jl", "jge", "jle", "jg",
         "jcxz", "jecxz", "jrcxz"}
RELOC_SIZES = {"call": 4, "jmp": 4}
for _c in REL32:
    RELOC_SIZES.setdefault(_c, 1)
RELOC_SIZES["imul"] = 1

TMP = os.path.join(NATIVE, "b4", "tmp")
OUT = os.path.join(NATIVE, "b4", "out")

# The six feature names the mission requires to match /proc/cpuinfo,
# in the order program 04 prints them.
SIX = ("sse2", "ssse3", "sse4_1", "avx", "avx2", "fma")

# (expected exit status, expected stdout) - fixed before running.
# 04's stdout is computed from /proc/cpuinfo at the start of main():
# the printed set must equal this machine's flags restricted to SIX.
PROGRAMS = {
    "01_syscalls": (41, b"B4 syscalls ok\n"),
    "02_threads": (42, b"B4 threads ok\n"),
    "03_timers": (43, b"B4 timers ok\n"),
    "04_features": (44, None),
    "05_sum": (45, b"B4 sum ok\n"),
    "06_barrier": (46, b"B4 barrier ok\n"),
}

# First data label of each program: the message bytes live after
# `.align 4096`, so the ELF can be split into R-X code and RW- data.
DATA_LABEL = {
    "01_syscalls": "msg",
    "02_threads": "msg",
    "03_timers": "msg",
    "04_features": "name_sse2",
    "05_sum": "msg_ok",
    "06_barrier": "msg_ok",
}

# The atomics unit: every atomic form the runtime uses, in one fixed
# source that must assemble byte identically with GNU as.
ATOMICS_SRC = """\
    .intel_syntax noprefix
    .text
    .global _start
_start:
    xchg rax, rbx
    xchg rbx, rax
    xchg eax, ecx
    xchg rcx, rdx
    xchg rdi, rsi
    xchg r8, r9
    xchg rax, [rbx]
    xchg [rbx], rax
    xchg rax, r8
    xchg r8, rax
    xchg eax, r8d
    xchg rsi, rdi
    xchg rbx, rax
    xchg dl, cl
    lock xadd dword ptr [rax], ecx
    lock xadd qword ptr [rbx], r8
    lock xadd byte ptr [rcx], dl
    lock cmpxchg qword ptr [rax], rdx
    lock cmpxchg dword ptr [rbx], ecx
    lock cmpxchg byte ptr [rcx], dl
    lock add dword ptr [rax], 1
    lock or qword ptr [rbx], 3
    pause
    mov rax, qword ptr [rip+buf]
loop1:
    lock xadd qword ptr [rax], rdx
    lock cmpxchg qword ptr [rax], rdx
    jne loop1
    ret
    .align 8
buf:
    .quad 0
"""


def say(msg):
    print(msg, flush=True)


def run(argv, **kw):
    kw.setdefault("timeout", 300)
    return subprocess.run(argv, capture_output=True, text=True, **kw)


def read_src(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def objcopy_text(obj):
    r = run(["objcopy", "-O", "binary", "-j", ".text", obj, obj + ".bin"])
    if r.returncode != 0:
        raise RuntimeError("objcopy failed: " + r.stderr)
    with open(obj + ".bin", "rb") as f:
        return f.read()


def gnu_link(mods):
    """GNU as + GNU ld .text of the modules (order = list order)."""
    for name, text in mods:
        p = os.path.join(TMP, name + ".gnu.s")
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        r = run(["as", "--64", "-o", p[:-2] + ".o", p])
        if r.returncode != 0:
            raise RuntimeError("GNU as rejected %s:\n%s" % (name, r.stderr))
    out = os.path.join(TMP, "gnu.elf")
    r = run(["ld", "-e", "_start", "-o", out] +
            [os.path.join(TMP, name + ".gnu.o") for name, _ in mods])
    if r.returncode != 0:
        raise RuntimeError("GNU ld failed: " + r.stderr)
    return objcopy_text(out), out


def text_section(elf):
    with open(elf, "rb") as f:
        data = f.read()
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]

    def sh(i):
        off = e_shoff + i * e_shentsize
        return struct.unpack_from("<IIQQQQIIQQ", data, off)

    str_off = sh(e_shstrndx)[4]
    for i in range(e_shnum):
        s = sh(i)
        name = data[str_off + s[0]:].split(b"\0", 1)[0].decode()
        if name == ".text":
            return data[s[4]:s[4] + s[5]], s[3], s[5]
    raise RuntimeError("no .text in " + elf)


# ld's own synthetic end-of-section markers: no source label stands for
# them, so they are not part of the symbol comparison.
LD_SYNTHETIC = ("__bss_start", "_edata", "_end", "__data_start",
                "__TMC_END__", "_edata__", "__bss_start__")


def gnu_symbols(elf):
    r = run(["nm", elf])
    if r.returncode != 0:
        raise RuntimeError("nm failed: " + r.stderr)
    syms = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3:
            addr, t, name = parts
            if t in ("T", "t") and name not in LD_SYNTHETIC:
                syms[name] = int(addr, 16)
    return syms


def our_link(mods, base):
    """Our B3 linker over the same modules: (text, entry, symbols, size)."""
    code, entry, symbols, _mods = link(mods, base=base)
    return code, entry, symbols, len(code)


def cpuinfo_expected():
    """Program 04's expected stdout: this machine's /proc/cpuinfo flags
    restricted to SIX, in the printed order (fixed rule - the printed
    set must match the CPU's actual flags)."""
    flags = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        return None
    return (" ".join(name for name in SIX if name in flags) + "\n").encode()


def check_atomics():
    """The atomics unit: our assembler must reproduce GNU as's bytes for
    every lock/xchg/xadd/cmpxchg/pause form the runtime uses."""
    src = os.path.join(TMP, "atomics.s")
    obj = os.path.join(TMP, "atomics.o")
    binf = os.path.join(TMP, "atomics.gnu.bin")
    with open(src, "w", encoding="utf-8") as f:
        f.write(ATOMICS_SRC)
    r = run(["as", "--64", "-o", obj, src])
    if r.returncode != 0:
        raise RuntimeError("GNU as rejected the atomics unit:\n" + r.stderr)
    r = run(["objcopy", "-O", "binary", "--only-section=.text", obj, binf])
    if r.returncode != 0:
        raise RuntimeError("objcopy failed: " + r.stderr)
    with open(binf, "rb") as f:
        gnu = f.read()
    ours, _entry = asm.assemble(ATOMICS_SRC, base=0)
    if ours != gnu:
        off = next((i for i in range(min(len(ours), len(gnu)))
                    if ours[i] != gnu[i]), min(len(ours), len(gnu)))
        say("ATOMICS MISMATCH at byte %d: ours %s, gnu %s"
            % (off, ours[max(0, off - 4):off + 8].hex(" "),
               gnu[max(0, off - 4):off + 8].hex(" ")))
        return False, len(ours)
    say("atomics unit: %d bytes == GNU as (lock xadd/cmpxchg/add/or, "
        "xchg forms, pause)" % len(ours))
    return True, len(ours)


def check_wx(tag, blob, elf_path, data_off, entry):
    """The emitted ELF must be W^X: an R-X PT_LOAD for the code and an
    RW- PT_LOAD for the data page, never a writable+executable segment
    and never a page shared by both (independent struct parse + readelf)."""
    e_entry = struct.unpack_from("<Q", blob, 24)[0]
    e_phoff = struct.unpack_from("<Q", blob, 32)[0]
    e_phentsize = struct.unpack_from("<H", blob, 54)[0]
    e_phnum = struct.unpack_from("<H", blob, 56)[0]
    phdrs = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        t, flags, offset, vaddr, _paddr, filesz, memsz, _al = \
            struct.unpack_from("<IIQQQQQQ", blob, off)
        phdrs.append((t, flags, offset, vaddr, filesz, memsz))
    code_end = elf64_writer.SPLIT_CODE_OFF + data_off
    assert len(phdrs) == 2, "%s: %d PT_LOADs" % (tag, len(phdrs))
    t, flags, offset, vaddr, filesz, memsz = phdrs[0]
    assert (t, flags, offset, vaddr) == (1, 5, 0, elf64_writer.BASE), \
        "%s: code segment is not R-X at the base: %r" % (tag, phdrs[0])
    assert filesz == memsz == code_end, \
        "%s: code segment size %d != %d" % (tag, filesz, code_end)
    t, flags, offset, vaddr, filesz, memsz = phdrs[1]
    assert (t, flags) == (1, 6), \
        "%s: data segment is not RW-: %r" % (tag, phdrs[1])
    assert offset == code_end, "%s: data offset %d != %d" % (
        tag, offset, code_end)
    assert vaddr == elf64_writer.BASE + code_end
    assert filesz == memsz == len(blob) - code_end
    assert e_entry == entry, "%s: entry 0x%x != _start 0x%x" % (
        tag, e_entry, entry)
    assert elf64_writer.BASE + elf64_writer.SPLIT_CODE_OFF <= e_entry \
        < elf64_writer.BASE + code_end, \
        "%s: entry 0x%x is not inside the code segment" % (tag, e_entry)
    r = subprocess.run(["readelf", "-lW", elf_path], capture_output=True,
                       text=True)
    assert r.returncode == 0 and not r.stderr, \
        "%s: readelf -lW complained: %r" % (tag, r.stderr)
    say("  ELF: R-X [0x%x,+%d) + RW- [0x%x,+%d), entry 0x%x (W^X)"
        % (elf64_writer.BASE, code_end,
           elf64_writer.BASE + code_end, filesz, e_entry))


def main():
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    for p in os.listdir(TMP):
        os.remove(os.path.join(TMP, p))
    runtime = read_src(os.path.join(HERE, "runtime.s"))
    failures = 0

    expected_04 = cpuinfo_expected()
    if expected_04 is None:
        say("b4_check: FATAL - /proc/cpuinfo unreadable; run on Linux "
            "(WSL/Kaggle), never bare Windows")
        return 2
    PROGRAMS["04_features"] = (44, expected_04)
    say("04_features expects stdout %r (cpuinfo flags in %s)"
        % (expected_04, list(SIX)))

    say("--- atomics unit ---")
    try:
        ok, _n = check_atomics()
    except Exception:
        say("atomics unit: FAILED (exception)")
        for line in traceback.format_exc().splitlines():
            say("  " + line)
        ok = False
    if not ok:
        failures += 1
        say("atomics unit: FAIL")

    for name in sorted(PROGRAMS):
        expect_exit, expect_out = PROGRAMS[name]
        # runtime.s FIRST: the program's `.align 4096` data block must be
        # the last thing in the image, or everything after it (the whole
        # runtime) would land in the non-executable data page.
        mods = [("runtime.s", runtime),
                ("main.s", read_src(os.path.join(HERE, "corpus", name, "main.s")))]
        say("--- %s: expect exit %d, stdout %r ---" % (name, expect_exit,
                                                       expect_out))
        gtext, gelf = gnu_link(mods)
        _, base, size = text_section(gelf)
        say("  GNU ld: .text at 0x%x, %d bytes" % (base, size))
        rt_text, rt_entry, rt_syms, rt_size = our_link(mods, base)
        if rt_size != size:
            say("  FAIL: size differs: ours %d vs GNU %d" % (rt_size, size))
            failures += 1
            continue
        if rt_text != gtext:
            first = next(i for i in range(len(rt_text)) if rt_text[i] != gtext[i])
            say("  FAIL: .text differs at byte %d (0x%x): ours 0x%02x vs GNU 0x%02x"
                % (first, base + first, rt_text[first], gtext[first]))
            failures += 1
            continue
        say("  .text byte-identical: %d bytes" % rt_size)
        gsyms = gnu_symbols(gelf)
        if rt_syms != gsyms:
            say("  FAIL: symbols differ:\n    ours: %s\n    GNU:  %s"
                % (sorted(rt_syms.items()), sorted(gsyms.items())))
            failures += 1
            continue
        say("  symbols match nm: entry %s at 0x%x" %
            ([k for k, v in rt_syms.items() if v == rt_entry] or ["?"], rt_entry))
        # Re-link at the runtime load address and wrap it with the W^X
        # split: the code page(s) stay R-X, the message bytes after the
        # `.align 4096` land in a separate RW- page (same trick as the B3
        # programs check).
        data_label = DATA_LABEL[name]
        base_run = elf64_writer.BASE + elf64_writer.SPLIT_CODE_OFF
        run_text, run_entry, run_syms, _ = link(mods, base=base_run)
        data_off = run_syms[data_label] - base_run
        our_elf_path = os.path.join(OUT, name + ".elf")
        blob = elf64_writer.emit_elf64(
            run_text, entry_off=run_entry - base_run, data_off=data_off)
        with open(our_elf_path, "wb") as f:
            f.write(blob)
        os.chmod(our_elf_path, 0o755)
        check_wx(name, blob, our_elf_path, data_off, run_entry)
        for label, elf_path in (("our", our_elf_path), ("GNU", gelf)):
            os.chmod(elf_path, 0o755)
            r = subprocess.run([elf_path], capture_output=True, timeout=300)
            if r.stdout != expect_out:
                say("  FAIL: %s ELF stdout %r != %r"
                    % (label, r.stdout, expect_out))
                failures += 1
            elif r.returncode != expect_exit:
                say("  FAIL: %s ELF exit %d != %d"
                    % (label, r.returncode, expect_exit))
                failures += 1
            else:
                say("  %s ELF ran: exit %d, stdout %r" % (label, r.returncode,
                                                          r.stdout))
    if failures:
        say("b4_check: FAIL (%d failures)" % failures)
        return 1
    say("b4_check: PASSED (%d programs)" % len(PROGRAMS))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
