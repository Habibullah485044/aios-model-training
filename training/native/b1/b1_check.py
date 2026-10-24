#!/usr/bin/env python3
"""b1_check.py - B1 pass check. Runs on Kaggle (Linux: file, readelf, objdump,
executing the ELF). Thresholds fixed before running, never loosened:
  1) emitted file is EXACTLY 132 bytes
  2) file(1): "ELF 64-bit LSB executable" and "x86-64"
  3) readelf -a: exit 0 with EMPTY stderr (no warnings); entry 0x400078;
     exactly one PT_LOAD, RWX, vaddr 0x400000, offset 0
  4) objdump -d -M intel: disassembly is exactly the 3 expected
     instructions (mov eax,0x3c / mov edi,0x2a / syscall)
  5) running it exits with status 42
  6) independent struct-level parse of every header field in Python
Exit codes: 0 pass, 1 check failed, 2 environment problem.
Results always land in b1/b1_results.txt (Kaggle empty-log quirk, D-003).
"""
import os
import re
import stat
import struct
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "b1_results.txt")
ELF = os.path.join(HERE, "b1_exit42.elf")

REPORT = []


def say(s):
    print(s, flush=True)
    REPORT.append(s)


def run(cmd, **kw):
    say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, **kw)
    return r.returncode, r.stdout or "", r.stderr or ""


def main():
    sys.path.insert(0, HERE)
    from elf64_writer import BASE, CODE_OFF, emit_elf64, exit_program

    say("=== B1 pass check: 132-byte exit42 ELF64 ===")
    blob = emit_elf64(exit_program(42))
    with open(ELF, "wb") as f:
        f.write(blob)
    say("emitted %s (%d bytes)" % (os.path.basename(ELF), len(blob)))
    assert len(blob) == 132, "size %d != 132" % len(blob)

    # (6) struct-level parse (independent of readelf)
    assert blob[:4] == b"\x7fELF"
    (e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags,
     e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum,
     e_shstrndx) = struct.unpack("<HHIQQQIHHHHHH", blob[16:64])
    assert e_type == 2 and e_machine == 62 and e_version == 1
    assert e_entry == BASE + CODE_OFF == 0x400078, hex(e_entry)
    assert e_phoff == 64 and e_phnum == 1 and e_phentsize == 56
    assert e_shoff == 0 and e_shnum == 0
    (p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz,
     p_align) = struct.unpack("<IIQQQQQQ", blob[64:120])
    assert p_type == 1 and p_flags == 7 and p_offset == 0
    assert p_vaddr == p_paddr == BASE and p_filesz == p_memsz == 132
    assert p_align == 0x1000 and len(blob) == 132
    say("struct parse: e_entry=0x%x PT_LOAD RWX base=0x%x filesz=%d OK"
        % (e_entry, p_vaddr, p_filesz))

    # (2) file(1)
    rc, out, err = run(["file", ELF])
    say(out.strip())
    assert rc == 0, "file(1) failed"
    assert "ELF 64-bit LSB executable" in out, out
    assert "x86-64" in out, out

    # (3) readelf: clean parse, entry point, one RWX PT_LOAD
    rc, out, err = run(["readelf", "-a", "-W", ELF])
    assert rc == 0, "readelf -a exit %d" % rc
    assert err == "", "readelf warnings: %r" % err
    m = re.search(r"Entry point address:\s*(0x[0-9a-f]+)", out)
    assert m and m.group(1) == "0x400078", out
    loads = [l for l in out.splitlines() if "LOAD" in l]
    assert len(loads) == 1, loads
    say("readelf -a: clean (empty stderr), entry 0x400078, 1 PT_LOAD")

    # (4) objdump -M intel: exact instruction sequence. The ELF is
    # sectionless (that is what keeps it at 132 bytes), so objdump -d has
    # no .text to walk - disassemble the exact bytes at the entry offset
    # as raw x86-64 instead.
    code = blob[CODE_OFF:]
    assert len(code) == 12, len(code)
    raw = ELF + ".code.bin"
    with open(raw, "wb") as f:
        f.write(code)
    rc, out, err = run(["objdump", "-D", "-b", "binary", "-m", "i386:x86-64",
                        "-M", "intel", raw])
    assert rc == 0, "objdump -b binary exit %d" % rc
    insns = []
    for line in out.splitlines():
        m = re.match(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s)+\s*(.+?)\s*$", line)
        if m:
            insns.append(re.sub(r"\s+", " ", m.group(1)))
    say("objdump -b binary -M intel (12 code bytes): %s" % insns)
    assert insns == ["mov eax,0x3c", "mov edi,0x2a", "syscall"], insns

    # (5) run it: exit status 42
    os.chmod(ELF, os.stat(ELF).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    rc, out, err = run([ELF])
    say("run: exit status = %d" % rc)
    assert rc == 42, "exit status %d != 42" % rc

    say("B1 PASSED")


def guarded_main():
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:
        say("B1 FAILED (exception):")
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
