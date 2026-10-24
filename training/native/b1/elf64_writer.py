#!/usr/bin/env python3
"""elf64_writer.py - B1: minimal ELF64 executable emitter (build script).

Emits a complete ET_EXEC ELF64 file for x86-64 Linux from raw machine-code
bytes: Ehdr + one PT_LOAD (RWX, base 0x400000) + code at file offset 0x78.
No section headers (the kernel loader only reads program headers; readelf
and objdump reconstruct everything from the phdr).

B1 deliverable: the 132-byte "exit 42" executable - the mission's fixed
pass target. Later parts (B2 assembler output, Part C engine) reuse
emit_elf64() with larger code payloads.
"""
import os
import struct

BASE = 0x400000
CODE_OFF = 64 + 56  # sizeof(Ehdr64) + sizeof(Phdr64) = 0x78

# W^X layout (data_off=...): [Ehdr 64B][2x Phdr 112B][pad to 4 KiB]
# [code ...]. Code starts on its own page at file offset/SPLIT_CODE_OFF
# 4096, so the R-X code segment ends exactly at a page boundary and the
# RW- data segment starts on the next one - the two PT_LOADs share no
# page, the same layout GNU ld produces with -z separate-code.
SPLIT_CODE_OFF = 4096


def _ehdr64(entry, phnum):
    e_ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    return e_ident + struct.pack(
        "<HHIQQQIHHHHHH",
        2,               # e_type    ET_EXEC
        62,              # e_machine EM_X86_64
        1,               # e_version
        entry,           # e_entry
        64,              # e_phoff
        0,               # e_shoff (no sections)
        0,               # e_flags
        64,              # e_ehsize
        56,              # e_phentsize
        phnum,           # e_phnum
        64,              # e_shentsize (GNU convention; shnum=0 so unused)
        0,               # e_shnum
        0,               # e_shstrndx
    )


def _phdr64(flags, offset, vaddr, size):
    return struct.pack(
        "<IIQQQQQQ",
        1,        # p_type   PT_LOAD
        flags,    # p_flags
        offset,   # p_offset
        vaddr,    # p_vaddr
        vaddr,    # p_paddr
        size,     # p_filesz
        size,     # p_memsz
        0x1000,   # p_align (vaddr % align == offset % align)
    )


def emit_elf64(code, base=BASE, entry_off=None, data_off=None):
    """Wrap raw x86-64 machine code in a loadable ELF64 executable.

    data_off=None (B1/B2/B3/B4 default, byte-stable): [Ehdr 64B]
    [Phdr 56B][code ...], ONE PT_LOAD RWX covering the whole file at
    `base`; e_entry = base + entry_off, entry_off defaulting to 0x78
    (start of code) - the B3 linker passes the real offset of _start
    when it is not the first byte of the linked image (a program linked
    in reverse module order starts elsewhere).

    data_off=<offset of the first data byte in `code`> (hardening):
    W^X split into TWO PT_LOADs - [0, data_off) R-X (code),
    [data_off, len(code)) RW- (data). The code must be assembled at
    `base` + SPLIT_CODE_OFF and data_off must be page aligned (put
    `.align 4096` before the data label in the source), otherwise the
    segments would share a page and the split would be a lie. Returns a
    4096 + len(code) byte file; e_entry = base + SPLIT_CODE_OFF +
    entry_off (entry_off default 0, i.e. _start at the top of code).
    """
    if data_off is None:
        if entry_off is None:
            entry_off = CODE_OFF
        size = CODE_OFF + len(code)
        return (_ehdr64(base + entry_off, 1)
                + _phdr64(7, 0, base, size) + code)
    if data_off <= 0 or data_off >= len(code):
        raise ValueError("data_off %r leaves no code or no data (%d bytes)"
                         % (data_off, len(code)))
    if data_off % 0x1000:
        raise ValueError("data_off %d is not page aligned - put .align 4096 "
                         "before the data label" % data_off)
    if entry_off is None:
        entry_off = 0
    code_end = SPLIT_CODE_OFF + data_off   # file offset of first data byte
    total = SPLIT_CODE_OFF + len(code)
    return (_ehdr64(base + SPLIT_CODE_OFF + entry_off, 2)
            + _phdr64(5, 0, base, code_end)          # R-X: header+pad+code
            + _phdr64(6, code_end, base + code_end, total - code_end)  # RW-
            + bytes(SPLIT_CODE_OFF - (64 + 112))     # pad header -> page
            + code)


def exit_program(code=42):
    """x86-64 machine code for exit(code) via raw syscall. 12 bytes."""
    return (b"\xb8\x3c\x00\x00\x00"                        # mov eax, 60 (sys_exit)
            + b"\xbf" + struct.pack("<I", code & 0xFFFFFFFF)   # mov edi, code
            + b"\x0f\x05")                                    # syscall


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "b1_exit42.elf"))
    ap.add_argument("--code", type=int, default=42)
    a = ap.parse_args()
    blob = emit_elf64(exit_program(a.code))
    with open(a.out, "wb") as f:
        f.write(blob)
    print("wrote %s: %d bytes, entry 0x%x" % (a.out, len(blob), BASE + CODE_OFF))


if __name__ == "__main__":
    main()
