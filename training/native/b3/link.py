#!/usr/bin/env python3
"""link.py - B3: our linker.

Places assembled modules exactly the way GNU ld does and resolves the
cross-module relocations, so the linked .text of a multi-module program
comes out byte identical to `as` + `ld`.

Layout rules (measured against binutils 2.46 and 2.47 - WSL Ubuntu and
the Windows PE build; see DECISIONS.md D-031..):
  - modules are concatenated in list order (verified with the order
    reversed);
  - module i starts at the first offset at or after the end of module i-1
    that is a multiple of the module's alignment. That alignment is gas's
    sh_addralign: the largest `.align N` in the module, 1 when it has
    none (measured with readelf);
  - the gap between two modules is filled with GNU ld's nop fill
    (asm.gap_fill: 10-byte nops first, canonical remainder last, measured
    k=0..63) - NOT the sequence gas pads `.align` with (asm.align_fill);
  - a reference from one module to another resolves to the target
    label's linked address; branches to labels in other modules always
    take the rel32 form (GNU as cannot know the distance while
    assembling, so it never emits rel8 for them either);
  - visibility is ELF's: only `.global`/`.globl` labels are visible to
    other modules, a module's own label shadows an external one, and an
    undefined or multiply defined name is a link error.

API
  link(sources, base=BASE) -> (code, entry, symbols, mods)
    sources  [(name, text), ...] module sources in link order
    code     bytes of the whole linked .text
    entry    linked address of _start
    symbols  {label: linked address} for every label defined exactly
             once across the modules
    mods     the Module objects: name / start / size / align / labels /
             stmt_map (for error reporting)
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NATIVE, "b2"))

from asm import Assembler, AsmError, gap_fill  # noqa: E402  (path above)

BASE = 0x401000  # where GNU ld's default script puts .text


class LinkError(Exception):
    pass


class Module(object):
    """One input module: source plus everything the linker learns about it."""

    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.label_names = set()
        self.exports = set()     # names named by .global/.globl
        self.align = 1           # gas sh_addralign = the largest .align
        self.start = 0           # offset of the module in the linked .text
        self.code = b""
        self.labels = {}         # label -> offset within the module
        self.stmt_map = []       # (lineno, source, offset, size) per stmt


def scan(m):
    """Label names, .global exports and the section alignment of one module."""
    a = Assembler()
    stmts, label_at = a.parse(m.text)
    m.label_names = set(label_at)
    for st in stmts:
        if st.kind == "align" and st.amount > m.align:
            m.align = st.amount
        if st.kind == "ignore" and st.mnem in (".global", ".globl"):
            for name in re.findall(r"[A-Za-z_.][A-Za-z0-9_.]*",
                                   st.text[len(st.mnem):]):
                m.exports.add(name)
    return m


def collect_exports(mods):
    """{label: defining module} for every .global label, ELF's rule: two
    modules exporting the same name is a multiple-definition error."""
    exported = {}
    for m in mods:
        for name in sorted(m.exports & m.label_names):
            if name in exported:
                raise LinkError("multiple definition of %r (%s and %s)"
                                % (name, exported[name].name, m.name))
            exported[name] = m
    return exported


def link(sources, base=BASE):
    mods = []
    for i, src in enumerate(sources):
        if isinstance(src, (tuple, list)):
            name, text = src
        else:
            name, text = "m%d" % (i + 1), src
        mods.append(scan(Module(name, text)))
    exported = collect_exports(mods)

    # Pass A: sizes and module-relative label offsets. External values
    # cannot change an encoding width (every cross-module fixup is
    # fixed-width, and a branch to an external label always takes the
    # rel32 form), so a placeholder address of 0 answers every external
    # reference here.
    for m in mods:
        extern = dict((n, 0) for n, owner in exported.items()
                      if owner is not m and n not in m.label_names)
        try:
            a = Assembler(0, extern)
            code, _ = a.assemble(m.text)
        except AsmError as e:
            raise LinkError("%s: %s" % (m.name, e))
        m.code = code
        m.labels = a.labels
        m.stmt_map = a.stmt_map

    # Placement: each module starts at an absolute address that is a
    # multiple of its own alignment (ld aligns sections in address
    # space, so the rule holds for any base); gaps belong to GNU ld's
    # nop fill.
    addr = base
    for m in mods:
        if m.align > 1:
            addr = (addr + m.align - 1) & ~(m.align - 1)
        m.start = addr - base
        addr += len(m.code)

    # Linked addresses of the exported labels.
    gsyms = {}
    for name, m in exported.items():
        gsyms[name] = base + m.start + m.labels[name]

    # Report table: every label defined exactly once (a local label may
    # legally repeat across modules and then has no single address).
    symbols, seen = {}, {}
    for m in mods:
        for name, o in m.labels.items():
            seen[name] = seen.get(name, 0) + 1
            symbols[name] = base + m.start + o
    for name, n in seen.items():
        if n > 1:
            del symbols[name]

    # Pass B: re-assemble with the real external addresses and splice the
    # modules together with ld's gap fill between them.
    out = bytearray()
    for m in mods:
        out += gap_fill(m.start - len(out))
        extern = dict((n, addr) for n, addr in gsyms.items()
                      if n not in m.label_names)
        try:
            a = Assembler(base + m.start, extern)
            code, _ = a.assemble(m.text)
        except AsmError as e:
            raise LinkError("%s: %s" % (m.name, e))
        if len(code) != len(m.code):
            raise LinkError("module %s changed size between passes" % m.name)
        out += code

    starts = [m for m in mods if "_start" in m.label_names]
    if not starts:
        raise LinkError("no _start in any module")
    if len(starts) > 1:
        raise LinkError("_start defined in %s"
                        % ", ".join(m.name for m in starts))
    entry = base + starts[0].start + starts[0].labels["_start"]
    return bytes(out), entry, symbols, mods


def main():
    ap = argparse.ArgumentParser(
        description="B3 linker: link .s modules into one .text image")
    ap.add_argument("modules", nargs="+", help="module sources, in link order")
    ap.add_argument("-o", "--out", help="write the linked .text here")
    ap.add_argument("--base", default=hex(BASE),
                    help="linked address of .text (default %s)" % hex(BASE))
    a = ap.parse_args()
    sources = []
    for path in a.modules:
        with open(path, "r", encoding="utf-8") as f:
            sources.append((os.path.basename(path), f.read()))
    code, entry, symbols, mods = link(sources, base=int(a.base, 0))
    for m in mods:
        print("module %-8s +0x%04x  size %-4d align %d"
              % (m.name, m.start, len(m.code), m.align))
    print("linked .text: %d bytes, _start at 0x%x" % (len(code), entry))
    for name in sorted(symbols):
        print("  %-14s 0x%x" % (name, symbols[name]))
    if a.out:
        with open(a.out, "wb") as f:
            f.write(code)
        print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
