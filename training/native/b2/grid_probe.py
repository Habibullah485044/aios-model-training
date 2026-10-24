#!/usr/bin/env python3
"""grid_probe.py - exhaustive auto-generated differential grid for B2.

NOT the mission pass check (that is b2_check.py over the corpus: linked on
Kaggle and executed). This is the encoding sweep: every supported
mnemonic x every operand form x the register and addressing variations
that break encoders - r8-r15, xmm8-15, ymm8-15, rsp/r12/rbp/r13 bases,
disp8/disp32, index*scale (incl. r12/r13 indexes), rip-relative, symbolic
absolute operands (bare label = memory; `offset`/`movabs` = address),
64-bit immediates - assembled by BOTH GNU as and asm.py and compared
byte for byte. Target: 0 differences.

Method (the corpus check minus the linker, which is Kaggle-only): GNU as
resolves same-section references (branches, [rip+label]) at assembly
time, so those bytes are compared directly. The only fields a
relocatable object leaves unresolved are absolute symbol references; for
each one this harness reads the relocation (offset, kind, symbol,
addend) and the symbol's section offset from GAS's own object file,
computes the value the linker would write, patches it into GAS's bytes,
and then requires full byte identity with our output - and requires our
absolute fixups to sit at exactly the same offsets. Every byte is
therefore checked, and the expected relocation values come from GAS's
layout, not from our own label arithmetic.

The grid only emits operand forms GNU as itself accepts (forms GAS
rejects - memory destinations for most VEX mnemonics, the EVEX-only
shift forms - are outside the documented subset). A GAS syntax error,
an AsmError, or any byte difference fails the run with the offending
source line. GNU as pads .text to its 16-byte alignment with nops; that
trailing padding (at most 15 bytes) is tolerated and verified to be
nops, exactly like b2_check.py.

Run anywhere GNU as exists: on the build PC point AIOS_BINU at a local
binutils directory (or have `as` on PATH), or on Kaggle (Linux). The
authoritative mission evidence remains b2_check.py; this exists to find
encoding bugs across the whole operand space without Kaggle round trips.
Exit codes: 0 = 0 differences, 1 = difference or assembly error, 2 =
environment problem (no GNU as).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import asm  # noqa: E402  (the assembler under test)

# --------------------------------------------------------------------------
# operand pools
# --------------------------------------------------------------------------
GP64 = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"]
GP32 = [r[:-1] + "d" if r != "rsp" else "esp" for r in GP64]
GP32 = ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
        "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"]
GP8 = ["al", "cl", "dl", "bl", "spl", "bpl", "sil", "dil",
       "r8b", "r9b", "r10b", "r11b", "r12b", "r13b", "r14b", "r15b"]
# src subsets: every width sees low regs, rsp/rbp specials and r8-r15
SUB64 = ["rax", "rcx", "rbx", "rsp", "rbp", "rsi", "rdi",
         "r8", "r9", "r12", "r13", "r15"]
SUB32 = ["eax", "ecx", "ebx", "esp", "ebp", "esi", "edi",
         "r8d", "r9d", "r12d", "r13d", "r15d"]
SUB8 = ["al", "cl", "bl", "spl", "bpl", "sil", "dil",
        "r8b", "r9b", "r12b", "r13b", "r15b"]
XMM = ["xmm%d" % i for i in range(16)]
YMM = ["ymm%d" % i for i in range(16)]
VSUB = [0, 7, 8, 15]          # vector reg indexes used in pair forms
VSUB2 = [0, 8]                # narrower vvvv/rm pool

BASES = ("rsp", "r12", "rbp", "r13")           # the SIB/REX specials
IDX = ("rax", "rbx", "r8", "r9", "r12", "r13")  # incl. bit3-set indexes


def mem_variants(size_kw, full=True):
    """Addressing-mode spellings: bases x disp8/disp32, index*scale, rip."""
    out = []
    for b in BASES:
        out.append("[%s]" % b)
        out.append("[%s+8]" % b)
        out.append("[%s-8]" % b)              # negative disp8
        out.append("[%s+0x1234]" % b)         # disp32
        out.append("[%s-0x1234]" % b)         # negative disp32
    if full:
        for b in ("rsp", "rbp"):
            for i in IDX:
                for s in (1, 2, 4, 8):
                    out.append("[%s+%s*%d]" % (b, i, s))
                    out.append("[%s+%s*%d+32]" % (b, i, s))
    out.append("[rip+grid_data]")
    out.append("[rip+grid_data+8]")
    if size_kw:
        return ["%s ptr %s" % (size_kw, m) for m in out]
    return out


IMM64 = ["1", "-1", "0x7f", "0x80", "-128", "0xff",
         "0x7fffffff", "0x80000000", "0xffffffff", "-0x80000000",
         "0x12345678", "0x123456789abcdef0", "0x8000000000000000",
         "0x7fffffffffffffff", "0xffffffff80000000", "0x0000000100000000"]
# 64-bit destinations only take an imm32 whose sign-extension round-trips
# (GNU as rejects the rest), so arith/test/qword-store immediates use this
IMM_ARITH = ["1", "-1", "0x7f", "0x80", "-128", "0xff",
             "0x7fffffff", "-0x80000000", "0x12345678", "-0x12345678",
             "0xffffffff80000000"]
# 32-bit destinations take the full raw imm32 range
IMM_R32 = ["1", "-1", "0x7fffffff", "0x80000000", "0xffffffff",
           "-0x80000000", "0x12345678"]
IMM8 = ["0", "1", "3", "17", "0x7f", "0x80", "0xff"]
SHIMM = ["0", "1", "2", "3", "17", "63"]
# per-mnemonic memory operand width for scalar SSE/AVX forms
SCALAR = {"movss": "dword", "movsd": "qword", "addss": "dword",
          "addsd": "qword", "subss": "dword", "subsd": "qword",
          "mulss": "dword", "mulsd": "qword", "divss": "dword",
          "divsd": "qword", "minss": "dword", "maxss": "dword",
          "cvtss2sd": "dword", "cvtsd2ss": "qword", "ucomiss": "dword",
          "ucomisd": "qword", "comiss": "dword", "comisd": "qword"}


def generate():
    """The grid: list of source lines (instructions and directives).

    Bare-label operands (`mov rax, grid_data`) are generated for BOTH
    object formats: GNU as's Intel syntax makes a bare label a MEMORY
    operand on ELF and PE/COFF alike (an absolute load with a 32-bit
    relocation), and `offset`/`movabs` spell the address forms. Earlier
    grid versions skipped the bare-label cases on PE/COFF because our
    assembler wrongly emitted movabs there; the assembler now matches
    GAS, so the cases are unconditional."""
    L = []
    add = L.append

    def branches():
        short_cc = ["e", "ne", "b", "ae", "s", "l", "ge", "g"]
        for i, cc in enumerate(short_cc):
            add("j%s jc_%d" % (cc, i))
            add("jc_%d:" % i)
        add("jmp jfar_far")
        add(".space 200")
        add("jfar_far:")
        add("call cal_0")
        add("cal_0:")
        for i, cc in enumerate(short_cc):
            add("j%s jnear_%d" % (cc, i))
            add(".space 200")
            add("jnear_%d:" % i)

    def gp():
        # mov r,r (every dest x the subset, all three widths)
        for dw, sw, pool in ((GP64, SUB64, None), (GP32, SUB32, None),
                             (GP8, SUB8, None)):
            for d in dw:
                for s in sw:
                    add("mov %s, %s" % (d, s))
        # mov r,mem and mov mem,r (full addressing sweep on qword)
        for w, kw in ((SUB64[:4] + ["r9", "r12", "r15"], "qword"),
                      (SUB32[:4] + ["r9d", "r12d", "r15d"], "dword"),
                      (SUB8[:4] + ["r9b", "r12b", "r15b"], "byte")):
            mems = mem_variants(kw, full=(kw == "qword"))
            mems_small = mem_variants(kw, full=False)
            use = mems if kw == "qword" else mems_small
            for d in w:
                for m in use:
                    add("mov %s, %s" % (d, m))
                    add("mov %s, %s" % (m, d))
        # mov r,imm (64-bit immediates incl. movabs and round-trip cases)
        for d in ("rax", "rcx", "r8", "r12", "r15"):
            for v in IMM64:
                add("mov %s, %s" % (d, v))
        for d in ("eax", "ecx", "r8d", "r12d", "r15d"):
            for v in IMM_R32:
                add("mov %s, %s" % (d, v))
        for d in ("al", "cl", "spl", "r8b", "r15b"):
            for v in ("0", "1", "0x7f", "0x80", "0xff", "-1"):
                add("mov %s, %s" % (d, v))
        # mov mem,imm (qword/dword take an imm32; byte takes an imm8)
        for m in ("qword ptr [rsp]", "qword ptr [r12+8]", "qword ptr [rbp-0x10]",
                  "qword ptr [rip+grid_data]"):
            for v in ("1", "-1", "0x12345678", "-0x12345678"):
                add("mov %s, %s" % (m, v))
        for m in ("dword ptr [rsp+4]", "dword ptr [rip+grid_data]"):
            for v in ("1", "-1", "0x12345678", "0x80000000", "0xffffffff"):
                add("mov %s, %s" % (m, v))
        for m in ("byte ptr [rsp]", "byte ptr [rip+grid_data]"):
            for v in ("1", "-1", "0xf0", "0xff"):
                add("mov %s, %s" % (m, v))
        # lea (every addressing form; size-less memory)
        for d in ("rax", "r9", "r13", "r15", "eax"):
            for m in mem_variants(None):
                add("lea %s, %s" % (d, m))
        # arith reg/mem/imm forms
        for mn in ("add", "sub", "and", "or", "xor", "cmp"):
            for d in GP64:
                for s in SUB64:
                    add("%s %s, %s" % (mn, d, s))
            for d in ("rax", "r9", "r13", "r15"):
                for m in mem_variants("qword"):
                    add("%s %s, %s" % (mn, d, m))
                    add("%s %s, %s" % (mn, m, d))
            for d in ("eax", "r9d", "r15d"):
                for m in mem_variants("dword", full=False):
                    add("%s %s, %s" % (mn, d, m))
            for d in ("rax", "r8", "r13"):
                for v in IMM_ARITH:
                    add("%s %s, %s" % (mn, d, v))
            for m in ("qword ptr [rsp]", "qword ptr [rip+grid_data]"):
                for v in ("1", "-1", "0x1234", "-0x1234", "0x12345678"):
                    add("%s %s, %s" % (mn, m, v))
            for m in ("dword ptr [rbp-8]", "dword ptr [rip+grid_data]"):
                for v in ("1", "-1", "0x1234", "0x12345678", "0x80000000"):
                    add("%s %s, %s" % (mn, m, v))
        # byte arith (the 8-bit opcode families)
        for mn in ("add", "sub", "and", "or", "xor", "cmp"):
            for d in GP8:
                add("%s %s, 5" % (mn, d))
            for m in ("byte ptr [rsp]", "byte ptr [rip+grid_data]",
                      "byte ptr [r13+1]"):
                add("%s %s, 0xf0" % (mn, m))
        # test
        for d in GP64:
            for s in ("rax", "rbx", "r8", "r12", "r15"):
                add("test %s, %s" % (d, s))
        for d in ("rax", "r8", "r15"):
            for v in ("1", "-1", "0x12345678", "-0x12345678",
                      "0x7fffffff", "-0x80000000"):
                add("test %s, %s" % (d, v))
        for m in ("qword ptr [rsp]", "qword ptr [rip+grid_data]",
                  "dword ptr [r12+4]", "byte ptr [rsp]"):
            add("test %s, 5" % m)
        # imul
        for d in GP64:
            for s in SUB64:
                add("imul %s, %s" % (d, s))
        for d in ("rax", "r9", "r15"):
            for m in mem_variants("qword", full=False):
                add("imul %s, %s" % (d, m))
            for s in ("rax", "r8"):
                for v in ("5", "0x1234", "-1", "0x7fffffff"):
                    add("imul %s, %s, %s" % (d, s, v))
        for r in GP64:
            add("imul %s" % r)
        # inc/dec/neg/not
        for mn in ("inc", "dec", "neg", "not"):
            for r in GP64:
                add("%s %s" % (mn, r))
            for m in ("qword ptr [rsp]", "qword ptr [rip+grid_data]",
                      "dword ptr [r13]", "byte ptr [rsp+1]"):
                add("%s %s" % (mn, m))
        # shifts
        for mn in ("shl", "shr", "sar"):
            for r in GP64:
                for v in SHIMM:
                    add("%s %s, %s" % (mn, r, v))
                add("%s %s, cl" % (mn, r))
            for m in ("qword ptr [rsp]", "qword ptr [rip+grid_data]",
                      "dword ptr [r12]"):
                add("%s %s, 3" % (mn, m))
        # movzx/movsx/movsxd
        for mn in ("movzx", "movsx"):
            for d in ("rax", "r8", "r15", "eax", "r9d"):
                for s in ("al", "cl", "spl", "dil", "r8b", "r15b"):
                    add("%s %s, %s" % (mn, d, s))
                for m in ("byte ptr [rsp]", "byte ptr [rip+grid_data]",
                          "word ptr [rsp]"):
                    add("%s %s, %s" % (mn, d, m))
        for d in ("rax", "r8", "r15"):
            for s in ("eax", "ecx", "r8d", "r15d"):
                add("movsxd %s, %s" % (d, s))
            add("movsxd %s, dword ptr [rsp]" % d)
        # push/pop/setcc/misc
        for r in GP64:
            add("push %s" % r)
        for r in reversed(GP64):
            add("pop %s" % r)
        for cc in ("e", "ne", "b", "ae", "s", "ns", "l", "ge", "g", "le"):
            for r in GP8:
                add("set%s %s" % (cc, r))
        for s in ("nop", "int3", "syscall", "cpuid", "pause", "cld", "std",
                  "ret"):
            add(s)

    def sse():
        lr = ["movaps", "movups", "movapd", "movupd", "movdqa", "movdqu",
              "addps", "addpd", "subps", "subpd", "mulps", "mulpd",
              "divps", "divsd", "minps", "maxps", "andps", "andpd",
              "andnps", "andnpd", "orps", "orpd", "xorps", "xorpd",
              "paddb", "paddw", "paddd", "paddq", "psubb", "psubw",
              "psubd", "psubq", "pand", "pandn", "por", "pxor", "pshufb",
              "punpcklbw", "punpcklwd", "punpckldq", "punpcklqdq",
              "cvtss2sd", "cvtsd2ss", "cvtdq2ps", "cvtps2dq", "cvttps2dq",
              "ucomiss", "ucomisd", "comiss", "comisd"]
        for mn in lr:
            for d in XMM:
                for s in (XMM[0], XMM[7], XMM[8], XMM[15]):
                    add("%s %s, %s" % (mn, d, s))
            mems = mem_variants(SCALAR.get(mn, "xmmword"), full=(mn in
                                 ("movaps", "movups", "movdqu", "pshufb",
                                  "pand", "pxor", "por")))
            for d in (XMM[0], XMM[8], XMM[15]):
                for m in mems:
                    add("%s %s, %s" % (mn, d, m))
        # store forms (mem <- reg) for the rl mnemonics
        for mn, kw in (("movaps", "xmmword"), ("movups", "xmmword"),
                       ("movdqa", "xmmword"), ("movdqu", "xmmword"),
                       ("movss", "dword"), ("movsd", "qword")):
            for m in mem_variants(kw, full=(kw == "xmmword")):
                add("%s %s, %s" % (mn, m, XMM[0]))
            add("%s %s, %s" % (mn, mem_variants(kw, full=False)[1], XMM[15]))
        # movss/movsd loads
        for mn, kw in (("movss", "dword"), ("movsd", "qword")):
            for d in (XMM[0], XMM[8], XMM[15]):
                for m in mem_variants(kw, full=False):
                    add("%s %s, %s" % (mn, d, m))
                add("%s %s, %s" % (mn, d, XMM[7]))
        # movd/movq GP exchanges
        for s in ("eax", "ecx", "r8d", "r15d"):
            add("movd xmm0, %s" % s)
            add("movd %s, xmm8" % s)
        for m in ("dword ptr [rsp]", "dword ptr [rip+grid_data]"):
            add("movd xmm1, %s" % m)
            add("movd %s, xmm9" % m)
        for s in ("rax", "rcx", "r8", "r12", "r15"):
            add("movq xmm0, %s" % s)
            add("movq %s, xmm8" % s)
        for m in ("qword ptr [rsp]", "qword ptr [rip+grid_data]"):
            add("movq xmm1, %s" % m)
            add("movq %s, xmm9" % m)
        add("movq xmm2, xmm3")
        add("movq xmm15, xmm14")
        # cvtsi2ss/sd with every source width form
        for mn in ("cvtsi2ss", "cvtsi2sd"):
            for d in ("xmm0", "xmm8", "xmm15"):
                for s in ("eax", "r8d", "r15d", "rax", "r8", "r15"):
                    add("%s %s, %s" % (mn, d, s))
                for m in ("dword ptr [rsp]", "qword ptr [rsp]",
                          "qword ptr [rip+grid_data]"):
                    add("%s %s, %s" % (mn, d, m))
        for mn in ("cvtss2si", "cvttsd2si", "cvttss2si"):
            for d in ("eax", "r8d", "r15d", "rax", "r8", "r15"):
                add("%s %s, xmm0" % (mn, d))
                add("%s %s, xmm15" % (mn, d))
        # group/immediate SSE shifts (register destination only: GNU as
        # rejects memory destinations for these in Intel syntax)
        for mn, imms in (("pslld", ("2", "8")), ("psrld", ("2", "8")),
                         ("psllq", ("1", "63")), ("psrlq", ("1", "63")),
                         ("pslldq", ("3",)), ("psrldq", ("3",))):
            for d in (XMM[0], XMM[7], XMM[8], XMM[15]):
                for v in imms:
                    add("%s %s, %s" % (mn, d, v))
        for mn in ("pslld", "psrld"):
            for d in (XMM[0], XMM[8]):
                for s in (XMM[0], XMM[7], XMM[15]):
                    add("%s %s, %s" % (mn, d, s))
        # mask extract (GNU as's Intel syntax takes a register source only
        # for these; our assembler also accepts a memory one)
        for mn in ("movmskps", "movmskpd"):
            for d in ("eax", "r8d", "r15d"):
                add("%s %s, xmm0" % (mn, d))
                add("%s %s, xmm15" % (mn, d))
        for d in ("eax", "ecx", "r8d", "r15d"):
            for s in ("xmm0", "xmm8", "xmm15", "ymm0", "ymm8", "ymm15"):
                add("vpmovmskb %s, %s" % (d, s))

    def avx():
        # moves
        for mn in ("vmovups", "vmovdqu"):
            for d in XMM:
                for s in (XMM[0], XMM[7], XMM[8], XMM[15]):
                    add("%s %s, %s" % (mn, d, s))
            for d in (XMM[0], XMM[8], XMM[15]):
                for m in mem_variants("xmmword", full=True):
                    add("%s %s, %s" % (mn, d, m))
                for m in mem_variants(None, full=False):
                    add("%s %s, %s" % (mn, d, m))
            for m in mem_variants("xmmword", full=False):
                add("%s %s, %s" % (mn, m, XMM[0]))
                add("%s %s, %s" % (mn, m, XMM[15]))
            for d in YMM:
                for s in (YMM[0], YMM[7], YMM[8], YMM[15]):
                    add("%s %s, %s" % (mn, d, s))
            for d in (YMM[0], YMM[8], YMM[15]):
                for m in mem_variants("ymmword", full=True):
                    add("%s %s, %s" % (mn, d, m))
            for m in mem_variants("ymmword", full=False):
                add("%s %s, %s" % (mn, m, YMM[0]))
        # broadcast
        for d in XMM:
            for s in (XMM[0], XMM[8], XMM[15]):
                add("vbroadcastss %s, %s" % (d, s))
        for d in YMM:
            add("vbroadcastss %s, %s" % (d, XMM[8]))
        for m in mem_variants("dword", full=False):
            add("vbroadcastss xmm0, %s" % m)
            add("vbroadcastss ymm15, %s" % m)
        # vpshufb (3-operand)
        for d in ("xmm0", "xmm7", "xmm8", "xmm15"):
            for v in ("xmm0", "xmm8"):
                for r in ("xmm0", "xmm7", "xmm15"):
                    add("vpshufb %s, %s, %s" % (d, v, r))
                for m in mem_variants("xmmword", full=False):
                    add("vpshufb %s, %s, %s" % (d, v, m))
        for d in ("ymm0", "ymm8", "ymm15"):
            for v in ("ymm0", "ymm8"):
                for r in ("ymm0", "ymm15"):
                    add("vpshufb %s, %s, %s" % (d, v, r))
                for m in mem_variants("ymmword", full=False):
                    add("vpshufb %s, %s, %s" % (d, v, m))
        # zero extends
        for d in XMM:
            for s in (XMM[0], XMM[8], XMM[15]):
                add("vpmovzxbd %s, %s" % (d, s))
                add("vpmovzxwd %s, %s" % (d, s))
        for d in YMM:
            for s in (XMM[0], XMM[8], XMM[15]):
                add("vpmovzxbd %s, %s" % (d, s))
                add("vpmovzxwd %s, %s" % (d, s))
        for d in (XMM[0], XMM[8], XMM[15]):
            for m in mem_variants("dword", full=False):
                add("vpmovzxbd %s, %s" % (d, m))
            for m in mem_variants("qword", full=False):
                add("vpmovzxwd %s, %s" % (d, m))
        for d in (YMM[0], YMM[8], YMM[15]):
            for m in mem_variants("qword", full=False):
                add("vpmovzxbd %s, %s" % (d, m))
            for m in mem_variants("xmmword", full=False):
                add("vpmovzxwd %s, %s" % (d, m))
        # 3-operand logical/arithmetic/FMA
        for mn in ("vpand", "vpor", "vpxor", "vaddps", "vmulps",
                   "vfmadd231ps", "vpcmpeqb"):
            for d in ("xmm0", "xmm7", "xmm8", "xmm15"):
                for v in ("xmm0", "xmm8"):
                    for r in ("xmm0", "xmm7", "xmm15"):
                        add("%s %s, %s, %s" % (mn, d, v, r))
                    for m in mem_variants("xmmword", full=False):
                        add("%s %s, %s, %s" % (mn, d, v, m))
            for d in ("ymm0", "ymm8", "ymm15"):
                for v in ("ymm0", "ymm8"):
                    for r in ("ymm0", "ymm15"):
                        add("%s %s, %s, %s" % (mn, d, v, r))
                    for m in mem_variants("ymmword", full=False):
                        add("%s %s, %s, %s" % (mn, d, v, m))
        # shifts: immediate form (register source) and count form
        for mn in ("vpslld", "vpsrlw"):
            for d in ("xmm0", "xmm7", "xmm8", "xmm15"):
                for s in ("xmm0", "xmm8"):
                    for v in IMM8:
                        add("%s %s, %s, %s" % (mn, d, s, v))
            for d in ("ymm0", "ymm8", "ymm15"):
                for v in ("0", "1", "16", "255"):
                    add("%s %s, ymm0, %s" % (mn, d, v))
            # count form: the count operand is always 128-bit (xmm or
            # xmmword PTR) even with a ymm destination, per GNU as
            for d in ("xmm0", "xmm8", "xmm15"):
                for v in ("xmm0", "xmm7", "xmm15"):
                    add("%s %s, %s, %s" % (mn, d, d, v))
                    add("%s %s, %s, %s" % (mn, d, v, v))
                for m in mem_variants("xmmword", full=False):
                    add("%s %s, %s, %s" % (mn, d, d, m))
            for d in ("ymm0", "ymm8", "ymm15"):
                for v in ("xmm0", "xmm8", "xmm15"):
                    add("%s %s, %s, %s" % (mn, d, d, v))
                for m in mem_variants("xmmword", full=False):
                    add("%s %s, %s, %s" % (mn, d, d, m))
        # lane extracts / permutes (trailing imm8)
        for d in ("xmm0", "xmm8", "xmm15"):
            for s in ("ymm0", "ymm8", "ymm15"):
                for v in ("0", "1", "255"):
                    add("vextractf128 %s, %s, %s" % (d, s, v))
        for m in mem_variants("xmmword", full=False):
            add("vextractf128 %s, ymm8, 1" % m)
        for d in ("ymm0", "ymm8", "ymm15"):
            for v in ("ymm0", "ymm8"):
                for r in ("ymm0", "ymm8", "ymm15"):
                    for k in ("0", "1", "0x31", "255"):
                        add("vperm2f128 %s, %s, %s, %s" % (d, v, r, k))
                for m in mem_variants("ymmword", full=False):
                    add("vperm2f128 %s, %s, %s, 1" % (d, v, m))
        add("vzeroupper")

    def data():
        add(".byte 0x11, 0x22, 0x33")
        add(".word 0x4455, 0x6677")
        add(".long 0x89abcdef, 0x12345678")
        add(".quad 0x1122334455667788, -1, 0x8000000000000000")
        add(".long grid_data")
        add(".quad grid_data")
        add(".quad grid_data+8")
        add(".space 17")
        # Bare-label operands: in GNU as's Intel syntax a bare label is
        # a MEMORY operand - an absolute address (mod=00 SIB 04 25 +
        # a disp32 with a 32-bit relocation) - on ELF and PE/COFF alike.
        for r in ("rax", "rcx", "r9", "r12", "r15"):
            add("mov %s, grid_data" % r)
            add("mov %s, grid_data+8" % r)
        for r in ("eax", "ebx", "r8d", "r15d"):
            add("mov %s, grid_data" % r)
        for r in ("al", "r8b"):
            add("mov %s, grid_data" % r)
        add("mov rax, [grid_data]")
        add("mov r15, qword ptr [grid_data]")
        add("mov qword ptr [grid_data], r15")
        add("mov dword ptr [grid_data], r9d")
        add("mov byte ptr [grid_data], al")
        # `offset label` puts the ADDRESS into an immediate
        for r in ("rax", "r8", "r15"):
            add("mov %s, offset grid_data" % r)
        for r in ("eax", "r8d", "r15d"):
            add("mov %s, offset grid_data" % r)
        # the 8-bit relocation of a byte destination only assembles when
        # the address fits the field, so those cases reference _start
        # (offset 0) rather than grid_data at the section's end
        for r in ("al", "r8b"):
            add("mov %s, offset _start" % r)
        add("movabs rax, offset grid_data")
        add("movabs r8, offset grid_data")
        # movabs: a 64-bit immediate, or an absolute moffs64 access to
        # the accumulator (loads and stores)
        add("movabs rax, grid_data")
        add("movabs rax, grid_data+8")
        add("movabs eax, grid_data")
        add("movabs al, grid_data")
        add("movabs qword ptr [grid_data], rax")
        add("movabs dword ptr [grid_data], eax")
        add("movabs byte ptr [grid_data], al")
        # the bare-label-is-memory rule applies to every instruction
        add("add rax, grid_data")
        add("cmp r15, grid_data")
        add("lea rax, [grid_data]")
        add("movaps xmm0, [grid_data]")
        add("add r9, offset grid_data")
        add("test r15, offset grid_data")

    branches()
    gp()
    sse()
    avx()
    data()
    return L


HEAD = ("\t.intel_syntax noprefix\n\t.text\n\t.global _start\n_start:\n")
TAIL = ("grid_data:\n\t.quad 0x1122334455667788\n\t.quad 0x99aabbccddeeff00\n")


# --------------------------------------------------------------------------
# tool discovery and comparison
# --------------------------------------------------------------------------
def tool_path(name):
    exe = name + (".exe" if os.name == "nt" else "")
    envd = os.environ.get("AIOS_BINU")
    cands = []
    if envd:
        cands.append(os.path.join(envd, exe))
    p = shutil.which(name)
    if p:
        cands.append(p)
    for tmp in (os.environ.get("TEMP"), os.environ.get("TMP"), "/tmp"):
        if tmp:
            cands.append(os.path.join(tmp, "binu", exe))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    return r.returncode, r.stdout or "", r.stderr or ""


RELOC_RE = re.compile(
    r"^([0-9a-f]{8,})\s+(?:IMAGE_REL_AMD64_(\w+)|R_X86_64_(\w+))\s+"
    r"([A-Za-z_.][\w.]*)\s*(?:([+-])(0x[0-9a-f]+|\d+))?")


def relocations(obj, objdump):
    """([(offset, size, symbol, addend, is_pc)], is_elf) from the object's
    relocation table. Works for both PE/COFF (the build PC's mingw
    binutils) and ELF (Kaggle's GNU as)."""
    rc, out, err = run([objdump, "-r", obj])
    if rc != 0:
        raise SystemExit("objdump -r failed: %s" % err[:200])
    is_elf = "IMAGE_REL_AMD64" not in out
    # kind -> (field size, is_pc_relative)
    kinds = {"ADDR64": (8, False), "ADDR32": (4, False),
             "ADDR32NB": (4, False), "REL32": (4, True),
             "64": (8, False), "32": (4, False), "32S": (4, False),
             "8": (1, False),
             "PC32": (4, True), "PLT32": (4, True), "PC64": (8, True)}
    out_relocs = []
    for line in out.splitlines():
        m = RELOC_RE.match(line)
        if not m:
            continue
        off = int(m.group(1), 16)
        kind = m.group(2) or m.group(3)
        if kind not in kinds:
            raise SystemExit("unhandled relocation kind %r in %r"
                             % (kind, line))
        sym = m.group(4)
        addend = int(m.group(6), 0) if m.group(6) else 0
        if m.group(5) == "-":
            addend = -addend
        size, is_pc = kinds[kind]
        out_relocs.append((off, size, sym, addend, is_pc))
    return out_relocs, is_elf


def symbols(obj, objdump):
    """{name: section offset} from the object's symbol table. Handles the
    ELF objdump -t columns, the classic PE columns, and mingw's bracketed
    PE layout `[ 20](sec  1)(fl 0x00)(ty 0)(scl 6) (nx 0) 0x... grid_data`."""
    rc, out, err = run([objdump, "-t", obj])
    if rc != 0:
        raise SystemExit("objdump -t failed: %s" % err[:200])
    syms = {}
    for line in out.splitlines():
        toks = line.split()
        if len(toks) < 2:
            continue
        name = toks[-1]
        if not re.fullmatch(r"[A-Za-z_.][\w.]*", name):
            continue
        if any("UND" in t for t in toks[:-1]):
            continue
        val = None
        for t in toks[:-1]:
            if re.fullmatch(r"0x[0-9a-f]+|[0-9a-f]{6,}", t):
                val = int(t, 16)
        if val is not None:
            syms[name] = val
    return syms


def main():
    as_tool = tool_path("as")
    objcopy = tool_path("objcopy")
    objdump = tool_path("objdump")
    missing = [n for n, p in (("as", as_tool), ("objcopy", objcopy),
                              ("objdump", objdump)) if p is None]
    if missing:
        print("GRID FATAL: GNU binutils not found: %s (set AIOS_BINU to a "
              "directory containing them)" % ", ".join(missing))
        return 2

    work = tempfile.mkdtemp(prefix="b2grid_")
    lines = generate()
    src = HEAD + "\n".join(lines) + "\n" + TAIL
    spath = os.path.join(work, "grid.s")
    opath = os.path.join(work, "grid.o")
    gbin = os.path.join(work, "gas.bin")
    obin = os.path.join(work, "ours.bin")
    with open(spath, "w", encoding="utf-8") as f:
        f.write(src)

    n_insns = sum(1 for l in lines
                  if l and not l.startswith((".", " ", "\t"))
                  and not l.endswith(":"))
    rc, out, err = run([as_tool, "--64", "-o", opath, spath])
    fmt = "PE/COFF"
    if rc == 0:
        frc, fout, _ = run([objdump, "-f", opath])
        if frc == 0 and re.search(r"file format\s+elf", fout):
            fmt = "ELF"
    print("=== B2 differential grid (target: %s) ===" % fmt)
    print("generated: %d lines (%d instruction cases), %d mnemonics"
          % (len(lines), n_insns,
             len(set(l.split()[0] for l in lines if l
                     and not l[0].isspace() and not l.startswith(".")
                     and not l.endswith(":")))))
    if rc != 0 or "rror" in err:
        print("GRID FAILED: GNU as rejected generated source:")
        for line in err.splitlines():
            print("  " + line)
            m = re.search(r":(\d+):", line)
            if m:
                n = int(m.group(1))
                # grid.s line n -> src line (HEAD has 4 lines)
                k = n - 5
                if 0 <= k < len(lines):
                    print("    grid line: %s" % lines[k])
        return 1
    run([objcopy, "-O", "binary", "--only-section=.text", opath, gbin])
    with open(gbin, "rb") as f:
        gas = f.read()

    # our side
    try:
        a = asm.Assembler(base=0)
        ours, entry = a.assemble(src)
    except asm.AsmError as e:
        print("GRID FAILED: our assembler rejected the generated source: %s"
              % e)
        return 1
    with open(obin, "wb") as f:
        f.write(ours)

    # Patch GAS's unresolved relocation fields with the values the linker
    # would write, computed from GAS's own symbol table and addends:
    #   absolute: S + A            PC-relative (ELF): S + A - P
    # PE/COFF same-section PC-relative fields are already final in the
    # object (GAS resolves them at assembly), so only ELF PC fields need
    # patching - either way the direct byte comparison then covers them.
    relocs, is_elf = relocations(opath, objdump)
    syms = symbols(opath, objdump)
    gp = bytearray(gas)
    problems = []
    for off, size, sym, addend, is_pc in relocs:
        if is_pc and not is_elf:
            # PE/COFF resolves same-section PC-relative references at
            # assembly time: the field already holds the final value and
            # the direct byte comparison below checks it
            continue
        symval = syms.get(sym)
        if symval is None and sym.startswith("."):
            symval = 0   # section symbol: the offset lives in the field
        if symval is None:
            problems.append("relocation against unknown symbol %r" % sym)
            continue
        # ELF RELA keeps the addend in the relocation entry (field is 0);
        # PE/COFF keeps it in the field itself - so add the field's
        # current value either way (it is 0 for ELF)
        field = int.from_bytes(gas[off:off + size], "little")
        val = symval + addend + field - (off if (is_pc and is_elf) else 0)
        gp[off:off + size] = (val & ((1 << (8 * size)) - 1)) \
            .to_bytes(size, "little")
    # our absolute fixups must sit exactly where GAS's absolute
    # relocations do (PC-relative ones are position-identical too, which
    # the byte comparison itself proves)
    our_abs = sorted((off, {"abs8": 1, "abs64": 8}.get(k, 4))
                     for (si, off, k, e, en) in a.fixups
                     if k in ("abs8", "abs32", "abs64"))
    gas_abs = sorted((off, size) for off, size, s, ad, pc in relocs
                     if not pc)
    if our_abs != gas_abs:
        problems.append("our absolute fixup offsets %s != GAS relocation "
                        "offsets %s" % (our_abs[:5], gas_abs[:5]))

    # full byte comparison (with GAS's relocation fields patched)
    gp = bytes(gp)
    if len(gp) < len(ours) or gp[:len(ours)] != ours:
        n = min(len(gp), len(ours))
        off = next((j for j in range(n) if gp[j] != ours[j]), n)
        for lineno, text, start, size in a.stmt_map:
            if start <= off < start + max(size, 1):
                problems.append("first byte difference at %d: grid line %d: "
                                "%s" % (off, lineno, text))
                break
        else:
            problems.append("first byte difference at %d (past the last "
                            "statement)" % off)
        problems.append("ours %s" % ours[max(0, off - 4):off + 8].hex(" "))
        problems.append("gas  %s" % gp[max(0, off - 4):off + 8].hex(" "))
    else:
        pad = gp[len(ours):]
        if len(pad) > 15:
            problems.append("%d trailing padding bytes > 15" % len(pad))
        elif pad:
            ppath = os.path.join(work, "pad.bin")
            with open(ppath, "wb") as f:
                f.write(pad)
            rc, out, err = run([objdump, "-D", "-b", "binary",
                                "-m", "i386:x86-64", "-M", "intel", ppath])
            for line in out.splitlines():
                m = re.match(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s)+\s*(\S.*)$",
                             line)
                if m and not (m.group(1).startswith("nop")
                              or m.group(1).startswith("xchg")):
                    problems.append("trailing bytes are not nops: %s"
                                    % m.group(1))

    print("bytes: ours=%d gas=%d, relocations verified=%d (%s object)"
          % (len(ours), len(gp), len(relocs),
             "ELF" if is_elf else "PE/COFF"))
    if problems:
        print("GRID FAILED: %d problem(s)" % len(problems))
        for p in problems:
            print("  " + p)
        return 1
    print("GRID: %d instruction cases, %d bytes, 0 differences"
          % (n_insns, len(ours)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
