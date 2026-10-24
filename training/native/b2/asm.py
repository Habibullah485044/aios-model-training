#!/usr/bin/env python3
"""asm.py - B2: x86-64 assembler. Intel-syntax .asm in, machine code out.

Pass check (b2_check.py, on Kaggle): every output byte identical to GNU as
over the test corpus in b2/corpus/*.s. Corpus files carry
`.intel_syntax noprefix` and assemble unmodified with `as --64` + `ld`; the
comparison uses the LINKED .text of both sides (relocations resolved, with
the load address handed to assemble()), so a relocatable .o (where GAS
leaves zeros behind relocations) cannot skew it, and each corpus program is
also executed to prove the bytes work. GNU as pads a section out to its
alignment with nops, so the check tolerates (and verifies) that trailing
padding, which is a section artifact and not part of an instruction.

API: assemble(text, base=0, extern=None) -> (code, entry_offset);
     assemble_map(text, base=0, extern=None) -> (code, entry, [(lineno,
     source, off, size)]) for pinpointing the first differing line.
     `base` is the address the code is loaded at and only affects
     absolute fixups. `extern` maps labels defined in OTHER modules
     (B3) to their linked absolute addresses; references to them are
     fixed up from it, and a branch to one always takes the rel32 form
     (GNU as cannot know an undefined symbol's distance).

Architecture: parse -> statements; layout (branch relaxation, grow-only
rel8->rel32 until fixpoint, GNU as semantics); emit with fixed-width
placeholders for label-dependent values; fixups patched afterwards.
Design guarantee: an instruction's byte length never depends on a label
value (mem disps are numeric or a fixed-width symbolic form: [rip+label]
is always disp32, a symbolic absolute is always a disp32/moffs64 field;
movabs is chosen at parse time; branches by the relaxation state).

GNU as conventions matched byte for byte (measured against binutils 2.47):
  - the 8-bit opcode families 88/8A/00..03/84/80/FE/F6/C0/D0/D2;
  - the A-register short forms for al/eax/rax immediates (04..3D, A8/A9);
  - the imm8 forms (80/83) whenever the value fits, and the sign-extended
    imm32 form for a 64-bit operand whose bit pattern round-trips
    (mov rax, 0xffffffff80000000 -> 48 C7 C0 00 00 00 80);
  - the D0/D1 short shift form for a count of 1;
  - movq: xmm<-m64 F3 0F 7E, m64<-xmm 66 0F D6, xmm<-r64 66 0F 6E,
    r64<-xmm 66 0F 7E, xmm<-xmm F3 0F 7E;
  - disp32 for every [base+disp] that does not fit disp8, [rbp]/[r13]
    needing an explicit disp8, SIB whenever the base is rsp/r12;
  - a bare label operand is MEMORY in Intel syntax (GNU as: mov rax, d
    loads from an absolute address = mod=00 SIB 04 25 + disp32 with a
    32-bit relocation), `offset d` puts the ADDRESS into an immediate
    (mov r64 -> 48 C7 C0 imm32, mov r32 -> B8 imm32, mov r8 -> B0 ib,
    each with a 32/8-bit relocation; movabs r64, offset d -> 48 B8
    imm64 with a 64-bit relocation; `offset <constant>` is the
    identity), and movabs takes a 64-bit immediate on any r64 or an
    absolute moffs64 accumulator access (load 48 A1/A1/A0, store
    48 A3/A3/A2); other registers or addresses are rejected, as GAS
    rejects them.

GNU as conventions matched for the VEX (AVX2/FMA) subset, again measured
against binutils 2.47:
  - the short C5 prefix is used exactly when the instruction is 0F-map,
    W=0 and neither the SIB index nor the r/m register needs bit 3; any
    other case (0F38/0F3A maps, W1, or X/B bits) gets the 3-byte C4 form;
  - VEX.vvvv = second operand for 3-operand forms, and for the
    shift-immediate group (0F 71/72 /n) it is the DESTINATION while
    ModRM.reg carries the /n extension (vpslld xmm4,xmm5,3 -> vvvv=4,
    /6, rm=5);
  - vfmadd231ps is 66 0F38.W0 B8 (the FMA opcode B8 = 98 + 2*0x10);
  - vzeroupper encodes as C5 F8 77, i.e. a VEX form, not plain 0F 77;
  - memory destinations are offered only where GNU as's Intel syntax
    accepts them (vextractf128); it rejects mem destinations for
    vpand/vpor/vpxor/vaddps/vpslld-imm, so we reject them too, and its
    shift-by-immediate template takes a register source only (the memory
    form is routed to an EVEX/AVX-512 encoding we do not implement);
  - the shift COUNT operand of vpslld/vpsrlw is always 128-bit (an xmm
    register or an xmmword PTR) even with a ymm destination, and the
    movmskps/movmskpd/vpmovmskb sources must be registers;
  - a two-register vmovups/vmovdqu is emitted in whichever direction
    (0F 10/11, 0F 6F/7F) avoids the 3-byte C4 prefix: the load form
    flips to the store form exactly when the source register has bit 3
    set and the destination does not (vmovups xmm0,xmm8 -> C5 78 11 C0;
    ties keep the load form).

Supported subset (everything the Part C engine ops need):
  GP: mov movabs movzx movsx movsxd lea push pop add sub and or xor cmp
      test imul
      inc dec neg not shl shr sar jmp jcc call ret setcc
      nop int3 syscall cpuid pause cld std
  SSE/SSSE3: movaps movups movapd movupd movdqa movdqu movss movsd movd
      movq addps/sd subps/sd mulps/sd divps/sd minps maxps andps andnps
      andpd orps orpd xorps xorpd paddb/w/d/q psubb/w/d/q pand por pxor
      pandn pslld psrld psllq psrlq pslldq psrldq pshufb punpcklbw/lwd/
      ldq/lqdq cvtsi2ss/sd cvtss2si cvttsd2si cvttss2si cvtss2sd cvtsd2ss
      cvtdq2ps cvtps2dq cvttps2dq ucomiss/sd comiss/sd movmskps/pd
  AVX/AVX2/FMA (VEX): vmovups vmovdqu vbroadcastss vpshufb vpmovzxbd
      vpmovzxwd vpand vpor vpxor vpslld vpsrlw vpcmpeqb vpmovmskb vaddps
      vmulps vfmadd231ps vextractf128 vperm2f128 vzeroupper (the Part C
      modern fast path reads BF16 through these)
  directives: .byte .word .long .quad (numeric; .quad/.long label allowed)
      .align N .space N .global (.intel_syntax .text .section .p2align
      .file .ident .type .size: accepted, ignored)

Restrictions: no 16-bit GP ops, no NUMERIC absolute [disp32] memory
operands (a symbolic absolute - bare label or [label] - is supported as
a fixed-width disp32), no
indirect call/jmp (register or memory target), no segment overrides, no
x87, no AVX outside the listed subset. `.space N` fills with zeros and
`.align N` fills with the exact GNU as nop sequence (both measured
against binutils - align_fill/gap_fill below); `.align N` also raises
the module's section alignment (gas: sh_addralign = largest .align),
which the B3 linker uses to place modules. `#` comments; `;` splits
statements.
"""
import re
import sys

GP = {
    "rax": (0, 64), "rcx": (1, 64), "rdx": (2, 64), "rbx": (3, 64),
    "rsp": (4, 64), "rbp": (5, 64), "rsi": (6, 64), "rdi": (7, 64),
    "r8": (8, 64), "r9": (9, 64), "r10": (10, 64), "r11": (11, 64),
    "r12": (12, 64), "r13": (13, 64), "r14": (14, 64), "r15": (15, 64),
    "eax": (0, 32), "ecx": (1, 32), "edx": (2, 32), "ebx": (3, 32),
    "esp": (4, 32), "ebp": (5, 32), "esi": (6, 32), "edi": (7, 32),
    "r8d": (8, 32), "r9d": (9, 32), "r10d": (10, 32), "r11d": (11, 32),
    "r12d": (12, 32), "r13d": (13, 32), "r14d": (14, 32), "r15d": (15, 32),
    "al": (0, 8), "cl": (1, 8), "dl": (2, 8), "bl": (3, 8),
    "spl": (4, 8), "bpl": (5, 8), "sil": (6, 8), "dil": (7, 8),
    "r8b": (8, 8), "r9b": (9, 8), "r10b": (10, 8), "r11b": (11, 8),
    "r12b": (12, 8), "r13b": (13, 8), "r14b": (14, 8), "r15b": (15, 8),
}
SIZE_KW = {"byte": 8, "word": 16, "dword": 32, "qword": 64, "xmmword": 128,
           "ymmword": 256}
CC = {"o": 0, "no": 1, "b": 2, "c": 2, "nae": 2, "ae": 3, "nb": 3, "nc": 3,
      "e": 4, "z": 4, "ne": 5, "nz": 5, "be": 6, "na": 6, "a": 7, "nbe": 7,
      "s": 8, "ns": 9, "p": 10, "pe": 10, "np": 11, "po": 11,
      "l": 12, "nge": 12, "ge": 13, "nl": 13, "le": 14, "ng": 14,
      "g": 15, "nle": 15}
ARITH = {"add": 0, "or": 1, "and": 4, "sub": 5, "xor": 6, "cmp": 7}
SHIFTS = {"shl": 4, "sal": 4, "shr": 5, "sar": 7}
SIMPLE = {"nop": b"\x90", "ret": b"\xc3", "int3": b"\xcc",
          "syscall": b"\x0f\x05", "cpuid": b"\x0f\xa2",
          "cld": b"\xfc", "std": b"\xfd", "pause": b"\xf3\x90"}

# GNU as / GNU ld nop fills, measured byte for byte against binutils
# 2.46 and 2.47 (pad lengths 1..126 for .align, 0..63 for ld gaps).
# CANON[n] is the canonical single x86 nop of n bytes.
CANON = {
    0: b"",
    1: b"\x90",
    2: b"\x66\x90",
    3: b"\x0f\x1f\x00",
    4: b"\x0f\x1f\x40\x00",
    5: b"\x0f\x1f\x44\x00\x00",
    6: b"\x66\x0f\x1f\x44\x00\x00",
    7: b"\x0f\x1f\x80\x00\x00\x00\x00",
    8: b"\x0f\x1f\x84\x00\x00\x00\x00\x00",
    9: b"\x66\x0f\x1f\x84\x00\x00\x00\x00\x00",
    10: b"\x66\x2e\x0f\x1f\x84\x00\x00\x00\x00\x00",
    11: b"\x66\x66\x2e\x0f\x1f\x84\x00\x00\x00\x00\x00",
}


def decomp_nops(m):
    """gas's nop decomposition for m bytes: the canonical nop of m%11
    bytes first, then m//11 eleven-byte nops (measured over 1..126)."""
    q, r = divmod(m, 11)
    return CANON[r] + CANON[11] * q


def code_align_fill(k):
    """gas .align pad of k bytes that follows CODE (measured k=1..4095,
    align_probe.py, incl. the 4096-byte page aligns of the B3 programs):
    nops only below 88; from 88 the pad opens with a jmp that skips the
    whole pad - short rel8 form while k-2 fits in a signed byte
    (k <= 129), near rel32 form from k = 130 on. The rest is always the
    canonical nop decomposition of the bytes behind the jmp."""
    if k <= 0:
        return b""
    if k >= 130:
        return bytes([0xE9]) + (k - 5).to_bytes(4, "little") \
            + decomp_nops(k - 5)
    if k >= 88:
        return bytes([0xEB, (k - 2) & 0xFF]) + decomp_nops(k - 2)
    return decomp_nops(k)


def align_fill(k, prev_kind):
    """Exact gas .align pad bytes for a pad of k bytes. After code
    (insn) the fill is code_align_fill(k); after .space/data a leading
    90 comes first (measured). prev_kind = the previous byte-emitting
    statement's kind, None at module start (where k is always 0). An
    align directly after a non-empty align is not measured and the
    corpus avoids it; it takes the data rule."""
    if k <= 0:
        return b""
    if prev_kind in ("skip", "data", "align"):
        return b"\x90" + code_align_fill(k - 1)
    return code_align_fill(k)


def gap_fill(k):
    """Exact GNU ld fill for a k-byte gap between two input .text
    sections (measured k=0..63): 10-byte nops first, canonical
    remainder last."""
    q, r = divmod(k, 10)
    return CANON[10] * q + CANON[r]

N = ()
P66 = (0x66,)
PF3 = (0xF3,)
PF2 = (0xF2,)
SSE = {
    "movaps": [("lr", N, (0x0F, 0x28)), ("rl", N, (0x0F, 0x29))],
    "movups": [("lr", N, (0x0F, 0x10)), ("rl", N, (0x0F, 0x11))],
    "movapd": [("lr", P66, (0x0F, 0x28))],
    "movupd": [("lr", P66, (0x0F, 0x10))],
    "movdqa": [("lr", P66, (0x0F, 0x6F)), ("rl", P66, (0x0F, 0x7F))],
    "movdqu": [("lr", PF3, (0x0F, 0x6F)), ("rl", PF3, (0x0F, 0x7F))],
    "movss": [("lr", PF3, (0x0F, 0x10)), ("rl", PF3, (0x0F, 0x11))],
    "movsd": [("lr", PF2, (0x0F, 0x10)), ("rl", PF2, (0x0F, 0x11))],
    "movd": [("lr", P66, (0x0F, 0x6E)), ("rl", P66, (0x0F, 0x7E))],
    # GNU as loads m64 into an xmm with F3 0F 7E, and only uses the
    # 66 0F 6E form when the source is a GP register.
    "movq": [("lm", PF3, (0x0F, 0x7E)),   # xmm <- m64
             ("sm", P66, (0x0F, 0xD6)),   # m64 <- xmm
             ("lr", P66, (0x0F, 0x6E)),   # xmm <- r64
             ("rl", P66, (0x0F, 0x7E))],  # r64 <- xmm
    "addps": [("lr", N, (0x0F, 0x58))],
    "addpd": [("lr", P66, (0x0F, 0x58))],
    "addss": [("lr", PF3, (0x0F, 0x58))],
    "addsd": [("lr", PF2, (0x0F, 0x58))],
    "subps": [("lr", N, (0x0F, 0x5C))],
    "subpd": [("lr", P66, (0x0F, 0x5C))],
    "subss": [("lr", PF3, (0x0F, 0x5C))],
    "subsd": [("lr", PF2, (0x0F, 0x5C))],
    "mulps": [("lr", N, (0x0F, 0x59))],
    "mulpd": [("lr", P66, (0x0F, 0x59))],
    "mulss": [("lr", PF3, (0x0F, 0x59))],
    "mulsd": [("lr", PF2, (0x0F, 0x59))],
    "divps": [("lr", N, (0x0F, 0x5E))],
    "divss": [("lr", PF3, (0x0F, 0x5E))],
    "divsd": [("lr", PF2, (0x0F, 0x5E))],
    "minps": [("lr", N, (0x0F, 0x5D))],
    "maxps": [("lr", N, (0x0F, 0x5F))],
    "minss": [("lr", PF3, (0x0F, 0x5D))],
    "maxss": [("lr", PF3, (0x0F, 0x5F))],
    "andps": [("lr", N, (0x0F, 0x54))],
    "andpd": [("lr", P66, (0x0F, 0x54))],
    "andnps": [("lr", N, (0x0F, 0x55))],
    "andnpd": [("lr", P66, (0x0F, 0x55))],
    "orps": [("lr", N, (0x0F, 0x56))],
    "orpd": [("lr", P66, (0x0F, 0x56))],
    "xorps": [("lr", N, (0x0F, 0x57))],
    "xorpd": [("lr", P66, (0x0F, 0x57))],
    "paddb": [("lr", P66, (0x0F, 0xFC))],
    "paddw": [("lr", P66, (0x0F, 0xFD))],
    "paddd": [("lr", P66, (0x0F, 0xFE))],
    "paddq": [("lr", P66, (0x0F, 0xD4))],
    "psubb": [("lr", P66, (0x0F, 0xF8))],
    "psubw": [("lr", P66, (0x0F, 0xF9))],
    "psubd": [("lr", P66, (0x0F, 0xFA))],
    "psubq": [("lr", P66, (0x0F, 0xFB))],
    "pand": [("lr", P66, (0x0F, 0xDB))],
    "pandn": [("lr", P66, (0x0F, 0xDF))],
    "por": [("lr", P66, (0x0F, 0xEB))],
    "pxor": [("lr", P66, (0x0F, 0xEF))],
    "pslld": [("ext", P66, (0x0F, 0x72), 6), ("lr", P66, (0x0F, 0xF2))],
    "psrld": [("ext", P66, (0x0F, 0x72), 2), ("lr", P66, (0x0F, 0xD2))],
    "psllq": [("ext", P66, (0x0F, 0x73), 6), ("lr", P66, (0x0F, 0xF3))],
    "psrlq": [("ext", P66, (0x0F, 0x73), 2), ("lr", P66, (0x0F, 0xD3))],
    "pslldq": [("ext", P66, (0x0F, 0x73), 7)],
    "psrldq": [("ext", P66, (0x0F, 0x73), 3)],
    "pshufb": [("lr", P66, (0x0F, 0x38, 0x00))],
    "punpcklbw": [("lr", P66, (0x0F, 0x60))],
    "punpcklwd": [("lr", P66, (0x0F, 0x61))],
    "punpckldq": [("lr", P66, (0x0F, 0x62))],
    "punpcklqdq": [("lr", P66, (0x0F, 0x6C))],
    "cvtsi2ss": [("lr", PF3, (0x0F, 0x2A))],
    "cvtsi2sd": [("lr", PF2, (0x0F, 0x2A))],
    "cvtss2si": [("gr", PF3, (0x0F, 0x2D))],
    "cvttsd2si": [("gr", PF2, (0x0F, 0x2C))],
    "cvttss2si": [("gr", PF3, (0x0F, 0x2C))],
    "cvtss2sd": [("lr", PF3, (0x0F, 0x5A))],
    "cvtsd2ss": [("lr", PF2, (0x0F, 0x5A))],
    "cvtdq2ps": [("lr", N, (0x0F, 0x5B))],
    "cvtps2dq": [("lr", P66, (0x0F, 0x5B))],
    "cvttps2dq": [("lr", PF3, (0x0F, 0x5B))],
    "ucomiss": [("lr", N, (0x0F, 0x2E))],
    "ucomisd": [("lr", P66, (0x0F, 0x2E))],
    "comiss": [("lr", N, (0x0F, 0x2F))],
    "comisd": [("lr", P66, (0x0F, 0x2F))],
    "movmskps": [("gr", N, (0x0F, 0x50))],
    "movmskpd": [("gr", P66, (0x0F, 0x50))],
}

# VEX-encoded AVX/AVX2/FMA forms: (kind, pp, map, w, opcode[, extension]).
# pp: 0 none, 1 = 66, 2 = F3; map: 1 = 0F, 2 = 0F38, 3 = 0F3A. Kinds:
#   lr   dest reg <- reg/mem        (vvvv unused)
#   rl   mem <- reg                 (vvvv unused)
#   lrr  dest <- vvvv, reg/mem      (3-operand, plus imm8 when lrrI)
#   lri  shift-imm group: dest in VEX.vvvv, reg/mem in rm, ext in reg
#   rli  rm <- reg field, imm8      (vextractf128)
#   gr   GP32 reg field <- reg/mem  (vpmovmskb)
#   none no operands                (vzeroupper)
VEX = {
    "vmovups": [("lr", 0, 1, 0, 0x10), ("rl", 0, 1, 0, 0x11)],
    "vmovdqu": [("lr", 2, 1, 0, 0x6F), ("rl", 2, 1, 0, 0x7F)],
    "vbroadcastss": [("lr", 1, 2, 0, 0x18)],
    "vpshufb": [("lrr", 1, 2, 0, 0x00)],
    "vpmovzxbd": [("lr", 1, 2, 0, 0x31)],
    "vpmovzxwd": [("lr", 1, 2, 0, 0x33)],
    "vpand": [("lrr", 1, 1, 0, 0xDB)],
    "vpor": [("lrr", 1, 1, 0, 0xEB)],
    "vpxor": [("lrr", 1, 1, 0, 0xEF)],
    "vpslld": [("lri", 1, 1, 0, 0x72, 6), ("lrr", 1, 1, 0, 0xF2)],
    "vpsrlw": [("lri", 1, 1, 0, 0x71, 2), ("lrr", 1, 1, 0, 0xD1)],
    "vpcmpeqb": [("lrr", 1, 1, 0, 0x74)],
    "vpmovmskb": [("gr", 1, 1, 0, 0xD7)],
    "vaddps": [("lrr", 0, 1, 0, 0x58)],
    "vmulps": [("lrr", 0, 1, 0, 0x59)],
    # FMA3 vfmadd231ps: dst = src1*src2 + dst; opcode B8 (98 + 2*16)
    "vfmadd231ps": [("lrr", 1, 2, 0, 0xB8)],
    # (the store-direction opcodes GNU as switches to for a two-register
    # move when that avoids the 3-byte C4 prefix - see enc_vex)
    "vextractf128": [("rli", 1, 3, 0, 0x19)],
    "vperm2f128": [("lrrI", 1, 3, 0, 0x06)],
    "vzeroupper": [("none", 0, 1, 0, 0x77)],
}


class AsmError(Exception):
    pass


def is_xmm(name):
    return name.startswith("xmm") and name[3:].isdigit() and int(name[3:]) < 16


def is_ymm(name):
    return name.startswith("ymm") and name[3:].isdigit() and int(name[3:]) < 16


class Op(object):
    __slots__ = ("kind", "num", "width", "xmm", "name", "rip", "expr",
                 "value", "size", "base", "index", "scale", "ofs")

    def __init__(self, kind, **kw):
        self.kind = kind
        for k in self.__slots__:
            if k != "kind":
                setattr(self, k, kw.get(k))


class Stmt(object):
    __slots__ = ("kind", "lineno", "text", "mnem", "ops", "values", "amount",
                 "near", "prefix")

    def __init__(self, kind, lineno, text, mnem=None, ops=None, values=None,
                 amount=None, prefix=None):
        self.kind = kind
        self.lineno = lineno
        self.text = text
        self.mnem = mnem
        self.ops = ops or []
        self.values = values or []
        self.amount = amount
        self.near = False
        self.prefix = prefix    # 0xF0 for `lock <insn>` (B4 atomics)


def split_operands(s):
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


_EXPR = re.compile(r"([A-Za-z_.][A-Za-z0-9_.]*|0x[0-9a-fA-F]+|[0-9]+)")
_IDENT = re.compile(r"[A-Za-z_.][A-Za-z0-9_.]*")
# a label reference with an optional numeric addend: label, label+8, label-4
_IDENT_OFF = re.compile(r"([A-Za-z_.][A-Za-z0-9_.]*)"
                        r"(?:\s*([+-])\s*(0x[0-9a-fA-F]+|[0-9]+))?\Z")


def is_label_ref(s):
    """True for a symbolic operand: `label`, `label+8`, `label-4`."""
    return bool(_IDENT_OFF.match(s.strip()))


def sym_to_mem(op):
    """A bare-label immediate becomes a memory operand: GNU as's Intel
    syntax treats a bare label as MEMORY (`mov rax, d` loads the value
    stored at d). `offset` operands carry their ofs flag and stay
    immediate; the caller exempts branch targets."""
    if op.kind == "imm" and op.value is None and not op.ofs:
        return Op("mem", rip=False, expr=op.expr, value=None, size=None,
                  base=None, index=None, scale=1)
    return op


def eval_num_expr(s):
    s = s.strip()
    total, sign, i, seen = 0, 1, 0, False
    if s[:1] in ("+", "-"):         # leading sign on the whole expression
        sign = -1 if s[0] == "-" else 1
        i = 1
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        m = _EXPR.match(s, i)
        if not m:
            raise AsmError("bad numeric expression %r" % s)
        t = m.group(1)
        if t[0].isdigit() or t.startswith("0x"):
            total += sign * int(t, 0)
        else:
            raise AsmError("label %r not allowed here (numeric only)" % t)
        seen = True
        i = m.end()
        if i < len(s):
            if s[i] == "+":
                sign = 1
                i += 1
            elif s[i] == "-":
                sign = -1
                i += 1
            else:
                raise AsmError("bad expression %r" % s)
    if not seen:
        raise AsmError("empty expression")
    return total


def parse_operand(tok):
    tok = tok.strip()
    low = tok.lower()
    if low in GP:
        num, width = GP[low]
        return Op("reg", num=num, width=width, name=low)
    if is_xmm(low):
        return Op("reg", num=int(low[3:]), width=128, xmm=True, name=low)
    if is_ymm(low):
        return Op("reg", num=int(low[3:]), width=256, xmm=True, name=low)
    mo = re.fullmatch(r"offset\s+(.+)", tok, re.I)
    if mo:
        rest = mo.group(1).strip()
        if is_label_ref(rest):
            return Op("imm", expr=rest, value=None, ofs=True)
        # GNU as treats `offset <constant>` as the constant itself
        return Op("imm", expr=None, value=eval_num_expr(rest))
    if "[" in tok:
        m = re.match(
            r"^(?:(byte|word|dword|qword|xmmword|ymmword)\s+PTR\s+)?\[(.*)\]\s*$",
            tok, re.S | re.I)
        if not m:
            raise AsmError("bad memory operand %r" % tok)
        size = SIZE_KW.get(m.group(1).lower()) if m.group(1) else None
        inner = m.group(2).strip()
        mm = re.match(r"^rip\s*\+\s*(.*)$", inner, re.I)
        if mm:
            expr = mm.group(1).strip()
            if not is_label_ref(expr):
                raise AsmError("bad rip expression %r" % expr)
            return Op("mem", rip=True, expr=expr, size=size)
        lm = _IDENT_OFF.match(inner.strip())
        if lm and lm.group(1).lower() not in GP:
            # symbolic absolute address (GNU as: `[label]` is a mod=00
            # SIB with neither base nor index + a disp32 relocation);
            # the field is fixed-width, so layout cannot depend on it.
            # A register name is never a label, so `[rsp+8]` etc. fall
            # through to the piece parser below.
            return Op("mem", rip=False, expr=inner.strip(), value=None,
                      size=size, base=None, index=None, scale=1)
        base = None
        index = None
        scale = 1
        disp = 0
        for piece in re.split(r"(?=[+-])", inner):
            piece = piece.strip()
            if not piece or piece == "+":
                continue
            neg = piece.startswith("-")
            if neg or piece.startswith("+"):
                piece = piece[1:].strip()
            sm = re.fullmatch(r"([A-Za-z0-9]+)\s*\*\s*([0-9]+)", piece)
            if sm:
                if neg or sm.group(1).lower() not in GP:
                    raise AsmError("bad index in %r" % tok)
                index = GP[sm.group(1).lower()][0]
                scale = int(sm.group(2))
                if scale not in (1, 2, 4, 8):
                    raise AsmError("bad scale in %r" % tok)
                continue
            if piece.lower() in GP:
                if base is not None:
                    raise AsmError("two base registers in %r" % tok)
                base = GP[piece.lower()][0]
                continue
            v = eval_num_expr(piece)
            disp += -v if neg else v
        return Op("mem", rip=False, expr=None, value=disp, size=size,
                  base=base, index=index, scale=scale)
    if is_label_ref(tok):
        return Op("imm", expr=tok, value=None)
    return Op("imm", expr=tok, value=eval_num_expr(tok))


def rex_byte(w, r, x, b):
    """REX value from FULL register numbers (0 for unused fields)."""
    return 0x40 | (w << 3) | (((r >> 3) & 1) << 2) | (((x >> 3) & 1) << 1) | \
        ((b >> 3) & 1)


def rex_maybe(v, force):
    """Emit REX only when it carries bits - or is required for spl..dil."""
    if v == 0x40 and not force:
        return b""
    return bytes([v])


# rl-direction opcodes for VEX moves that have both directions: GNU as
# emits whichever of load/store avoids the 3-byte C4 prefix for a
# two-register operand pair (vmovups xmm0,xmm8 -> C5 78 11 C0).
VEX_MOVE = {"vmovups": 0x11, "vmovdqu": 0x7F}


def vex_prefix(vvvv, l, pp, vmap, w, r, x, b):
    """VEX prefix bytes. vvvv = full number of the VEX.vvvv register, or
    None to leave the field at 1111; r/x/b = full numbers of the
    ModRM.reg, SIB index and ModRM.r/m registers; l = 1 for 256-bit
    forms; pp = mandatory prefix (0 none, 1 = 66, 2 = F3, 3 = F2);
    vmap = 1 (0F), 2 (0F38) or 3 (0F3A); w = VEX.W. The short C5 form is
    used exactly when GNU as uses it: 0F map, W=0 and neither the index
    nor the r/m register needs bit 3."""
    rb = (r >> 3) & 1
    xb = (x >> 3) & 1
    bb = (b >> 3) & 1
    vv = 15 if vvvv is None else (~vvvv) & 15
    if vmap == 1 and w == 0 and xb == 0 and bb == 0:
        return bytes([0xC5, ((1 ^ rb) << 7) | (vv << 3) | (l << 2) | pp])
    return bytes([0xC4, ((1 ^ rb) << 7) | ((1 ^ xb) << 6) | ((1 ^ bb) << 5)
                  | vmap, (w << 7) | (vv << 3) | (l << 2) | pp])


def byte_reg(op):
    """True for the 8-bit registers that need a REX prefix to exist."""
    return (op is not None and op.kind == "reg" and not op.xmm
            and op.width == 8 and 4 <= op.num <= 7)


def rm_width(op):
    """Width of an r/m operand: its PTR size in memory, else the register."""
    return op.size if op.kind == "mem" else op.width


def sign_extend(value, bits):
    """The low `bits` bits of value, read as a signed number."""
    x = value & ((1 << bits) - 1)
    return x - (1 << bits) if x & (1 << (bits - 1)) else x


def fits_imm(value, dest_bits, imm_bits):
    """True when an `imm_bits`-wide immediate, sign-extended to the
    destination size, encodes exactly the same bit pattern as `value`.
    GNU as always picks the shortest such form, e.g.
    mov rax, 0xffffffff80000000 -> 48 C7 C0 00 00 00 80 and
    add rax, 0xffffffffffffffff -> 48 83 C0 FF."""
    mask = (1 << dest_bits) - 1
    return (value & mask) == (sign_extend(value, imm_bits) & mask)


def is_accumulator(op):
    """True for al/eax/rax (register number 0). GNU as encodes immediate
    forms of the accumulator with the dedicated A-register opcodes (04..3D,
    A8/A9) instead of the generic 80/81/F6/F7 forms and we must match it."""
    return op is not None and op.kind == "reg" and not op.xmm and op.num == 0 \
        and op.width in (8, 32, 64)


def check_size(op, want):
    """An explicit byte/word/dword/qword PTR must agree with the other
    operand's width (GNU as rejects a mismatch, so do we)."""
    if op.kind == "mem" and op.size is not None and op.size != want:
        raise AsmError("PTR size %d does not match operand width %d"
                       % (op.size, want))


def modrm_mem(op, reg):
    """(modrm+sib bytes, x_bit, b_bit, disp_kind) - disp appended later."""
    if op.rip:
        return (bytes([(reg & 7) << 3 | 5]), 0, 0, "rip")
    base, index, scale = op.base, op.index, op.scale
    disp = op.value or 0
    if base is None:
        if index is None and op.expr:
            # symbolic absolute address: mod=00, SIB with neither base
            # nor index (GNU as: 04 25 disp32 + a 32-bit relocation)
            return (bytes([(reg & 7) << 3 | 4, 0x25]), 0, 0, "a32")
        raise AsmError("memory operand needs a base register")
    if index is None and (base & 7) != 4:
        if (base & 7) == 5 and disp == 0:
            return (bytes([0x40 | ((reg & 7) << 3) | (base & 7)]),
                    0, base, "dz")
        if disp == 0:
            mod = 0
        elif -128 <= disp <= 127:
            mod = 1
        else:
            mod = 2  # 0b10 = disp32 (a bare 10 here would overflow the byte)
        kind = {0: None, 1: "d8", 2: "d32"}[mod]
        return (bytes([(mod << 6) | ((reg & 7) << 3) | (base & 7)]),
                0, base, kind)
    if index == 4:  # rsp is never a valid index; r12 is (REX.X escapes
        # the SIB "no index" encoding, which is why the check is on the
        # full register number and not on its low bits)
        raise AsmError("rsp cannot be an index")
    ss = {1: 0, 2: 1, 4: 2, 8: 3}[scale]
    idx = (index & 7) if index is not None else 4
    x_reg = index if index is not None else 0
    if (base & 7) == 5 and disp == 0 and index is not None:
        mod, kind = 1, "dz"
    elif disp == 0:
        mod, kind = 0, None
    elif -128 <= disp <= 127:
        mod, kind = 1, "d8"
    else:
        mod, kind = 2, "d32"  # 0b10: the mod field is shifted below
    sib = (ss << 6) | (idx << 3) | (base & 7)
    return (bytes([(mod << 6) | ((reg & 7) << 3) | 4, sib]),
            x_reg, base, kind)


def disp_bytes(kind, value):
    if kind is None:
        return b""
    if kind == "dz":
        return b"\x00"
    if kind == "d8":
        if not (-128 <= value <= 127):
            raise AsmError("disp8 overflow (%d)" % value)
        return bytes([value & 0xFF])
    if kind == "d32":
        return (value & 0xFFFFFFFF).to_bytes(4, "little")
    raise AsmError("bad disp kind")


class Assembler(object):
    def __init__(self, base=0, extern=None):
        self.labels = {}
        self.fixups = []      # (stmt_index, off_in_stmt, kind, expr, end_in_stmt)
        self.base = base      # load address of offset 0 (absolute fixups)
        self.stmt_map = []    # (lineno, source, offset, size) per statement
        self.extern = extern or {}   # other-module label -> linked address

    def assemble(self, text):
        stmts, label_at = self.parse(text)
        addrs = self.layout(stmts, label_at)
        code, entry = self.emit(stmts, addrs)
        self.apply_fixups(code)
        return bytes(code), entry

    # ---------- parsing ----------
    def parse(self, text):
        stmts = []
        label_at = {}
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].rstrip()
            for piece in line.split(";"):
                piece = piece.strip()
                if not piece:
                    continue
                while True:
                    m = re.match(r"^([A-Za-z_.][A-Za-z0-9_.]*):\s*", piece)
                    if not m:
                        break
                    label_at[m.group(1)] = len(stmts)
                    piece = piece[m.end():]
                if not piece:
                    continue
                toks = piece.split(None, 1)
                mnem = toks[0].lower()
                rest = toks[1] if len(toks) > 1 else ""
                prefix = None
                if mnem == "lock":
                    # gas Intel syntax: `lock xadd qword ptr [rbx], r8` -
                    # the lock byte (F0) is a prefix on the instruction
                    # that follows (B4 atomics).
                    sub = rest.split(None, 1)
                    if not sub or sub[0].startswith("."):
                        raise AsmError("line %d: lock needs an instruction"
                                       % lineno)
                    prefix = 0xF0
                    mnem = sub[0].lower()
                    rest = sub[1] if len(sub) > 1 else ""
                if mnem in (".byte", ".word", ".long", ".quad"):
                    stmts.append(Stmt("data", lineno, piece, mnem,
                                      values=split_operands(rest)))
                elif mnem == ".align":
                    n = int(rest.strip().split(",")[0], 0)
                    if n <= 0 or n & (n - 1):
                        raise AsmError("line %d: .align must be a power of two"
                                       % lineno)
                    stmts.append(Stmt("align", lineno, piece, amount=n))
                elif mnem in (".space", ".skip"):
                    stmts.append(Stmt("skip", lineno, piece,
                                      amount=int(rest.strip().split(",")[0], 0)))
                elif mnem.startswith("."):
                    stmts.append(Stmt("ignore", lineno, piece, mnem))
                else:
                    ops = [parse_operand(t) for t in split_operands(rest)] \
                        if rest else []
                    # GNU as Intel syntax: a bare label is a MEMORY
                    # operand. Only branch targets stay immediate, and
                    # `offset` marks an operand as the address itself.
                    if not (mnem in ("jmp", "call")
                            or (len(mnem) > 1 and mnem[0] == "j"
                                and mnem[1:] in CC)):
                        ops = [sym_to_mem(o) for o in ops]
                    stmts.append(Stmt("insn", lineno, piece, mnem, ops,
                                      prefix=prefix))
        return stmts, label_at

    # ---------- pass 1: layout with branch relaxation ----------
    def _is_branch(self, st):
        if st.kind != "insn":
            return False
        if st.mnem in ("jmp", "call"):
            return st.mnem == "jmp"
        return len(st.mnem) > 1 and st.mnem[0] == "j" and st.mnem[1:] in CC

    def layout(self, stmts, label_at):
        for _ in range(len(stmts) + 2):
            addrs = []
            addr = 0
            for st in stmts:
                addrs.append(addr)
                addr += len(self.encode(st, addr, resolve=False))
            labels = {}
            for name, idx in label_at.items():
                labels[name] = addrs[idx] if idx < len(addrs) else addr
            changed = False
            for i, st in enumerate(stmts):
                if st.near or not self._is_branch(st):
                    continue
                op = st.ops[0]
                m = _IDENT_OFF.match(op.expr.strip()) if op.expr else None
                if m is not None and m.group(1) not in labels:
                    # Not defined in this module: GNU as leaves an
                    # undefined symbol's distance unknown, so it always
                    # emits the rel32 form (B3 cross-module branches).
                    if m.group(1) in self.extern:
                        st.near = True
                        changed = True
                        continue
                    raise AsmError("line %d: unknown label %r"
                                   % (st.lineno, op.expr))
                try:
                    target = self.label_offset(op.expr, labels)
                except AsmError:
                    raise AsmError("line %d: unknown label %r"
                                   % (st.lineno, op.expr))
                rel = target - (addrs[i]
                                + len(self.encode(st, addrs[i], resolve=False)))
                if rel < -128 or rel > 127:
                    st.near = True
                    changed = True
            if not changed:
                self.labels = labels
                self.total = addr
                return addrs
        raise AsmError("branch relaxation did not converge")

    # ---------- pass 2: emit ----------
    def emit(self, stmts, addrs):
        code = bytearray()
        self.fixups = []
        self.stmt_map = []
        prev_kind = None   # kind of the last statement that emitted bytes
        for i, st in enumerate(stmts):
            start = len(code)
            chunk = self.encode(st, addrs[i], resolve=True, sidx=i,
                                prev_kind=prev_kind)
            code += chunk
            if chunk:
                prev_kind = st.kind
            # A rip-relative displacement is relative to the END of the
            # whole instruction, and instructions with a trailing
            # immediate (pslld r/m,imm8; vperm2f128/vextractf128 ...,imm8;
            # mov [rip+label],imm) extend past the disp field, so the
            # statement end - not field+4 - is the reference point.
            j = len(self.fixups) - 1
            while j >= 0 and self.fixups[j][0] == i:
                if self.fixups[j][2] == "rip":
                    f = self.fixups[j]
                    self.fixups[j] = (f[0], f[1], f[2], f[3], len(code))
                j -= 1
            if st.kind != "ignore":
                self.stmt_map.append((st.lineno, st.text, start,
                                      len(code) - start))
        entry = self.labels.get("_start", 0)
        return code, entry

    def apply_fixups(self, code):
        for (sidx, off, kind, expr, end) in self.fixups:
            s = self.label_offset(expr)
            if kind == "rel8":
                rel = s - end
                if not (-128 <= rel <= 127):
                    raise AsmError("rel8 out of range for %r" % expr)
                code[off] = rel & 0xFF
            elif kind == "rel32":
                code[off:off + 4] = ((s - end) & 0xFFFFFFFF).to_bytes(4, "little")
            elif kind == "rip":
                if end is None:
                    raise AsmError("rip fixup without an instruction end")
                code[off:off + 4] = ((s - end) & 0xFFFFFFFF) \
                    .to_bytes(4, "little")
            elif kind == "abs8":
                code[off] = (s + self.base) & 0xFF
            elif kind == "abs64":
                code[off:off + 8] = (s + self.base).to_bytes(8, "little")
            elif kind == "abs32":
                code[off:off + 4] = (s + self.base).to_bytes(4, "little")
            else:
                raise AsmError("bad fixup kind %r" % kind)

    # ---------- encoding ----------
    def encode(self, st, addr, resolve, sidx=-1, prev_kind=None):
        """Full statement bytes, lock prefix included. The body below
        encodes the instruction; for `lock <insn>` the F0 byte goes in
        front (after a leading 0x66 operand-size prefix - the order gas
        emits). Body fixups are statement-relative, so they shift by the
        inserted prefix byte."""
        n0 = len(self.fixups)
        body = self.encode_inner(st, addr, resolve, sidx, prev_kind)
        pfx = getattr(st, "prefix", None)
        if pfx is None or st.kind != "insn" or not body:
            return body
        if body[:1] == b"\x66":
            out = body[:1] + bytes([pfx]) + body[1:]
        else:
            out = bytes([pfx]) + body
        if resolve:
            for i in range(n0, len(self.fixups)):
                f = self.fixups[i]
                self.fixups[i] = (f[0], f[1] + 1, f[2], f[3],
                                  None if f[4] is None else f[4] + 1)
        return out

    def encode_inner(self, st, addr, resolve, sidx=-1, prev_kind=None):
        """Return the exact bytes of one statement. With resolve=False no
        fixups are recorded and label-dependent fields stay zero; sizes are
        identical in both modes (design guarantee). Internal offsets are
        statement-relative; fx() translates them to absolute code offsets."""
        mnem = st.mnem
        ops = st.ops
        out = bytearray()

        def fx(off, kind, expr, end=None):
            if resolve:
                self.fixups.append((sidx, addr + off, kind, expr,
                                    addr + end if end is not None else None))

        if st.kind == "data":
            w = {".byte": 1, ".word": 2, ".long": 4, ".quad": 8}[mnem]
            for v in st.values:
                if is_label_ref(v):
                    if w not in (4, 8):
                        raise AsmError("label reference needs .long or .quad")
                    fx(len(out), "abs64" if w == 8 else "abs32", v)
                    val = self.labels.get(v, 0) if resolve else 0
                    out += (val & ((1 << (8 * w)) - 1)).to_bytes(w, "little")
                else:
                    out += (eval_num_expr(v) & ((1 << (8 * w)) - 1)) \
                        .to_bytes(w, "little")
            return bytes(out)
        if st.kind in ("align", "skip", "ignore"):
            # Padding must have its real size in the layout pass, otherwise
            # every label after it is misplaced (and branch relaxation lies).
            if st.kind == "ignore":
                return b""
            if st.kind == "skip":
                return b"\x00" * st.amount
            # .align fills with gas's nop sequence (context dependent;
            # the fill BYTES do not affect the pad length, so the layout
            # pass gets the size right with any context).
            return align_fill((-addr) % st.amount, prev_kind)

        if mnem in SIMPLE and not ops:
            return SIMPLE[mnem]

        op1 = ops[0] if len(ops) > 0 else None
        op2 = ops[1] if len(ops) > 1 else None

        # ---- branches ----
        if mnem == "call":
            if op1 is None or op1.kind != "imm":
                raise AsmError("call register/indirect not supported")
            fx(len(out) + 1, "rel32", op1.expr, len(out) + 5)
            return b"\xe8\x00\x00\x00\x00"
        if mnem == "jmp" or (mnem[0] == "j" and mnem[1:] in CC):
            if op1 is None or op1.kind != "imm":
                raise AsmError("%s needs a label or constant target" % mnem)
            if mnem == "jmp":
                if st.near:
                    fx(len(out) + 1, "rel32", op1.expr, len(out) + 5)
                    return b"\xe9\x00\x00\x00\x00"
                fx(len(out) + 1, "rel8", op1.expr, len(out) + 2)
                return b"\xeb\x00"
            cc = CC[mnem[1:]]
            if st.near:
                fx(len(out) + 2, "rel32", op1.expr, len(out) + 6)
                return bytes([0x0F, 0x80 + cc, 0, 0, 0, 0])
            fx(len(out) + 1, "rel8", op1.expr, len(out) + 2)
            return bytes([0x70 + cc, 0])

        # ---- push/pop ----
        if mnem in ("push", "pop") and op1 is not None and op1.kind == "reg" \
                and not op1.xmm and op1.width == 64:
            return rex_maybe(rex_byte(0, 0, 0, op1.num), False) + \
                bytes([(0x50 if mnem == "push" else 0x58) + (op1.num & 7)])

        # ---- setcc ----
        sm = re.fullmatch(r"set([a-z]+)", mnem)
        if sm and sm.group(1) in CC and op1 is not None \
                and op1.kind == "reg" and op1.width == 8:
            return rex_maybe(rex_byte(0, 0, 0, op1.num), byte_reg(op1)) + \
                bytes([0x0F, 0x90 + CC[sm.group(1)], 0xC0 | (op1.num & 7)])

        # ---- lea ----
        if mnem == "lea":
            if op1 is None or op1.kind != "reg" or op1.xmm \
                    or op1.width not in (32, 64) or op2 is None \
                    or op2.kind != "mem":
                raise AsmError("bad lea operands")
            w = 1 if op1.width == 64 else 0
            self.rm_reg_stmt(out, 0x8D, w, op1, op2, False, fx)
            return bytes(out)

        # ---- mov / movabs ----
        if mnem in ("mov", "movabs"):
            return self.enc_mov(op1, op2, out, fx, resolve,
                                force64=(mnem == "movabs"))

        # ---- movzx / movsx / movsxd ----
        if mnem in ("movzx", "movsx"):
            if op1 is None or op2 is None or op1.kind != "reg" or op1.xmm \
                    or op1.width not in (32, 64):
                raise AsmError("bad %s operands" % mnem)
            w2 = op2.width if op2.kind == "reg" else op2.size
            if w2 not in (8, 16):
                raise AsmError("%s source needs an 8/16-bit register or a "
                               "PTR size" % mnem)
            check_size(op2, w2)
            w = 1 if op1.width == 64 else 0
            opcode = {("movzx", 8): 0xB6, ("movzx", 16): 0xB7,
                      ("movsx", 8): 0xBE, ("movsx", 16): 0xBF}[(mnem, w2)]
            if op2.kind == "reg":
                return rex_maybe(rex_byte(w, op1.num, 0, op2.num),
                                 byte_reg(op2)) + \
                    bytes([0x0F, opcode,
                           0xC0 | ((op1.num & 7) << 3) | (op2.num & 7)])
            self.rm_reg_stmt(out, bytes([0x0F, opcode]), w, op1,
                             op2, byte_reg(op2), fx)
            return bytes(out)
        if mnem == "movsxd":
            if op1 is None or op2 is None or op1.kind != "reg" \
                    or op1.width != 64:
                raise AsmError("bad movsxd operands")
            if op2.kind == "reg":
                if op2.xmm or op2.width != 32:
                    raise AsmError("movsxd source must be r/m32")
                return rex_maybe(rex_byte(1, op1.num, 0, op2.num), False) + \
                    bytes([0x63, 0xC0 | ((op1.num & 7) << 3) | (op2.num & 7)])
            check_size(op2, 32)
            self.rm_reg_stmt(out, 0x63, 1, op1, op2, False, fx)
            return bytes(out)

        # ---- arith ----
        if mnem in ARITH:
            ext = ARITH[mnem]
            if op1 is not None and op2 is not None \
                    and op1.kind in ("reg", "mem") and op2.kind == "reg" \
                    and not op2.xmm:
                # r/m, r  (00/01 + ext<<3): the register sets the width
                check_size(op1, op2.width)
                if op1.kind == "reg" and op1.width != op2.width:
                    raise AsmError("%s width mismatch" % mnem)
                if op2.width == 16:
                    raise AsmError("16-bit %s not supported" % mnem)
                w = 1 if op2.width == 64 else 0
                force = byte_reg(op1) or byte_reg(op2)
                if op1.kind == "reg":
                    return rex_maybe(rex_byte(w, op2.num, 0, op1.num), force) + \
                        bytes([(0x00 if op2.width == 8 else 0x01) | (ext << 3),
                               0xC0 | ((op2.num & 7) << 3) | (op1.num & 7)])
                self.rm_reg_stmt(out, (0x00 if op2.width == 8 else 0x01)
                                 | (ext << 3), w, op2, op1, force, fx)
                return bytes(out)
            if op1 is not None and op2 is not None and op1.kind == "reg" \
                    and not op1.xmm and op2.kind == "mem":
                # r, r/m  (02/03 + ext<<3)
                check_size(op2, op1.width)
                if op1.width == 16:
                    raise AsmError("16-bit %s not supported" % mnem)
                w = 1 if op1.width == 64 else 0
                self.rm_reg_stmt(out, (0x02 if op1.width == 8 else 0x03)
                                 | (ext << 3), w, op1, op2, byte_reg(op1), fx)
                return bytes(out)
            if op1 is not None and op2 is not None \
                    and op1.kind in ("reg", "mem") and op2.kind == "imm":
                width = rm_width(op1)
                if width is None:
                    raise AsmError("%s [mem], imm needs a byte/word/dword/qword "
                                   "PTR size" % mnem)
                if width == 16:
                    raise AsmError("16-bit %s not supported" % mnem)
                w = 1 if width == 64 else 0
                accum = is_accumulator(op1)
                if op2.value is None:
                    # Symbolic immediate: its value is unknown while laying
                    # out, so GNU as picks the generic imm32 form (81 /ext).
                    if width == 8:
                        raise AsmError("%s label immediate needs a 32/64-bit "
                                       "operand" % mnem)
                    if accum:
                        out += rex_maybe(rex_byte(w, 0, 0, 0), False) + \
                            bytes([(ext << 3) | 5])
                    else:
                        self.rm_stmt(out, 0x81, w, ext, op1, False, fx)
                    fx(len(out), "abs32", op2.expr)
                    out += b"\x00\x00\x00\x00"
                    return bytes(out)
                v = op2.value
                if width == 8:
                    if not (-0x80 <= v <= 0xFF):
                        raise AsmError("%s imm8 out of range (%d)" % (mnem, v))
                    if accum:
                        out += bytes([(ext << 3) | 4, v & 0xFF])
                    else:
                        self.rm_stmt(out, 0x80, w, ext, op1, byte_reg(op1), fx)
                        out += bytes([v & 0xFF])
                    return bytes(out)
                if width == 64:
                    fit8 = fits_imm(v, 64, 8)
                    fit32 = fits_imm(v, 64, 32)
                else:
                    fit8 = -0x80 <= v <= 0x7F
                    fit32 = -0x80000000 <= v <= 0xFFFFFFFF
                if fit8:
                    self.rm_stmt(out, 0x83, w, ext, op1, byte_reg(op1), fx)
                    out += bytes([v & 0xFF])
                    return bytes(out)
                if not fit32:
                    raise AsmError("%s imm out of range (%d)" % (mnem, v))
                if accum:
                    out += rex_maybe(rex_byte(w, 0, 0, 0), False) + \
                        bytes([(ext << 3) | 5])
                else:
                    self.rm_stmt(out, 0x81, w, ext, op1, byte_reg(op1), fx)
                out += (v & 0xFFFFFFFF).to_bytes(4, "little")
                return bytes(out)
            raise AsmError("bad %s operands" % mnem)

        # ---- test ----
        if mnem == "test":
            if op1 is not None and op2 is not None \
                    and op1.kind == "reg" and not op1.xmm \
                    and op2.kind == "reg" and not op2.xmm:
                if op1.width != op2.width or op1.width == 16:
                    raise AsmError("bad test operands")
                w = 1 if op1.width == 64 else 0
                return rex_maybe(rex_byte(w, op2.num, 0, op1.num),
                                 byte_reg(op1) or byte_reg(op2)) + \
                    bytes([0x84 if op1.width == 8 else 0x85,
                           0xC0 | ((op2.num & 7) << 3) | (op1.num & 7)])
            if op1 is not None and op2 is not None and op1.kind in ("reg", "mem") \
                    and not op1.xmm and op2.kind == "imm":
                width = rm_width(op1)
                if width is None:
                    raise AsmError("test [mem], imm needs a byte/dword/qword "
                                   "PTR size")
                if width == 16:
                    raise AsmError("16-bit test not supported")
                w = 1 if width == 64 else 0
                accum = is_accumulator(op1)
                if width == 8:
                    if op2.value is None:
                        raise AsmError("test imm8 must be numeric")
                    if accum:
                        out += bytes([0xA8, op2.value & 0xFF])
                    else:
                        self.rm_stmt(out, 0xF6, 0, 0, op1, False, fx)
                        out += bytes([op2.value & 0xFF])
                    return bytes(out)
                if accum:
                    out += rex_maybe(rex_byte(w, 0, 0, 0), False) + bytes([0xA9])
                else:
                    self.rm_stmt(out, 0xF7, w, 0, op1, False, fx)
                if op2.value is None:
                    fx(len(out), "abs32", op2.expr)
                    out += b"\x00\x00\x00\x00"
                else:
                    out += (op2.value & 0xFFFFFFFF).to_bytes(4, "little")
                return bytes(out)
            raise AsmError("bad test operands")

        # ---- xchg (measured vs gas: accumulator short form 0x90+r
        # whenever one operand is rax/eax - either source order -,
        # otherwise 0x87/0x86 with op1 in r/m and op2 in reg; with one
        # memory operand the register is always in the reg field) ----
        if mnem == "xchg":
            if len(ops) != 2:
                raise AsmError("xchg needs two operands")
            a1, a2 = ops
            if a1.kind == "reg" and a2.kind == "reg":
                if a1.xmm or a2.xmm:
                    raise AsmError("xchg xmm not supported")
                if a1.width != a2.width or a1.width not in (8, 32, 64):
                    raise AsmError("bad xchg operands")
                if a1.width == 8:
                    return rex_maybe(rex_byte(0, a2.num, 0, a1.num),
                                     byte_reg(a1) or byte_reg(a2)) + \
                        bytes([0x86,
                               0xC0 | ((a2.num & 7) << 3) | (a1.num & 7)])
                if a1.num == 0 or a2.num == 0:
                    other = a2 if a1.num == 0 else a1
                    w = 1 if a1.width == 64 else 0
                    return rex_maybe(rex_byte(w, 0, 0, other.num), False) + \
                        bytes([0x90 + (other.num & 7)])
                w = 1 if a1.width == 64 else 0
                return rex_maybe(rex_byte(w, a2.num, 0, a1.num),
                                 byte_reg(a1) or byte_reg(a2)) + \
                    bytes([0x87, 0xC0 | ((a2.num & 7) << 3) | (a1.num & 7)])
            if a1.kind == "reg" and not a1.xmm and a2.kind == "mem":
                reg, mem = a1, a2
            elif a2.kind == "reg" and not a2.xmm and a1.kind == "mem":
                reg, mem = a2, a1
            else:
                raise AsmError("bad xchg operands")
            # gas infers the memory size from the register operand; an
            # explicit PTR must still agree (check_size).
            if reg.width == 16:
                raise AsmError("bad xchg operands")
            check_size(mem, reg.width)
            w = 1 if reg.width == 64 else 0
            self.rm_reg_stmt(out, bytes([0x86 if reg.width == 8 else 0x87]),
                             w, reg, mem, False, fx)
            return bytes(out)

        # ---- xadd / cmpxchg, Intel order (r/m, reg); measured vs gas:
        # 0F C1 / 0F B1, byte forms 0F C0 / 0F B0, register in reg.
        # word is refused: gas puts 0x66 in front of the lock byte and
        # nothing in the runtime uses it. ----
        if mnem in ("xadd", "cmpxchg"):
            if len(ops) != 2 or ops[1].kind != "reg" or ops[1].xmm:
                raise AsmError("%s needs (r/m, reg)" % mnem)
            dst, src = ops
            if dst.kind == "reg" and dst.xmm:
                raise AsmError("%s needs (r/m, reg)" % mnem)
            if src.width == 16:
                raise AsmError("%s word not supported" % mnem)
            check_size(dst, src.width)
            w = 1 if src.width == 64 else 0
            op_hi = 0xC1 if mnem == "xadd" else 0xB1
            opcode = bytes([0x0F, op_hi - 1 if src.width == 8 else op_hi])
            self.rm_reg_stmt(out, opcode, w, src, dst, False, fx)
            return bytes(out)

        # ---- imul ----
        if mnem == "imul":
            if len(ops) == 2 and ops[0].kind == "reg" and not ops[0].xmm \
                    and ops[1].kind in ("reg", "mem"):
                op1, op2 = ops
                w = 1 if op1.width == 64 else 0
                if op2.kind == "reg":
                    if op2.xmm or op1.width != op2.width:
                        raise AsmError("bad imul operands")
                    return rex_maybe(rex_byte(w, op1.num, 0, op2.num), False) + \
                        bytes([0x0F, 0xAF,
                               0xC0 | ((op1.num & 7) << 3) | (op2.num & 7)])
                check_size(op2, op1.width)
                self.rm_reg_stmt(out, bytes([0x0F, 0xAF]), w, op1, op2, False, fx)
                return bytes(out)
            if len(ops) == 3 and ops[0].kind == "reg" and ops[1].kind == "reg" \
                    and ops[2].kind == "imm":
                op1, op2, op3 = ops
                if op3.value is None:
                    raise AsmError("imul immediate must be numeric")
                v = op3.value
                w = 1 if op1.width == 64 else 0
                modrm = 0xC0 | ((op1.num & 7) << 3) | (op2.num & 7)
                if -0x80 <= v <= 0x7F:
                    return rex_maybe(rex_byte(w, op1.num, 0, op2.num), False) + \
                        bytes([0x6B, modrm, v & 0xFF])
                out += rex_maybe(rex_byte(w, op1.num, 0, op2.num), False) \
                    + bytes([0x69, modrm])
                out += (v & 0xFFFFFFFF).to_bytes(4, "little")
                return bytes(out)
            if len(ops) == 1 and ops[0].kind in ("reg", "mem"):
                width = rm_width(ops[0])
                if width is None:
                    raise AsmError("imul [mem] needs a size PTR")
                if width == 16:
                    raise AsmError("16-bit imul not supported")
                w = 1 if width == 64 else 0
                self.rm_stmt(out, 0xF6 if width == 8 else 0xF7, w, 5, ops[0],
                             False, fx)
                return bytes(out)
            raise AsmError("bad imul operands")

        # ---- inc/dec/neg/not ----
        if mnem in ("inc", "dec", "neg", "not"):
            width = rm_width(op1)
            if width is None:
                raise AsmError("%s [mem] needs a byte/word/dword/qword PTR size"
                               % mnem)
            if width == 16:
                raise AsmError("16-bit %s not supported" % mnem)
            ext = {"inc": 0, "dec": 1, "not": 2, "neg": 3}[mnem]
            if mnem in ("inc", "dec"):
                opcode = 0xFE if width == 8 else 0xFF
            else:
                opcode = 0xF6 if width == 8 else 0xF7
            w = 1 if width == 64 else 0
            self.rm_stmt(out, opcode, w, ext, op1, byte_reg(op1), fx)
            return bytes(out)

        # ---- shifts ----
        if mnem in SHIFTS:
            ext = SHIFTS[mnem]
            width = rm_width(op1)
            if width is None:
                raise AsmError("shift [mem] needs a byte/word/dword/qword PTR "
                               "size")
            if width == 16:
                raise AsmError("16-bit shift not supported")
            w8 = width == 8
            w = 1 if width == 64 else 0
            if op2.kind == "imm":
                if op2.value is None:
                    raise AsmError("shift count must be numeric")
                v = op2.value
                if not (0 <= v <= 63):
                    raise AsmError("shift count out of range (%d)" % v)
                # GNU as uses the D0/D1 short form for a count of 1
                if v == 1:
                    self.rm_stmt(out, 0xD0 if w8 else 0xD1, w, ext, op1,
                                 byte_reg(op1), fx)
                    return bytes(out)
                self.rm_stmt(out, 0xC0 if w8 else 0xC1, w, ext, op1,
                             byte_reg(op1), fx)
                out += bytes([v])
                return bytes(out)
            if op2.kind == "reg" and op2.name == "cl":
                self.rm_stmt(out, 0xD2 if w8 else 0xD3, w, ext, op1,
                             byte_reg(op1), fx)
                return bytes(out)
            raise AsmError("bad shift operands")

        # ---- SSE ----
        if mnem in SSE:
            return self.enc_sse(mnem, ops, out, fx, resolve)

        # ---- AVX/AVX2/FMA (VEX) ----
        if mnem in VEX:
            return self.enc_vex(mnem, ops, out, fx, resolve)

        raise AsmError("unknown instruction %r" % mnem)

    def imm(self, op, resolve):
        """Value of an immediate. A label resolves to base + offset, i.e. its
        linked address, which is what GNU as + ld put in the instruction."""
        if op.value is not None:
            return op.value
        if resolve:
            try:
                return self.label_offset(op.expr) + self.base
            except AsmError:
                return None
        return None

    def label_offset(self, expr, table=None):
        """Offset of a label reference, optional addend included. `table`
        lets the layout pass resolve against candidate addresses. Labels
        from other modules (the B3 linker's extern map) answer in the
        same coordinate space: linked address minus `base`."""
        table = self.labels if table is None else table
        m = _IDENT_OFF.match(expr.strip())
        if not m:
            raise AsmError("unknown label %r" % expr)
        if m.group(1) in table:
            v = table[m.group(1)]
        elif m.group(1) in self.extern:
            v = self.extern[m.group(1)] - self.base
        else:
            raise AsmError("unknown label %r" % expr)
        if m.group(2):
            n = int(m.group(3), 0)
            v += n if m.group(2) == "+" else -n
        return v

    def rm_head(self, w, ext, op):
        """REX bytes for rm-form (no reg-field register)."""
        if op.kind == "reg":
            return rex_maybe(rex_byte(w, 0, 0, op.num), byte_reg(op))
        _, xb, bb, _ = modrm_mem(op, ext)
        return rex_maybe(rex_byte(w, 0, xb, bb), byte_reg(op))

    def rm_tail(self, out, ext, op, fx):
        """Append modrm (+ SIB + displacement) for an r/m operand whose REX
        and opcode have already been written into `out`. Records the
        rip-relative fixup, which no other helper can see."""
        if op.kind == "reg":
            out += bytes([0xC0 | ((ext & 7) << 3) | (op.num & 7)])
            return out
        mb, _, _, kind = modrm_mem(op, ext)
        out += mb
        self.append_disp(out, op, kind, fx)
        return out

    def append_disp(self, out, op, kind, fx):
        """Displacement of a memory operand; a rip-relative one is a fixup."""
        if kind == "rip":
            fx(len(out), "rip", op.expr)
            out += b"\x00\x00\x00\x00"
        elif kind == "a32":
            fx(len(out), "abs32", op.expr)
            out += b"\x00\x00\x00\x00"
        else:
            out += disp_bytes(kind, op.value or 0)
        return out

    def rm_stmt(self, out, opcode, w, ext, op, force, fx):
        """Append REX + opcode + modrm + displacement of an r/m-form
        instruction (the reg field carries an opcode extension)."""
        out += self.rm_head(w, ext, op) + bytes([opcode])
        self.rm_tail(out, ext, op, fx)
        return out

    def rm_reg_stmt(self, out, opcode, w, reg, op, force, fx):
        """Append an r/m,reg-form instruction: REX (reg field = register),
        opcode (possibly two bytes), modrm, displacement."""
        ob = opcode if isinstance(opcode, (bytes, bytearray)) else bytes([opcode])
        if op.kind == "reg":
            out += rex_maybe(rex_byte(w, reg.num, 0, op.num), force) + ob + \
                bytes([0xC0 | ((reg.num & 7) << 3) | (op.num & 7)])
            return out
        mb, xb, bb, kind = modrm_mem(op, reg.num & 7)
        out += rex_maybe(rex_byte(w, reg.num, xb, bb), force) + ob + mb
        self.append_disp(out, op, kind, fx)
        return out

    def enc_mov(self, op1, op2, out, fix, resolve, force64=False):
        if op1 is None or op2 is None:
            raise AsmError("bad mov operands")
        # ---- movabs: a 64-bit immediate on any r64, or an absolute
        # moffs64 accumulator access - GNU as accepts nothing else
        # (loads 48 A1 / A1 / A0, stores 48 A3 / A3 / A2, then an
        # 8-byte address field with a 64-bit relocation) ----
        if force64:
            if op1.kind == "mem" and op2.kind == "reg":
                if op2.xmm or op2.num != 0 or op2.width not in (8, 32, 64):
                    raise AsmError("movabs store needs the accumulator "
                                   "(al/eax/rax)")
                if op1.size is not None and op1.size != op2.width:
                    raise AsmError("movabs PTR size mismatch")
                if op1.rip or op1.expr is None or op1.base is not None \
                        or op1.index is not None:
                    raise AsmError("movabs memory operand must be an "
                                   "absolute address")
                out += rex_maybe(rex_byte(1 if op2.width == 64 else 0,
                                          0, 0, 0), False)
                out += bytes([0xA2 if op2.width == 8 else 0xA3])
                fix(len(out), "abs64", op1.expr)
                out += b"\x00" * 8
                return bytes(out)
            if op1.kind != "reg" or op1.xmm or op1.width not in (8, 32, 64):
                raise AsmError("bad movabs operands")
            if op2.kind == "mem":
                if op1.num != 0:
                    raise AsmError("movabs load needs the accumulator "
                                   "(al/eax/rax)")
                if op2.size is not None and op2.size != op1.width:
                    raise AsmError("movabs PTR size mismatch")
                if op2.rip or op2.expr is None or op2.base is not None \
                        or op2.index is not None:
                    raise AsmError("movabs memory operand must be an "
                                   "absolute address")
                out += rex_maybe(rex_byte(1 if op1.width == 64 else 0,
                                          0, 0, 0), False)
                out += bytes([0xA0 if op1.width == 8 else 0xA1])
                fix(len(out), "abs64", op2.expr)
                out += b"\x00" * 8
                return bytes(out)
            if op2.kind != "imm":
                raise AsmError("bad movabs operands")
            if op1.width != 64:
                raise AsmError("movabs immediate needs a 64-bit register")
            # the 64-bit immediate path below honors force64 by always
            # emitting the full 10-byte form
        # ---- mov r/m, r  (88/89 /r) - also the reg,reg form ----
        if op2.kind == "reg" and not op2.xmm and op1.kind in ("reg", "mem"):
            if op2.width == 16:
                raise AsmError("16-bit mov not supported")
            check_size(op1, op2.width)
            if op1.kind == "reg" and op1.width != op2.width:
                raise AsmError("mov width mismatch")
            w = 1 if op2.width == 64 else 0
            opc = 0x88 if op2.width == 8 else 0x89
            force = byte_reg(op1) or byte_reg(op2)
            if op1.kind == "reg":
                return rex_maybe(rex_byte(w, op2.num, 0, op1.num), force) + \
                    bytes([opc, 0xC0 | ((op2.num & 7) << 3) | (op1.num & 7)])
            self.rm_reg_stmt(out, opc, w, op2, op1, force, fix)
            return bytes(out)
        # ---- mov r, r/m  (8A/8B /r) ----
        if op1.kind == "reg" and not op1.xmm and op2.kind == "mem":
            if op1.width == 16:
                raise AsmError("16-bit mov not supported")
            check_size(op2, op1.width)
            w = 1 if op1.width == 64 else 0
            self.rm_reg_stmt(out, 0x8A if op1.width == 8 else 0x8B, w, op1,
                             op2, byte_reg(op1), fix)
            return bytes(out)
        # ---- mov reg, imm ----
        if op1.kind == "reg" and not op1.xmm and op2.kind == "imm":
            if op1.width == 8:
                if op2.value is None:
                    # `mov r8, offset label`: the address in an 8-bit
                    # field with an 8-bit relocation (B0, like GNU as)
                    out += rex_maybe(rex_byte(0, 0, 0, op1.num),
                                     byte_reg(op1)) + \
                        bytes([0xB0 + (op1.num & 7)])
                    fix(len(out), "abs8", op2.expr)
                    out += b"\x00"
                    return bytes(out)
                if not (-0x80 <= op2.value <= 0xFF):
                    raise AsmError("mov r8 imm out of range (%d)" % op2.value)
                return rex_maybe(rex_byte(0, 0, 0, op1.num), byte_reg(op1)) + \
                    bytes([0xB0 + (op1.num & 7), op2.value & 0xFF])
            if op1.width == 64:
                if force64:
                    # movabs always emits the full 10-byte form (GNU as
                    # never shrinks it to imm32)
                    out += rex_maybe(rex_byte(1, 0, 0, op1.num), False) + \
                        bytes([0xB8 + (op1.num & 7)])
                    if op2.value is None:
                        fix(len(out), "abs64", op2.expr)
                        out += b"\x00" * 8
                    else:
                        out += (op2.value & 0xFFFFFFFFFFFFFFFF) \
                            .to_bytes(8, "little")
                    return bytes(out)
                if op2.value is None:
                    # `mov r64, offset label`: GNU as uses the
                    # sign-extended imm32 form with a 32-bit relocation
                    # (48 C7 C0 imm32), not movabs
                    out += rex_maybe(rex_byte(1, 0, 0, op1.num), False) + \
                        bytes([0xC7, 0xC0 | (op1.num & 7)])
                    fix(len(out), "abs32", op2.expr)
                    out += b"\x00\x00\x00\x00"
                    return bytes(out)
                if fits_imm(op2.value, 64, 32):
                    out += rex_maybe(rex_byte(1, 0, 0, op1.num), False) + \
                        bytes([0xC7, 0xC0 | (op1.num & 7)])
                    out += (op2.value & 0xFFFFFFFF).to_bytes(4, "little")
                    return bytes(out)
                # movabs: a 64-bit constant
                out += rex_maybe(rex_byte(1, 0, 0, op1.num), False) + \
                    bytes([0xB8 + (op1.num & 7)])
                out += (op2.value & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
                return bytes(out)
            if op1.width == 32:
                out += rex_maybe(rex_byte(0, 0, 0, op1.num), False) + \
                    bytes([0xB8 + (op1.num & 7)])
                if op2.value is None:
                    # `mov r32, offset label`: B8 imm32 + a 32-bit
                    # relocation (GNU as)
                    fix(len(out), "abs32", op2.expr)
                    out += b"\x00" * 4
                else:
                    out += (op2.value & 0xFFFFFFFF).to_bytes(4, "little")
                return bytes(out)
            raise AsmError("16-bit mov imm not supported")
        # ---- mov r/m, imm  (C6/C7 /0) ----
        if op1.kind == "mem" and op2.kind == "imm":
            size = op1.size
            if size not in (8, 32, 64):
                raise AsmError("mov [mem], imm needs byte/dword/qword PTR")
            if op2.value is None:
                raise AsmError("mov [mem], label is not supported")
            w = 1 if size == 64 else 0
            self.rm_stmt(out, 0xC6 if size == 8 else 0xC7, w, 0, op1, False, fix)
            n = 1 if size == 8 else 4
            out += (op2.value & ((1 << (8 * n)) - 1)).to_bytes(n, "little")
            return bytes(out)
        raise AsmError("bad mov operands")

    def enc_sse(self, mnem, ops, out, fx, resolve):
        op1 = ops[0]
        op2 = ops[1] if len(ops) > 1 else None
        # movd/movq and cvtsi2ss/sd take a GP register source operand
        gp_ok = mnem in ("movd", "movq", "cvtsi2ss", "cvtsi2sd")
        # movq xmm, xmm is F3 0F 7E in GNU as; the 66 0F 6E form below is a
        # move from a GP register and would assemble to the wrong thing.
        if mnem == "movq" and op1.kind == "reg" and op1.xmm \
                and op2 is not None and op2.kind == "reg" and op2.xmm:
            return bytes(PF3) + rex_maybe(rex_byte(0, op1.num, 0, op2.num),
                                          False) + \
                bytes([0x0F, 0x7E, 0xC0 | ((op1.num & 7) << 3) | (op2.num & 7)])
        for form in SSE[mnem]:
            kind, pfx, opc = form[0], form[1], form[2]
            ext = form[3] if len(form) > 3 else 0
            regf = rm = None
            if kind == "lr" and op1.kind == "reg" and op1.xmm and op2 is not None:
                if op2.kind == "mem" or (op2.kind == "reg" and op2.xmm) \
                        or (gp_ok and op2.kind == "reg" and not op2.xmm):
                    regf, rm = op1, op2
            elif kind == "lm" and op1.kind == "reg" and op1.xmm \
                    and op2 is not None and op2.kind == "mem":
                regf, rm = op1, op2
            elif kind == "sm" and op1.kind == "mem" and op2 is not None \
                    and op2.kind == "reg" and op2.xmm:
                regf, rm = op2, op1
            elif kind == "rl" and op2 is not None and op2.kind == "reg" \
                    and op2.xmm:
                if op1.kind == "mem" or (op1.kind == "reg" and op1.xmm) \
                        or (gp_ok and op1.kind == "reg" and not op1.xmm):
                    regf, rm = op2, op1
            elif kind == "ext" and op1.kind == "reg" and op1.xmm \
                    and op2 is not None and op2.kind == "imm":
                regf, rm = ext, op1
            elif kind == "gr" and op1.kind == "reg" and not op1.xmm \
                    and op2 is not None and \
                    ((op2.kind == "reg" and op2.xmm) or op2.kind == "mem"):
                regf, rm = op1, op2
            else:
                continue
            if rm is None:
                continue
            w = 0
            if mnem in ("cvtsi2ss", "cvtsi2sd"):
                if rm.kind == "mem":
                    if rm.size == 64:
                        w = 1
                    elif rm.size == 32:
                        w = 0
                    else:
                        continue   # needs an explicit dword/qword PTR
                elif rm.xmm:
                    continue
                elif rm.width == 64:
                    w = 1
                elif rm.width != 32:
                    continue
            if mnem in ("cvtss2si", "cvttsd2si", "cvttss2si"):
                if op1.width == 64:
                    w = 1
            if mnem == "movq" and rm.kind == "reg" and not rm.xmm:
                if rm.width == 64:
                    w = 1
                elif rm.width != 32:
                    continue
            if mnem == "movd" and rm.kind == "reg" and not rm.xmm \
                    and rm.width != 32:
                continue
            rf = regf if isinstance(regf, int) else regf.num
            if rm.kind == "reg":
                out += bytes(pfx) + rex_maybe(rex_byte(w, rf, 0, rm.num),
                                              False) + bytes(opc) + \
                    bytes([0xC0 | ((rf & 7) << 3) | (rm.num & 7)])
            else:
                mb, xb, bb, dkind = modrm_mem(rm, rf)
                out += bytes(pfx) + rex_maybe(rex_byte(w, rf, xb, bb), False) \
                    + bytes(opc) + mb
                self.append_disp(out, rm, dkind, fx)
            if kind == "ext":
                v = self.imm(op2, resolve)
                if v is None or not (0 <= v <= 255):
                    raise AsmError("sse imm out of range")
                out += bytes([v])
            return bytes(out)
        raise AsmError("no form of %r matches operands" % mnem)


    def enc_vex(self, mnem, ops, out, fx, resolve):
        """VEX-encoded instructions (VEX table). Field placement matches
        GNU as byte for byte; see the docstring conventions."""
        op1 = ops[0] if len(ops) > 0 else None
        op2 = ops[1] if len(ops) > 1 else None
        op3 = ops[2] if len(ops) > 2 else None
        op4 = ops[3] if len(ops) > 3 else None

        def vec_reg(op):
            return op is not None and op.kind == "reg" and op.xmm

        def mem_or_vec(op):
            return op is not None and (op.kind == "mem"
                                       or (op.kind == "reg" and op.xmm))

        for form in VEX[mnem]:
            kind, pp, vmap, w, opc = form[:5]
            ext = form[5] if len(form) > 5 else 0
            vvvv = None
            regf = 0         # ModRM.reg: a register number or an extension
            rm = None
            imm = None
            no_modrm = False
            if kind == "none":
                if ops:
                    continue
                no_modrm = True
            elif kind == "lr":
                if len(ops) != 2 or not vec_reg(op1) or not mem_or_vec(op2):
                    continue
                regf, rm = op1.num, op2
            elif kind == "rl":
                if len(ops) != 2 or op1.kind != "mem" or not vec_reg(op2):
                    continue
                regf, rm = op2.num, op1
            elif kind == "lrr":
                if len(ops) != 3 or not vec_reg(op1) or not vec_reg(op2) \
                        or not mem_or_vec(op3):
                    continue
                vvvv, regf, rm = op2.num, op1.num, op3
            elif kind == "lrrI":
                if len(ops) != 4 or not vec_reg(op1) or not vec_reg(op2) \
                        or not mem_or_vec(op3) or op4.kind != "imm":
                    continue
                vvvv, regf, rm, imm = op2.num, op1.num, op3, op4
            elif kind == "lri":
                # shift by immediate: GNU as's VEX template takes a
                # REGISTER source (its mem form goes to an EVEX/AVX-512
                # template, outside this subset), so we require one too
                if len(ops) != 3 or not vec_reg(op1) or op2.kind != "reg" \
                        or not op2.xmm or op3.kind != "imm":
                    continue
                vvvv, regf, rm, imm = op1.num, ext, op2, op3
            elif kind == "rli":
                if len(ops) != 3 or not mem_or_vec(op1) or not vec_reg(op2) \
                        or op3.kind != "imm":
                    continue
                regf, rm, imm = op2.num, op1, op3
            elif kind == "gr":
                if len(ops) != 2 or op1.kind != "reg" or op1.xmm \
                        or op1.width != 32 or not mem_or_vec(op2):
                    continue
                regf, rm = op1.num, op2
            else:
                raise AsmError("bad VEX form kind %r" % kind)

            # ---- width rules (GNU as rejects a mismatch, so do we) ----
            if kind == "lr" and mnem == "vbroadcastss":
                if rm.kind == "reg":
                    if rm.width != 128:
                        raise AsmError("vbroadcastss source must be xmm")
                else:
                    check_size(rm, 32)
            elif kind == "lr" and mnem in ("vpmovzxbd", "vpmovzxwd"):
                if rm.kind == "reg":
                    if rm.width != 128:
                        raise AsmError("%s source must be xmm" % mnem)
                else:
                    wide = op1.width == 256
                    want = {("vpmovzxbd", False): 32, ("vpmovzxbd", True): 64,
                            ("vpmovzxwd", False): 64,
                            ("vpmovzxwd", True): 128}[(mnem, wide)]
                    check_size(rm, want)
            elif kind == "lr":
                if rm.kind == "reg":
                    if rm.width != op1.width:
                        raise AsmError("%s width mismatch" % mnem)
                else:
                    check_size(rm, op1.width)
            elif kind == "rl":
                check_size(op1, op2.width)
            elif kind in ("lrr", "lrrI"):
                if op1.width != op2.width:
                    raise AsmError("%s width mismatch" % mnem)
                if mnem in ("vpslld", "vpsrlw"):
                    # shift count form: GNU as's Intel syntax takes an XMM
                    # count register (or an xmmword PTR) whatever the
                    # destination width is
                    if rm.kind == "reg":
                        if rm.width != 128:
                            raise AsmError("%s count source must be xmm"
                                           % mnem)
                    else:
                        check_size(rm, 128)
                elif rm.kind == "reg":
                    if rm.width != op1.width:
                        raise AsmError("%s width mismatch" % mnem)
                else:
                    check_size(rm, op1.width)
            elif kind == "lri":
                if rm.width != op1.width:
                    raise AsmError("%s width mismatch" % mnem)
            elif kind == "rli":
                if op2.width != 256:
                    raise AsmError("%s source must be ymm" % mnem)
                if op1.kind == "reg" and op1.width != 128:
                    raise AsmError("%s destination must be xmm" % mnem)
                check_size(op1, 128)

            # GNU as emits the SHORTER direction for a two-register VEX
            # move: the load form needs the 3-byte C4 prefix when its r/m
            # register has bit 3 set, and the store form needs it when the
            # other one does - so when the load needs C4 and the store
            # does not, it flips to the store opcode with the fields
            # swapped (ties keep the load form). Width checks above already
            # ran on the original operands.
            if kind == "lr" and rm.kind == "reg" and mnem in VEX_MOVE \
                    and (rm.num >> 3) & 1 and not (regf >> 3) & 1:
                opc = VEX_MOVE[mnem]
                regf, rm = rm.num, op1

            v = self.imm(imm, resolve) if imm is not None else None
            if imm is not None and (v is None or not (0 <= v <= 255)):
                raise AsmError("%s imm8 out of range" % mnem)
            # VEX.L comes from the vector register width: 256-bit forms
            # (dest, vvvv, rm or the ymm source of vextractf128)
            l = 1 if any(vec_reg(o) and o.width == 256
                         for o in (op1, op2, op3)) else 0
            if no_modrm:
                out += vex_prefix(vvvv, l, pp, vmap, w, 0, 0, 0) \
                    + bytes([opc])
                return bytes(out)
            if rm.kind == "reg":
                out += vex_prefix(vvvv, l, pp, vmap, w, regf, 0, rm.num) \
                    + bytes([opc, 0xC0 | ((regf & 7) << 3) | (rm.num & 7)])
            else:
                mb, xb, bb, dkind = modrm_mem(rm, regf & 7)
                out += vex_prefix(vvvv, l, pp, vmap, w, regf, xb, bb) \
                    + bytes([opc]) + mb
                self.append_disp(out, rm, dkind, fx)
            if imm is not None:
                out += bytes([v])
            return bytes(out)
        raise AsmError("no form of %r matches operands" % mnem)


def assemble(text, base=0, extern=None):
    """Assemble Intel-syntax text -> (code_bytes, entry_offset).

    `base` is the address the code is loaded at. Only absolute fixups
    (.long/.quad label, mov r32/r64 with a label immediate) depend on it;
    branches and [rip+label] operands are position independent.
    `extern` (B3) maps other modules' labels to linked addresses.
    """
    return Assembler(base, extern).assemble(text)


def assemble_map(text, base=0, extern=None):
    """(code, entry, stmts) with stmts = [(lineno, source, offset, size)].
    The pass check uses the map to name the first line that differs from
    GNU as instead of just reporting a byte offset."""
    a = Assembler(base, extern)
    code, entry = a.assemble(text)
    return code, entry, a.stmt_map


def main():
    if len(sys.argv) < 2:
        print("usage: asm.py file.asm [-o out.bin] [--base 0x...]")
        return 2
    src = sys.argv[1]
    out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else src + ".bin"
    base = 0
    if "--base" in sys.argv:
        base = int(sys.argv[sys.argv.index("--base") + 1], 0)
    with open(src, "r", encoding="utf-8") as f:
        code, entry = assemble(f.read(), base)
    with open(out, "wb") as f:
        f.write(code)
    print("%s -> %s: %d bytes, entry 0x%x" % (src, out, len(code), entry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
