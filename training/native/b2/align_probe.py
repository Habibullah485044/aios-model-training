#!/usr/bin/env python3
"""align_probe.py - measure GNU as's `.align` pad bytes in .text for
LARGE pads. The B3 programs align their data segment to 4096 so the ELF
writer can split R-X/RW- segments on a page boundary; pads that big were
never measured (code_align_fill was derived for k=1..126), and at
k>=130 the short jmp no longer fits, so gas must switch forms.

Method: for each pad size k, assemble a source of exactly 4096-k bytes
of instructions followed by `.align 4096`, with GNU as; extract the pad
bytes (bytes [4096-k, 4096) of .text) and compare against the candidate
rule. The probe also prints the structure of the pad for the sample
sizes (leading jmp form + canonical-nop piece lengths) so the rule can
be read off the data.

Linux only (WSL/Kaggle, D14). Exit 0 if the candidate matches every
probed k, 1 otherwise. Results also land in b2/align_probe_results.txt.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "align_probe_results.txt")

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

REPORT = []


def say(line):
    print(line, flush=True)
    REPORT.append(line)


def decomp_nops(m):
    q, r = divmod(m, 11)
    return CANON[r] + CANON[11] * q


def candidate(k):
    """Candidate rule for gas's .align pad of k bytes after CODE."""
    if k <= 0:
        return b""
    if k < 88:
        return decomp_nops(k)
    if k <= 129:
        return bytes([0xEB, (k - 2) & 0xFF]) + decomp_nops(k - 2)
    # k >= 130: near jmp skipping the whole pad, then the remainder
    return bytes([0xE9]) + (k - 5).to_bytes(4, "little") + tail(k - 5)


def tail(m):
    """Fill after the 5-byte near jmp: measured below (placeholder
    hypotheses are printed against the data)."""
    # Hypothesis A: canonical remainder first, then 11-byte nops
    return decomp_nops(m)


def gas_pad(k, tmp):
    """gas's pad bytes for a k-byte .align 4096 after (4096-k) bytes of
    code, from as + objcopy of .text."""
    o = 4096 - k
    src = os.path.join(tmp, "pad_%d.s" % k)
    with open(src, "w", encoding="utf-8") as f:
        f.write("\t.text\n")
        f.write("\tnop\n" * o)
        f.write("\t.align 4096\n")
        f.write("after:\n\tnop\n")
    obj = os.path.join(tmp, "pad_%d.o" % k)
    r = subprocess.run(["as", "--64", "-o", obj, src],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("as failed for k=%d: %s" % (k, r.stderr))
    binf = os.path.join(tmp, "pad_%d.bin" % k)
    r = subprocess.run(["objcopy", "-O", "binary", "--only-section=.text",
                        obj, binf], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("objcopy failed for k=%d: %s" % (k, r.stderr))
    with open(binf, "rb") as f:
        data = f.read()
    return data[o:4096]


def structure(pad, k):
    """Readable structure: leading jmp form + the canonical piece
    lengths of the rest (greedy largest-first match)."""
    i = 0
    lead = "nops"
    if pad[:1] == b"\xeb":
        rel = pad[1] if pad[1] < 128 else pad[1] - 256
        lead = "EB rel8 %+d (lands %s)" % (rel, "at pad end" if i + 2 + rel == k
                                           else "at %d" % (2 + rel))
        i = 2
    elif pad[:1] == b"\xe9":
        rel = int.from_bytes(pad[1:5], "little", signed=True)
        lead = "E9 rel32 %+d (lands %s)" % (rel, "at pad end" if 5 + rel == k
                                            else "at %d" % (5 + rel))
        i = 5
    pieces = []
    while i < len(pad):
        for n in range(11, 0, -1):
            if pad[i:i + n] == CANON[n]:
                pieces.append(n)
                i += n
                break
        else:
            return lead, None, pad[i:i + 16].hex(" ")
    return lead, pieces, None


def short_struct(pieces):
    """[4, 11, 11, ..., 6] -> '4 + 11x368 + 6'."""
    if pieces is None:
        return "?"
    from itertools import groupby
    out = []
    for val, grp in groupby(pieces):
        n = len(list(grp))
        out.append(str(val) if n == 1 else "%dx%d" % (val, n))
    return " + ".join(out)


def main():
    if not sys.platform.startswith("linux"):
        print("FATAL: run this under WSL/Kaggle (Linux), never bare Windows")
        return 2
    tmp = tempfile.mkdtemp(prefix="align_probe_")
    sample = [1, 2, 10, 64, 86, 87, 88, 89, 100, 126, 127, 128, 129, 130,
              131, 132, 133, 140, 150, 200, 255, 256, 257, 258, 300, 500,
              1000, 2000, 4063, 4095]
    say("=== align probe: GNU as .align 4096 pad in .text, %d sample "
        "sizes ===" % len(sample))
    fails = 0
    for k in sample:
        pad = gas_pad(k, tmp)
        lead, pieces, raw = structure(pad, k)
        cand = candidate(k)
        ok = pad == cand
        if not ok:
            fails += 1
        say("k=%-5d %-40s %s%s" % (
            k, lead, short_struct(pieces),
            "" if ok else "   CANDIDATE MISMATCH (cand %d bytes: %s)"
            % (len(cand), cand[:12].hex(" ") + "...")))
        if raw is not None:
            say("      unparsed tail: %s" % raw)
    say("sample: %d/%d pads match the candidate rule" % (
        len(sample) - fails, len(sample)))
    # exhaustive sweep where gas is cheap to generate
    sweep_fail = 0
    sweep = list(range(1, 300)) + [511, 512, 513, 1023, 1024, 1025,
                                   2047, 2048, 2049, 4063, 4095]
    for k in sweep:
        if pad_eq(k, tmp):
            sweep_fail += 1
    say("sweep: %d cases, %d mismatches" % (len(sweep), sweep_fail))
    code = 0 if (fails == 0 and sweep_fail == 0) else 1
    say("align probe %s" % ("PASSED" if code == 0 else "FAILED"))
    try:
        with open(RESULTS, "w") as f:
            f.write("\n".join(REPORT) + "\n")
    except OSError:
        pass
    return code


def pad_eq(k, tmp):
    return gas_pad(k, tmp) != candidate(k)


if __name__ == "__main__":
    sys.exit(main())
