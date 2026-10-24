#!/usr/bin/env python3
"""make_a3bench_data.py - build a3bench.bin for the A3 old-PC benchmark.

File format (little-endian; parsed by a3/bench.c):
    "A3B1" | u32 nmat | per matrix:
        u32 rows, u32 cols, u32 n_exc,
        15B exponent table,
        bf16[rows*cols]   (u16 LE weights, verbatim from weights.bin),
        sa[rows*cols]     ((sign<<7)|mant bytes, verbatim from packed.bin),
        sb[rows*cols/2]   (planar nibbles, 64-weight groups, verbatim from packed.bin),
        exc[n_exc*5]      (u32 LE flat index + u8 true exponent, strictly ascending;
                           packed.bin's 4-byte count prefix is NOT copied - bench.c
                           takes n_exc from the matrix header)

Modes:
  --real (default): slice the first gate/up/down matrices of the real Qwen2.5-3B
    export until --target bytes of BF16 are collected.
    Inputs:  $AIOS_TMP/qwen2_5_3b/{weights.bin,manifest.txt}
             $AIOS_TMP/packed/{packed.bin,packed_index.txt}
    After writing, a numpy decoder re-derives every BF16 bit from the packed
    streams in the file and asserts 0 mismatches (the answer key for the
    Windows run, A1/A4-style threshold: 0).
  --synthetic: small random matrices with a realistic exponent mix plus a
    uniform tail that forces escape records; no model files needed. Used to
    smoke-test bench64.exe/bench32.exe end to end (bit-identity of the three
    kernels + float64 reference) without shipping 158 MB of real weights.
"""
import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase2"))
from pack import pick_table, pack_codes_planar  # noqa: E402  (proven A1 logic)

GROUP = 64


def pack_matrix(u16):
    """(rows, cols) uint16 weights -> (table15, sa, sb_bytes, exc_idx, exc_exp).

    Identical math to phase2/pack.py (the A1 packer that passed 0-mismatch
    unpack verification): sign/mant to stream A, per-tensor top-15 exponent
    table to codes, planar nibble packing to stream B, escapes to a sorted
    exception list.
    """
    rows, cols = u16.shape
    assert cols % GROUP == 0, "cols %d not a multiple of %d" % (cols, GROUP)
    sign = ((u16 >> 15) & 1).astype(np.uint8)
    exp = ((u16 >> 7) & 0xFF).astype(np.uint8)
    mant = (u16 & 0x7F).astype(np.uint8)
    sa = ((sign << 7) | mant).astype(np.uint8)
    counts = np.bincount(exp.ravel(), minlength=256).astype(np.int64)
    table, code_of = pick_table(counts)
    codes = code_of[exp]
    exc_mask = codes == 15
    exc_idx = np.flatnonzero(exc_mask).astype("<u4")
    exc_exp = exp[exc_mask].astype(np.uint8)
    sb = pack_codes_planar(codes.ravel().copy(), rows, cols)
    assert len(sb) == rows * cols // 2, (len(sb), rows * cols // 2)
    return table, sa, sb, exc_idx, exc_exp


def decode_bits(table, sa, sb, exc_idx, exc_exp, nrows, cols):
    """Decode the packed streams back to bf16 bits (uint16, row-major).

    Same reconstruction the C kernels do in registers: codes -> table lookup,
    escape lanes patched from the exception list, sign/mant from stream A.
    """
    codes_b = np.frombuffer(sb, dtype=np.uint8).reshape(nrows, cols // GROUP, 32)
    low = codes_b & 0x0F
    high = codes_b >> 4
    # planar: byte i holds codes i (low) and i+32 (high) of the 64-group
    codes = np.concatenate([low, high], axis=2).reshape(-1)
    n_exc = int(exc_idx.size)
    assert int((codes == 15).sum()) == n_exc, "escape count mismatch"
    exp = table[np.minimum(codes, 14)].astype(np.uint8)
    if n_exc:
        exp[exc_idx.astype(np.int64)] = exc_exp
    sa_a = np.frombuffer(sa, dtype=np.uint8)
    sign = (sa_a >> 7).astype(np.uint16)
    mant = (sa_a & 0x7F).astype(np.uint16)
    return (sign << 15) | (exp.astype(np.uint16) << 7) | mant


def exc_records_bytes(exc_idx, exc_exp):
    n = int(exc_idx.size)
    rec = np.empty((n, 5), dtype=np.uint8)
    rec[:, 0] = exc_idx & 0xFF
    rec[:, 1] = (exc_idx >> 8) & 0xFF
    rec[:, 2] = (exc_idx >> 16) & 0xFF
    rec[:, 3] = (exc_idx >> 24) & 0xFF
    rec[:, 4] = exc_exp
    return rec.tobytes()


def write_matrix(fh, u16, name, log):
    rows, cols = u16.shape
    table, sa, sb, exc_idx, exc_exp = pack_matrix(u16)
    n = rows * cols
    n_exc = int(exc_idx.size)
    # cross-check BEFORE writing: decode must reproduce every bf16 bit
    got = decode_bits(table, sa, sb, exc_idx, exc_exp, rows, cols)
    bad = int((got != u16.ravel().astype(np.uint16)).sum())
    if bad:
        raise AssertionError("%s: packed decode mismatched %d/%d bits" % (name, bad, n))
    if n_exc:
        idx64 = exc_idx.astype(np.int64)
        assert np.all(np.diff(idx64) > 0), "%s: exception indices not ascending" % name
    fh.write(struct.pack("<III", rows, cols, n_exc))
    fh.write(table.astype(np.uint8).tobytes())
    fh.write(u16.astype("<u2").tobytes())
    fh.write(sa.tobytes())
    fh.write(sb)
    fh.write(exc_records_bytes(exc_idx, exc_exp))
    log("  %-28s rows=%-6d cols=%-6d n_exc=%-8d decode_bad=%d" % (name, rows, cols, n_exc, bad))
    return n, n_exc


def read_at(path, off, n, fh):
    fh.seek(off)
    b = fh.read(n)
    if len(b) != n:
        raise IOError("short read at %d+%d in %s" % (off, n, path))
    return b


def real(out_path, tmp, target, log):
    d = os.path.join(tmp, "qwen2_5_3b")
    pk = os.path.join(tmp, "packed")
    man = []
    with open(os.path.join(d, "manifest.txt")) as f:
        for line in f:
            idx, name, dtype, rows, cols, off, nb = line.split()
            assert dtype == "bf16", dtype
            man.append((name, int(rows), int(cols), int(off), int(nb)))
    pidx = {}
    with open(os.path.join(pk, "packed_index.txt")) as f:
        for line in f:
            parts = line.split()
            pidx[parts[0]] = parts
    picked = []
    total = 0
    for name, rows, cols, off, nb in man:
        if not (name.endswith(".gate.weight") or name.endswith(".up.weight")
                or name.endswith(".down.weight")):
            continue
        if total >= target:
            break
        assert nb == rows * cols * 2, (name, nb, rows, cols)
        assert name in pidx, "no packed record for %s (name mismatch?)" % name
        picked.append((name, rows, cols, off))
        total += rows * cols * 2
    if not picked:
        raise AssertionError("no gate/up/down matrices found in manifest")
    log("picked %d matrices, %.1f MB bf16 (target %.1f MB)"
        % (len(picked), total / 1e6, target / 1e6))
    n_total = 0
    exc_total = 0
    with open(out_path, "wb") as out, \
            open(os.path.join(d, "weights.bin"), "rb") as wf, \
            open(os.path.join(pk, "packed.bin"), "rb") as pf:
        out.write(b"A3B1")
        out.write(struct.pack("<I", len(picked)))
        for name, rows, cols, off in picked:
            p = pidx[name]
            n = int(p[1])
            assert n == rows * cols, (name, n, rows, cols)
            assert int(p[9]) == off, "%s: orig_off %s != manifest off %d" % (name, p[9], off)
            tbl_off, sa_off, sb_off, exc_off = (int(p[4]), int(p[5]), int(p[6]), int(p[7]))
            n_exc = int(p[8])
            tbl = read_at(os.path.join(pk, "packed.bin"), tbl_off, 15, pf)
            bf16 = read_at(os.path.join(d, "weights.bin"), off, n * 2, wf)
            sa = read_at(os.path.join(pk, "packed.bin"), sa_off, n, pf)
            sb = read_at(os.path.join(pk, "packed.bin"), sb_off, n // 2, pf)
            exc = read_at(os.path.join(pk, "packed.bin"), exc_off + 4, n_exc * 5, pf) \
                if n_exc else b""
            # cross-check the sliced bytes reproduce the weights exactly
            u16 = np.frombuffer(bf16, dtype="<u2").reshape(rows, cols)
            got = decode_bits(np.frombuffer(tbl, dtype=np.uint8), sa, sb,
                              np.frombuffer(exc, dtype=np.uint8).reshape(n_exc, 5)[:, 0:4]
                              .copy().view("<u4").reshape(-1) if n_exc else np.zeros(0, "<u4"),
                              np.frombuffer(exc, dtype=np.uint8).reshape(n_exc, 5)[:, 4]
                              .copy() if n_exc else np.zeros(0, np.uint8),
                              rows, cols)
            bad = int((got != u16.ravel()).sum())
            if bad:
                raise AssertionError("%s: sliced decode mismatched %d/%d bits" % (name, bad, n))
            if n_exc:
                recs = np.frombuffer(exc, dtype=np.uint8).reshape(n_exc, 5)
                idx = (recs[:, 0].astype(np.int64) | (recs[:, 1].astype(np.int64) << 8)
                       | (recs[:, 2].astype(np.int64) << 16) | (recs[:, 3].astype(np.int64) << 24))
                assert np.all(np.diff(idx) > 0), "%s: exceptions not ascending" % name
            out.write(struct.pack("<III", rows, cols, n_exc))
            out.write(tbl)
            out.write(bf16)
            out.write(sa)
            out.write(sb)
            out.write(exc)
            log("  %-28s rows=%-6d cols=%-6d n_exc=%-8d decode_bad=%d"
                % (name, rows, cols, n_exc, bad))
            n_total += n
            exc_total += n_exc
    return len(picked), n_total, exc_total


def synthetic(out_path, log, seed=1234):
    rng = np.random.RandomState(seed)
    specs = [(256, 256), (128, 320)]  # cols multiples of 64
    with open(out_path, "wb") as out:
        out.write(b"A3B1")
        out.write(struct.pack("<I", len(specs)))
        n_total = 0
        exc_total = 0
        for i, (rows, cols) in enumerate(specs):
            exp = rng.normal(122.0, 7.0, size=(rows, cols))
            tail = rng.uniform(0, 1, size=(rows, cols)) < 0.02
            exp[tail] = rng.randint(0, 256, size=int(tail.sum()))
            exp = np.clip(exp, 0, 255).astype(np.uint16)
            sign = rng.randint(0, 2, size=(rows, cols)).astype(np.uint16)
            mant = rng.randint(0, 128, size=(rows, cols)).astype(np.uint16)
            u16 = (sign << 15) | (exp << 7) | mant
            n, n_exc = write_matrix(out, u16.astype(np.uint16), "synthetic%d" % i, log)
            n_total += n
            exc_total += n_exc
    return len(specs), n_total, exc_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("real", "synthetic"), default="real")
    ap.add_argument("--out", default=os.path.join(HERE, "a3bench.bin"))
    ap.add_argument("--target", type=int, default=90_177_536,
                    help="collect gate/up/down matrices until this many BF16 bytes "
                         "(default 90177536 = 2 x 11008x2048 matrices, ~158 MB file, "
                         "~350 MB RAM in bench.c - safe for a DDR2-era PC)")
    ap.add_argument("--tmp", default=os.environ.get("AIOS_TMP", "/tmp/aios_native"))
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    log("=== make_a3bench_data (%s) ===" % args.mode)
    if args.mode == "real":
        nmat, n_total, exc_total = real(args.out, args.tmp, args.target, log)
    else:
        nmat, n_total, exc_total = synthetic(args.out, log)
    sz = os.path.getsize(args.out)
    log("matrices       : %d" % nmat)
    log("weights        : %d" % n_total)
    log("exceptions     : %d" % exc_total)
    log("file bytes     : %d (%.1f MB)" % (sz, sz / 1e6))
    log("bf16 bytes     : %d (in-file raw stream)" % (n_total * 2))
    with open(os.path.join(HERE, "a3bench_stats.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
