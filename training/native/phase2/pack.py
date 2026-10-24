#!/usr/bin/env python3
"""Phase 2 - packer for the lossless 4-bit exponent format (§4).

Reads the BF16 export (weights.bin + manifest.txt) and writes, per tensor:
  - a 15-entry exponent table (code 0..14 = the 15 most common exponents;
    code 15 = escape -> the true exponent lives in the exception list)
  - stream A: sign+mantissa byte per weight, a = (sign << 7) | mantissa7
  - stream B: 4-bit exponent codes, planar nibble packing in 64-weight
    groups inside each row (low nibble = weight i, high nibble = weight i+32)
  - exception list: sorted by flat index, (uint32 index, uint8 true exponent)

Outputs:
  packed.bin      concatenated tensor records (little-endian)
  packed_index.txt  one line per tensor:
      name n n_rows row_len tbl_off sa_off sb_off exc_off n_exc orig_off

Row lengths for Qwen2.5-3B (2048, 11008, 256) are all multiples of 64, so
every 64-weight group is full. 1-D tensors are packed as a single row.
"""
import os

import numpy as np

TMP = os.environ.get("AIOS_TMP", "/tmp/aios_native")
EXPORT_DIR = os.path.join(TMP, "qwen2_5_3b")
PACK_DIR = os.environ.get("AIOS_PACK_DIR", os.path.join(TMP, "packed"))
GROUP = 64


def read_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            idx, name, dtype, r, c, off, nb = line.split()
            assert dtype == "bf16", dtype
            rows.append((int(idx), name, int(r), int(c), int(off), int(nb)))
    return rows


def shapes(rows, cols):
    """(n_rows, row_len) for planar packing. 1-D tensors -> single row."""
    if cols > 1:
        n_rows, row_len = rows, cols
    else:
        n_rows, row_len = 1, rows
    assert row_len % GROUP == 0, "row_len %d not a multiple of %d" % (row_len, GROUP)
    return n_rows, row_len


def pick_table(counts):
    """Top-15 exponents by (count desc, exponent asc). Returns (table, code_of)."""
    table = np.zeros(15, dtype=np.uint8)
    code_of = np.full(256, 15, dtype=np.uint8)  # 15 = escape
    exps = np.arange(256)
    order = np.lexsort((exps, -counts))  # primary: -count asc == count desc
    top = [int(e) for e in order if counts[e] > 0][:15]
    for i, e in enumerate(top):
        table[i] = e
        code_of[e] = i
    return table, code_of


def pack_codes_planar(codes, n_rows, row_len):
    """Planar nibble packing. codes: uint8 array length n_rows*row_len, values 0..15."""
    g = codes.reshape(n_rows, row_len // GROUP, GROUP)
    low = g[..., :32]
    high = g[..., 32:]
    byts = ((high.astype(np.uint8) << 4) | low).reshape(n_rows, row_len // GROUP, 32)
    return np.ascontiguousarray(byts).reshape(-1).tobytes()


def main():
    os.makedirs(PACK_DIR, exist_ok=True)
    man = read_manifest(os.path.join(EXPORT_DIR, "manifest.txt"))
    raw = np.memmap(os.path.join(EXPORT_DIR, "weights.bin"), dtype=np.uint16, mode="r")

    idx_lines = []
    total_bytes = 0
    total_n = 0
    total_exc = 0

    with open(os.path.join(PACK_DIR, "packed.bin"), "wb") as pf:
        off = 0
        for i, (m_idx, name, rows, cols, orig_off, nb) in enumerate(man):
            assert nb == rows * cols * 2, (name, nb, rows, cols)
            n = rows * cols
            n_rows, row_len = shapes(rows, cols)

            u16 = np.asarray(raw[orig_off // 2: orig_off // 2 + n], dtype=np.uint16)
            sign = ((u16 >> 15) & 1).astype(np.uint8)
            exp = ((u16 >> 7) & 0xFF).astype(np.uint8)
            mant = (u16 & 0x7F).astype(np.uint8)

            stream_a = ((sign << 7) | mant).astype(np.uint8)

            counts = np.bincount(exp, minlength=256).astype(np.int64)
            table, code_of = pick_table(counts)
            codes = code_of[exp]

            exc_mask = codes == 15
            exc_idx = np.flatnonzero(exc_mask).astype("<u4")
            exc_exp = exp[exc_mask].astype(np.uint8)
            n_exc = int(exc_idx.size)

            stream_b = pack_codes_planar(codes, n_rows, row_len)

            # record layout
            tbl_off = off
            pf.write(table.tobytes())
            off += 15

            sa_off = off
            pf.write(stream_a.tobytes())
            off += n

            sb_off = off
            pf.write(stream_b)
            off += len(stream_b)

            exc_off = off
            pf.write(np.uint32(n_exc).astype("<u4").tobytes())
            off += 4
            if n_exc:
                rec = np.empty(n_exc * 5, dtype=np.uint8)
                rec[0::5] = (exc_idx & 0xFF).astype(np.uint8)
                rec[1::5] = ((exc_idx >> 8) & 0xFF).astype(np.uint8)
                rec[2::5] = ((exc_idx >> 16) & 0xFF).astype(np.uint8)
                rec[3::5] = ((exc_idx >> 24) & 0xFF).astype(np.uint8)
                rec[4::5] = exc_exp
                pf.write(rec.tobytes())
                off += n_exc * 5

            idx_lines.append("%s %d %d %d %d %d %d %d %d %d"
                             % (name, n, n_rows, row_len, tbl_off, sa_off, sb_off,
                                exc_off, n_exc, orig_off))
            total_bytes = off
            total_n += n
            total_exc += n_exc

            if i % 100 == 0:
                print("  packed %d/%d tensors (exc so far %d)" % (i, len(man), total_exc),
                      flush=True)
            del u16, sign, exp, mant, stream_a, codes, stream_b

    with open(os.path.join(PACK_DIR, "packed_index.txt"), "w") as f:
        f.write("\n".join(idx_lines) + "\n")

    print("tensors packed      : %d" % len(man))
    print("total weights       : %d" % total_n)
    print("total exceptions    : %d" % total_exc)
    print("packed.bin bytes    : %d (%.4f GB)" % (total_bytes, total_bytes / 1e9))
    print("bits per weight     : %.4f" % (total_bytes * 8.0 / total_n))
    print("theoretical bytes   : %d" % (total_n + total_n // 2 + 15 * len(man)
                                        + 4 * len(man) + 5 * total_exc))
    with open(os.path.join(PACK_DIR, "pack_stats.txt"), "w") as f:
        f.write("tensors %d\nweights %d\nexceptions %d\nbytes %d\nbits_per_weight %.4f\n"
                % (len(man), total_n, total_exc, total_bytes, total_bytes * 8.0 / total_n))


if __name__ == "__main__":
    main()
