#!/usr/bin/env python3
"""Phase 1 - measure the real data. No speed work yet.

For every tensor in the exported BF16 weights.bin of Qwen2.5-3B:
  - distinct 8-bit exponent values
  - exponent entropy (bits)
  - top-15 coverage (fraction of weights whose exponent is in the 15 most
    common values) -> these are the ones that fit a 4-bit code
  - exception count = weights whose exponent is NOT in the top 15
  - predicted packed size for the 4-bit format (12 bits/weight + 24 bits per
    exception, per the brief's costing)
  - ideal entropy size for reference (8 + exponent entropy)

Pass: a table per tensor type + totals. Expect ~12.0 bits/weight overall.

Reads the already-exported weights.bin + manifest.txt, so it needs no model
download and only a memmap (low RAM).
"""
import os
import sys
from collections import defaultdict

import numpy as np

TMP = os.environ.get("AIOS_TMP", "/tmp/aios_native")
EXPORT_DIR = os.path.join(TMP, "qwen2_5_3b")
RESULTS = os.environ.get("AIOS_RESULTS",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.txt"))

TOP_K = 15
EXC_BITS = 24.0  # brief's costing for one exception entry


def type_key(name):
    """layers.3.q.weight -> layers.N.q.weight ; embed/norm unchanged."""
    parts = name.split(".")
    return ".".join("N" if p.isdigit() else p for p in parts)


def read_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            idx, name, dtype, r, c, off, nb = line.split()
            assert dtype == "bf16", dtype
            rows.append((name, int(r), int(c), int(off), int(nb)))
    return rows


def metrics(counts):
    n = counts.sum()
    if n == 0:
        return None
    p = counts[counts > 0] / n
    entropy = float(-(p * np.log2(p)).sum())
    order = np.sort(counts)[::-1]
    top15 = float(order[:TOP_K].sum())
    n_exc = int(n - top15)
    bits_4 = 12.0 + EXC_BITS * n_exc / n
    bits_ideal = 8.0 + entropy
    return {
        "n": int(n),
        "distinct": int((counts > 0).sum()),
        "entropy": entropy,
        "top15": top15 / n,
        "n_exc": n_exc,
        "exc_pct": 100.0 * n_exc / n,
        "bits_4": bits_4,
        "bits_ideal": bits_ideal,
    }


def main():
    man = read_manifest(os.path.join(EXPORT_DIR, "manifest.txt"))
    raw = np.memmap(os.path.join(EXPORT_DIR, "weights.bin"), dtype=np.uint16, mode="r")

    per_type_counts = defaultdict(lambda: np.zeros(256, dtype=np.int64))
    per_tensor = []
    total_counts = np.zeros(256, dtype=np.int64)

    for i, (name, r, c, off, nb) in enumerate(man):
        chunk = raw[off // 2: off // 2 + r * c]
        assert chunk.size == r * c, (name, chunk.size, r * c)
        exp = ((chunk >> 7) & 0xFF).astype(np.uint8)
        counts = np.bincount(exp, minlength=256).astype(np.int64)
        per_type_counts[type_key(name)] += counts
        total_counts += counts
        per_tensor.append((name, type_key(name), r * c, counts))
        if i % 100 == 0:
            print("  %d/%d tensors" % (i, len(man)), flush=True)

    lines = []
    lines.append("tensors: %d   total params: %d" % (len(man), int(total_counts.sum())))
    lines.append("")

    # ---- per tensor-type table -------------------------------------------
    lines.append("== per tensor type ==")
    hdr = ("%-42s %8s %9s %8s %8s %8s %9s %9s"
           % ("type", "params", "distinct", "entropy", "top15%", "exc%", "bits4", "bitsIdeal"))
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for t in sorted(per_type_counts):
        m = metrics(per_type_counts[t])
        if m is None:
            continue
        lines.append("%-42s %8d %9d %8.3f %8.3f %8.4f %9.3f %9.3f"
                     % (t, m["n"], m["distinct"], m["entropy"], 100 * m["top15"],
                        m["exc_pct"], m["bits_4"], m["bits_ideal"]))

    # ---- totals ----------------------------------------------------------
    tm = metrics(total_counts)
    lines.append("")
    lines.append("== totals (all %d tensors, %d params) ==" % (len(man), tm["n"]))
    lines.append("distinct exponents      : %d" % tm["distinct"])
    lines.append("exponent entropy        : %.4f bits" % tm["entropy"])
    lines.append("top-15 coverage         : %.5f%%" % (100 * tm["top15"]))
    lines.append("exception count         : %d (%.5f%%)" % (tm["n_exc"], tm["exc_pct"]))
    lines.append("predicted packed size   : %.4f bits/weight  (4-bit format)" % tm["bits_4"])
    lines.append("ideal entropy size      : %.4f bits/weight  (reference)" % tm["bits_ideal"])
    lines.append("BF16                    : 16.0000 bits/weight")
    lines.append("predicted vs BF16       : %.4fx smaller" % (16.0 / tm["bits_4"]))
    lines.append("")
    n_params = tm["n"]
    packed_bytes = int(np.ceil(n_params * tm["bits_4"] / 8.0))
    lines.append("predicted packed bytes  : %d (%.3f GB)" % (packed_bytes, packed_bytes / 1e9))
    lines.append("bf16 bytes (weights.bin): %d (%.3f GB)" % (n_params * 2, n_params * 2 / 1e9))

    # ---- the 5 tensors named in the brief, for cross-check ---------------
    lines.append("")
    lines.append("== cross-check vs brief (per-tensor, full tensor) ==")
    # exporter names -> brief names: embed_tokens->embed, gate_proj->gate, etc.
    brief_names = ["embed", "layers.0.gate.weight", "layers.10.down.weight",
                   "layers.10.k.weight", "layers.10.q.weight"]
    by_name = {name: (counts) for name, _tk, _n, counts in per_tensor}
    for bn in brief_names:
        if bn in by_name:
            m = metrics(by_name[bn])
            lines.append("%-42s distinct=%d entropy=%.3f top15=%.4f%% bits4=%.3f"
                         % (bn, m["distinct"], m["entropy"], 100 * m["top15"], m["bits_4"]))

    out = "\n".join(lines)
    print(out)
    with open(RESULTS, "w") as f:
        f.write(out + "\n")
    print("\nwrote", RESULTS)


if __name__ == "__main__":
    main()
