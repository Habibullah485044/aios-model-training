#!/usr/bin/env python3
"""make_manifest_3b.py - build the 3B manifest.txt WITHOUT downloading 6 GB.

C1's pass check must run against the REAL 3B model too (D16: the same binary
must load Qwen2.5-0.5B-Instruct AND Qwen2.5-3B-Instruct). Downloading the
6.17 GB weights.bin again is wasteful: the manifest only needs each tensor's
SHAPE, and the HF repo publishes model.safetensors.index.json which maps
every tensor name to its shard file - the shapes come from the safetensors
header of each shard, which we can read with a ranged HTTP request (the
header is at the start of the file; the tensor DATA is never downloaded).

So: fetch config.json + the index, then read each shard's safetensors header
(via Range: bytes=0-8 then bytes=8-8+hdrlen) to recover the exact shapes.
From that we reproduce kernel/export_weights.py's tensor ORDER and write
manifest.txt with the same line format:
    "idx name dtype rows cols offset_bytes nbytes"

and verify the total against the mission's fixed numbers:
    434 tensors, 3,085,938,688 params, 6,171,877,376 bytes,
    embed_tokens at 0, layers.0.self_attn.q_proj.weight at 622,333,952,
    model.norm.weight at 6,171,873,280.
"""
import argparse
import json
import os
import sys
import urllib.request

BASE = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main"


def fetch(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", "bytes=%d-%s" % (start, end if end is not None else ""))
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def safetensors_header(url):
    """Return {name: {"dtype":.., "shape":[..]}} from a shard's header."""
    n = int.from_bytes(fetch(url, 0, 7), "little")
    raw = fetch(url, 8, 8 + n - 1)
    hdr = json.loads(raw)
    hdr.pop("__metadata__", None)
    return hdr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="Qwen/Qwen2.5-3B-Instruct")
    a = ap.parse_args()

    base = "https://huggingface.co/%s/resolve/main" % a.repo
    cfg = json.loads(fetch(base + "/config.json"))
    index = json.loads(fetch(base + "/model.safetensors.index.json"))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print("config: %d layers, d_model %d, n_head %d, n_kv_head %d, vocab %d, tied %s"
          % (cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"],
             cfg["num_key_value_heads"], cfg["vocab_size"],
             cfg.get("tie_word_embeddings", False)))
    print("index: %d tensors in %d shards" % (len(weight_map), len(shards)))

    # ---- shapes: read every shard's header (no tensor data) ----
    shapes = {}
    dtypes = {}
    for shard in shards:
        hdr = safetensors_header(base + "/" + shard)
        print("  %s: %d tensors (header only)" % (shard, len(hdr)))
        for name, meta in hdr.items():
            shapes[name] = meta["shape"]
            dtypes[name] = meta["dtype"]

    # ---- reproduce kernel/export_weights.py's ORDER ----
    # The exporter's names are SHORT (embed, layers.N.q.weight, ...); the
    # HF names are long (model.embed_tokens.weight, ...). Map short -> long.
    def hf(layer, attr):
        return "model.layers.%d.%s" % (layer, attr)

    order = []          # (short_name, hf_name)
    order.append(("embed", "model.embed_tokens.weight"))
    for i in range(cfg["num_hidden_layers"]):
        order.append(("layers.%d.input_layernorm.weight" % i,
                      hf(i, "input_layernorm.weight")))
        order.append(("layers.%d.q.weight" % i, hf(i, "self_attn.q_proj.weight")))
        if hf(i, "self_attn.q_proj.bias") in shapes:
            order.append(("layers.%d.q.bias" % i, hf(i, "self_attn.q_proj.bias")))
        order.append(("layers.%d.k.weight" % i, hf(i, "self_attn.k_proj.weight")))
        if hf(i, "self_attn.k_proj.bias") in shapes:
            order.append(("layers.%d.k.bias" % i, hf(i, "self_attn.k_proj.bias")))
        order.append(("layers.%d.v.weight" % i, hf(i, "self_attn.v_proj.weight")))
        if hf(i, "self_attn.v_proj.bias") in shapes:
            order.append(("layers.%d.v.bias" % i, hf(i, "self_attn.v_proj.bias")))
        order.append(("layers.%d.o.weight" % i, hf(i, "self_attn.o_proj.weight")))
        order.append(("layers.%d.post_attention_layernorm.weight" % i,
                      hf(i, "post_attention_layernorm.weight")))
        order.append(("layers.%d.gate.weight" % i, hf(i, "mlp.gate_proj.weight")))
        order.append(("layers.%d.up.weight" % i, hf(i, "mlp.up_proj.weight")))
        order.append(("layers.%d.down.weight" % i, hf(i, "mlp.down_proj.weight")))
    order.append(("norm", "model.norm.weight"))
    if not cfg.get("tie_word_embeddings", False):
        order.append(("lm_head", "lm_head.weight"))

    # ---- write manifest.txt in the exporter's exact format ----
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "manifest.txt")
    offset = 0
    n_params = 0
    rows_out = []
    for idx, (short, long_name) in enumerate(order):
        if long_name not in shapes:
            raise SystemExit("tensor %s missing from the shards" % long_name)
        dtype = dtypes[long_name]
        if dtype != "BF16":
            raise SystemExit("tensor %s dtype %s (expected BF16)" % (long_name, dtype))
        shape = shapes[long_name]
        rows = int(shape[0])
        cols = 1
        for d in shape[1:]:
            cols *= int(d)
        nbytes = rows * cols * 2
        rows_out.append("%d %s bf16 %d %d %d %d" % (idx, short, rows, cols, offset, nbytes))
        offset += nbytes
        n_params += rows * cols

    with open(mpath, "w") as f:
        f.write("\n".join(rows_out) + "\n")

    # ---- the mission's fixed facts must match ----
    checks = [
        ("tensor count", len(order), 434),
        ("params", n_params, 3085938688),
        ("total bytes", offset, 6171877376),
    ]
    ok = True
    for label, got, want in checks:
        flag = "OK" if got == want else "MISMATCH"
        if got != want:
            ok = False
        print("%-14s %16d  expected %16d  %s" % (label, got, want, flag))
    # fixed byte offsets from MISSION.md 3.1
    for short, want_off in (("embed", 0),
                             ("layers.0.q.weight", 622333952),
                             ("norm", 6171873280)):
        line = [l for l in rows_out if l.split()[1] == short]
        got_off = int(line[0].split()[5]) if line else None
        flag = "OK" if got_off == want_off else "MISMATCH"
        if got_off != want_off:
            ok = False
        print("offset %-18s %16s expected %16d  %s" % (short, got_off, want_off, flag))
    print("wrote %s" % mpath)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
