#!/usr/bin/env python3
"""export_weights.py - export an HF causal LM to the AIOS native binary format.

Phase 0: weights.bin holds BF16 (Qwen's native format). The C oracle converts
every number to FP32 on load; BF16 -> FP32 is bit-exact, so the file is
lossless and half the size of the old FP32 export.

Outputs written into --out:
    config.txt     key/value model config read by the C oracle
    manifest.txt   "idx name dtype rows cols offset_bytes nbytes" per tensor
    weights.bin    raw BF16 bytes, tensors concatenated in manifest order
    ref_tokens.txt one token id per line (via write_reference)
    ref_logits.bin FP32 logits [n_tokens, n_vocab]   (via write_reference)

Tensor export order (the contract with the C oracle):
    embed,
    per layer: input_layernorm, q.weight, q.bias, k.weight, k.bias,
               v.weight, v.bias, o.weight, post_attention_layernorm,
               gate, up, down,
    model.norm, lm_head (only when embeddings are untied).
"""
import argparse
import os

import numpy as np
import torch

SUPPORTED_MODEL_TYPES = ("llama", "qwen2")


def check_model_config(cfg):
    assert cfg.model_type in SUPPORTED_MODEL_TYPES, "unsupported model_type %r" % cfg.model_type
    assert cfg.hidden_act == "silu", "unsupported activation %r" % cfg.hidden_act
    rs = getattr(cfg, "rope_scaling", None)
    # older transformers: None/{}; newer: {'rope_type': 'default', 'rope_theta': ...}
    rope_type = rs.get("rope_type") or rs.get("type") if isinstance(rs, dict) else None
    assert rs is None or rs == {} or rope_type == "default", \
        "only default RoPE is supported, got %r" % rs


def ordered_tensors(model):
    """Return [(name, tensor)] in the contract order. Asserts unsupported structures."""
    cfg = model.config
    check_model_config(cfg)
    base = model.model
    out = [("embed", base.embed_tokens.weight)]
    for i in range(cfg.num_hidden_layers):
        layer = base.layers[i]
        attn, mlp = layer.self_attn, layer.mlp
        assert attn.o_proj.bias is None, "o_proj bias is not supported"
        assert mlp.gate_proj.bias is None and mlp.up_proj.bias is None and mlp.down_proj.bias is None, \
            "MLP bias is not supported"
        p = "layers.%d." % i
        out.append((p + "input_layernorm.weight", layer.input_layernorm.weight))
        out.append((p + "q.weight", attn.q_proj.weight))
        if attn.q_proj.bias is not None:
            out.append((p + "q.bias", attn.q_proj.bias))
        out.append((p + "k.weight", attn.k_proj.weight))
        if attn.k_proj.bias is not None:
            out.append((p + "k.bias", attn.k_proj.bias))
        out.append((p + "v.weight", attn.v_proj.weight))
        if attn.v_proj.bias is not None:
            out.append((p + "v.bias", attn.v_proj.bias))
        out.append((p + "o.weight", attn.o_proj.weight))
        out.append((p + "post_attention_layernorm.weight", layer.post_attention_layernorm.weight))
        out.append((p + "gate.weight", mlp.gate_proj.weight))
        out.append((p + "up.weight", mlp.up_proj.weight))
        out.append((p + "down.weight", mlp.down_proj.weight))
    out.append(("norm", base.norm.weight))
    if not cfg.tie_word_embeddings:
        out.append(("lm_head", model.lm_head.weight))
    return out


def get_rope_theta(cfg):
    """rope_theta lives on cfg in older transformers, in rope_scaling in newer."""
    th = getattr(cfg, "rope_theta", None)
    if th is not None:
        return float(th)
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict) and rs.get("rope_theta") is not None:
        return float(rs["rope_theta"])
    return 10000.0  # HF default


def write_config(model, out_dir, n_params):
    cfg = model.config
    d_head = cfg.hidden_size // cfg.num_attention_heads
    lines = [
        ("model_type", cfg.model_type),
        ("n_layer", cfg.num_hidden_layers),
        ("d_model", cfg.hidden_size),
        ("n_head", cfg.num_attention_heads),
        ("n_kv_head", cfg.num_key_value_heads),
        ("d_head", d_head),
        ("d_ff", cfg.intermediate_size),
        ("n_vocab", cfg.vocab_size),
        ("rms_eps", repr(float(cfg.rms_norm_eps))),
        ("rope_theta", repr(get_rope_theta(cfg))),
        ("tied", 1 if cfg.tie_word_embeddings else 0),
        ("n_params", n_params),
    ]
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        for k, v in lines:
            f.write("%s %s\n" % (k, v))


def export(model, out_dir):
    """Write config.txt, manifest.txt, weights.bin (BF16). Returns total param count."""
    os.makedirs(out_dir, exist_ok=True)
    tensors = ordered_tensors(model)
    n_params = 0
    offset = 0
    with open(os.path.join(out_dir, "weights.bin"), "wb") as wf, \
         open(os.path.join(out_dir, "manifest.txt"), "w") as mf:
        for idx, (name, t) in enumerate(tensors):
            t = t.detach().to(torch.bfloat16).cpu().contiguous()
            arr = t.view(torch.uint16).numpy()  # identical bits, 16-bit view
            nbytes = int(arr.nbytes)
            wf.write(arr.tobytes())
            shape = tuple(t.shape)
            rows = int(shape[0])
            cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            mf.write("%d %s bf16 %d %d %d %d\n" % (idx, name, rows, cols, offset, nbytes))
            offset += nbytes
            n_params += rows * cols
    write_config(model, out_dir, n_params)
    return n_params


def write_reference(model, token_ids, out_dir):
    """Write ref_tokens.txt and ref_logits.bin (FP32, from a full-precision model)."""
    model.eval()
    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids).logits[0].float().cpu().numpy()
    with open(os.path.join(out_dir, "ref_tokens.txt"), "w") as f:
        for t in token_ids:
            f.write("%d\n" % int(t))
    logits.astype(np.float32).tofile(os.path.join(out_dir, "ref_logits.bin"))
    return logits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy")
    ap.add_argument("--n-tokens", type=int, default=8)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(args.prompt)[: args.n_tokens]
    assert len(ids) == args.n_tokens, "prompt tokenized to only %d tokens" % len(ids)
    print("reference tokens:", ids)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    n = export(model, args.out)
    print("exported %d params -> %s" % (n, args.out))
    model = model.float()
    write_reference(model, ids, args.out)
    print("reference logits written")


if __name__ == "__main__":
    main()
