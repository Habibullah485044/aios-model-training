"""Fine-tune HuggingFaceTB/SmolLM2-135M-Instruct on the AIOS project corpus.

Runs on a Kaggle GPU kernel (dataset mounted at /kaggle/input/aios-corpus).
No refusal/safety-specific training is added: the model learns only the
project text. Authority for actions comes from PEP grants in the code, not
from the model.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def find_data_dir() -> Path:
    """Locate the mounted corpus. Kaggle may mount it under a different
    slug, so search /kaggle/input instead of hardcoding one path."""
    env = os.environ.get("AIOS_DATA")
    if env:
        return Path(env)
    root = Path("/kaggle/input")
    print(f"inputs under {root}: {sorted(str(p) for p in root.rglob('*'))[:20]}", flush=True)
    for hit in root.rglob("train.txt"):
        return hit.parent
    raise FileNotFoundError("train.txt not found under /kaggle/input")


DATA_DIR = None  # set in main()
OUT_DIR = Path(os.environ.get("AIOS_OUT", "/kaggle/working/aios-smol"))
BLOCK = 512
BATCH = 8
EPOCHS = 1
LR = 5e-5
SEED = 1337
SAMPLE_PROMPTS = [
    "What is the AIOS task ledger?",
    "How does the PEP grant check work?",
    "Which command completes a task?",
]


class ChunkDataset(Dataset):
    """Pre-tokenized, fixed-length blocks for causal LM training."""

    def __init__(self, ids: list[int]):
        n = len(ids) // BLOCK
        self.blocks = torch.tensor(ids[: n * BLOCK]).view(n, BLOCK)

    def __len__(self) -> int:
        return self.blocks.size(0)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.blocks[i]


def encode(tok, path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        out = model(input_ids=batch, labels=batch)
        total += out.loss.item() * batch.numel()
        count += batch.numel()
    model.train()
    return total / max(count, 1)


def main() -> None:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={MODEL_ID}", flush=True)

    global DATA_DIR
    DATA_DIR = find_data_dir()
    print(f"data_dir={DATA_DIR}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # fp32 master weights: the fp16 GradScaler cannot unscale bf16 grads.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    train_ids = encode(tok, DATA_DIR / "train.txt")
    val_ids = encode(tok, DATA_DIR / "val.txt")
    train_ds, val_ds = ChunkDataset(train_ids), ChunkDataset(val_ids)
    print(f"train_blocks={len(train_ds)} val_blocks={len(val_ds)}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / total_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    log = []
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_dl:
            batch = batch.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(input_ids=batch, labels=batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0 or step == total_steps:
                rec = {"step": step, "total": total_steps,
                       "train_loss": round(loss.item(), 4),
                       "elapsed_s": round(time.time() - t0, 1)}
                log.append(rec)
                print(json.dumps(rec), flush=True)

    val_loss = evaluate(model, val_dl, device)
    print(json.dumps({"val_loss": round(val_loss, 4),
                      "val_ppl": round(math.exp(val_loss), 2)}), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    samples = {}
    model.eval()
    for p in SAMPLE_PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        if isinstance(enc, torch.Tensor):
            ids = enc
        else:
            ids = enc["input_ids"]
        out = model.generate(ids.to(device), max_new_tokens=120, do_sample=False)
        samples[p] = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    (OUT_DIR / "train_report.json").write_text(
        json.dumps({"log": log, "val_loss": val_loss, "samples": samples}, indent=2),
        encoding="utf-8")
    print(json.dumps(samples, indent=2), flush=True)


if __name__ == "__main__":
    main()
