"""Build the project training corpus (text only, secrets scrubbed).

Sources: root *.md, docs/**/*.md, and the task ledger (one JSON record per
line, rendered as readable text). Output:
  training/data/train.txt, training/data/val.txt, training/data/stats.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "training" / "data"
MAX_FILE_BYTES = 1_000_000          # skip huge generated files
MAX_TOTAL_BYTES = 25_000_000        # keep the upload small for Kaggle
VAL_FRACTION = 0.02
SEED = 1337

# Lines that look like credentials are dropped before anything leaves the machine.
SECRET_PATTERNS = [
    re.compile(r"KGAT_[A-Za-z0-9]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*\S{8,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
]


def is_secret_line(line: str) -> bool:
    return any(p.search(line) for p in SECRET_PATTERNS)


def scrub(text: str) -> tuple[str, int]:
    kept, dropped = [], 0
    for line in text.splitlines():
        if is_secret_line(line):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept).strip() + "\n", dropped


def iter_markdown() -> list[Path]:
    files = list(REPO.glob("*.md")) + list((REPO / "docs").rglob("*.md"))
    return sorted(set(files))


def ledger_records() -> list[str]:
    ledger = REPO / "docs" / "tasks" / "MASTER_TASK_LEDGER.jsonl"
    if not ledger.exists():
        return []
    out = []
    for raw in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        # Render as "key: value" lines so the model learns the project's vocabulary.
        out.append("\n".join(f"{k}: {v}" for k, v in rec.items()))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    docs, seen = [], set()
    stats = {"files_read": 0, "files_skipped_size": 0, "files_skipped_dup": 0,
             "lines_dropped_secret": 0, "bytes": 0, "ledger_records": 0}

    for path in iter_markdown():
        if path.stat().st_size > MAX_FILE_BYTES:
            stats["files_skipped_size"] += 1
            continue
        text, dropped = scrub(path.read_text(encoding="utf-8", errors="replace"))
        stats["lines_dropped_secret"] += dropped
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen or not text.strip():
            stats["files_skipped_dup"] += 1
            continue
        seen.add(digest)
        docs.append(f"# FILE: {path.relative_to(REPO).as_posix()}\n{text}")
        stats["files_read"] += 1

    for rec in ledger_records():
        text, dropped = scrub(rec)
        stats["lines_dropped_secret"] += dropped
        docs.append(f"# LEDGER RECORD\n{text}")
        stats["ledger_records"] += 1

    random.Random(SEED).shuffle(docs)
    corpus, total = [], 0
    for d in docs:
        size = len(d.encode("utf-8"))
        if total + size > MAX_TOTAL_BYTES:
            break
        corpus.append(d)
        total += size
    stats["bytes"] = total

    n_val = max(1, int(len(corpus) * VAL_FRACTION))
    val, train = corpus[:n_val], corpus[n_val:]
    (OUT / "train.txt").write_text("\n\n".join(train), encoding="utf-8")
    (OUT / "val.txt").write_text("\n\n".join(val), encoding="utf-8")
    stats["train_docs"], stats["val_docs"] = len(train), len(val)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
