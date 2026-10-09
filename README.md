# AIOS Model Training

Fine-tunes a small instruct model on this project's own documentation and
task ledger. Runs on a Kaggle GPU kernel.

## Contents

| File | Purpose |
| --- | --- |
| `build_corpus.py` | Builds `train.txt` / `val.txt` from the AIOS repo's Markdown docs and `MASTER_TASK_LEDGER.jsonl`, dropping credential-looking lines. |
| `train.py` | Full fine-tune of `HuggingFaceTB/SmolLM2-135M-Instruct` (134.5M params) with fp32 master weights + fp16 autocast, 1 epoch, block size 512. Prints loss every 50 steps, then val loss/perplexity and sample answers. |
| `kernel-metadata.json` | Kaggle kernel config: GPU + internet enabled, corpus dataset mounted. |

## Corpus

Built locally from the AIOS repo: ~19.9 MB, 7,245 docs + 10,062 ledger
records (16,961 train / 346 val docs). The built text files are **not**
committed here (they are the project's internals); build them with
`build_corpus.py` and upload via `kaggle datasets create -p <dir>`.

## Running on Kaggle

```bash
kaggle kernels push -p <dir-with-kernel-metadata-and-train.py>
kaggle kernels status <owner>/aios-smol-train
kaggle kernels output <owner>/aios-smol-train -p <out-dir>
```

Model weights are written to `/kaggle/working/aios-smol/` and reported in
`train_report.json` (loss curve, val perplexity, sample generations).

## Note on authority

The model learns project text only; no refusal/safety training is added or
removed. What the model is *allowed to do* is enforced by PEP grants and
audit rows in the AIOS code, not by the model itself.
