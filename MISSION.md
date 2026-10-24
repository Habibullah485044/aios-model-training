# AIOS ENGINE — MASTER MISSION v2 (autonomous run)

> **Save this file as `MISSION.md` in the project root and never delete it.**
> **Re-read it after every context compaction, restart or lost connection** — it is the
> only complete copy of the plan. It contains the goal, what is done, verified facts, every
> remaining part with fixed pass-checks, and the rules for working **without stopping**.
>
> v2 = the original mission + the user's PLAN CHANGE (we build an **assembler**, and the
> engine is **hand-written assembly**) + the current status (12 Oct 2026).

**Order of work (quick view):**
~~A1 packer → A2 kernel race~~ → A3 old-PC package (user action, don't wait) → ~~A4
packed-model check → Gate A~~ → ~~B1 ELF writer~~ → B2 assembler (one Kaggle run left) →
**B3 programs ← NEXT** → B4 runtime → C1 config/manifest loader → C2 ops → C3 full model →
C4 text generation → C5 speed → C6 old-PC path → D1 CLI → D2 tokenizer → D3 Windows .exe →
D4 AIOS.

---

## 0. HOW TO RUN THIS MISSION — READ FIRST

### 0.1 Never-stop rule
- Work through the parts in order. Inside a part, do the steps in order.
- **Do not stop to ask the user** for confirmation, approval or opinions. Every normal choice
  already has a default in §5. If something isn't covered, choose the option that is
  (1) lossless/correct, (2) simplest, (3) reversible — log it in `DECISIONS.md` and keep going.
- After each step, append a short entry to `PROGRESS.md` and immediately start the next step.
- If you can't find a definition or a rule, **re-read `MISSION.md` first** — the whole plan is
  in it. If it truly isn't covered, decide by D12 and log it; don't ask.

### 0.2 The ONLY reasons to stop and ask
1. An action that could destroy user data or can't be undone: deleting files outside this
   project, deleting Kaggle kernels/datasets, force-pushing, overwriting anything you didn't create.
2. You need money, a paid service, or a credential you don't have.
3. Every remaining step depends on a BLOCKED step (no independent work left anywhere).
   → write the failure report and stop.

Everything else: keep working.

### 0.3 When the user must do something physically (e.g. run a test on their old PC)
Prepare everything (package + 3-line instructions), write it to `USER_ACTIONS.md`, tell the
user in one short message, and **continue with the next independent step**. Do not wait.
When results arrive, log them and apply any decision rule that depends on them.

### 0.4 When a check fails
- Up to **5 focused debug attempts**. Each attempt: written hypothesis → one change → re-run.
  Log every attempt in `FAILURES.md`.
- Use the debug aids in this file (smaller test cases, layer-by-layer dumps, the reference
  unpacker, the oracle).
- Still failing after 5: mark the step **BLOCKED** in `PROGRESS.md` with evidence, continue with
  the next step that does not depend on it, and come back when new information appears.
- **Never loosen a threshold.** Never mark a check passed without the measured number.

### 0.5 Files to keep in the project root
- `MISSION.md` — this file: the plan. Read-only; don't edit it unless the user asks.
- `PROGRESS.md` — per step: what was built, which check, threshold, measured result, files.
- `DECISIONS.md` — every default or judgment call, with a one-line reason.
- `FAILURES.md` — failed attempts, hypotheses, evidence.
- `USER_ACTIONS.md` — anything the user must do, with exact instructions.

### 0.6 Reporting
After each **part** (not each step), send the user a short report using the template in §8,
then continue immediately.

---

## 1. What we are building (plain English)

- A **Python-free, fast program** that runs **Qwen2.5-3B-Instruct** on a normal or old CPU.
  The engine is **hand-written x86-64 assembly** (`.asm` files, Intel syntax), built into a
  program by **our own assembler + ELF writer** — no C compiler, no GNU ld, no Python inside
  the engine. AIOS runs it as a subprocess.
- On a CPU, speed is limited mostly by **how many bytes of weights are read from RAM per
  generated word**. So we:
  (a) store weights in BF16 — **done**;
  (b) test lossless packing to ~12 bits/weight — **Part A, done: LOSS on CPU** (packed ran at
      0.30–0.45× of BF16) → the engine reads **BF16**; packed stays a storage-only format;
  (c) build the toolchain and the engine — **Parts B–C**;
  (d) ship it to AIOS — **Part D**.
- **Correctness first, speed second.** Our tricks must never change the model's answers.

**Where things run (current setup):**
- **Dev PC** (where you work): Windows 10 22H2 (build 19045), **Intel i5-2540M (Sandy
  Bridge): AVX yes, AVX2/FMA no**. It has **WSL Ubuntu with binutils 2.46** — use it for every
  Linux check (running ELF files, GNU as/ld/objdump/readelf). It can run the **old (SSSE3)
  path** natively, but **not** the modern AVX2+FMA path.
- **Kaggle** (Linux, AMD EPYC with AVX2+FMA): runs the **modern path**, anything needing the
  real 6.17 GB `weights.bin`, and PyTorch reference runs. There is no Kaggle CLI/key on the dev
  PC — the user runs Kaggle kernels from the Kaggle website (put each run in
  `USER_ACTIONS.md`). **Never ask for the Kaggle API key.**
- **Old target PC**: DDR2 era, Windows, no compiler — receives the Windows `.exe` (Part D3)
  and the A3 benchmark.

---

## 2. Where we are (DONE — verified)

**Phase 0 (BF16 base) — PASS**
- `export_weights.py` writes BF16 `weights.bin` for Qwen2.5-3B-Instruct: exactly
  **6,171,877,376 bytes**, **434 tensors**, **3,085,938,688 params**.
- `aios_infer.c` (oracle: C99, accumulates in **double**, reads BF16 → FP32 on load) compiles
  clean with `gcc -std=c99 -Wall -Wextra -Werror -O2`.
- Tiny random Qwen2 vs PyTorch: tied 2.027e-6, untied 1.945e-6 (threshold 1e-5), argmax ✓.
- Every tensor BF16→FP32 vs PyTorch FP32: **0 mismatched bits** / 3,085,938,688.
- Real Qwen2.5-3B vs PyTorch: max abs diff **6.306e-5** (threshold 1e-3), argmax ✓ at all 8 positions.

**Phase 1 (measurement) — PASS**
- Exponent entropy (whole model) 2.7951 bits · top-15 coverage 99.93851% · exceptions
  **1,897,535** (0.06149%) · predicted 12.0148 bits/weight ≈ 4.635 GB (exceptions costed at
  24 bits) · per-tensor distinct exponents 20–39.

**Files:** `export_weights.py`, `aios_infer.c`, `compile_check.py`, `real_check.py`,
`measure.py`, `build_kernels.py`, `results.txt`, `phase1_results.txt`.

**Part A — DONE (A3 pending as a user action)**
- A1 packer: 0 mismatched bits; size matches the prediction.
- A2 kernel race (Kaggle, EPYC): all configs bit-identical; **PACKED/BF16 = 0.30–0.45 → LOSS**
  on every config; BF16/FP32 = 1.00–1.50.
- A4: packed oracle bit-identical to the BF16 oracle (0 differing bits of 1,215,488 logits).
- **Gate A = BF16.** Under the Gate A rules, A3 can no longer change the engine format; it
  only gives real speed numbers from the old PC.

**Part B — status**
- **B1 — PASSED** (WSL): 132-byte ELF64, header checks + real execution, exits 42.
- **B2 — PASSED on Linux except one leg**: `asm.py` (Intel-syntax assembler) — corpus 6/6
  files byte-identical (1,841 bytes); auto-generated grid **27,791 cases, 0 differences**
  (ELF and PE). Remaining: **execute `06_avx2.s` on an AVX2 machine (Kaggle)** — queued in
  `USER_ACTIONS.md`. Don't wait for it; continue with B3.
- Known assembler semantics (match GAS exactly): a bare label is a **memory operand**
  (`mov rax, d` loads the value at d); `offset d` is the address; `movabs` exists.

---

## 3. Fixed facts (verified — build on these, do not re-derive)

### 3.1 Model (Qwen2.5-3B-Instruct `config.json`)
hidden 2048 · layers 36 · heads 16 · KV heads 2 · head_dim 128 · KV dim 256 · intermediate
11008 · vocab 151936 · **tied embeddings** (embed_tokens is also the output layer) ·
rope_theta 1,000,000 · rms_norm_eps 1e-6 · max_position_embeddings 32768 · bos 151643 ·
eos 151645 · activation silu · native dtype BF16.

BF16 byte offsets in `weights.bin` (tensor-order contract): `embed_tokens` at **0**;
`layers.0.self_attn.q_proj.weight` at **622,333,952**; `model.norm.weight` (last) at
**6,171,873,280**.

KV cache in FP32: **73,728 bytes per token** (≈ 302 MB for 4,096 tokens).

**Model licence note (user's decision, not yours):** Qwen2.5-3B is under a research-only
licence. If the product is sold, the shipped model will likely switch to an Apache-2.0 size
(e.g. Qwen2.5-1.5B-Instruct). So the engine must **never hard-code sizes** — everything comes
from config/manifest at runtime, and the same binary must run other Qwen2.5 sizes (C3b).

### 3.2 BF16 ↔ FP32
`fp32_bits = bf16_bits << 16` (exact). BF16 = 1 sign · 8 exponent · 7 mantissa bits.
`fp32_bits = (sign << 31) | (exp << 23) | (mant7 << 16)`.

### 3.3 Packed format basis (measured)
4-bit exponent codes with a per-tensor table of the 15 most common exponents (codes 0–14),
code 15 = escape. 3-bit codes are **not** worth it (top-7 covers only ~96%). Ideal entropy
coding ≈ 10.6–10.8 bits/weight (optional, later).

### 3.4 CPU instruction levels (verified with GNU as)
- **Core 2 (DDR2 era):** SSSE3 → `PSHUFB` ✓, `PMADDUBSW` ✓, but `PMOVZX*` ✗ (that is SSE4.1,
  Penryn/Nehalem and newer). On SSSE3-only CPUs widen bytes with `PUNPCKLBW`/`PUNPCKLWD`
  against a zero register.
- **FMA3 and AVX2:** Haswell (2013) and newer. Old CPUs: use `MULPS` + `ADDPS`.
- **Sandy/Ivy Bridge (like the dev PC):** AVX but **no AVX2, no FMA** → they use the old
  path. (An extra AVX-only path is optional and only after C6 passes.)
- FMA rounds once, MUL+ADD rounds twice → tiny differences. **Bit-identical comparisons are
  only valid within the same instruction path.**
- Always detect features at runtime with `CPUID`; never run a path the CPU lacks.

### 3.5 Linux x86-64 (own toolchain)
- `syscall`: number in `rax`; args `rdi, rsi, rdx, r10, r8, r9`; result in `rax`; it destroys
  `rcx` and `r11`; errors: `rax` in −4095..−1 (= −errno).
- Numbers: read 0 · write 1 · open 2 · close 3 · fstat 5 · lseek 8 · mmap 9 · mprotect 10 ·
  munmap 11 · madvise 28 · clone 56 · exit 60 · futex 202 · sched_getaffinity 204 ·
  clock_gettime 228 · exit_group 231 · getrandom 318.
- Flags: O_RDONLY 0 · O_WRONLY 1 · O_CREAT 0x40 · O_TRUNC 0x200 · PROT_READ 1 · PROT_WRITE 2 ·
  MAP_PRIVATE 2 · MAP_ANONYMOUS 0x20 · SEEK_END 2 · CLOCK_MONOTONIC 1.
- Threads: clone flags CLONE_VM 0x100 | CLONE_FS 0x200 | CLONE_FILES 0x400 | CLONE_SIGHAND
  0x800 | CLONE_THREAD 0x10000 | CLONE_SYSVSEM 0x40000 = **0x50F00**; each thread needs its own
  mmap'd stack; FUTEX_WAIT 0, FUTEX_WAKE 1, FUTEX_PRIVATE_FLAG 128; end the process with
  exit_group.
- At program entry: `[rsp]` = argc, `[rsp+8]` = argv[0], `[rsp+16]` = argv[1], …; rsp is
  16-byte aligned.

### 3.6 ELF64 executable
ELF header 64 bytes; program headers 56 bytes each; no section headers needed. e_type 2 (EXEC),
e_machine 62, load address 0x400000, entry = 0x400000 + 64 + 56 × (number of program headers).
PT_LOAD = 1 with flags R=4, W=2, X=1; `p_offset ≡ p_vaddr (mod 0x1000)`; `p_memsz > p_filesz`
→ the extra bytes are zero-filled. Add a PT_GNU_STACK (0x6474e551) header with flags RW, as
GNU ld does. Sanity: exit(42) = `B8 3C 00 00 00 BF 2A 00 00 00 0F 05` → a 132-byte file.

### 3.7 x86-64 encoding rules
Byte order: legacy/mandatory prefix → REX → opcode → ModR/M → SIB → displacement → immediate.
REX = `0100WRXB` (W = 64-bit; R/X/B extend reg/index/base). Base rsp/r12 needs a SIB byte.
Base rbp/r13 needs a displacement (mod=00, rm=101 means RIP-relative). SIB index=100 means
"no index". Mandatory prefixes (66/F2/F3) go **before** REX. AVX/AVX2/FMA use the VEX prefix (C5
two-byte / C4 three-byte; R, X, B and vvvv stored inverted; L=1 for 256-bit). **Every encoding
must match GNU as byte-for-byte.**

---

## 4. Global rules

1. **Lossless only.** No weight value may change.
2. **Thresholds in this file are fixed.** Never loosen them.
3. **Comparisons:**
   - The oracle accumulates in double, so fast float32 code can't match it bit-for-bit. Compare
     fast code to the oracle/PyTorch **with tolerance**: max abs diff < 1e-3, argmax matches at
     all positions.
   - **Bit-identical** is required only between variants that share the same accumulation
     structure **and** the same instruction path (e.g. FP32 vs BF16 vs packed kernels in one build).
4. **Fair benchmarks:** strong SIMD baseline; same thread count; data ≥ max(4×L3, 1 GB) on
   Kaggle; 3 warm-up runs, then report best-of-10 and median; always report CPU model, feature
   flags, cache sizes and measured RAM read bandwidth.
5. **Product path:** the engine is **hand-written `.asm`** built only by **our assembler +
   ELF/PE writer**. GNU as/ld/objdump are allowed only as **test references** (byte
   comparisons). Python is allowed only for build scripts and tests — **never inside the
   engine**. C (with intrinsics, OpenMP) is allowed only for the oracle, Part A research
   benchmarks and test harnesses. gcc/MinGW/zig may build research tools, never the product.
6. **Kaggle practice:** big files in `/tmp` (not `/kaggle/working`, except the optional weights
   kernel in §5 D11); `set -o pipefail`; check `df -h /tmp` before downloads; run PyTorch and
   native code in separate subprocesses; write results to result files and read them from the
   kernel output; regenerate bundles with `build_kernels.py` after every edit and verify bundle
   == source.
7. No git commits or pushes unless the user asks.

---

## 5. Default decisions (already made — don't ask)

- **D1** Target order: Linux ELF x86-64 first; Windows PE x86-64 second (Part D3).
- **D2** Two instruction paths: **modern** = AVX2+FMA; **old** = SSSE3, no FMA, no SSE4.1.
  CPUID picks at runtime.
- **D3** Engine weight format: decided automatically by **Gate A**. BF16 is always supported.
- **D4** Correctness decoding is greedy. Sampling (temperature/top-p) only after Part C passes.
- **D5** Context cap v1: 4,096 tokens. KV cache in FP32 for v1.
- **D6** Engine I/O v1: token IDs (D1). Native tokenizer comes in D2.
- **D7** Test prompts (Part C4), chat template applied with the HF tokenizer (Kaggle or WSL)
  (`add_generation_prompt=True`):
  - P1: `What is the capital of Pakistan?`
  - P2: `Write a Python function that reverses a string.`
  - P3: `Explain why the sky is blue in two sentences.`
- **D8** Near-tie rule: if our greedy token differs from PyTorch's at step k and PyTorch's
  top-1 minus top-2 logit gap at step k is **< 2e-3**, record a near-tie (accepted; compare only
  up to step k). Otherwise it is a bug.
- **D9** Old-PC benchmark data: 6 matrices of shape 2048×11008 (≈ 270 MB BF16) — larger than
  any DDR2-era cache.
- **D10** Generality: the engine never hard-codes sizes; it reads config + manifest at runtime.
- **D11** Optional Kaggle speed-up: a dedicated `aios-weights` kernel writes the BF16 and packed
  files to `/kaggle/working` once; other kernels attach it via `kernel_sources` in
  `kernel-metadata.json`; never pull its output to the PC. If this doesn't work, keep
  downloading per run.
- **D12** Anything else unspecified: lossless → simplest → reversible; log it.
- **D13** No separate linker. Each program/engine is **one assembly unit**: a top `.asm` file
  that pulls shared code in with GAS's `.include "file"` directive (add it to `asm.py` if
  missing — it keeps GAS byte-comparisons possible). Our assembler + ELF writer turn that one
  unit into one executable.
- **D14** Where checks run: Linux checks on the dev PC's **WSL**; anything needing **AVX2/FMA**
  or the real **6.17 GB weights.bin** on **Kaggle** (queue it in `USER_ACTIONS.md`, don't wait).
  A check that needs a CPU feature the machine lacks prints `SKIPPED: no <feature>` — that is
  never counted as a pass; the step stays open until that leg runs on capable hardware.

---

## 6. THE PARTS

### PART A — Lossless packing (the possible discovery) — DONE except A3

- **A1 packer, A2 kernel race, A4 packed-model check: PASSED.** Result: **LOSS on CPU**
  (PACKED/BF16 = 0.30–0.45 on every config) → **Gate A = BF16**. The packed format stays a
  storage-only format (lossless, 1.33× smaller). Details are in `PROGRESS.md`.
- Gate A rule for later: when A3 results arrive, record them; they cannot switch the engine
  to PACKED (only a modern-path WIN could, and that was a LOSS).

**A3 — Old-PC benchmark package (USER ACTION — don't wait)**
- `a3/bench.c` (kernel32-only, no CRT) is written but has never been built.
- Build it **in WSL** with MinGW-w64 (`apt install mingw-w64`): **x86_64 and i686** builds of
  the A2 old-path benchmark, `-march=core2`, statically linked (no DLLs). (The Kaggle bundle
  `aios-native-a3build` stays as a backup route.)
- The `.exe`:
  - prints CPU brand and feature flags (CPUID);
  - refuses paths the CPU doesn't support;
  - loads one contestant at a time (FP32 expanded in memory from BF16);
  - runs single-thread and all-threads;
  - writes `results.txt`.
- `run.bat` tries the 64-bit exe, falls back to 32-bit, and pauses at the end.
- Data: D9 (≈ 270 MB BF16 plus its packed version).
- **Smoke-test the `.exe` directly on the dev PC (it is Windows)** before handing it over.
- Make the zip and write in `USER_ACTIONS.md`: "1) copy the zip to the old PC 2) unzip it
  3) double-click run.bat 4) send me results.txt". Then continue with Part B.

### PART B — Our own toolchain (foundation for the product)

**B1 — ELF writer** (§3.6) — **PASSED (WSL).** **PASS:** exit42 is 132 bytes and exits with
code 42; `readelf -h` and `readelf -l` are clean.

**B2 — x86-64 assembler** (`asm.py`: Intel-syntax `.asm` text in → machine code out; labels,
forward/backward fixups, branch relaxation) — **PASSED on Linux; the `06_avx2.s` execution leg
still has to run on Kaggle** (AVX2 hardware).
- Coverage:
  - **GPR:** mov (incl. 64-bit immediate), add, sub, imul, and, or, xor, shl, shr, sar, inc,
    dec, neg, cmp, test, lea, push, pop, jmp, jcc, call, ret, syscall, cpuid, rdtsc.
  - **SSE/SSSE3:** movups, movaps, movdqu, movd, movq, movss, addps, mulps, subps, divps, maxps,
    minps, sqrtss/sqrtps, xorps, andps, shufps, cvtdq2ps, cvttps2dq, pshufb, pmaddubsw,
    punpcklbw/punpckhbw, punpcklwd/punpckhwd, pand, por, pxor, psllw/pslld, psrlw/psrld, paddd,
    pcmpeqb, pmovmskb, comiss.
  - **AVX2/FMA (VEX):** vmovups, vmovdqu, vbroadcastss, vpshufb, vpmovzxbd, vpmovzxwd, vpand,
    vpor, vpxor, vpslld, vpsrlw, vpcmpeqb, vpmovmskb, vaddps, vmulps, vfmadd231ps, vextractf128,
    vperm2f128, vzeroupper.
- Memory forms: [base], [base+disp8], [base+disp32], [base+index*scale+disp], RIP-relative,
  including every special case in §3.7.
- **PASS:** automated differential test vs GNU as across many register/memory combinations
  (including r8–r15, xmm8–15/ymm8–15, rsp/r12/rbp/r13 bases, disp8/disp32, 64-bit immediates)
  → **0 mismatches**. Report the case count. *(Measured: 27,791 cases, 0 differences;
  corpus 6/6 files byte-identical.)*

**B3 — Small programs, written in `.asm`, built only by our assembler + ELF writer** ← NEXT
- Every program is an `.asm` source file. Build = our assembler → our ELF writer → run in WSL.
  No GNU as/ld in the build; shared code via `.include` (D13).
- Extra check for every B3/B4 program: assemble the same source with GNU as and confirm the code
  bytes match ours (reuse the `b2_check` machinery).
- The programs:
  - **exit42** — exits with code 42 (this time from `.asm` source).
  - **hello** — prints a fixed text stored in the data segment; output byte-exact.
  - **sumloop** — adds 1..1000 and prints `500500`, using our own integer-to-text code.
  - **weights reader** — `read_weights <weights.bin> <out.bin>`: open, lseek(SEEK_END), mmap
    the whole file read-only, and copy the first 64 bytes at three offsets to `out.bin`:
    embed_tokens (0), layers.0.q_proj (622,333,952), model.norm (6,171,873,280).
- **Test file for the weights reader:** in WSL, make a **sparse** file of exactly
  6,171,877,376 bytes (same size as the real `weights.bin`) with known random 64-byte blocks
  written at the three offsets. This tests offsets above 4 GB without downloading anything.
  Also queue one run on Kaggle with the real `weights.bin` (`USER_ACTIONS.md`; don't wait).
- **PASS:** every output is exact; `out.bin` matches the input file at those offsets byte for
  byte; a wrong path gives exit code 2 (ENOENT) with no crash.

**B4 — Runtime pieces, written in `.asm`** (shared by the engine via `.include`)
- CPUID feature detection, a timer (clock_gettime, CLOCK_MONOTONIC), an anonymous-mmap
  allocator, a thread pool (clone + futex with each thread's own mmap'd stack; CPU count from
  sched_getaffinity), and a barrier.
- **PASS:**
  - a parallel sum of 1..10^8 is correct (= 5,000,000,050,000,000) with every thread count
    from 1 to N (N = CPU count);
  - 10,000 barrier rounds complete without deadlock;
  - the printed features (at least sse2, ssse3, sse4_1, avx, avx2, fma) match the
    `/proc/cpuinfo` flags.
- Run it in WSL. Also queue one Kaggle run (different CPU, more cores); don't wait.

### PART C — The engine (hand-written `.asm`, built by our assembler)

**How to develop Part C (defaults — don't ask):**
- **D15 Correctness path first:** write C2–C4 on the **old path (SSSE3, no FMA, no SSE4.1)**
  first, because the dev PC can run it. Add the modern AVX2+FMA path in C5 and verify it on
  Kaggle. CPUID picks the path at runtime.
- **D16 Small model first:** develop and check C1–C4 with **Qwen2.5-0.5B-Instruct** in WSL
  (≈1 GB BF16 `weights.bin`; build the C oracle with gcc inside WSL). Then run the same checks
  with the real **3B** on Kaggle. Both must pass with the **same binary** — only the model files
  change.
- **D17 RoPE tables:** the exporter (Python, offline) writes them as `rope.bin` (float32
  cos/sin for 4,096 positions, from config's rope_theta and head_dim) next to `weights.bin`;
  the engine loads it like the weights. No sin/cos code inside the engine.

**C1 — Config/manifest loader (written in `.asm`, runs inside the engine)**
- Reads `config.txt` + the manifest at runtime and computes every tensor's offset, shape and
  format. No hard-coded sizes.
- **PASS:**
  - every computed offset matches the manifest, and the sizes sum exactly to the weights
    file size (3B: 434 tensors);
  - the same binary loads the Qwen2.5-0.5B-Instruct config/manifest correctly.

**C2 — Ops in `.asm` + unit tests**
- Each op is hand-written `.asm`, wrapped as a tiny standalone test program (input file →
  output file), compared against float64 numpy. The ops:
  - embedding gather
  - RMSNorm
  - linear/matvec (BF16) with optional bias
  - RoPE (rotate-half; tables from `rope.bin`, D17)
  - our own `exp` (range reduction + polynomial)
  - softmax
  - causal GQA attention (heads from config; 3B: 16 query heads share 2 KV heads, 8 each)
  - SiLU×up (SwiGLU)
  - residual add
  - final norm
  - lm_head (tied)
- **PASS (fixed):**
  - elementwise/norm ops: max relative error ≤ 1e-5;
  - reductions (matvec, attention): max |diff| ≤ 1e-4 × max|ref|;
  - `exp`: max relative error ≤ 2e-6 over [−87, 88]; exp(very negative) = 0; no NaN.

**C3 — Full forward pass**
- **PASS** vs PyTorch and vs the oracle: max abs diff < 1e-3, argmax ✓ at all 8 positions.
- Debug aid: per-layer hidden-state dumps in **both** the engine and the oracle; compare layer
  by layer to find the first divergence.
- **C3b generality:** Qwen2.5-0.5B-Instruct **and** Qwen2.5-3B-Instruct both pass these
  thresholds with the **same binary** (D16).

**C4 — Generation loop**
- KV cache (FP32, ctx 4096), greedy; stop at eos (151645) or after N new tokens.
- **PASS:** P1–P3 (§5 D7), 32 new tokens each, token-exact vs PyTorch greedy, except near-ties
  per D8 (log them).

**C5 — Speed** (only after C3 and C4 pass)
- Add the **modern AVX2+FMA path** (CPUID-dispatched), threads (B4), fused ops; BF16 weights
  (Gate A).
- Report prefill tokens/s and decode tokens/s, single-thread and all-threads: modern path on
  Kaggle, old path on the dev PC (WSL) and on Kaggle; compare against the oracle.
- C3 and C4 must pass on the **modern path too** (Kaggle).
- Optional reference: llama.cpp on the same model and format. Skip it if setup takes more than
  1 hour.
- **Rule: after every speed change, re-run C3 and C4.**

**C6 — Old-PC path (hardening)**
- The old path already exists (D15). Now prove it never uses an instruction a Core 2 lacks:
  scan the old-path code for anything newer than SSSE3 (SSE4.1+, AVX, FMA) → must be **0**.
- Force-run the old path on Kaggle: C3 and C4 must pass there too.
- Package it for the user's old PC after D3 (USER ACTION — don't wait).

### PART D — Shipping and AIOS

**D1 — Engine CLI**
- `aios_engine <model_dir> [--max-new N] [--threads T]`
- Reads token IDs (space-separated) from stdin; writes generated IDs to stdout on one line;
  timings go to stderr.
- Exit codes: 0 ok, 2 bad arguments, 3 model load error, 4 runtime error.
- **PASS:** scripted tests for every exit code; P1–P3 outputs identical to C4.

**D2 — Native tokenizer** (hand-written `.asm`, Python-free)
- Qwen2 byte-level BPE from `tokenizer.json` (pre-tokenizer + merges + special tokens), plus
  the chat template.
- **PASS:**
  - 2,000 strings (English, Urdu, Roman Urdu, code, numbers, emoji, whitespace edge cases)
    → token IDs exactly equal to the HF tokenizer;
  - decode(encode(s)) == s for all of them;
  - chat-template output identical to `apply_chat_template` on 20 conversations.
- Then add a `--text` mode to the CLI.

**D3 — Windows .exe (PE) output**
- PE32+ writer with an import table for kernel32.dll: GetStdHandle, ReadFile, WriteFile,
  CreateFileW, GetFileSizeEx, CreateFileMappingW, MapViewOfFile, VirtualAlloc, CreateThread,
  WaitForSingleObject and ExitProcess, as needed.
- Windows x64 calling convention: args in rcx, rdx, r8, r9; 32-byte shadow space; rsp 16-byte
  aligned at each call.
- **PASS:** if wine is available on Kaggle, P1–P3 must be identical to the Linux build. Then
  USER ACTION: run it on the user's PC (both paths) and report tokens/s.

**D4 — AIOS integration**
- When you reach D4, write in `USER_ACTIONS.md` a request for the AIOS repo/location (the only
  information we don't have yet).
- Meanwhile, write the subprocess contract doc, an example caller and an end-to-end test script.
- **PASS:** AIOS sends text and gets the same reply as the D2/D1 path; report end-to-end latency.

---

## 7. Threshold table (all fixed — never loosen)

| Check | Threshold |
|---|---|
| A1 unpack vs BF16 | 0 mismatched bits; exceptions = 1,897,535 |
| A2 kernels, same build | bit-identical outputs |
| A2 vs float64 reference | max abs diff / max abs y_ref < 1e-4 |
| A2 PACKED vs BF16 | ≥ 1.10 WIN · 1.00–1.10 MARGINAL · < 1.00 LOSS |
| A4 packed oracle vs BF16 oracle | bit-identical logits |
| B1 exit42 | 132 bytes, exit code 42 |
| B2 assembler vs GNU as | 0 mismatches |
| B3 weights reader | byte-exact at 3 offsets; wrong path → exit 2 |
| B4 parallel sum | exactly 5,000,000,050,000,000 for every thread count 1..N |
| B4 barrier | 10,000 rounds, no deadlock |
| B4 CPU features | printed flags = `/proc/cpuinfo` flags |
| C1 loader | every offset = manifest; sizes sum = weights file size |
| C2 elementwise/norm ops | max relative error ≤ 1e-5 |
| C2 matvec/attention | max abs diff ≤ 1e-4 × max abs ref |
| C2 exp | max relative error ≤ 2e-6 on [−87, 88] |
| C3/C3b full model (0.5B and 3B, both paths) | max abs diff < 1e-3; argmax 8/8 |
| C4 generation | token-exact for 32 tokens, near-tie gap < 2e-3 allowed |
| C6 old-path instruction scan | 0 instructions newer than SSSE3 |
| D2 tokenizer | 100% identical IDs on 2,000 strings; round-trip exact |

---

## 8. Report template (send after each part, then continue)

```
PART X — DONE / PARTIAL / BLOCKED
Steps completed:
Checks (threshold → measured):
Decisions made (DECISIONS.md):
Failures / blocked steps (FAILURES.md):
User actions pending (USER_ACTIONS.md):
Starting next: Part Y
```

---

## 9. Definition of done

- A Python-free engine — **hand-written `.asm` built by our own assembler** — runs
  Qwen2.5-3B-Instruct (and the same binary runs Qwen2.5-0.5B-Instruct), passing C3, C4 and D2
  on both the modern and old instruction paths.
- Windows .exe and Linux builds both pass.
- AIOS calls it as a subprocess, text in → text out.
- Part A's result (WIN / MARGINAL / LOSS on Kaggle and on the old PC) is measured and recorded
  honestly.
- `PROGRESS.md`, `DECISIONS.md` and `FAILURES.md` tell the whole story with real numbers.
