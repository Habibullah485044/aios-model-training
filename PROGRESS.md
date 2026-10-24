# PROGRESS — AIOS engine mission

Format per step: what was built · which check · threshold · measured result · files.

## MISSION STATUS — 6 of 18 measured PASS, 7 of 18 built (~33%)
- The mission has 18 steps: A1–A4, B1–B4, C1–C6, D1–D4.
- Measured PASS: **A1** (packer + reference unpacker), **A2** (kernel race), **A4**
  (packed oracle), **B1** (ELF64 writer — executed on Linux), **B3** (linker —
  byte-identical to `as`+`ld`, executed on Linux), **B4** (runtime — 6 programs
  byte-identical to `as`+`ld` AND executed with fixed targets on Linux), plus the
  **Gate A** decision. **6 of 18 ≈ 33%.**
- **B2** (assembler, incl. the AVX2/FMA subset and the auto-generated differential
  grid) is **PASSED on Linux (WSL on this PC)** except one leg: `06_avx2.s` is
  byte-identical but could not be EXECUTED here (this CPU has no AVX2), so per the
  user's rule B2 counts as PASSED only after it runs on Kaggle's AVX2 hardware.
  Everything else measured: corpus 6/6 files byte-identical (1841 bytes, 0
  differing), programs 01–05 executed with exit codes 1–5, grid 27,791 cases /
  156,887 bytes / 0 differences on BOTH the PE/COFF (Windows) and ELF (WSL)
  toolchains.
- **B3** (our linker) is **PASSED on Linux (WSL on this PC)**: 3 multi-module corpus
  programs in BOTH link orders + an 82-case gap/align grid, **10,762 bytes compared,
  0 differing**, every inter-module gap size 0..63 covered, symbols matched against
  `nm -n`, and all 6 linked programs executed with their marker exit codes.
  **6 of 18 ≈ 33%** have artifacts.
- Linux now runs on this PC itself: Ubuntu 26.04 in WSL2, installed
  non-interactively with no admin click (binutils 2.46 + python3 already in the
  image) — see USER_ACTIONS.md for the exact re-run commands.
- Not started: **A3** (old-PC benchmark — demoted to a pure measurement: under Gate A it
  can no longer change the engine format, it only reports real speed numbers from the
  DDR2-era PC; its build kernel is written and bundled, waiting to be run),
  **Part C** (hand-written `.asm` engine), **Part D** (runtime/packaging).
- Part A was the experiment; the building (B, C, D) has started with B1/B2.

---

## PLAN CHANGE (user, after Gate A) — ASSEMBLER, not compiler
- B2 is now a real **assembler**: reads `.asm` files (Intel syntax), emits machine code
  via our encoder. Pass check unchanged: every byte identical to GNU as.
- Part C: **no Python code generator**. The engine is written **by hand in `.asm` files**
  — one file per operation (embedding, RMSNorm, BF16 matvec, RoPE, attention, softmax/exp,
  SwiGLU, residual add, final norm, lm_head) + a main loop. Our assembler + ELF writer
  build them into the engine.
- C1 is now a **config/manifest loader written in assembly**: reads config.txt + manifest
  at runtime, computes all 434 tensor offsets (must match manifest; total = weights file
  size). No model sizes hard-coded in the engine.
- D2 tokenizer: also **written in assembly** (same pass checks).
- All pass checks/thresholds (C2–C6, D1–D4) stay exactly as fixed.
- Python is allowed ONLY for build scripts and tests on Kaggle — never inside the engine.
- Consequence for B4/C2: the runtime pieces (threads, syscalls, timers) and every op are
  now assembly source under our own assembler; gcc-built code remains only as test
  oracles and harnesses.

---

## Phase 0 — BF16 base — PASS
- Built `export_weights.py` (BF16 export) + `aios_infer.c` (C99 oracle, double accumulation).
- Checks: gcc `-std=c99 -Wall -Wextra -Werror -O2` clean · tiny Qwen2 tied/untied < 1e-5 (measured 2.027e-6 / 1.945e-6, argmax ok) · every tensor BF16→FP32 vs torch `.float()` 0 mismatched bits / 3,085,938,688 · real 3B vs PyTorch max abs diff 6.306e-5 < 1e-3, argmax 8/8.
- weights.bin = 6,171,877,376 bytes, 434 tensors, 3,085,938,688 params.
- Files: `training/native/kernel/export_weights.py`, `training/native/runtime/aios_infer.c`, `training/native/{compile_check,real_check}/*.py`.

## Phase 1 — measurement — PASS
- Built `measure.py` (exponent stats over all 434 tensors, memmap).
- Result: entropy 2.7951 bits · top-15 coverage 99.93851% · exceptions 1,897,535 (global table) · 12.0148 bits/weight predicted ≈ 4.635 GB.
- Files: `training/native/phase1/measure.py`, `phase1_results.txt`.

## Phase 2 = A1 — packer + reference unpacker — PASS
- Built `pack.py` (numpy packer, per-tensor 15-entry exponent table, planar nibble packing, sorted exceptions, per-tensor offset index) + `unpack_ref.c` (scalar C99 reference unpacker = Phase 3 answer key).
- Checks: gcc compile clean · unpack vs original BF16 **0 mismatched bits / 3,085,938,688** · size == arithmetic prediction exactly.
- Measured: **1,358,322 exceptions** (per-tensor tables) vs Phase 1 global 1,897,535 → **4,635,707,888 bytes (4.6357 GB)** = 12.018 bits/weight = 1.3314× smaller than BF16. See DECISIONS.md for the exception-count note.
- Files: `training/native/phase2/{pack.py,unpack_ref.c,phase2_check.py}`, `phase2_results.txt`.

## A2 — kernel race — PASS (Kaggle v19). Verdict: PACKED = LOSS
- Machine: AMD EPYC 7B12, 4 threads, L3 16 MB, RAM read probe 17.6 GB/s. Data: 23 real gate/up/down matrices, 518,520,832 weights (1.037 GB BF16 / 781 MB packed). Both builds `-Werror` clean; old .o has **0** AVX fingerprints (objdump scan), modern 48 vec-regs/253 VEX mnemonics.
- Correctness (threshold → measured): 3 kernels bit-identical **within each build** — 0 differing of 190,464 elements, all 4 configs (threshold: 0). vs float64 ref: rel **7.2e-7** (modern) / **1.7e-6** (old), threshold < 1e-4. PASS.
- Speed (median of 10 timed runs, threshold table A2; all numbers from v19, same class in v20):
  | config | fp32 GB/s | BF16 GB/s | PACKED GB/s | bf16/fp32 best/med | packed/bf16 best/med | RAM probe med | verdict |
  |---|---|---|---|---|---|---|---|
  | modern 1t | 20.22 | 11.64 | 3.84 | 1.152 / **1.182** | 0.438 / **0.429** | 17.39 | LOSS |
  | modern all | 46.25 | 38.94 | 7.75 | 1.684 / **1.519** | 0.264 / **0.300** | 16.60 | LOSS |
  | old 1t | 14.33 | 7.46 | 2.39 | 1.041 / **1.050** | 0.425 / **0.427** | 16.67 | LOSS |
  | old all | 36.94 | 18.80 | 4.82 | 1.018 / **1.025** | 0.340 / **0.344** | 16.94 | LOSS |
  GB/s = bytes of that format actually fetched per second (rows-at-once shares the x-vector loads; each weight is still read from RAM exactly once per sweep, so the byte counts are real DRAM traffic). RAM probe = single-thread read of one 64-byte line per 16 floats over the 2.07 GB buffer, best/median of 10.
- Memory-bound check: BF16 was **faster** than FP32 in all 4 configs (bf16/fp32 1.02–1.68), so the mission's "≥4 GB all-threads memory-bound run" condition was NOT triggered. Note 4-thread configs fetch at 36–46 GB/s vs the 1-thread probe's ~17 GB/s (aggregate multi-core DRAM bandwidth), and BF16-all (38.9 GB/s) sits below fp32-all (46.3 GB/s): the byte ratio is 2.0 but the speed ratio is only 1.52–1.68, i.e. even at 4 threads the BF16 path is part conversion-compute, not purely bandwidth.
- Interpretation (scoped to the measured machine): PACKED reads 25% fewer bytes than BF16 (781 MB vs 1037 MB per sweep) yet takes ~3× longer in **all four configs on this shared AMD EPYC 7B12 (Kaggle, ≤4 threads)** — the nibble-split/PSHUFB/escape-patch decode is compute-bound and dominates the byte savings. This is one machine, not a universal x86 result; the A3 package will measure the user's DDR2-era PC directly and is the authority for that era. What A1+A2+A4 establish: lossless 12-bit packing shrinks storage 1.33× with bit-identical results, but on this CPU it does **not** speed up matvec. Gate A → engine uses BF16.
- Files: `training/native/phase3/{race.c,race_check.py}`, `phase3_results.txt` (v19, Kaggle). Failures v16–v18 in FAILURES.md (harness AVX-scan false positive, manifest short-name filter, NULL-pointer realloc — all fixed at the cause).

## A4 — packed oracle — PASS (Kaggle v20)
- unpack_ref reproduced weights.bin from packed.bin: **0 mismatched bits / 3,085,938,688**, size equal (6,171,877,376 B).
- Double-accumulation oracle run on the unpacked weights vs Phase 0 oracle on weights.bin: **0 differing bits of 1,215,488 logits** (8×151936 floats). Threshold: 0. PASS.
- Cross-tool probe for A3 packaging: x86_64-w64-mingw32-gcc / i686 / wine / wine64 all **not found** on Kaggle → the Windows .exe for the old PC must come from our own PE writer (D3). A3 user action is therefore scheduled after D3; noted in USER_ACTIONS.md.
- Files: `training/native/phase4/phase4_check.py`, `phase4_results.txt` (v20, Kaggle).

## Gate A — DECISION: engine uses BF16 weights (packed kept as a storage option)
- A2 measured LOSS on every config of the tested machine (shared AMD EPYC 7B12, Kaggle, ≤4 threads; packed/bf16 median 0.30–0.44, threshold table: <1.00 = LOSS), on two independent runs (v19: 0.300/0.429/0.427/0.344; v20: 0.321/0.352 all-threads).
- A1+A4 proved the format lossless (0 mismatched bits; bit-identical logits), so packed remains valid for download/disk footprint (1.33× smaller), but the runtime will read BF16. The old-PC measurement (A3) will be recorded honestly when the exe exists (after D3); it can only revisit this gate if it shows ≥1.10 there, which the Kaggle old-path numbers (0.35–0.43) make very unlikely.

## B1 — ELF64 writer — PASSED (Linux, WSL on this PC, 2026-10-12)
- Built `elf64_writer.py` (ET_EXEC: Ehdr + one RWX PT_LOAD at 0x400000, code at
  offset 0x78, no section headers) + `b1_check.py` (fixed thresholds: 132 bytes exactly;
  `file` says ELF 64-bit LSB executable x86-64; `readelf -a` exit 0 with EMPTY stderr,
  entry 0x400078, exactly one RWX PT_LOAD at vaddr 0x400000 offset 0; `objdump -M intel`
  disassembles the 12 code bytes to exactly mov eax,0x3c / mov edi,0x2a / syscall;
  running the file exits 42; plus an independent struct-level parse of every header
  field in Python).
- Local pre-verification (mingw-w64 binutils 2.47 `file`/`readelf`/`objdump`): 132-byte
  file, every header field as specified, readelf clean, disassembly exactly the three
  instructions.
- Linux run (WSL Ubuntu 26.04, binutils 2.46): **B1 PASSED** — the 132-byte file:
  `file`/`readelf -a` (empty stderr)/`objdump -M intel` exactly as specified, entry
  0x400078, one RWX PT_LOAD at vaddr 0x400000, the 12 code bytes disassemble to
  exactly mov eax,0x3c / mov edi,0x2a / syscall, and **the linked file runs and exits
  42** — the mission check now has real Linux evidence, not only pre-verification.
- Files: `training/native/b1/{elf64_writer.py,b1_check.py}`.

## B2 — assembler — PASSED on Linux except the 06_avx2.s execution (no AVX2 on this CPU)
- Built `asm.py` (Intel-syntax x86-64 assembler: GP + the SSE/SSSE3 subset + the
  **AVX/AVX2/FMA (VEX) subset** Part C's modern BF16 path needs, `.byte/.word/.long/
  .quad`, `.space`, `.align`, branch relaxation rel8->rel32, rip-relative and absolute
  fixups with an image base, `assemble_map()` for per-line diagnostics) +
  `corpus/01..06*.s` (6 self-contained, runnable `_start` programs, exit markers 1-6)
  + `b2_check.py` + `grid_probe.py`.
- VEX subset = the 19 mission mnemonics: vmovups vmovdqu vbroadcastss vpshufb
  vpmovzxbd vpmovzxwd vpand vpor vpxor vpslld vpsrlw vpcmpeqb vpmovmskb vaddps vmulps
  vfmadd231ps vextractf128 vperm2f128 vzeroupper (`corpus/06_avx2.s`, marker 6, runs
  fault-free on AVX2 hardware - the Kaggle EPYC has AVX2+FMA).
- Pass criterion (unchanged from the mission): **every byte identical to GNU as**.
  `b2_check.py` assembles each corpus file with `as --64`, links it with `ld -e _start`
  (relocations resolved), extracts the linked `.text` with objcopy, assembles the same
  source with our assembler at ld's `.text` VMA, requires 0 differing bytes, and then
  RUNS both linked programs and requires both to exit with the file's marker status.
- Differential evidence, measured locally against real GNU binutils 2.47 (pre-
  verification, not the mission check; see DECISIONS D-016/D-020/D-022): corpus
  **6/6 files byte-identical, 1840 code bytes, 0 differing** (relocation offsets ==
  our fixups, GAS's alignment padding verified to be nops), and the auto-generated
  grid `b2/grid_probe.py`: **27,747 instruction cases, 156,548 bytes, 0 differences**
  (every supported mnemonic x every operand form x r8-r15 / xmm8-15 / ymm8-15 /
  rsp-r12-rbp-r13 bases / disp8 / disp32 / index*scale incl. r12+r13 indexes /
  rip-relative / 64-bit immediates; absolute relocations verified against GAS's own
  symbol table). The grid also runs on Kaggle (ELF objects, full coverage including
  `mov reg, symbol`); it is chained into Part B's B2 step.
- GNU as conventions matched for VEX (all measured, D-020): the short C5 prefix exactly
  when 0F-map + W=0 + no X/B bit3; dest-in-VEX.vvvv for the shift-immediate group;
  vfmadd231ps = 66 0F38.W0 **B8**; vzeroupper = C5 F8 77; GAS's shorter-direction rule
  for two-register vmovups/vmovdqu; the shift count operand is always 128-bit; mask
  extracts take register sources only; memory destinations only for vextractf128;
  `mov reg, symbol` = movabs+ADDR64 on ELF (what b2_check compares) vs an absolute-
  memory ADDR32 form on PE/COFF.
- Bugs the local comparison found and fixed at the cause (never by loosening), on top
  of the earlier ones: r12-as-index rejected (modrm_mem checked the low 3 bits, but
  only actual rsp is an invalid index - REX.X makes SIB index=100 mean r12);
  rip-relative displacements were computed from field+4 instead of the END of the
  instruction (wrong whenever a trailing imm8 follows the disp: vperm2f128,
  vextractf128, pslld-imm with a memory source, `mov [rip+label], imm`); the vpslld
  count-form opcode was transposed with psrld's (D2 vs F2).
- The bare-label bug the user caught on real Linux GAS (D-025): our assembler encoded
  `mov reg, label` as movabs (the ADDRESS) while GAS's Intel syntax makes a bare label
  a MEMORY operand (`mov rax, d` = 48 8B 04 25 + a 32-bit relocation = the VALUE at
  d). Now matched exactly, `offset` is supported (mov r64 -> 48 C7 C0 imm32, mov r32
  -> B8 imm32, mov r8 -> B0 ib with an 8-bit relocation), and `movabs` is implemented
  as a mnemonic (imm64 on any r64; accumulator moffs64 loads 48 A1/A1/A0 and stores
  48 A3/A3/A2), with every GAS-rejected form rejected the same way. Measured against
  GNU as 2.47 (PE) and 2.46 (ELF, WSL); the grid's cases are unconditional now - the
  old PE/ELF "difference" was our own bug, not a GAS backend difference.
- First Linux run (WSL Ubuntu 26.04, binutils 2.46, 2026-10-12): corpus **6/6 files
  byte-identical (1841 bytes, 0 differing, exact section length), programs 01-05
  execute with exit codes 1-5, 06_avx2.s byte-identical with "SKIPPED: no AVX2"**
  (this CPU is an Intel i5-2540M: AVX yes, AVX2/FMA no - so per the user's rule the
  06 execution must happen on Kaggle before B2 counts as PASSED; b2_check prints the
  note loudly and stays exit 0). Grid on ELF objects: **27,791 instruction cases,
  156,887 bytes, 0 differences, 47 relocations verified** - the same numbers as the
  PE/COFF run on Windows. Two failures on the way, both fixed at the cause
  (FAILURES.md): the harness crashed on ELF's empty nop-pad (ELF .text has alignment
  1 and NO padding at all), and corpus 01 itself had a latent bug - its
  `[rsp+rax*8]` qword store clobbered the dword the load section later reads into
  ecx and uses as an index, so BOTH builds segfaulted ~2 GB off the stack.
- Files: `training/native/b2/{asm.py,b2_check.py,grid_probe.py,corpus/*.s}`, bundle
  `kaggle/aios-native-partb` (B1 + B2 + corpus + grid embedded),
  `partb/run_all.py` (B1 then B2; the B2 step runs b2_check then grid_probe).

## B3 — our linker (multi-module placement + cross-module relocations) · PASSED
- Built `b3/link.py`: links assembled modules exactly like GNU ld - concatenation in
  list order, each module at the first address that is a multiple of its section
  alignment (gas sh_addralign = largest `.align`, 1 if none), gaps filled with ld's
  nop fill (asm.gap_fill - different from gas's `.align` pad), ELF visibility for
  cross-module names (`.global` exports, module-local labels shadow, undefined is an
  error, duplicate export is a multiple-definition error). Every relocation kind
  works cross-module: rel32 call/jmp/jcc (always near for external targets, like GNU
  as), rip-relative, `offset label`, movabs, [label] disp32, .long/.quad data.
- Check `b3/b3_check.py`: byte identity with `as`+`ld` on the whole linked .text over
  3 corpus programs (01_xmod cross-module relocs, 02_align shadowing + align
  placement, 03_order link-order sensitivity) in **both link orders**, plus an
  82-case generated gap/align grid whose .align 64 sweep must produce every
  inter-module gap size 0..63. Also: entry points equal, `nm -n` symbols match, and
  both ELFs (ours via the B1 writer with the real _start entry) run and exit with the
  marker.
- Measured (WSL Ubuntu 26.04, binutils 2.46, 2026-10-14): **B3 PASSED** - 3 programs
  x2 orders + 82 grid cases, **10,762 bytes compared, 0 differing**; 01_xmod 348
  bytes fwd / 345 rev, 02_align 87/91, 03_order 403/386 (reverse order shifts
  everything - and is still byte-identical), all marker exits 1/2/3 correct on both
  sides, gap sizes 0..63 all covered. `partb/run_all.py` now runs B1, B2, B3:
  **PART B PASSED (3 steps)**.
- Files: `training/native/b3/{link.py,b3_check.py,corpus/01_xmod/*.s,
  corpus/02_align/*.s, corpus/03_order/*.s}`; `b1/elf64_writer.py` gained the
  `entry_off` parameter (D-035); `partb/run_all.py` + the partb Kaggle bundle
  regenerated (25 embedded files, byte-identical to sources).

## B4 — the runtime (DONE, upgraded 2026-10-18)
- `training/native/b4/runtime.s` (459 lines) holds the whole runtime, hand-written in
  the B2 dialect: write_str/exit/exit_group/mmap_anon/spawn_task/futex_wait/futex_wake/
  futex_wake_all/now_ns/sleep_ns, plus the allocator (heap_init/heap_alloc — a 16-byte
  bump allocator with lock cmpxchg on the next pointer), the REAL thread pool
  (spawn_worker = clone(0x50F00) on 64 KiB heap stacks, pool_worker park/consume loop,
  pool_init/pool_run generation-counted work queue), barrier(), cpu_count() and
  cpu_features() (CPUID). See D-036/D-037/D-039/D-040/D-041/D-042.
- The assembler (`b2/asm.py`) gained every scalar/atomic form the runtime uses:
  `lock` add/or/xadd/cmpxchg (the F0 byte after 0x66, in gas's order), all xchg forms
  (accumulator short form, generic reg/reg, memory) and pause — a 104-byte unit is
  byte-identical to GNU as.
- Corpus: `01_syscalls` (raw write+exit), `02_threads` (3 clone/futex tasks, slot sum
  = 66), `03_timers` (CLOCK_MONOTONIC around a 50 ms nanosleep, fixed 40 ms floor),
  `04_features` (CPUID vs this machine's /proc/cpuinfo flags), `05_sum` (parallel sum
  of 1..100,000,000 for EVERY thread count T=1..N), `06_barrier` (10,000 barrier rounds
  over all N workers).
- Measured (WSL Ubuntu 26.04, binutils 2.46, 2026-10-18): **B4 PASSED** - all 6
  programs byte-identical to `as`+`ld` (1087/1241/1129/1305/1314/1236 bytes of .text),
  nm symbols match, and BOTH ELFs of every program run to the fixed target: exits
  41/42/43/44/45/46 with the exact stdout (04 mirroring this CPU's flags). Full
  suite: **PART B PASSED (4 steps)** - B1, B2, B3, B4. The partb Kaggle bundle was
  regenerated (33 embedded files, all verified byte-identical to their sources).
- Two measured traps cost a debug cycle, both now recorded: `sched_getaffinity` is
  syscall **204** (203 is sched_SETaffinity, whose zero-mask call returns EINVAL — the
  first cpu_count returned 0 and 05/06 exited 96) and it returns the number of mask
  bytes it filled in, so the bit count must stop there (D-041); `barrier()`'s
  generation must live in a futex-preserved register because the futex helpers zero
  r8/r10 and every waiter released early (D-042).
- Files: `training/native/b4/{runtime.s,b4_check.py,corpus/01_syscalls/main.s,
  corpus/02_threads/main.s, corpus/03_timers/main.s, corpus/04_features/main.s,
  corpus/05_sum/main.s, corpus/06_barrier/main.s}`; `b2/asm.py` gained the atomics;
  `partb/run_all.py` runs the whole suite.

## PART C — environment facts measured 2026-10-18 (before writing C1)
- WSL Ubuntu here has binutils + python3 only: **no gcc/cc/clang/zig, no numpy/torch, and
  no network** (apt cannot reach archive.ubuntu.com). The C2 C-oracle build and any model
  export therefore cannot happen in WSL on this PC.
- Windows Python 3.14.6 has **numpy 2.5.1 and torch 2.13.0+cpu**, huggingface.co is
  reachable, and `kernel/export_weights.py` needs only torch - so the 0.5B export
  (config.txt, manifest.txt, weights.bin) and every float64 numpy reference can be
  produced on Windows while the engine still builds and runs as a Linux ELF in WSL.
  `transformers`/`huggingface_hub`/`safetensors` are not installed yet (that is the next
  step, a local install rather than a Kaggle round trip).
- C1 plan: export Qwen2.5-0.5B-Instruct on Windows, write the `.asm` config/manifest
  loader (parse config.txt + manifest.txt, compute every offset/shape, verify the sizes
  sum exactly to weights.bin), then check it against BOTH manifests - 0.5B locally and the
  3B's 434-tensor manifest from the Kaggle export (D16).

## BUILT, NOT YET RUN — Kaggle kernels waiting on a run
- `aios-native-partb` (new): from-scratch kernel that checks **B1** then **B2**
  (b2_check over the 6 corpus files, then the differential grid) and writes
  `b1/b1_results.txt` and `b2/b2_results.txt`. Already ran under WSL on this PC (B1
  PASSED; B2 all-but-06); its remaining value is the Kaggle run that EXECUTES
  06_avx2.s on AVX2 hardware to close B2 (USER_ACTIONS.md).
- `aios-native-a3build`: A3 bench (zig cc PE32/PE32+ build, static scans, synthetic data,
  zip). Its embedded `bench.c` was refreshed to the current source by this regeneration
  (the bundle had been generated before the last bench.c edit).
- `aios-native-real-check`: regenerated for the same reason (stale embedded `bench.c`).

## (archived) A2 build log — earlier Kaggle versions
- Built `race.c` (901 lines): three matvec kernels (FP32-expanded / BF16 / PACKED) sharing one loop skeleton (ROWS=4 rows-at-once, VL=8 modern / VL=4 old, ascending 64-weight groups) so the three outputs must be **bit-identical** within one build. Packed path decodes nibbles + PSHUFB table + escape patch **in registers**, never writing unpacked weights to RAM. Old build uses PUNPCKLBW/PUNPCKLWD (no PMOVZX/SSE4.1), MULPS+ADDPS (no FMA).
- Built `race_check.py` harness: compiles both builds `-Werror`, scans **our .o files** with objdump (old path must contain ZERO AVX references = "forced"), runs 4 configs ({modern,old}×{1,all threads}) on gate/up/down matrices ≥ max(4×L3, 1 GB), verifies bit-identity + ref rel < 1e-4 per run, classifies packed_vs_bf16 (median): ≥1.10 WIN · 1.00–1.10 MARGINAL · <1.00 LOSS.
- Pre-verified locally (no local gcc): pure-Python simulation of the SIMD decode byte math vs the unpack_ref formula — 600/600 exact, 32,986 escape records exercised, both VL paths.
- Chained into `real_check.py` as step (7); bundle regenerated (9 files, byte-identical to sources); pushed Kaggle v16.
- Files: `training/native/phase3/{race.c,race_check.py}`, `phase3_results.txt` (pending).

