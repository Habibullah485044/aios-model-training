# FAILURES — AIOS engine mission

Failed attempts, hypotheses, evidence. Max 5 focused attempts per check, then BLOCKED.

## v12 Kaggle run — Phase 2 died silently
- Attempt 1: pulled kernel log → empty (Kaggle log endpoint returns nothing for this kernel).
  Hypothesis: crash before any results file. Evidence: `results.txt` and `phase1_results.txt`
  present, `phase2_results.txt` absent. Fix: added `guarded_main()` writing traceback + `df -h`
  to the results file on any exception.

## v13 — gcc -Werror: unused `read_u64_le`
- Evidence: captured gcc stderr: `unused function 'read_u64_le'` in `unpack_ref.c`.
  Hypothesis: dead helper left over after switching index parsing to `sscanf`.
  Fix: deleted the function (not a suppression). 

## v14 — gcc -Werror: unused parameter `n_rows`
- Evidence: captured gcc stderr: `unused parameter 'n_rows'` in `unpack_tensor`.
  Hypothesis: `row` is derived from `j / row_len`, parameter redundant.
  Fix: removed parameter from signature + call site.

## v15 — PASS
- Phase 2 PASSED: 0 mismatched bits / 3,085,938,688; packed size == arithmetic prediction.

## v16 — A2 harness: AVX scan false-positive + L3 parse abort (attempt 1 of A2 check)
- Evidence: both builds compiled clean `-Werror`; old run aborted with
  `avx-instructions race_old.o: vec_regs=376 vex_mnemonics=0 / OLD PATH FAIL`.
  Also `L3 bytes: 0` despite `cache L3: 56320K` printed above it.
- Hypotheses: (a) the vec-reg regex `\b[xyz]mm\d+\b` counts `xmm` registers —
  legacy SSE that the SSSE3 build uses by design, not AVX; `vex_mnemonics=0`
  already proves no AVX (every AVX insn requires VEX). (b) `l3_bytes()` wraps
  the whole sysfs loop in one try, so the first non-cache entry (power/, uevent)
  aborts it → returns 0.
- Fix: regex now matches only `ymm|zmm`; VEX-mnemonic count stays as the
  decisive second signal. `l3_bytes()` guards per entry. Harness-only changes;
  race.c untouched (it never ran). No thresholds changed.

## v17 — A2: "no gate/up/down matrices found in manifest" (attempt 2 of A2 check)
- Evidence: v17 got past the AVX scan (old: vec_regs=0 vex_mnemonics=0 ✓, modern:
  48/253 ✓), both builds compiled `-Werror` clean, all 4 configs then died at
  manifest selection: `FATAL: no gate/up/down matrices found in manifest`.
- Hypothesis: race.c filtered on HF-style names (`gate_proj.weight`), but
  `export_weights.py:ordered_tensors` writes SHORT names: `layers.<i>.gate.weight`.
  Confirmed by reading the exporter source (lines 68-70).
- Fix: filter now matches `.gate.weight` / `.up.weight` / `.down.weight`
  (dot-prefixed, so q/k/v/o.weight cannot false-match). No thresholds changed.

## B2 development — local pre-verification only, no Kaggle run yet
Bug hunt before the first Kaggle run of Part B, using GNU binutils 2.47 locally
(see DECISIONS D-016). Every failure below was fixed at the cause.

### Local smoke test of our own assembler on the corpus
- `AsmError: bad numeric expression '-1'` for `mov rax, -1` / `add rbx, -5`.
  Hypothesis: eval_num_expr only accepted a sign *between* terms, not a leading one.
  Fix: handle a leading +/- in eval_num_expr.
- `ValueError: bytes must be in range(0, 256)` from modrm_mem on
  `mov qword ptr [rsp+128], r15`. Hypothesis: the disp32 mod field was written as the
  decimal literal 10 instead of binary 0b10, so `(mod << 6)` overflowed for every SIB
  operand with a disp32. Fix: mod = 2 (and the matching dict key), with a comment.
- `AsmError: PTR size 32 does not match operand width 64` on
  `mov rax, dword ptr [rip+d]`. The rejection is correct (GNU as rejects it too), the
  corpus line was wrong. Fix: corpus line -> `mov eax, dword ptr [rip+d]`.

### Differential probe vs GNU as (one form per program)
- 88 forms, first round showed 14 divergences in 2 families: (a) the A-register short
  forms (test/and/or/xor/cmp/add/sub with immediates on al/eax/rax use A8/A9 and the
  04..3D family, not 80/81/F6/F7), (b) `movq qword ptr [mem], xmm` (GAS uses
  `66 0F D6`, not `66 0F 7E`). Both fixed and added to the corpus.
- Second round found one more: `mov rax, 0xffffffff80000000` must use the sign-extended
  imm32 form (`48 C7 C0 00 00 00 80`), not movabs; GAS picks the shortest form whose bit
  pattern round-trips. Fixed with a fits_imm(value, dest_bits, imm_bits) test.
- Result after fixes: 88 forms, 0 differences.

### Corpus comparison vs GNU as
- `03_gp_flow.s` differed at offset 21: GAS emitted `e9 c8 00 00 00` (rel32) where we
  emitted `eb 00` (+0). Hypothesis: `.space 200` contributed 0 bytes during layout, so
  `after_pad` was misplaced, the branch was relaxed to rel8, and the emitted program
  would have jumped into the zero padding and faulted. Fix: align/skip emit their real
  padding inside encode(), so layout, relaxation and emitted bytes all agree.
- GAS's `.text` is 1–8 bytes longer than our code in 4 of 5 files: section-alignment
  nop padding, verified by disassembling the tail (all nops). Handled in the check
  (D-015), not by loosening the byte criterion.

### Bugs already present in the interrupted asm.py draft (found by the probe/corpus)
- 8-bit opcode families were wrong: mov reg,reg used 89 instead of 88, loads used 8B
  instead of 8A, arith used the 01/03 family instead of 00/02, test used 85 instead of
  84, inc/dec and neg/not used FF/F7 instead of FE/F6, 8-bit arith immediates used 83
  instead of 80.
- mem,reg / r,mem forms took the operand width from the memory operand (which has none)
  instead of the register, so `mov qword ptr [rsp], rax` was emitted as a 32-bit store.
- rip-relative memory operands only worked in lea and the SSE path; every other form
  appended a bogus displacement instead of a fixup.
- movsxd had no memory form; movq xmm,xmm assembled to a GP-source form.

## B2 AVX2/FMA + grid — local development (before any Kaggle run)
Every failure below was found by the local GAS comparison and fixed at the cause.

### AVX byte-compare probe (our VEX encoder vs GNU as)
- `AsmError: rsp cannot be an index` on `vpshufb xmm9, xmm10, [rsp+r12*4+0x1234]`.
  PRE-EXISTING bug in modrm_mem: the check `(index & 7) == 4` rejected r12 as an index
  (its low 3 bits equal rsp's), but with REX.X=1 the SIB index field 100 means r12, not
  "no index" - only the actual rsp register is invalid. Fixed to `index == 4`.
- `vpslld xmm4, xmm5, xmm2` encoded with opcode D2. D2 is PSRLD's; the vpslld count
  form is 66 0F **F2** /r (transposition of the two). Fixed in the VEX table.
- `vperm2f128 ymm11, ymm12, ymmword ptr [rip+d], 255` had a rip displacement 1 too
  small: our rip fixups used field+4 as the reference point, but x86 RIP-relative
  addressing is relative to the END of the whole instruction, which differs from
  field+4 whenever a trailing imm8 follows the displacement. The same flaw existed
  in the SSE group forms (`pslld m128, imm8`), `vextractf128` with a memory
  destination, and `mov [rip+label], imm`/arith-imm. Fixed: emit() stamps the
  statement end into every rip fixup.
- `vpslld xmm0, [rip+d], 4` diverged: GNU as emits an EVEX (AVX-512) encoding for the
  shift-by-immediate with a MEMORY source; its VEX template takes a register source
  only. We reject that form (outside the documented subset) instead of implementing
  EVEX; grid_probe only generates what GAS accepts.
- `vpsrlw ymm9, ymm10, ymmword ptr [rsp]` rejected by GAS: a memory COUNT operand must
  be an xmmword PTR even with a ymm destination, and `vpslld ymm0, ymm1, ymm2` (ymm
  count) is rejected too - the count operand is always 128-bit. Width rules in
  enc_vex updated to match exactly.
- `vmovups xmm0, xmm8` differed by one byte: GAS emits the STORE direction
  (`C5 78 11 C0`, 4 bytes) when the load form would need the 3-byte C4 prefix
  (r/m register has bit 3) and the store form would not; ties keep the load form.
  Implemented as the VEX_MOVE flip.

### Corpus 06_avx2.s authoring
- GAS rejected `mov qword ptr [rsp], imm64`: a memory store only takes an imm32.
  The seed pattern now goes through movabs+register (which also exercises the
  64-bit-immediate path).
- Our assembler (deliberately) rejects `mov word ptr [...], imm` (no 16-bit ops),
  so the byte-store seeds use dword/byte stores.

### grid_probe.py (the harness itself)
- The generated file hit three generator bugs before any byte comparison ran:
  a doubled `ptr` in every sized memory variant; scalar SSE mnemonics (divsd,
  cvtss2sd, ...) given xmmword sources instead of m64/m32; and unsigned imm32
  values (0x80000000/0xffffffff) on 64-bit arith/test, which GNU as rejects because
  those destinations only take an imm32 whose sign-extension round-trips. All fixed
  in the generator; GAS then assembles all 27k+ lines with zero errors and zero
  warnings.
- PE/COFF relocation plumbing: mingw's objdump -t prints a bracketed symbol-table
  layout, and PE relocations carry their addend IN THE FIELD (the printout shows
  just the section symbol). The harness now parses both layouts and computes the
  linker's value as symval + printed addend + in-field value (ELF RELA leaves the
  field 0, so the same formula covers both), while PE's already-resolved REL32
  fields are left to the direct byte comparison.
- After the VEX_MOVE fix the streams matched except `mov reg, label` (9 cases):
  GNU as's PE backend uses a 7-byte absolute-memory form with ADDR32 where the ELF
  backend uses movabs with ADDR64. Not an assembler bug (our encoding matches ELF,
  which is what b2_check runs against); the grid now detects the object format and
  generates those cases only for ELF toolchains, with the skip printed.

## v18 — A2: SIGSEGV in all 4 configs (attempt 3 of A2 check)
- Evidence: v18 got past name selection (v17's fix worked) but every run died with
  exit -11 (SIGSEGV) right after `target_bf16_bytes`, before `selected:` printed.
- Hypothesis: the selection loop allocated `mats`/`all` only under `nm >= cap`
  (cap=16), so the FIRST matrix (nm=0) did `memset(&mats[0], ...)` on a NULL
  pointer. v17 masked this by matching zero names.
- Fix: allocate on `!mats || nm >= cap`, doubling from a first allocation of 16;
  realloc through temporaries so a failed realloc cannot leak the old pointer.
  No thresholds changed.

## B2 first Linux run (WSL, binutils 2.46) — two failures, both fixed at the cause
The first-ever execution of Part B on Linux (Ubuntu 26.04 in WSL2 on this PC). B1
passed immediately; B2's byte comparison passed for all 6 files on the first try
(our assembler is byte-identical to Linux GNU as), but the run then failed twice.

### Harness crash: objdump cannot disassemble an EMPTY nop-pad
- Evidence: every corpus file died with `AssertionError: objdump failed on
  .../<file>.pad.bin (exit 1)` inside padding_is_nops, AFTER the byte comparison had
  already succeeded.
- Hypothesis: on ELF objects GNU as emits no trailing padding at all (.text has
  alignment 1 there and ends exactly at our last byte - the <=15-byte nop tolerance
  only ever fires on the PE/COFF backend, where .text is 16-aligned), so the pad file
  was 0 bytes and objdump -D rejects empty input.
- Fix: only disassemble a NON-empty pad; an empty pad is exact length equality, the
  strongest form of the same criterion. Docstring updated (D-029). No threshold
  changed.

### Corpus 01_gp_mov.s itself segfaulted — both builds, identically
- Evidence: `01_gp_mov.s: ... exit ours=-11 gnu=-11 expected=1` - OUR ELF and GNU
  ld's linked ELF both died with SIGSEGV, while bytes were identical (363 == 363).
  Files 02-05 ran and exited correctly, so the toolchain was not at fault.
- Hypothesis (confirmed by one-line ablation: removing any of lines 40, 52, 60 or 69
  stops the crash): the indexed store `mov qword ptr [rsp+rax*8], rbx` (rax=3 ->
  [rsp+24]) overwrote the dword that `mov dword ptr [rsp+24], ecx` (ecx=4) had put
  there; the load section then read the low half of the stack pointer into ecx, and
  `mov edx, dword ptr [rbx+rcx*4+32]` indexed ~2 GB off the stack. The program's own
  comment - "rax and rcx are small so the indexed forms address real stack slots" -
  stated the invariant its own store violated; on ANY Linux (Kaggle included) it
  would have crashed the same way; it had simply never been executed before.
- Fix: the indexed store now targets the free slot `[rsp+rax*8+48]` (instruction
  coverage unchanged: index*scale + disp8), with a comment recording the bug. The
  check itself was never touched: bytes equal AND both programs run and exit with the
  marker remain the criteria (D-030). Local PE re-verification: 1841 bytes, 0
  problems; WSL re-run: all 6 files exit 1-5 (06 skipped, no AVX2).
