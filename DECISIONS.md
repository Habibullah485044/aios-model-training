# DECISIONS — AIOS engine mission

Every default / judgment call, one-line reason.

- **D-001** Phase 2 = A1 packer uses **per-tensor** 15-entry exponent tables (not one global
  table). Reason: §5/A1 specify a per-tensor table; it is what the format actually stores.
- **D-002** A1 exception count measured **1,358,322**, not Phase 1's 1,897,535. Reason: Phase 1's
  figure was computed with a single GLOBAL top-15 table over the whole model; per-tensor tables
  are more selective, so the fixed "1,897,535 exactly" threshold (defined as "same definition as
  Phase 1") is reported alongside the per-tensor count with the explanation — no threshold was
  loosened, both numbers are stated.
- **D-003** `phase2_check.py` gets a `guarded_main()` failure path (traceback + disk state into
  the captured results file). Reason: Kaggle's log endpoint returns empty for this kernel, so a
  bare crash shows nothing anywhere.
- **D-004** Harness captures gcc stderr into the results file. Reason: same empty-log quirk made
  three compile errors invisible until stderr was captured.
- **D-005** Fix compile errors at the cause (unused `read_u64_le`, unused `n_rows` param) instead
  of adding `-Wno-*` suppressions. Reason: mission rule "never loosen"; `-Werror` stays on.
- **D-006** Phase 2 harness is chained into `real_check.py` as step (6) so it reuses the exported
  `weights.bin` in the same kernel run. Reason: avoids a second 6 GB model download.
- **D-007** A2 bit-identical checks compare **packed vs BF16 fast kernels** (same float32
  accumulation, decode-only difference); comparisons vs the double-accumulating oracle stay
  tolerance-based (< 1e-4 rel). Reason: user rule change + §4.3 (FMA vs MUL+ADD never match).
- **D-008** A2 classification uses the **median** of 10 timed runs (best-of-10 also reported).
  Reason: median is robust to scheduler outliers on shared Kaggle workers; the mission fixes
  the threshold (1.10/1.00) but not the statistic, and median is the more conservative choice.
- **D-009** Gate A: the engine reads **BF16** weights. Packed format is kept as a storage/
  distribution option (1.33× smaller, proven lossless by A1+A4) but is not used at runtime.
  Reason: A2 measured packed/bf16 = 0.30–0.44 (LOSS) on every config of the one machine
  tested (shared AMD EPYC 7B12, Kaggle, ≤4 threads), two runs; the decode work dominated
  the 25% byte saving there. Threshold table rule: <1.00 = LOSS. Claim scoped to the
  measured machine — the A3 old-PC benchmark measures the user's DDR2-era CPU directly
  and will be recorded honestly; if it shows ≥1.10 there, the gate is revisited.
- **D-010** A3 (old-PC benchmark .exe) is deferred until after D3. Reason: Kaggle has no
  mingw/wine cross-tools (probed v20) and the user's PC has no compiler; our own PE writer
  (D3) is the planned way to build Windows executables. USER_ACTIONS.md records the plan.
- **D-011** A4 verifies the unpacked file as a full `weights.bin` replacement (symlinked into
  a second model dir) rather than comparing per-tensor arrays. Reason: exercises the exact
  oracle code path end-to-end; bit-identical logits is the mission's stated criterion.
- **D-012** PLAN CHANGE (user): B2 = a real assembler (.asm Intel-syntax text in, machine
  code out, byte-identical to GNU as); Part C = hand-written `.asm` sources built by our
  assembler+ELF writer (no Python codegen); C1 = runtime config/manifest loader in asm;
  D2 tokenizer in asm; Python only for build scripts/tests. All thresholds unchanged.
- **D-013** A3 benchmark exe built with zig cc (x86_64-windows-gnu + x86-windows-gnu,
  -mcpu=core2, kernel32-only) instead of waiting for D3's PE writer. Reason: user verified
  the zig path on Kaggle (PE32+/PE32, CreateThread, pshufb present, 0 AVX/SSE4.1, only
  KERNEL32.dll imported) — unblocks the old-PC measurement months early.
- **D-014** B2's corpus is 5 files (`b2/corpus/01..05*.s`), one clear area per file
  (GP moves, GP arithmetic, control flow/data, SSE moves, SSE arithmetic), each a
  complete `_start` program that exits with a marker status. Ground rules: assembles
  unmodified with GNU as (`.intel_syntax noprefix`), no `.align`/`.p2align`, no 16-bit
  operands, no absolute `[disp32]`, no indirect call/jmp, and every memory access is a
  valid stack slot at run time. Reason: the check must be able to do more than compare
  bytes — it links both sides and runs them, so a corpus file that faults would fail a
  check the assembler itself cannot fix.
- **D-015** The byte comparison is against the LINKED `.text` with our assembler given
  ld's `.text` address as `base`, and GNU as's trailing section-alignment padding is
  tolerated (<= 15 bytes) but verified to disassemble to nops. Reason: relocations must
  be resolved (a relocatable .o leaves zeros behind them), absolute values depend on the
  load address, and the padding is a section artifact rather than an instruction; the
  mission's criterion is that the assembled code is byte-identical.
- **D-016** B2 was pre-verified locally with a real GNU toolchain installed for this
  purpose (MSYS2 mingw-w64 binutils 2.47: as/objdump/objcopy/readelf; its ld cannot emit
  ELF) using two harnesses that are NOT part of the mission: a per-form differential
  probe (88 forms, 0 differences) and a corpus comparison that normalizes GAS's
  relocation fields and asserts our fixups sit exactly where GAS's relocations do.
  Reason: each Kaggle round trip is slow and user-mediated, so encoding bugs should be
  found locally; the recorded evidence is still the Kaggle check, and the local run
  cannot cover the linked-ELF comparison or program execution (Windows).
- **D-017** GNU as's encoding *choices* were measured, not assumed, and asm.py now
  matches them: A-register short forms for al/eax/rax immediates (04..3D, A8/A9), the
  imm8 forms (80/83) whenever the value fits, the sign-extended imm32 form for a 64-bit
  operand whose bit pattern round-trips (`mov rax, 0xffffffff80000000` -> `48 C7 C0 00 00
  00 80`), the D0/D1 shift-by-one form, and the movq form table (xmm<-m64 `F3 0F 7E`,
  m64<-xmm `66 0F D6`, xmm<-r64 `66 0F 6E`, r64<-xmm `66 0F 7E`, xmm<-xmm `F3 0F 7E`).
  Reason: byte identity with GAS is the pass criterion, and three of these (accumulator
  forms, movq m64<-xmm, the imm32 round-trip) were the opposite of the obvious choice.
- **D-018** Bundle regeneration refreshed `aios-native-a3build` and
  `aios-native-real-check` as well as the new `aios-native-partb`. Reason: their embedded
  `bench.c` was stale (written before the last bench.c edit); a bundle must be a
  byte-identical reproduction of the sources.
- **D-019** `.align N`/`.space N` fill with zeros in our assembler, while GNU as fills
  `.text` with nops, so those directives stay out of the byte-compared corpus and the
  divergence is recorded here. Reason: reproducing GAS's multi-byte nop padding is a
  rabbit hole with no effect on the engine (our own ELF writer uses explicit placement),
  and hiding the difference inside the corpus would be dishonest. Part C can revisit it
  if it ever needs GAS-identical alignment padding.
- **D-020** B2's AVX2/FMA subset is exactly the 19 mission mnemonics, and every VEX
  encoding choice was measured against GNU as 2.47 rather than assumed: the short C5
  prefix exactly when the instruction is 0F-map with W=0 and neither the SIB index nor
  the r/m register needs bit 3 (any other case gets C4); the shift-immediate group
  (0F 71/72 /n) puts the DESTINATION in VEX.vvvv and the /n extension in ModRM.reg;
  vfmadd231ps is 66 0F38.W0 **B8** (not 9A); vzeroupper is `C5 F8 77`; a two-register
  vmovups/vmovdqu flips to the store opcode when the load form would need C4 and the
  store would not (`vmovups xmm0,xmm8` -> `C5 78 11 C0`, ties keep the load); the
  vpslld/vpsrlw COUNT operand is always 128-bit (xmm register or xmmword PTR) even
  with a ymm destination; movmskps/movmskpd/vpmovmskb take register sources only;
  memory destinations are rejected for vpand/vpor/vpxor/vaddps/vpslld-imm but allowed
  for vextractf128; the shift-by-immediate MEMORY-source form is not implemented
  because GNU as routes it to an EVEX/AVX-512 template outside this subset. Reason:
  byte identity with GAS is the criterion, and six of these were the opposite of the
  obvious encoding.
- **D-021** Corpus grows to 6 files: `06_avx2.s` (marker 6) exercises all 19 VEX
  mnemonics including high registers, rip-relative sources and index*scale addressing,
  with values chosen so the program runs fault-free on AVX2 hardware and exits 6.
  Same ground rules as D-014 (no `.align`, stack-only memory). Reason: b2_check not
  only compares bytes but EXECUTES both linked programs, so the AVX file must be
  runnable, and the Kaggle EPYC has AVX2+FMA.
- **D-022** The hand-written 88-form differential probe was replaced by an
  auto-generated grid, `b2/grid_probe.py`: every supported mnemonic x every operand
  form x the register/addressing variations (r8-r15, xmm8-15, ymm8-15, rsp/r12/rbp/r13
  bases, disp8/disp32, index*scale incl. r12/r13 indexes, rip-relative, 64-bit
  immediates). Local result: 27,747 cases, 156,548 bytes, 0 differences. Absolute
  relocations are verified by patching GAS's fields with the values computed from
  GAS's OWN symbol table + addends, then requiring full byte identity, so every byte
  is checked and the expected relocation values do not come from our own label
  arithmetic. A GAS syntax error fails the run: the grid only emits forms GAS accepts.
  PE/COFF vs ELF difference: GNU as's PE backend spells `mov reg, symbol` as a 7-byte
  absolute-memory load with a 32-bit relocation, while the ELF backend uses movabs +
  64-bit relocation - the encoding our assembler emits and the one b2_check compares
  against on Kaggle, so those cases are generated only for ELF-style toolchains.
  Reason: coverage of the whole operand space is the point of the exercise, and the
  target-format quirk must not be papered over.
- **D-023** Part B's B2 step runs `b2_check.py` and then `grid_probe.py` (still one
  B1/B2 step list, no new mission step). Reason: the grid applies the same pass
  criterion (byte identity with GNU as) to a far larger operand space; it is
  evidence for B2, not a separate mission milestone, and the mission step list stays
  B1..B4.
- **D-024** This machine CAN run Linux checks itself: Windows 10 build 19045 (22H2)
  with WSL2 already installed (`wsl --status` -> Default Version 2) but no distro.
  USER_ACTIONS.md carries the exact one-command setup; once a distro exists, Part B's
  pass checks and the grid run locally (`/mnt/c/...`) and Kaggle is only needed for
  the model-scale kernels (A2/A3/real-check). No Kaggle API key is ever requested;
  the bundles stay ready to push from the Kaggle UI.
- **D-025** `mov reg, label` now means what GNU as's Intel syntax means: a bare label
  is a MEMORY operand - an absolute load (mod=00 SIB 04 25 + disp32 with a 32-bit
  relocation), not movabs. `offset label` puts the ADDRESS in an immediate and is
  implemented in all four widths GAS accepts (r64 -> 48 C7 C0 imm32, r32 -> B8 imm32,
  r8 -> B0 ib with an 8-bit relocation; `movabs r64, offset` -> 48 B8 imm64), and the
  `movabs` mnemonic is now real (imm64 on any r64; accumulator-only moffs64 loads
  48 A1/A1/A0 and stores 48 A3/A3/A2 from an absolute address; every form GAS rejects
  - non-accumulator movabs, movabs imm on non-r64, movabs with a register/rip
  address, movabs reg,reg - is rejected identically). Reason: the old encoder emitted
  movabs for a bare label, i.e. it computed the ADDRESS where GAS loads the VALUE -
  a silent semantic difference invisible to byte comparison because no corpus or grid
  line used a bare label; the user caught it against real Linux GAS. Every encoding
  choice above was then measured against GNU as 2.47 (PE) and confirmed on 2.46 (ELF).
- **D-026** The grid generates the bare-label/offset/movabs cases UNCONDITIONALLY for
  both object formats and the ELF-gating (detect_elf + skip note) is deleted. Reason:
  the PE/ELF "difference" those workarounds were built on was our own movabs bug, not
  a GAS backend difference - both backends emit the identical memory form, so the
  cases now run everywhere (27,791 cases, 0 differences on BOTH toolchains). The
  8-bit-offset cases reference `_start` (offset 0) rather than grid_data: GAS checks
  a 1-byte relocation field at assembly time and rejects a value that does not fit.
- **D-027** The Linux checks now run on THIS PC: Ubuntu 26.04 installed into WSL2
  non-interactively (`wsl --install -d Ubuntu --no-launch`; no admin prompt, no
  restart, no first-run UI - the distro starts as root), and the image already ships
  binutils 2.46 + python3, so no apt step is needed. Each `wsl.exe` invocation boots
  a fresh VM with a tmpfs /tmp, so a check must run inside ONE invocation (scratch
  state does not survive between calls; results live under /mnt/c). Kaggle is then
  needed only for the model-scale kernels and AVX2 execution.
- **D-028** This PC's CPU is an Intel Core i5-2540M (Sandy Bridge): AVX yes, AVX2/FMA
  no. b2_check.py reads /proc/cpuinfo once; on a non-AVX2 CPU it still byte-compares
  `06_avx2.s` (the bytes are the pass criterion) but skips executing it, printing
  "SKIPPED: no AVX2" plus a NOTE that the file must execute on AVX2 hardware before
  B2 counts as PASSED (the user's explicit rule). Exit code stays 0 because the byte
  criterion passed and the skip is stated, never hidden; the run_all/PART B lines
  carry the same note.
- **D-029** ELF objects have no trailing nop padding: GNU as's ELF .text gets
  alignment 1 and ends exactly at our last byte (the <=15-byte nop tolerance only
  ever fires on the PE/COFF backend, where .text is aligned to 16). b2_check now
  disassembles the pad only when it is non-empty - objdump cannot read an empty file,
  which crashed the first Linux run. No threshold changed; zero trailing bytes is
  exact length equality, the strongest form of the same criterion.
- **D-030** Corpus `01_gp_mov.s`'s latent bug was fixed at the cause, in the program:
  its indexed qword store `mov [rsp+rax*8], rbx` landed on [rsp+24] and clobbered the
  dword the load section later reads into ecx and uses as the index of
  `[rbx+rcx*4+32]`, so BOTH linked builds (ours and GNU ld's, byte-identical) died
  ~2 GB off the stack with SIGSEGV. Found by one-line ablation in WSL (removing any of
  lines 40/52/60/69 stops the crash); fixed by moving the store to the free slot
  `[rsp+rax*8+48]`, keeping the indexed-form coverage. Reason: the check (bytes equal
  AND both programs run and exit with the marker) was never at fault - the test
  program was; loosening the check would have hidden a real execution criterion.
- **D-031** B3 module placement, measured against binutils 2.46/2.47 with readelf and
  a reversed-order link: modules are concatenated in list order; module i starts at
  the first address at/after the end of module i-1 that is a multiple of the module's
  section alignment - gas's sh_addralign, i.e. the LARGEST `.align N` in the module
  (1 when it has none, measured per .align combination). The rule is in absolute
  address space, so it holds for any .text base.
- **D-032** The gap between two linked modules is filled with GNU ld's nop fill:
  10-byte nops first, canonical remainder last (asm.gap_fill, measured k=0..63).
  That is NOT gas's `.align` pad (asm.align_fill: canonical decomposition with
  11-byte nops, and a leading 90 after .space/data context) - both sequences were
  captured  byte for byte and both are needed. The 64-case .align 64 gap sweep proves
  gap_fill against ld for every gap size 0..63 in the linked image.
- **D-038** D13 is CHANGED by the user: B3 keeps the separate linker
  (`b3/link.py`). Reason: D13 prescribed a single assembly unit pulled
  together with `.include` and no linker, but the linker was built and
  verified instead (byte-identical placement + relocations vs `as`+`ld`,
  3 programs x 2 link orders + 82-case grid, 10,762 bytes, 0 differing -
  D-034), and the user's instruction is to KEEP it. Programs are therefore
  built as one or more modules: our assembler -> our linker (when there is
  more than one module) -> our ELF writer. `.include` is not needed.
- **D-033** Cross-module relocations follow ELF visibility: only `.global`/`.globl`
  labels are visible to other modules; a module's own label shadows an external one
  (corpus 02_align pins both directions: module 1 reads its own local `shadow`=11
  while module 2 exports `shadow`=33 which module 3 must see); an undefined name is a
  link error and a name exported by two modules is a multiple-definition error. A
  branch to a label in another module always takes the rel32 form - GNU as cannot
  know the distance at assembly time so it never emits rel8 for it either.
- **D-034** b3_check.py's criterion is byte identity with `as` + `ld` on the whole
  linked .text (3 corpus programs x BOTH link orders + an 82-case generated
  gap/align grid), plus equal entry points, matching `nm -n` symbols, and both ELFs
  exiting with the program marker. Every relocation kind is exercised cross-module:
  rel32 call/jmp/jcc, rip-relative, `offset label` abs32, movabs abs64, [label]
  disp32, .long/.quad data. WSL result: 3 programs x2 orders + 82 grid cases,
  10,762 bytes compared, 0 differing, gap sizes 0..63 all covered.
- **D-035** `emit_elf64` grew an `entry_off` parameter (default CODE_OFF, unchanged
  for B1's 132-byte target): a B3 program linked in reverse module order does not
  start with _start, so the runtime ELF must set e_entry to the real offset instead
  of assuming code offset 0. The B3 check builds its runtime image at the B1 load
  address with the true entry - the bytes are only proven equal at ld's base, but the
  linked program still has to work when it runs.
- **D-036** The B4 runtime (`b4/runtime.s`) is raw Linux x86-64 syscall services
  written in the B2 dialect: write/exit, mmap, clone, futex, clock_gettime and
  nanosleep wrappers. Tasks are spawned with `clone(CLONE_VM|SIGCHLD, own_stack)`:
  a VM-sharing task (true shared memory, its own stack from mmap) whose entry is the
  program-defined `thread_body(arg1, arg2)` - a fixed direct call, since the subset
  has no indirect call. CLONE_THREAD was rejected: with `exit` (not exit_group) the
  process status would be the LAST thread to die (nondeterministic), while the
  SIGCHLD form makes the parent's own exit the process status and joins race-free
  via futex flags (store flag -> FUTEX_WAKE; FUTEX_WAIT re-checks in a loop, so no
  atomics are needed in the subset).
- **D-037** b4_check.py's criterion: each two-module program (runtime.s + main.s)
  must be byte-identical to `as` + `ld` with matching nm symbols (ld's synthetic
  `__bss_start/_edata/_end` excluded - no source label stands for them), AND both
  ELFs must run against fixed targets fixed before running: 01_syscalls prints
  "B4 syscalls ok\n" / exit 41, 02_threads prints "B4 threads ok\n" / exit 42 (3
  tasks write slots (i+1)*11, parent futex-joins and checks sum=66), 03_timers
  prints "B4 timers ok\n" / exit 43 (CLOCK_MONOTONIC delta of a 50 ms nanosleep
  must clear a fixed 40 ms floor). Mutable state lives only in registers, stack and
  mmap - never in label data - because GNU ld maps .text R-X while B1's RWX single
  segment would tolerate it; identical behavior on both sides is only honest if
  neither can write the image.
- **D-039** B4's threading model is REAL threads, not just VM-sharing tasks: the pool
  workers are spawned with `clone(0x50F00 = CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|
  CLONE_THREAD|CLONE_SYSVSEM)` on 64 KiB heap stacks, a worker ends with `sys_exit(60)`
  (that thread only) and the main thread ends every program with `exit_group(231)`, so
  the process status is always the main thread's and cannot depend on thread teardown
  order. Jobs are dispatched through a generation-counted work queue in heap memory
  (`pool` block: n/gen/job_arg/active/done + stacks[n]) because the subset has no
  indirect call (D-036): every worker calls the program-defined `pool_job`; `pool_run`
  publishes job_arg+active, bumps gen, wakes every parked worker and then futex-waits
  on `done` until `active` workers finished. Mutable state lives only in heap/stack/
  registers, never in label data (D-037 still holds).
- **D-040** B4's pass criteria are extended by three programs on top of D-037, with the
  targets fixed before running: `04_features` prints this machine's `/proc/cpuinfo`
  flags restricted to (sse2, ssse3, sse4_1, avx, avx2, fma) in that order and exits 44
  (the check computes the expected string from `/proc/cpuinfo` itself, so the program
  must mirror the CPU, not a constant); `05_sum` sums 1..100,000,000 over a `lock xadd`
  slice cursor for EVERY thread count T=1..N (N from sched_getaffinity) and verifies
  5,000,000,050,000,000 each time, exit 45; `06_barrier` runs 10,000 sense-barrier
  rounds over all N workers and verifies gen=10000/count=0 afterwards, exit 46 - a
  deadlock trips the check's per-program timeout instead of passing.
- **D-041** Measured while debugging 05/06: on x86-64 `sched_getaffinity` is syscall
  **204** (`203` is sched_SETaffinity - a zero-mask setaffinity returns EINVAL, which is
  what made the first cpu_count return 0). Confirmed against glibc's own wrapper
  (`mov $0xcc,%eax` in libc). It also returns the number of mask bytes it FILLED IN,
  not 0, so the bit count must run over exactly those bytes: the rest of the stack
  buffer is untouched garbage and must never be counted.
- **D-042** `barrier()` must keep the arriving thread's generation in a register the
  futex helpers do not touch: `futex_wait`/`futex_wake_all` zero r8 (uaddr2) and r10
  (timeout), so a generation living in r8 was read back as 0 and every waiter released
  early - 06_barrier hung. The generation now lives in r12 (pushed/popped around the
  call), matching the runtime's rule that rbx/r12-r15 survive every service call.
