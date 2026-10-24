# USER ACTIONS — AIOS engine mission

Anything the user must do, with exact instructions.

## DONE (2026-10-12) — WSL distro installed by the agent; the Linux checks run on this PC
- Installed non-interactively, no admin prompt and no restart needed:
  `wsl --install -d Ubuntu --no-launch` (Ubuntu 26.04; the image already ships
  binutils 2.46 and python3, so no apt step was required). The distro runs as root.
- To re-run the checks yourself (ONE wsl invocation; each `wsl.exe` call boots a
  fresh VM, so scratch state in /tmp does not survive between calls):
  `wsl -d Ubuntu -u root -- bash -lc 'cd "/mnt/c/Users/OBSESSION/Desktop/AIMODEL TEST" && python3 training/native/partb/run_all.py && python3 training/native/b2/grid_probe.py'`
- Result on this PC (Intel i5-2540M, AVX yes, AVX2/FMA no):
  `PART B PASSED (2 steps)` — B1 PASSED (exit-42 ELF runs); B2 all 6 corpus files
  byte-identical (1841 bytes, 0 differing), programs 01-05 exit 1-5, `06_avx2.s`
  executed only as bytes with `SKIPPED: no AVX2`; grid **27,791 instruction cases,
  156,887 bytes, 0 differences** (ELF target — same numbers as the Windows PE run).

## ACTION NEEDED (1 kernel) — B4 on different hardware (required by Part B)
- Push/run `training/native/kaggle/aios-native-partb` on Kaggle (UI, no API key).
  The bundle is current (regenerated 2026-10-18, 33 embedded files, all verified
  byte-identical to their sources).
- Why: B4's spec says "run it in WSL, and also queue one Kaggle run (different CPU,
  more cores); don't wait". WSL on this PC has 4 CPUs; Kaggle's EPYC has more, which
  exercises `cpu_count` / `pool_init` with a different N and a different CPUID set.
- Expected: `B1 PASSED`, `B2 PASSED` (no AVX2 skip note), `GRID: ... 0 differences`,
  `B4 PASSED (6 programs)`, `PART B PASSED (4 steps)`.
- Send back `partb_results.txt` plus `b1/b1_results.txt`, `b2/b2_results.txt`,
  `b4/b4_results.txt`.

## ACTION NEEDED (2 kernels, unchanged) — Kaggle evidence for the mission record
- No Kaggle API key is ever needed from you; push/run the bundles from the Kaggle UI
  as before. The bundles below are current (regenerated after every source edit).

### 1. `training/native/kaggle/aios-native-partb` — closes the last B2 leg
- What it does: runs the B1 pass check (132-byte exit-42 ELF64), the B2 pass check
  (6 corpus files assembled with GNU as, linked, compared byte for byte with our
  assembler, and both linked programs executed), then the auto-generated
  differential grid (27.7k instruction cases vs GNU as).
- Why it still matters: on this PC `06_avx2.s` could only be byte-compared (no AVX2
  here). Kaggle's EPYC has AVX2+FMA, so this kernel EXECUTES it — which per the
  mission rule is what makes B2 count as PASSED.
- Send back `partb_results.txt` plus `b1/b1_results.txt` and `b2/b2_results.txt`.
- Expected: `B1 PASSED`, `B2 PASSED` (no skip note on AVX2 hardware), `GRID: ...
  0 differences`, `PART B PASSED (2 steps)`.

### 2. `training/native/kaggle/aios-native-a3build` — run this to get the A3 zip
- What it does: zig-builds `bench64.exe`/`bench32.exe`, scans them statically
  (pshufb present, 0 ymm/zmm, 0 AVX, 0 SSE4.1+, imports exactly KERNEL32.dll), builds
  `a3bench.bin` from synthetic data, and packages `aios-oldpc-bench.zip`.
- Send back `a3/a3_results.txt`; if it passes, the zip is ready for the A3 measurement.

## Planned (after the A3 zip exists) — A3: old-PC benchmark
- Once `aios-native-a3build` has passed you will get `aios-oldpc-bench.zip` here:
  1. Download and unzip `aios-oldpc-bench.zip` (link will be here).
  2. Double-click `run.bat`.
  3. Send back `results_64bit.txt` and `results_32bit.txt`.
- Nothing to do today. Work continues without this; the result only refines an already
  measured verdict (Gate A chose BF16 after packed measured 0.30–0.44× on Kaggle).
