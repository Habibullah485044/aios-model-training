#!/usr/bin/env python3
"""run_all.py - Part B pass checks, chained in mission order. Every check
runs on Kaggle (Linux: file/readelf/objdump, GNU as, executing the ELF
files), or locally under WSL once a distro exists (USER_ACTIONS.md) -
never on bare Windows. Python is used only to build and test, never
inside the engine. Writes partb_results.txt at the native root.
Exit 0 only if every step passes.
"""
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(HERE)
RESULTS = os.path.join(NATIVE, "partb_results.txt")

STEPS = [
    ("B1 ELF64 writer (132-byte exit42)", ["b1/b1_check.py"]),
    # B2 = the mission pass check over the corpus, then the auto-generated
    # differential grid (same criterion - byte identity with GNU as - over
    # the whole operand space; D-023), then the align-probe (gas .align
    # pad bytes incl. the 4096-byte page aligns of the B3 programs).
    # Still one B2 step.
    ("B2 assembler (byte-identical to GNU as)",
     ["b2/b2_check.py", "b2/grid_probe.py", "b2/align_probe.py"]),
    # B3 = our linker: multi-module placement + cross-module relocations,
    # byte identical to as+ld (both link orders + a gap/align grid),
    # then the four mission programs built ONLY by our assembler + ELF
    # writer (byte-identical code, W^X segments, fixed behavior targets
    # incl. the sparse 6.17 GB weights file).
    ("B3 linker (byte-identical to GNU ld)",
     ["b3/b3_check.py", "b3/programs_check.py"]),
    # B4 = the runtime services (raw syscalls, clone/futex threads,
    # CLOCK_MONOTONIC timers) hand-written in asm: byte identical to
    # as+ld AND executed against fixed stdout/exit targets.
    ("B4 runtime (syscalls/threads/timers)", ["b4/b4_check.py"]),
]


def main():
    lines = []
    if os.path.exists(RESULTS):
        os.remove(RESULTS)
    for name, scripts in STEPS:
        print("=== %s ===" % name, flush=True)
        if not all(os.path.exists(os.path.join(NATIVE, s)) for s in scripts):
            print("STEP MISSING: %s" % ", ".join(scripts), file=sys.stderr)
            sys.exit(2)
        failed = None
        for rel in scripts:
            r = subprocess.run([sys.executable, os.path.join(NATIVE, rel)],
                               cwd=NATIVE)
            if r.returncode != 0:
                failed = (rel, r.returncode)
                break
        if failed:
            status = "FAIL (%s exit %d)" % failed
        else:
            status = "PASS"
        print("%s: %s" % (name, status), flush=True)
        lines.append("%s: %s" % (name, status))
        if failed:
            lines.append("PART B FAILED")
            with open(RESULTS, "a") as f:
                f.write("\n".join(lines) + "\n")
            print("STEP FAILED: %s (%s exit %d)" % ((name,) + failed),
                  file=sys.stderr)
            sys.exit(failed[1])
    lines.append("PART B PASSED (%d steps)" % len(STEPS))
    with open(RESULTS, "a") as f:
        f.write("\n".join(lines) + "\n")
    print("PART B PASSED (%d steps)" % len(STEPS), flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        with open(RESULTS, "a") as f:
            f.write("run_all crashed:\n" + traceback.format_exc())
        raise
