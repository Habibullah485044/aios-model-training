#!/usr/bin/env python3
"""Temporary harness: print the CPU count and the parallel sum 05_sum
computes for EVERY thread count T = 1..N. Deleted after recording the
numbers in PROGRESS.md. Patches 05's main.s to write N and each total
as raw 8-byte little-endian words to stdout."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "b1"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "b2"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "b3"))
import b4_check  # noqa: E402  (reuses its linkers/emitter)
import elf64_writer  # noqa: E402

SRC = os.path.join(HERE, "corpus", "05_sum", "main.s")
p = os.path.join(HERE, "tmp", "05_measure.elf")

with open(SRC, "r", encoding="utf-8") as f:
    text = f.read()

# 1) report N right after sched_getaffinity returns it
old = "    mov r12, rax                 # N\n"
assert old in text
text = text.replace(old, old +
    "    mov qword ptr [rip+outbuf], rax\n"
    "    lea rdi, [rip+outbuf]\n"
    "    mov esi, 8\n"
    "    call write_str\n", 1)

# 2) report each T's total (still verified against the fixed constant)
old = "    mov rax, qword ptr [rbx+8]\n    movabs rcx, 5000000050000000\n"
assert old in text
text = text.replace(old,
    "    mov rax, qword ptr [rbx+8]\n"
    "    mov qword ptr [rip+outbuf], rax\n"
    "    lea rdi, [rip+outbuf]\n"
    "    mov esi, 8\n"
    "    call write_str\n"
    "    mov rax, qword ptr [rbx+8]\n"
    "    movabs rcx, 5000000050000000\n", 1)

# 3) the 8-byte scratch word lives in the RW data page
text = text.replace("msg_ok:\n", "outbuf:\n    .byte 0,0,0,0,0,0,0,0\nmsg_ok:\n", 1)

# runtime FIRST, like b4_check: the program's `.align 4096` data page
# must be the last thing in the image.
mods = [("runtime.s", b4_check.read_src(os.path.join(HERE, "runtime.s"))),
        ("main.s", text)]
base_run = elf64_writer.BASE + elf64_writer.SPLIT_CODE_OFF
run_text, run_entry, run_syms, _ = b4_check.link(mods, base=base_run)
data_off = run_syms["outbuf"] - base_run   # page-aligned data page
blob = elf64_writer.emit_elf64(run_text, entry_off=run_entry - base_run,
                               data_off=data_off)
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, "wb") as f:
    f.write(blob)
os.chmod(p, 0o755)

import subprocess  # noqa: E402
r = subprocess.run([p], capture_output=True, timeout=300)
out = r.stdout
n = int.from_bytes(out[:8], "little")
print("cpu_count N = %d, exit %d, %d bytes of stdout"
      % (n, r.returncode, len(out)))
assert len(out) >= 8 + 8 * n, out
for t in range(1, n + 1):
    v = int.from_bytes(out[8 * t:8 * t + 8], "little")
    print("  T=%d: %d  (0x%016X) %s"
          % (t, v, v, "ok" if v == 5000000050000000 else "WRONG"))
print("trailing:", out[8 + 8 * n:])
