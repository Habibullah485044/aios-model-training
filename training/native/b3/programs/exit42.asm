# b3 program 1 - exit42: exits with status 42, built from .asm source by
# our assembler + our ELF writer (never GNU as/ld in the build). The
# check also assembles this source with GNU as and requires the code
# bytes to match ours, and requires the emitted image to be byte
# identical to B1's hand-coded 132-byte exit42 (same 12 code bytes).
    .intel_syntax noprefix
    .text
    .global _start
_start:
    mov eax, 60                     # sys_exit
    mov edi, 42
    syscall
