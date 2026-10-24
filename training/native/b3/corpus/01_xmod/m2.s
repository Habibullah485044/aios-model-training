# b3 corpus 01 module 2 - called code, with a local label that must NOT
# leak to the other modules (not .global) and enough padding to push
# module 3's placement around.
    .intel_syntax noprefix
    .text
    .global helper
helper:
    mov eax, 7
    ret
    .space 150
local_only:
    .long 0x55aa55aa
