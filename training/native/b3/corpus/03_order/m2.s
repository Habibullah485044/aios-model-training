# b3 corpus 03 module 2 - a leaf function called in a loop, plus an
# .align 64 whose nop pad must match gas byte for byte inside the module.
    .intel_syntax noprefix
    .text
    .global fn_b
fn_b:
    mov eax, 2
    ret
    .align 64
pad64:
    .byte 0xcc
