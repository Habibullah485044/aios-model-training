# b3 corpus 01 module 3 - exit path and the data modules 1 refers to,
# with .align 16 so this module starts on a 16-byte boundary and the gap
# before it is GNU ld's nop fill.
    .intel_syntax noprefix
    .text
    .global the_end
    .global counter
    .global seed
the_end:
    mov edi, 1
    mov eax, 60
    syscall
    .align 16
counter:
    .long 42
seed:
    .quad 0x1122334455667788
