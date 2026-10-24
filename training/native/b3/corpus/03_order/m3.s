# b3 corpus 03 module 3 - the exit path. Its .space 70 forces module 1's
# jmp to fn_c to stay rel32 whichever order the modules are linked in.
    .intel_syntax noprefix
    .text
    .global fn_c
    .space 70
fn_c:
    mov edi, 3
    mov eax, 60
    syscall
