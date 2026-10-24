# b3 corpus 03 module 1 - link-order sensitivity. This program is also
# linked REVERSED by the check, so every cross-module reference must
# follow the modules wherever they land. `call fn_b` and the .quad data
# below point at both other modules; the local `loop` label is module
# private. Exit status 3 comes from fn_c in module 3.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    xor r12d, r12d
    mov ecx, 4
loop:
    call fn_b                    # module 2
    add r12d, eax
    dec ecx
    jnz loop
    cmp r12d, 8
    jne fail
    jmp fn_c                     # module 3, rel32, ends the program
fail:
    mov edi, 99
    mov eax, 60
    syscall
    .space 130
tbl:
    .quad fn_b
    .quad fn_c
    .long _start
