# b2 corpus 03 - control flow and data directives: short/near jmp and jcc,
# call/ret, labels, .byte/.word/.long/.quad (including label references, which
# is where the code's load address enters). The long jump is forced past 127
# bytes with .space so both assemblers must choose the rel32 form. Exit
# status 3.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    xor eax, eax
    mov ecx, 3
loop_top:
    dec ecx
    jnz loop_top
    cmp eax, 0
    jne taken
    mov edx, 1
taken:
    jmp after_pad
    .space 200
after_pad:
    call helper
    test eax, eax
    js done
    mov eax, 0
done:
    mov edi, 3
    mov eax, 60
    syscall

helper:
    push rbp
    mov rbp, rsp
    mov eax, 7
    neg eax
    pop rbp
    ret

label_data:
    .long _start
    .quad _start
    .byte 0x11
    .word 0x2222
    .long 0x7fffffff
