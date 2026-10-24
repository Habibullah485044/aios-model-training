# b2 corpus 02 - GP arithmetic and logic: add/or/and/sub/xor/cmp in every
# operand form, test, imul (1/2/3 operand), inc/dec/neg/not, shifts. Memory
# operands are valid stack slots; exit status 2.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    sub rsp, 256
    mov rbp, rsp
    mov rbx, rsp
    xor ecx, ecx
    xor edx, edx
    xor eax, eax

    # r/m, r - 64/32/8 bit
    add rax, rbx
    add r15, r14
    or rsi, rdi
    and r8, r9
    sub r10, r11
    xor r12, r13
    cmp rcx, rdx
    add eax, ebx
    or r8d, r9d
    cmp r10d, r11d
    add al, bl
    xor r8b, r9b
    cmp spl, bpl
    add qword ptr [rsp+8], rax
    or qword ptr [rsp+16], r15
    sub dword ptr [rsp+24], ecx
    xor byte ptr [rsp+28], al

    # r, r/m
    add rax, qword ptr [rsp+8]
    sub r8, qword ptr [rsp+16]
    and ecx, dword ptr [rsp+24]
    or dl, byte ptr [rsp+28]

    # r/m, imm8 (83 form) and imm32 (81 form for non-accumulators)
    add rax, 5
    add rbx, -5
    and rsi, 0x7f
    or r8, 100
    sub rdx, 0x1234
    xor r9d, 0x1000
    cmp r10d, 100
    cmp r11, 0x7fffffff
    add qword ptr [rsp+32], 3
    add dword ptr [rsp+40], 0x1000
    sub qword ptr [rsp+48], 1
    xor dword ptr [rsp+56], 0xff
    and byte ptr [rsp+60], 0x0f

    # test
    test rax, rbx
    test r8, r9
    test ecx, edx
    test dl, cl
    test rdx, 0x1234
    test rdx, 5
    test dword ptr [rsp+32], 0x10000
    test byte ptr [rsp+60], 0x0f

    # imul
    imul rax, rbx
    imul rcx, qword ptr [rsp+8]
    imul rax, rbx, 5
    imul rdx, rcx, 0x1234
    imul r8

    # inc / dec / neg / not
    inc rax
    dec rbx
    neg rdx
    not rsi
    inc dword ptr [rsp+24]
    dec qword ptr [rsp+8]
    neg r9
    not r10d
    inc byte ptr [rsp+28]

    # shifts: count 1 (D1 form), imm8, cl, and memory
    shl rax, 3
    shr rbx, 1
    sar rdx, 63
    shl r8d, 1
    shr r11d, 4
    shl r10, cl
    shr r12d, cl
    sar r13, cl
    shr qword ptr [rsp+8], 2

    mov edi, 2
    mov eax, 60
    syscall
