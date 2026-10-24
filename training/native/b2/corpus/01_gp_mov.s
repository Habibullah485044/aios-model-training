# b2 corpus 01 - GP data movement: mov (all widths and operand forms), lea,
# zero/sign extension. Ground rules: assembles unmodified with GNU as
# (.intel_syntax noprefix); no .align/.p2align; no 16-bit operands; no
# absolute [disp32]; every memory access is a valid stack slot at run time so
# the bytes can also be executed (exit status 1).
    .intel_syntax noprefix
    .text
    .global _start
_start:
    sub rsp, 256
    mov rbp, rsp
    mov r12, rsp
    mov r13, rsp
    mov rbx, rsp
    xor ecx, ecx
    xor edx, edx

    # reg, reg - 64/32/8 bit, low and high register numbers, REX needs
    mov rax, rbx
    mov r15, r14
    mov eax, ebx
    mov r15d, r14d
    mov al, bl
    mov r8b, r9b
    mov sil, dil

    # reg, imm - imm32 sign-extended, movabs, 32-bit, 8-bit
    mov rax, 42
    mov rax, -1
    mov rax, 0x123456789abcdef0
    mov r9, 0x7fffffff
    mov r9, -0x80000000
    mov ecx, 0xdeadbeef
    mov r10d, 7
    mov al, 0x7f
    mov r8b, 0x20

    # mem, reg - stores with disp0/disp8/disp32, rbp/r12/r13 bases, SIB
    # (rax and rcx are small so the indexed forms address real stack slots)
    mov rax, 3
    mov rcx, 4
    mov qword ptr [rsp+8], rax
    mov qword ptr [rsp+16], r8
    mov dword ptr [rsp+24], ecx
    mov byte ptr [rsp+28], al
    mov qword ptr [rsp+32], r15
    mov qword ptr [rsp+128], r15
    mov qword ptr [rbp], rax
    mov qword ptr [r12], rax
    mov qword ptr [r13+8], rax
    mov qword ptr [rsp-8], rax
    # (rax*8+48 lands on a free slot: a qword store over [rsp+24] would
    # clobber the dword that the load section below reads into ecx and
    # then uses as an index - a real bug this program had until the
    # first Linux run crashed it)
    mov qword ptr [rsp+rax*8+48], rbx
    mov dword ptr [rbx+rcx*4+32], edx

    # reg, mem - loads, same addressing forms
    mov rax, 3
    mov rcx, 4
    mov rax, qword ptr [rsp+8]
    mov r8, qword ptr [rsp+16]
    mov ecx, dword ptr [rsp+24]
    mov al, byte ptr [rsp+28]
    mov r15, qword ptr [rsp+32]
    mov r15, qword ptr [rbp]
    mov r15, qword ptr [r12]
    mov r15, qword ptr [r13+8]
    mov r15, qword ptr [rsp-8]
    mov rax, 3
    mov r15, qword ptr [rsp+rax*8]
    mov edx, dword ptr [rbx+rcx*4+32]

    # mem, imm
    mov qword ptr [rsp+40], 1
    mov qword ptr [rsp+48], -1
    mov dword ptr [rsp+56], 0x40000000
    mov byte ptr [rsp+60], 0x0f

    # lea - disp forms and rip-relative data (no memory access)
    lea rax, [rsp+8]
    lea rdx, [rbx+rcx*4+32]
    lea rsi, [rip+bench_data]
    lea rdi, [r13+8]

    # zero / sign extension
    movzx eax, byte ptr [rsp]
    movzx r10d, word ptr [rsp+2]
    movzx r11, byte ptr [rsp+3]
    movsx edx, byte ptr [rsp+4]
    movsx r8, word ptr [rsp+6]
    movsxd r9, dword ptr [rsp+8]
    movzx ecx, al
    movsx edx, bl

    # rip-relative loads of the data below
    mov rax, qword ptr [rip+bench_data]
    mov eax, dword ptr [rip+bench_data+8]

    mov edi, 1
    mov eax, 60
    syscall

bench_data:
    .quad 0x0123456789abcdef
    .long 0x00000000
