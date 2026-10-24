# b3 program 3 - sumloop: adds 1..1000 (result 500500) and prints
# "500500\n" with our own integer-to-text code: no division, the decimal
# digits come out by repeated subtraction against the place table in the
# data segment (100000 .. 1, most significant first), written left to
# right into a stack buffer.
# Pass target (fixed before running): stdout exactly "500500\n" (7
# bytes), exit status 0.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    xor eax, eax                    # sum = 0
    mov ecx, 1
sum_loop:
    add rax, rcx
    inc ecx
    cmp ecx, 1001
    jne sum_loop                    # rax = 500500

    sub rsp, 32                     # digit buffer, written ascending
    lea rcx, [rsp]                  # next write position
    lea r10, [rip+places]           # the data segment
    xor r11d, r11d                  # place index 0..5
place_loop:
    cmp r11, 6
    je digits_done
    mov r8, qword ptr [r10+r11*8]
    xor r9d, r9d                    # digit = 0
digit_loop:
    cmp rax, r8
    jb digit_done
    sub rax, r8
    inc r9d
    jmp digit_loop
digit_done:
    add r9b, 48                     # '0' + digit
    mov byte ptr [rcx], r9b
    inc rcx
    inc r11
    jmp place_loop
digits_done:
    mov byte ptr [rcx], 10          # '\n'
    mov edi, 1
    mov rsi, rsp                    # 6 digits + '\n'
    mov edx, 7
    mov eax, 1                      # sys_write
    syscall
    xor edi, edi
    mov eax, 60                     # sys_exit(0)
    syscall
    .align 4096                    # data starts on its own page (W^X split)
places:
    .quad 100000
    .quad 10000
    .quad 1000
    .quad 100
    .quad 10
    .quad 1
