# b3 corpus 01 module 1 - cross-module references of every relocation
# kind: call/jmp to another module's labels (always the rel32 form, like
# GNU as with an undefined symbol), rip-relative loads of another
# module's data, absolute forms (offset label, movabs, [label] disp32)
# and .long/.quad data relocations. Exit status 1 comes from the_end in
# module 3; fail exits 99.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    call helper                    # module 2, rel32
    cmp eax, 7
    jne fail
    lea rsi, [rip+seed]            # module 3, rip
    mov rax, qword ptr [rsi]
    movabs r8, offset seed         # module 3, abs64
    cmp rax, qword ptr [r8]
    jne fail
    mov ecx, offset counter        # module 3, abs32
    mov ebx, dword ptr [counter]   # module 3, [label] disp32
    cmp ebx, dword ptr [rcx]
    jne fail
    mov r9d, dword ptr [rip+counter]
    cmp r9d, 42
    jne fail
    jmp the_end                    # module 3, rel32 (never rel8)
fail:
    mov edi, 99
    mov eax, 60
    syscall
    .space 60
msg:
    .long helper                   # module 2, abs32 data
    .long the_end                  # module 3, abs32 data
    .quad seed                     # module 3, abs64 data
