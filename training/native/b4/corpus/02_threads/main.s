# b4 corpus 02 - threads: three VM-sharing tasks (clone CLONE_VM|SIGCHLD,
# own stacks from mmap) each write their slot (i+1)*11 and set a futex
# flag; the parent joins every flag with FUTEX_WAIT and sums the slots.
# Pass target (fixed before running): stdout is exactly "B4 threads ok\n"
# (14 bytes) and the exit status is 42 (98 if the sum is not 66).
    .intel_syntax noprefix
    .text
    .global _start
    .global thread_body
    .global pool_job
_start:
    mov edi, 0x9000             # arena: slots+flags at 0..47, 3 stacks above
    call mmap_anon
    mov r12, rax
    xor r13d, r13d
spawn_loop:
    cmp r13d, 3
    jge join_phase
    mov rax, r13                # stack top = base + 0x2000*(i+2)
    add rax, 2
    shl rax, 13
    add rax, r12
    mov rdi, r12
    mov rsi, r13
    mov rdx, rax
    call spawn_task
    inc r13d
    jmp spawn_loop

join_phase:
    xor r13d, r13d
join_loop:
    cmp r13d, 3
    jge sum_phase
    lea r14, [r12+r13*8+24]     # &flag[i]
wait_flag:
    cmp dword ptr [r14], 1
    je flag_ok
    mov rdi, r14
    xor esi, esi                # expected: flag still 0
    call futex_wait
    jmp wait_flag
flag_ok:
    inc r13d
    jmp join_loop

sum_phase:
    mov rax, qword ptr [r12]
    add rax, qword ptr [r12+8]
    add rax, qword ptr [r12+16]
    cmp rax, 66
    jne bad
    lea rdi, [rip+msg]
    mov esi, 14
    call write_str
    mov edi, 42
    call exit
bad:
    mov edi, 98
    call exit

thread_body:                    # rdi=arena base, rsi=index
    mov rax, rsi
    inc rax
    imul rax, rax, 11
    mov qword ptr [rdi+rsi*8], rax
    mov qword ptr [rdi+rsi*8+24], 1
    lea rdi, [rdi+rsi*8+24]
    call futex_wake
    ret

pool_job:                       # pool entry; this program runs none
    ret

    .align 4096                 # data page: RW- (W^X split)
msg:
    .byte 66, 52, 32, 116, 104, 114, 101, 97, 100, 115, 32, 111, 107, 10
