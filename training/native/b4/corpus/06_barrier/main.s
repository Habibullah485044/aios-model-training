# b4 corpus 06 - barrier: N workers (N = CPU count from
# sched_getaffinity) run 10,000 sense-barrier rounds through the
# runtime's barrier() - lock xadd arrivals, futex sleep/wake on the
# block's generation word, the last arrival resets the count, bumps the
# generation and wakes every sleeper. The main thread runs the job on
# all N workers once (pool_run) and verifies the block afterwards.
# Pass target (fixed before running): stdout "B4 barrier ok\n" (14
# bytes) and exit status 46; 98 if the block did not land on
# gen=10000/count=0 (a deadlock makes the check time out instead),
# 96 on setup failure.
    .intel_syntax noprefix
    .text
    .global _start
    .global thread_body
    .global pool_job
_start:
    call cpu_count               # -> rax = N (0 = failure)
    test rax, rax
    je fail_setup
    mov r12, rax                 # N
    mov rdi, 0x400000            # 4 MiB heap
    call heap_init
    test rax, rax
    je fail_setup
    mov r13, rax                 # heap
    mov rdi, r13
    mov rsi, r12
    call pool_init               # n workers
    test rax, rax
    je fail_setup
    mov r14, rax                 # pool
    mov rdi, r13
    mov rsi, 16
    call heap_alloc              # barrier block: +0 count, +4 gen, +8 n
    test rax, rax
    je fail_setup
    mov rbx, rax
    mov dword ptr [rbx+8], r12d  # n for pool_job
    mov rdi, r14
    mov rsi, rbx                 # job_arg = barrier block
    mov rdx, r12                 # active = N: every worker arrives
    call pool_run
    cmp dword ptr [rbx+4], 10000 # 10,000 rounds completed
    jne barrier_bad
    cmp dword ptr [rbx], 0       # last round reset the count
    jne barrier_bad
    lea rdi, [rip+msg_ok]
    mov esi, 14
    call write_str
    mov edi, 46
    call exit_group
barrier_bad:
    mov edi, 98
    call exit_group
fail_setup:
    mov edi, 96
    call exit_group

pool_job:                       # rdi = barrier block (worker entry)
    mov r13d, 10000
barrier_round:
    mov esi, dword ptr [rdi+8]   # n
    call barrier                 # barrier preserves rdi (block)
    dec r13d
    jne barrier_round
    ret

thread_body:                    # task entry; this program spawns none
    ret

    .align 4096                 # data page: RW- (W^X split)
msg_ok:
    .byte 66, 52, 32, 98, 97, 114, 114, 105, 101, 114, 32, 111, 107, 10  # B4 barrier ok\n
