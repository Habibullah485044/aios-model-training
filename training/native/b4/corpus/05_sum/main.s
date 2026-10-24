# b4 corpus 05 - sum: a parallel sum of 1..100,000,000
# (5,000,000,050,000,000) over the thread pool, for EVERY thread count T
# from 1 to N (N = CPU count from sched_getaffinity). Workers pull
# 1,000,000-wide slices from a shared cursor with lock xadd and add each
# slice sum to a shared total, also with lock xadd; the main thread
# zeroes the shared block, runs pool_run(T) and verifies the total after
# every T. A fresh pull covers [old, old+slice) clamped to 100,000,001,
# so the union is always 0+1+..+100,000,000 exactly.
# Pass target (fixed before running): stdout "B4 sum ok\n" (10 bytes)
# and exit status 45; 98 if any total is wrong, 96 on setup failure.
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
    mov rdi, 0x400000            # 4 MiB heap: pool block + n*64K stacks + sum block
    call heap_init
    test rax, rax
    je fail_setup
    mov r13, rax                 # heap
    mov rdi, r13
    mov rsi, 16
    call heap_alloc              # sum block: +0 cursor, +8 total
    test rax, rax
    je fail_setup
    mov rbx, rax
    mov rdi, r13
    mov rsi, r12
    call pool_init               # n workers on their own stacks
    test rax, rax
    je fail_setup
    mov r14, rax                 # pool
    mov r15d, 1                  # T = 1..N
t_loop:
    cmp r15, r12
    jg sum_ok
    mov qword ptr [rbx], 0       # cursor = 0 (all workers are parked)
    mov qword ptr [rbx+8], 0     # total = 0
    mov rdi, r14
    mov rsi, rbx                 # job_arg = sum block
    mov rdx, r15                 # active = T
    call pool_run
    mov rax, qword ptr [rbx+8]
    movabs rcx, 5000000050000000
    cmp rax, rcx
    jne sum_bad
    inc r15d
    jmp t_loop
sum_ok:
    lea rdi, [rip+msg_ok]
    mov esi, 10
    call write_str
    mov edi, 45
    call exit_group              # every worker thread must end too
sum_bad:
    mov edi, 98
    call exit_group
fail_setup:
    mov edi, 96
    call exit_group

pool_job:                       # rdi = sum block (worker entry)
pull_loop:
    mov ecx, 1000000             # slice width
    lock xadd qword ptr [rdi], rcx   # rcx = old cursor; cursor += 1000000
    cmp rcx, 100000001           # old >= limit: the work is done
    jae pull_done
    lea rdx, [rcx+1000000]       # hi = old + slice
    cmp rdx, 100000001
    jbe slice_hi_ok
    mov rdx, 100000001           # clamp at the limit
slice_hi_ok:
    xor eax, eax
slice_sum:
    cmp rcx, rdx
    jae slice_summed
    add rax, rcx
    inc rcx
    jmp slice_sum
slice_summed:
    lock xadd qword ptr [rdi+8], rax   # total += slice sum
    jmp pull_loop
pull_done:
    ret

thread_body:                    # task entry; this program spawns none
    ret

    .align 4096                 # data page: RW- (W^X split)
msg_ok:
    .byte 66, 52, 32, 115, 117, 109, 32, 111, 107, 10   # B4 sum ok\n
