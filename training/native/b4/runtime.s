# B4 runtime - raw Linux x86-64 syscall services, hand-written in the
# B2 assembler dialect (Intel syntax, no libc). Assembled by our
# assembler and linked by our B3 linker into the corpus programs; the
# same source must also assemble and link with GNU as + GNU ld byte for
# byte (b4_check.py compares the whole linked .text).
#
# Services (SysV arg registers, syscall numbers in eax):
#   write_str(rdi=ptr, rsi=len)      write(1, ptr, len)
#   exit(rdi=code)                   exit(code) - does not return
#   mmap_anon(rdi=len) -> rax        anonymous RW mapping
#   spawn_task(rdi=arg1, rsi=arg2, rdx=child_stack_top)
#       clone(CLONE_VM|SIGCHLD): a VM-sharing task (its own stack) that
#       runs thread_body(arg1, arg2) and then exits 0. Returns the child
#       pid to the parent (rax != 0), 0 inside the child path.
#       Every program must define thread_body (its task entry), even if
#       it spawns no tasks (a plain `ret` stub).
#   futex_wait(rdi=addr, rsi=expected)  FUTEX_WAIT until *addr != expected
#   futex_wake(rdi=addr)                FUTEX_WAKE one waiter
#   futex_wake_all(rdi=addr)            FUTEX_WAKE every waiter
#   now_ns() -> rax                   CLOCK_MONOTONIC in nanoseconds
#   sleep_ns(rdi=nsec)                nanosleep
#   exit_group(rdi=code)              exit_group(code) - ends every
#                                     thread of the process (B4 real
#                                     threads end this way; workers end
#                                     with sys_exit/60 instead)
#   heap_init(rdi=bytes) -> rax       anonymous-mmap allocator arena
#                                     (0 = failure); the control block
#                                     (next/end) lives IN the arena
#   heap_alloc(rdi=heap, rsi=size) -> rax
#       16-byte bump allocation over the arena, lock cmpxchg on the
#       next pointer (0 = arena full)
#   spawn_worker(rdi=pool, rsi=index, rdx=stack_top) -> rax
#       clone(0x50F00 = CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|
#       CLONE_THREAD|CLONE_SYSVSEM): a REAL thread of this process that
#       parks in pool_worker(pool, index) on its own mmap'd stack.
#       Parent gets the tid; the worker ends with sys_exit(60) and the
#       program's main thread must finish with exit_group(231) so the
#       process status stays the main thread's (D-039). Direct call to
#       pool_worker - the subset has no indirect call (D-036), so job
#       dispatch is the program-defined pool_job (like thread_body).
#   pool_init(rdi=heap, rsi=n) -> rax
#       pool block + n 64 KiB stacks from the heap, spawns n workers
#       (0 = failure). Block layout (heap memory, never label data):
#         +0  dword n        +4  dword gen    (workers futex on gen)
#         +8  qword job_arg  +16 dword active  +20 dword done
#         +64 qword stacks[n]
#   pool_run(rdi=pool, rsi=job_arg, rdx=active)
#       publish job_arg + active, gen++, wake every worker, then wait
#       (futex on done) until `active` workers finished pool_job
#   barrier(rdi=block, rsi=n)         sense barrier; block = +0 count,
#       +4 gen: all n parties arrive (lock xadd), the last one resets
#       count, bumps gen and wakes every sleeper
#   cpu_count() -> rax                sched_getaffinity CPU count
#       (syscall 204 = sched_getaffinity; 203 is sched_SETaffinity.
#       The syscall returns the number of mask bytes it filled in, so
#       the bit count runs over exactly those bytes - the rest of the
#       stack buffer is untouched and must not be counted)
#   cpu_features() -> rax             CPUID mask: 1 sse2, 2 ssse3,
#       4 sse4_1, 8 fma, 16 avx, 32 avx2
#
# Registers: only rax, rdi, rsi, rdx, r10, r8, r9, rcx, r11 are touched
# (rcx/r11 are syscall clobbers), so r12-r15 and rbx/rbp survive every
# call. spawn_task preserves r12/r13 in the parent path (the child
# consumes them, then exits without returning). The B4 services below
# preserve rbx and r12-r15 as well (they push what they use).
    .intel_syntax noprefix
    .text

    .global write_str
    .global exit
    .global mmap_anon
    .global spawn_task
    .global futex_wait
    .global futex_wake
    .global futex_wake_all
    .global now_ns
    .global sleep_ns
    .global exit_group
    .global heap_init
    .global heap_alloc
    .global spawn_worker
    .global pool_init
    .global pool_run
    .global barrier
    .global cpu_count
    .global cpu_features

write_str:                      # rdi=ptr, rsi=len
    mov rdx, rsi
    mov rsi, rdi
    mov edi, 1
    mov eax, 1                  # sys_write
    syscall
    ret

exit:                           # rdi=code, does not return
    mov eax, 60                 # sys_exit
    syscall

mmap_anon:                      # rdi=len -> rax=base
    mov rsi, rdi
    mov edi, 0                  # addr = NULL
    mov edx, 3                  # PROT_READ|PROT_WRITE
    mov r10, 0x22               # MAP_PRIVATE|MAP_ANONYMOUS
    mov r8, -1                  # fd
    xor r9d, r9d                # offset
    mov eax, 9                  # sys_mmap
    syscall
    ret

spawn_task:                     # rdi=arg1, rsi=arg2, rdx=child stack top
    push r12
    push r13
    mov r12, rdi
    mov r13, rsi
    mov rsi, rdx
    mov rdi, 0x111              # CLONE_VM | SIGCHLD
    mov rdx, 0                  # parent_tidptr
    mov r10, 0                  # child_tidptr
    mov r8, 0                   # newtls
    mov eax, 56                 # sys_clone
    syscall
    test rax, rax
    jz task_child
    pop r13
    pop r12
    ret
task_child:                     # fresh stack: rsi at clone time
    and rsp, -16
    mov rdi, r12
    mov rsi, r13
    call thread_body            # defined by the program (external here)
    mov edi, 0
    call exit

futex_wait:                     # rdi=addr, rsi=expected value
    mov rdx, rsi
    xor esi, esi                # FUTEX_WAIT
    mov r10, 0                  # no timeout
    mov r8, 0
    mov eax, 202                # sys_futex
    syscall
    ret

futex_wake:                     # rdi=addr
    mov esi, 1                  # FUTEX_WAKE
    mov edx, 1                  # one waiter
    mov r10, 0
    mov r8, 0
    mov eax, 202                # sys_futex
    syscall
    ret

now_ns:                         # -> rax = monotonic nanoseconds
    sub rsp, 16
    mov edi, 1                  # CLOCK_MONOTONIC
    mov rsi, rsp
    mov eax, 228                # sys_clock_gettime
    syscall
    mov rax, qword ptr [rsp]
    mov rcx, qword ptr [rsp+8]
    imul rax, rax, 1000000000
    add rax, rcx
    add rsp, 16
    ret

sleep_ns:                       # rdi=nanoseconds (< 1e9)
    sub rsp, 32
    mov qword ptr [rsp], 0      # tv_sec
    mov qword ptr [rsp+8], rdi  # tv_nsec
    mov rdi, rsp
    xor esi, esi                # no remainder
    mov eax, 35                 # sys_nanosleep
    syscall
    add rsp, 32
    ret

futex_wake_all:                 # rdi=addr, wake every waiter
    mov esi, 1                  # FUTEX_WAKE
    mov edx, 0x7fffffff
    xor r10d, r10d
    xor r8d, r8d
    mov eax, 202                # sys_futex
    syscall
    ret

exit_group:                     # rdi=code, ends every thread
    mov eax, 231                # sys_exit_group
    syscall

heap_init:                      # rdi=bytes -> rax=base (0 = failure)
    push rbx
    mov rbx, rdi                # arena size
    call mmap_anon
    test rax, rax
    js heap_fail                # mmap returned -errno
    mov rcx, rax                # control block INSIDE the arena:
    add rcx, rbx
    mov qword ptr [rax+8], rcx  #   +8 end = base + bytes
    lea rcx, [rax+32]
    mov qword ptr [rax], rcx    #   +0 next = base + 32 (16-aligned)
    pop rbx
    ret
heap_fail:
    xor eax, eax
    pop rbx
    ret

heap_alloc:                     # rdi=heap, rsi=size -> rax (0 = full)
    mov rdx, rsi                # size
    mov rcx, qword ptr [rdi+8]  # end (immutable)
    mov rax, qword ptr [rdi]    # candidate next
heap_try:
    mov r8, rax
    add r8, rdx
    cmp r8, rcx
    ja heap_full
    lock cmpxchg qword ptr [rdi], r8   # [heap]==rax -> [heap]=r8
    jne heap_try                # failure reloads rax with [heap]
    ret                         # rax = old next = allocation
heap_full:
    xor eax, eax
    ret

spawn_worker:                   # rdi=pool, rsi=index, rdx=stack top
    push r12
    push r13
    mov r12, rdi                # pool
    mov r13, rsi                # index
    mov rsi, rdx                # child stack
    mov rdi, 0x50f00            # CLONE_VM|FS|FILES|SIGHAND|THREAD|SYSVSEM
    xor edx, edx                # parent_tid
    xor r10d, r10d              # child_tid
    xor r8d, r8d                # tls
    mov eax, 56                 # sys_clone
    syscall
    test rax, rax
    jne worker_parent
worker_child:                   # a real thread on the fresh stack
    and rsp, -16
    mov rdi, r12
    mov rsi, r13
    call pool_worker            # defined in this file (external use)
    mov edi, 0
    mov eax, 60                 # sys_exit: end THIS thread only
    syscall
worker_parent:
    pop r13
    pop r12
    ret                         # rax = tid (or -errno)

pool_worker:                    # rdi=pool, rsi=index (thread entry)
    mov r12, rdi                # pool
    mov r13, rsi                # index
    xor r14d, r14d              # last generation consumed
worker_park:
    mov eax, dword ptr [r12+4]  # gen
    cmp eax, r14d
    je worker_sleep             # no new job yet
    mov r14d, eax               # consume this generation
    cmp r13d, dword ptr [r12+16] # index < active?
    jae worker_park             # not called this run - wait for the next
    mov rdi, qword ptr [r12+8]  # job_arg
    push r12                    # keep loop state across the job
    push r13
    push r14
    call pool_job               # program-defined (direct call, D-036)
    pop r14
    pop r13
    pop r12
    mov ecx, 1
    lock xadd dword ptr [r12+20], ecx   # done++ -> ecx = old
    lea rdi, [r12+20]
    call futex_wake             # wake the main thread waiting on done
    jmp worker_park
worker_sleep:
    lea rdi, [r12+4]            # &gen
    mov esi, r14d               # sleep while gen == last consumed
    call futex_wait
    jmp worker_park

pool_init:                      # rdi=heap, rsi=n -> rax (0 = failure)
    push r12
    push r13
    push r14
    push r15
    mov r12, rdi                # heap
    mov r13, rsi                # n
    mov rdi, r12
    mov rsi, r13
    shl rsi, 3
    add rsi, 64                 # block: 64B header + n stack slots
    call heap_alloc
    test rax, rax
    je pool_fail
    mov r14, rax                # pool block (fresh page: zero)
    mov dword ptr [r14], r13d    # n
    mov dword ptr [r14+4], 0     # gen = 0
    mov dword ptr [r14+16], 0    # active = 0
    mov dword ptr [r14+20], 0    # done = 0
    xor r15d, r15d
pool_stk_loop:
    cmp r15, r13
    jge pool_spawn
    mov rdi, r12
    mov esi, 0x10000            # 64 KiB per worker stack
    call heap_alloc
    test rax, rax
    je pool_fail
    mov qword ptr [r14+r15*8+64], rax
    inc r15
    jmp pool_stk_loop
pool_spawn:
    xor r15d, r15d
pool_spawn_loop:
    cmp r15, r13
    jge pool_ok
    mov rax, qword ptr [r14+r15*8+64]
    lea rdx, [rax+0x10000]      # stack top (heap_alloc is 16-aligned)
    mov rdi, r14
    mov rsi, r15
    call spawn_worker
    test rax, rax
    js pool_fail
    inc r15
    jmp pool_spawn_loop
pool_ok:
    mov rax, r14
    pop r15
    pop r14
    pop r13
    pop r12
    ret
pool_fail:
    xor eax, eax
    pop r15
    pop r14
    pop r13
    pop r12
    ret

pool_run:                       # rdi=pool, rsi=job_arg, rdx=active
    push rbx
    push r15
    mov rbx, rdi                # pool
    mov r15d, edx               # active
    mov dword ptr [rbx+20], 0   # done = 0 (before publishing gen)
    mov qword ptr [rbx+8], rsi  # job_arg
    mov dword ptr [rbx+16], r15d # active
    add dword ptr [rbx+4], 1    # gen++: the job exists now
    lea rdi, [rbx+4]
    call futex_wake_all         # every parked worker
pool_wait:
    mov eax, dword ptr [rbx+20] # done
    cmp eax, r15d
    je pool_run_ok
    mov esi, eax                # expected value for FUTEX_WAIT
    lea rdi, [rbx+20]
    call futex_wait
    jmp pool_wait
pool_run_ok:
    pop r15
    pop rbx
    ret

barrier:                        # rdi=block (+0 count, +4 gen), rsi=n
    push rdi                    # [rsp+16] = block
    push rsi                    # [rsp+8]  = n
    push r12                    # [rsp]    = my generation; r12 survives
                                # the futex helpers (they zero r8/r10)
    mov r12d, dword ptr [rdi+4] # my generation
    mov ecx, 1
    lock xadd dword ptr [rdi], ecx   # count++ -> ecx = old
    lea rax, [rsi-1]            # last party is n-1
    cmp ecx, eax
    je barrier_last
barrier_wait:
    mov rdi, qword ptr [rsp+16]
    cmp dword ptr [rdi+4], r12d # generation moved on?
    jne barrier_out
    lea rdi, [rdi+4]            # &gen
    mov esi, r12d               # sleep while gen == mine
    call futex_wait
    jmp barrier_wait
barrier_last:
    mov rdi, qword ptr [rsp+16]
    mov dword ptr [rdi], 0      # count = 0: the next round may start
    add dword ptr [rdi+4], 1    # gen++ BEFORE waking (no lost wakeup)
    lea rdi, [rdi+4]
    call futex_wake_all
barrier_out:
    pop r12
    pop rsi
    pop rdi
    ret

cpu_count:                      # -> rax = CPUs in the affinity mask
    sub rsp, 136                # 128-byte mask (CPU_SETSIZE)
    xor edi, edi                # pid 0: this thread
    mov esi, 128
    mov rdx, rsp
    mov eax, 204                # sys_sched_getaffinity
    syscall
    test rax, rax
    js cpu_count_fail
    mov r10, rax                # bytes of mask the kernel filled in
    xor eax, eax                # bit count
    mov r9, rsp                 # current dword
    mov r8, rsp
    add r8, r10                 # end of the filled mask
cpu_word_loop:
    cmp r9, r8
    je cpu_count_done
    mov edx, dword ptr [r9]
    mov ecx, 32
cpu_bit_loop:
    shr edx, 1
    jnc cpu_bit_no
    inc eax
cpu_bit_no:
    dec ecx
    jne cpu_bit_loop
    add r9, 4
    jmp cpu_word_loop
cpu_count_done:
    add rsp, 136
    ret
cpu_count_fail:
    add rsp, 136
    xor eax, eax
    ret

cpu_features:                   # -> rax CPUID mask (FEAT_* in the program)
    push rbx
    xor ecx, ecx
    mov eax, 1
    cpuid                        # ecx = leaf1 features, edx = leaf1 edx
    xor r10d, r10d              # result mask
    test edx, 0x04000000        # sse2: edx bit 26
    je feat_no_sse2
    or r10d, 1
feat_no_sse2:
    test ecx, 0x200             # ssse3: ecx bit 9
    je feat_no_ssse3
    or r10d, 2
feat_no_ssse3:
    test ecx, 0x80000           # sse4_1: ecx bit 19
    je feat_no_sse41
    or r10d, 4
feat_no_sse41:
    test ecx, 0x1000            # fma: ecx bit 12
    je feat_no_fma
    or r10d, 8
feat_no_fma:
    test ecx, 0x10000000        # avx: ecx bit 28
    je feat_no_avx
    or r10d, 16
feat_no_avx:
    xor ecx, ecx
    mov eax, 7
    cpuid                        # ebx = leaf7 ebx
    test ebx, 0x20              # avx2: ebx bit 5
    je feat_no_avx2
    or r10d, 32
feat_no_avx2:
    mov eax, r10d
    pop rbx
    ret
