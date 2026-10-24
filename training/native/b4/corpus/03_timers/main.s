# b4 corpus 03 - timers: CLOCK_MONOTONIC around a 50 ms nanosleep.
# Pass target (fixed before running): stdout is exactly "B4 timers ok\n"
# (13 bytes) and the exit status is 43, provided the measured delta is at
# least 40 ms (fixed floor, generous vs 50 ms sleep); otherwise 99.
    .intel_syntax noprefix
    .text
    .global _start
    .global thread_body
    .global pool_job
_start:
    call now_ns
    mov r12, rax
    mov edi, 50000000           # 50 ms
    call sleep_ns
    call now_ns
    sub rax, r12
    cmp rax, 40000000           # >= 40 ms floor
    jb bad
    lea rdi, [rip+msg]
    mov esi, 13
    call write_str
    mov edi, 43
    call exit
bad:
    mov edi, 99
    call exit

thread_body:                    # task entry; this program spawns none
    ret

pool_job:                       # pool entry; this program runs none
    ret

    .align 4096                 # data page: RW- (W^X split)
msg:
    .byte 66, 52, 32, 116, 105, 109, 101, 114, 115, 32, 111, 107, 10
