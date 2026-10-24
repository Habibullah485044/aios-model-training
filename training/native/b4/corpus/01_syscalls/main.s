# b4 corpus 01 - syscalls: raw write(1, ...) and exit(code) through the
# runtime module. Pass target (fixed before running): stdout is exactly
# "B4 syscalls ok\n" (15 bytes) and the exit status is 41.
    .intel_syntax noprefix
    .text
    .global _start
    .global thread_body
    .global pool_job
_start:
    lea rdi, [rip+msg]
    mov esi, 15
    call write_str
    mov edi, 41
    call exit

thread_body:                    # task entry; this program spawns none
    ret

pool_job:                       # pool entry; this program runs none
    ret

    .align 4096                 # data page: RW- (W^X split)
msg:
    .byte 66, 52, 32, 115, 121, 115, 99, 97, 108, 108, 115, 32, 111, 107, 10
