# b4 corpus 04 - features: CPUID (leaf 1 + leaf 7) through the runtime's
# cpu_features(), printing the feature names this CPU has, in a fixed
# order, space separated. The check compares the printed set against
# /proc/cpuinfo's flags - the two must match exactly (at least sse2,
# ssse3, sse4_1, avx, avx2, fma on hardware that has them).
# Pass target (fixed before running): stdout == the /proc/cpuinfo flags
# restricted to those six names in the printed order, space separated,
# with one "\n" at the end; exit status 44.
    .intel_syntax noprefix
    .text
    .global _start
    .global thread_body
    .global pool_job
_start:
    call cpu_features            # -> eax: 1 sse2, 2 ssse3, 4 sse4_1, 8 fma, 16 avx, 32 avx2
    mov r12d, eax
    xor r13d, r13d               # 0 = nothing printed yet
    test r12d, 1
    je f_sse2
    lea rdi, [rip+name_sse2]
    mov esi, 4
    call wflag
f_sse2:
    test r12d, 2
    je f_ssse3
    lea rdi, [rip+name_ssse3]
    mov esi, 5
    call wflag
f_ssse3:
    test r12d, 4
    je f_sse41
    lea rdi, [rip+name_sse41]
    mov esi, 6
    call wflag
f_sse41:
    test r12d, 16
    je f_avx
    lea rdi, [rip+name_avx]
    mov esi, 3
    call wflag
f_avx:
    test r12d, 32
    je f_avx2
    lea rdi, [rip+name_avx2]
    mov esi, 4
    call wflag
f_avx2:
    test r12d, 8
    je f_fma
    lea rdi, [rip+name_fma]
    mov esi, 3
    call wflag
f_fma:
    lea rdi, [rip+name_nl]
    mov esi, 1
    call write_str
    mov edi, 44
    call exit

wflag:                            # rdi=name, rsi=len; r13 = "printed" flag
    cmp r13d, 0
    je wflag_first
    push rdi
    push rsi
    lea rdi, [rip+name_sp]
    mov esi, 1
    call write_str
    pop rsi
    pop rdi
wflag_first:
    mov r13d, 1
    call write_str
    ret

thread_body:                    # task entry; this program spawns none
    ret

pool_job:                       # pool entry; this program runs none
    ret

    .align 4096                 # data page: RW- (W^X split)
name_sse2:
    .byte 115, 115, 101, 50                # sse2
name_ssse3:
    .byte 115, 115, 115, 101, 51           # ssse3
name_sse41:
    .byte 115, 115, 101, 52, 95, 49        # sse4_1
name_avx:
    .byte 97, 118, 120                     # avx
name_avx2:
    .byte 97, 118, 120, 50                 # avx2
name_fma:
    .byte 102, 109, 97                     # fma
name_sp:
    .byte 32
name_nl:
    .byte 10
