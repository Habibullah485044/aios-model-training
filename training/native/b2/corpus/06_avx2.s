# 06_avx2.s - AVX2 + FMA coverage (B2). The Part C modern fast path reads
# BF16 through these forms, so the assembler must emit them byte identical
# to GNU as. The instructions are data-only: computed values do not affect
# the exit marker. The file just has to assemble identically and execute
# fault-free on AVX2 hardware, exiting 6. No .align: section padding would
# not be byte-comparable (D-019).
    .intel_syntax noprefix
    .text
    .global _start
_start:
    sub rsp, 544                  # 16-aligned scratch area (544 = 34*16)

    # ---- seed scratch through GP stores (imm64 needs movabs via a
    # register: a memory store only takes an imm32, as GNU as enforces) ----
    mov rax, 0x3f8000003f800000
    mov qword ptr [rsp], rax
    mov rax, 0x4000000040400000
    mov qword ptr [rsp+8], rax
    mov rax, 0x40a0000040800000
    mov qword ptr [rsp+16], rax
    mov rax, 0x40c0000040e00000
    mov qword ptr [rsp+24], rax
    mov dword ptr [rsp+32], 0x3f000000
    mov byte ptr [rsp+36], 0x56
    mov byte ptr [rsp+37], 0x78
    mov r8, 1
    mov dword ptr [rsp+r8*4+40], 0x40400000

    # ---- moves: load/store, both widths, high registers, addressing ----
    vmovups ymm0, ymmword ptr [rsp]
    vmovups ymm15, ymmword ptr [rsp+16]
    vmovups ymmword ptr [rsp+64], ymm0
    vmovups ymmword ptr [rsp+96], ymm15
    vmovups xmm2, xmm3
    vmovups [rsp+r8*8+128], ymm0
    vmovdqu xmm1, xmmword ptr [rsp]
    vmovdqu xmm14, xmmword ptr [rsp+8]
    vmovdqu xmmword ptr [rsp+160], xmm1
    vmovdqu xmmword ptr [rsp+176], xmm14
    vbroadcastss ymm4, dword ptr [rsp]
    vbroadcastss xmm5, dword ptr [rsp+4]
    vbroadcastss ymm6, xmm1
    vbroadcastss xmm7, dword ptr [rip+fp_one]

    # ---- integer path (the BF16 unpack: shuffle/zero-extend/logic) ----
    vpshufb ymm8, ymm0, ymm1
    vpshufb xmm9, xmm14, xmm15
    vpmovzxbd ymm10, xmm1
    vpmovzxbd xmm11, dword ptr [rsp]
    vpmovzxwd ymm12, xmm14
    vpmovzxwd xmm13, qword ptr [rsp+8]
    vpand ymm2, ymm0, ymm1
    vpand xmm3, xmm14, [rsp]
    vpor ymm8, ymm2, ymm4
    vpor xmm9, xmm3, xmm5
    vpxor ymm10, ymm0, ymm0
    vpxor xmm11, xmm1, [rsp+16]
    vpslld ymm12, ymm0, 3
    vpslld xmm13, xmm1, 7
    vpsrlw ymm2, ymm1, 4
    vpsrlw xmm3, xmm14, 8
    vpcmpeqb ymm4, ymm0, ymm15
    vpcmpeqb xmm5, xmm1, xmm14
    vpmovmskb eax, ymm4
    vpmovmskb r9d, xmm5

    # ---- float path (packed add/mul + FMA, the matvec core) ----
    vaddps ymm6, ymm0, ymm15
    vaddps xmm7, xmm1, xmm14
    vmulps ymm8, ymm6, ymm0
    vmulps xmm9, xmm7, xmm1
    vfmadd231ps ymm10, ymm8, ymm6
    vfmadd231ps xmm11, xmm9, xmm7
    vfmadd231ps ymm12, ymm10, ymmword ptr [rsp]
    vextractf128 xmm13, ymm6, 1
    vextractf128 xmmword ptr [rsp+192], ymm10, 0
    vperm2f128 ymm14, ymm6, ymm10, 0x31
    vperm2f128 ymm15, ymm8, ymmword ptr [rsp+16], 1
    vzeroupper

    add rsp, 544
    mov edi, 6
    mov eax, 60
    syscall

# rip-relative broadcast source (read-only float constants)
fp_one:
    .long 0x3f800000
    .long 0x40000000
