# b2 corpus 05 - SSE arithmetic, logic, min/max, conversions and compares
# (the per-op math the engine needs: mul/add/sub for FP32 kernels, conversions
# for BF16 widening, compares for max/softmax). Exit status 5.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    sub rsp, 256
    mov rbx, rsp
    xor eax, eax

    # packed/scalar arithmetic, reg,reg and mem forms
    addps xmm0, xmm1
    addpd xmm2, xmm3
    addss xmm4, xmm5
    addsd xmm6, xmm7
    subps xmm8, xmm9
    subpd xmm10, xmm11
    subss xmm12, xmm13
    subsd xmm14, xmm15
    mulps xmm0, xmm2
    mulpd xmm1, xmm3
    mulss xmm4, xmm6
    mulsd xmm5, xmm7
    divps xmm8, xmm10
    divss xmm9, xmm11
    divsd xmm12, xmm14
    addps xmm0, xmmword ptr [rsp]
    mulpd xmm1, xmmword ptr [rsp+16]
    mulsd xmm2, qword ptr [rsp]

    # min / max
    minps xmm0, xmm1
    maxps xmm2, xmm3
    minss xmm4, xmm5
    maxss xmm6, xmm7

    # logical
    andps xmm8, xmm9
    andpd xmm10, xmm11
    andnps xmm12, xmm13
    andnpd xmm14, xmm15
    orps xmm0, xmm1
    orpd xmm2, xmm3
    xorps xmm4, xmm5
    xorpd xmm6, xmm7

    # integer <-> scalar conversions
    cvtsi2ss xmm0, eax
    cvtsi2sd xmm1, rax
    cvtsi2sd xmm2, dword ptr [rsp]
    cvtss2si eax, xmm3
    cvtss2si rax, xmm4
    cvttsd2si ecx, xmm5
    cvttsd2si rdx, xmm6
    cvttss2si r8d, xmm7
    cvtss2sd xmm8, xmm9
    cvtsd2ss xmm10, xmm11
    cvtdq2ps xmm12, xmm13
    cvtps2dq xmm14, xmm15
    cvttps2dq xmm0, xmm1

    # compares and mask extraction
    ucomiss xmm2, xmm3
    ucomisd xmm4, xmm5
    comiss xmm6, xmm7
    comisd xmm8, xmm9
    movmskps eax, xmm10
    movmskpd edx, xmm11

    mov edi, 5
    mov eax, 60
    syscall
