# b2 corpus 04 - SSE move and pack/unpack/shuffle subset (the forms the engine
# uses to decode packed weights: movdqu/movq/pshufb/punpck*). Aligned moves
# use 16-byte-aligned stack slots so the file is also executable (exit
# status 4).
    .intel_syntax noprefix
    .text
    .global _start
_start:
    sub rsp, 256
    mov rbx, rsp
    xor eax, eax

    # packed moves, reg,reg / reg,mem / mem,reg
    movaps xmm0, xmm1
    movaps xmm2, xmmword ptr [rsp]
    movaps xmmword ptr [rsp+16], xmm3
    movapd xmm4, xmm5
    movups xmm6, xmmword ptr [rsp+32]
    movups xmmword ptr [rsp+48], xmm7
    movupd xmm8, xmmword ptr [rsp+64]
    movdqa xmm9, xmm10
    movdqa xmmword ptr [rsp+80], xmm11
    movdqu xmm12, xmmword ptr [rsp+96]
    movdqu xmmword ptr [rsp+112], xmm13

    # scalar moves
    movss xmm14, dword ptr [rsp]
    movss dword ptr [rsp+128], xmm15
    movsd xmm0, qword ptr [rsp]
    movsd qword ptr [rsp+136], xmm1

    # GP <-> xmm
    movd xmm2, eax
    movd eax, xmm3
    movd xmm4, dword ptr [rsp]
    movq xmm5, rbx
    movq rbx, xmm6
    movq xmm7, xmm8
    movq xmm9, qword ptr [rsp]

    # SSSE3/SSE2 byte shuffles and unpack (split-nibble decode path)
    pshufb xmm0, xmm1
    pshufb xmm2, xmmword ptr [rsp+16]
    punpcklbw xmm3, xmm4
    punpcklwd xmm5, xmm6
    punpckldq xmm7, xmm8
    punpcklqdq xmm9, xmm10

    # bitwise and integer add/sub
    pand xmm11, xmm12
    pandn xmm13, xmm14
    por xmm15, xmm0
    pxor xmm1, xmm1
    paddb xmm2, xmm3
    paddw xmm4, xmm5
    paddd xmm6, xmm7
    paddq xmm8, xmm9
    psubb xmm10, xmm11
    psubw xmm12, xmm13
    psubd xmm14, xmm15
    psubq xmm0, xmm1

    # shifts: imm8 form, xmm form, byte-shift forms
    pslld xmm2, 4
    psrld xmm3, 4
    psllq xmm4, 8
    psrlq xmm5, 8
    pslldq xmm6, 2
    psrldq xmm7, 2
    pslld xmm8, xmm9
    psrld xmm10, xmm11

    mov edi, 4
    mov eax, 60
    syscall
