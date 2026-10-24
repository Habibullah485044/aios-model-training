# b3 program 2 - hello: prints a fixed text stored in the data segment,
# output byte-exact. The text is the 14 bytes "Hello, world!\n" spelled
# out as .byte values (the dialect has no .string directive). The data
# segment starts at `msg`, page aligned so the check can split the
# emitted ELF into an R-X code segment and an RW- data segment that
# share no page.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    mov edi, 1                      # stdout
    lea rsi, [rip+msg]              # the data segment
    mov edx, 14                     # length
    mov eax, 1                      # sys_write
    syscall
    xor edi, edi
    mov eax, 60                     # sys_exit(0)
    syscall
    .align 4096                    # data starts on its own page (W^X split)
msg:
    .byte 72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 10
