# b3 program 4 - read_weights <weights.bin> <out.bin>:
#   open the weights file, lseek(SEEK_END) for its size, mmap the whole
#   file read-only, then copy the first 64 bytes at each of three
#   offsets to out.bin, in ascending offset order:
#     embed_tokens      (0)
#     layers.0.q_proj   (622,333,952)
#     model.norm        (6,171,873,280)   <- above 4 GB
# out.bin is exactly 192 bytes: the three 64-byte blocks concatenated.
# Pass targets (fixed before running):
#   - out.bin matches the input file at those offsets byte for byte;
#   - a wrong path gives exit status 2 (ENOENT) with no crash;
#   - wrong argc also exits 2 (bad arguments, D1 convention).
# On any open failure the exit status is the positive errno (-rax).
# The three offsets live in the data segment (`offs`), page aligned so
# the check maps it RW- separate from the R-X code.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    mov rax, qword ptr [rsp]        # argc
    cmp rax, 3
    jne bad_args
    mov r14, qword ptr [rsp+16]     # argv[1] = weights.bin
    mov r13, qword ptr [rsp+24]     # argv[2] = out.bin

    mov rdi, r14                    # open(argv[1], O_RDONLY)
    xor esi, esi
    mov eax, 2
    syscall
    test rax, rax
    js fail
    mov r12, rax                    # input fd

    mov rdi, r12                    # lseek(fd, 0, SEEK_END) -> size
    xor esi, esi
    mov edx, 2
    mov eax, 8
    syscall
    test rax, rax
    js fail
    mov r15, rax                    # file size

    xor edi, edi                    # mmap(0, size, PROT_READ, MAP_PRIVATE, fd, 0)
    mov rsi, r15
    mov edx, 1
    mov r10d, 2
    mov r8, r12
    xor r9d, r9d
    mov eax, 9
    syscall
    test rax, rax
    js fail
    mov r15, rax                    # mapped address (size no longer needed)

    mov rdi, r13                    # open(argv[2], O_CREAT|O_WRONLY|O_TRUNC, 0644)
    mov esi, 577
    mov edx, 420
    mov eax, 2
    syscall
    test rax, rax
    js fail
    mov r13, rax                    # output fd

    xor r12d, r12d                  # block index
write_loop:
    cmp r12, 3
    je done
    lea rdi, [rip+offs]
    mov rsi, qword ptr [rdi+r12*8]  # offset of this block
    add rsi, r15                    # buf = map + offset (write wants it in rsi)
    mov rdi, r13                    # out fd
    mov edx, 64                     # count (write wants it in rdx)
    mov eax, 1                      # sys_write
    syscall
    test rax, rax
    js fail
    inc r12
    jmp write_loop

done:
    xor edi, edi
    mov eax, 60                     # sys_exit(0)
    syscall

bad_args:
    mov edi, 2
    mov eax, 60
    syscall

fail:
    neg rax                         # -errno -> positive errno
    mov edi, eax
    mov eax, 60
    syscall
    .align 4096                    # data starts on its own page (W^X split)
offs:
    .quad 0
    .quad 622333952
    .quad 6171873280
