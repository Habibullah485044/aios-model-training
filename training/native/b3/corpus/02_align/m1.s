# b3 corpus 02 module 1 - label shadowing and module alignment. This
# module has NO .align (alignment 1), defines its own LOCAL `shadow`,
# and still reads it although module 2 exports a global `shadow` - the
# module's own label must win, like ELF's. Exit status 2.
    .intel_syntax noprefix
    .text
    .global _start
_start:
    call fn2                       # module 2
    cmp eax, 33
    jne fail
    mov eax, dword ptr [rip+shadow]   # THIS module's local shadow = 11
    cmp eax, 11
    jne fail
    mov edi, 2
    mov eax, 60
    syscall
fail:
    mov edi, 99
    mov eax, 60
    syscall
shadow:
    .long 11
