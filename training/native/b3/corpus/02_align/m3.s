# b3 corpus 02 module 3 - alignment 8 and the global binding test: the
# bare `shadow` here is undefined in this module, so it binds to module
# 2's exported 33, not to module 1's local 11.
    .intel_syntax noprefix
    .text
    .global fn3
    .align 8
fn3:
    mov eax, dword ptr [rip+shadow]   # module 2's global shadow = 33
    ret
