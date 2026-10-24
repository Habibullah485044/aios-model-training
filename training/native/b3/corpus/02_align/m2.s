# b3 corpus 02 module 2 - .align 32 raises this module's section
# alignment (gas sh_addralign = largest .align), so the linker must start
# it on a 32-byte boundary and fill the gap with ld's nop fill. Exports
# the global `shadow` (33) that module 3 must see but module 1 must not.
    .intel_syntax noprefix
    .text
    .global fn2
    .global shadow
    .align 32
fn2:
    call fn3                       # module 3
    ret
shadow:
    .long 33
