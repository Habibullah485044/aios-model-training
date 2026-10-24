/* unpack_ref.c - Phase 2 scalar reference unpacker + verifier.
 *
 * Plain C99, single-threaded, written to be obviously correct. This is the
 * answer key for Phase 3: the fast SIMD kernel must reproduce its output
 * bit-for-bit.
 *
 * Format (per tensor), all little-endian:
 *   table[15]        exponent for codes 0..14 (code 15 = escape)
 *   stream A[n]      a = (sign << 7) | mantissa7, one byte per weight
 *   stream B[n/2]    4-bit exponent codes, planar: within each 64-weight
 *                    group of a row, byte i low nibble = code of weight i,
 *                    byte i high nibble = code of weight i+32
 *   uint32 n_exc     exception count
 *   exceptions[n_exc]  (uint32 flat index, uint8 true exponent), sorted
 *
 * Reconstructed BF16 bit pattern: (sign << 15) | (exponent << 7) | mantissa7.
 *
 * Usage: unpack_ref <packed_dir> <orig_weights.bin> [unpacked_out.bin]
 *   packed_dir holds packed.bin + packed_index.txt.
 *   Verifies every unpacked weight against the original BF16 bits. Exits 0
 *   only if all match. If unpacked_out.bin is given, also writes the
 *   concatenated unpacked BF16 tensors there.
 *
 * Compile: gcc -std=c99 -Wall -Wextra -Werror -O2 unpack_ref.c -o unpack_ref
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NAME_MAX 128
#define GROUP 64

typedef struct {
    char name[NAME_MAX];
    uint64_t n;
    uint64_t n_rows;
    uint64_t row_len;
    uint64_t tbl_off;
    uint64_t sa_off;
    uint64_t sb_off;
    uint64_t exc_off;
    uint64_t n_exc;
    uint64_t orig_off;
} Meta;

static void die(const char *msg)
{
    fprintf(stderr, "unpack_ref: fatal: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p)
        die("out of memory");
    return p;
}

static void read_at(FILE *f, uint64_t off, void *dst, size_t nbytes)
{
    if (fseek(f, (long)off, SEEK_SET) != 0)
        die("fseek failed");
    if (fread(dst, 1, nbytes, f) != nbytes)
        die("short read");
}

/* Unpack one tensor into out[] as BF16 bit patterns. row_len is the number of
 * weights per row; n_rows is implied (n / row_len) and not needed here. */
static void unpack_tensor(const uint8_t *table, const uint8_t *sa, const uint8_t *sb,
                          const uint8_t *exc, uint64_t n_exc, uint64_t n,
                          uint64_t row_len, uint16_t *out)
{
    uint64_t group_bytes = row_len / GROUP * (GROUP / 2); /* bytes per row in stream B */
    uint64_t exc_cur = 0;
    uint64_t j;

    for (j = 0; j < n; j++) {
        uint8_t a = sa[j];
        uint8_t sign = a >> 7;
        uint8_t mant = a & 0x7F;
        uint64_t row = j / row_len;
        uint64_t col = j - row * row_len;
        uint64_t grp = col / GROUP;
        uint64_t pos = col - grp * GROUP;
        const uint8_t *chunk = sb + row * group_bytes + grp * (GROUP / 2);
        uint8_t code, e;

        if (pos < 32)
            code = chunk[pos] & 0x0F;
        else
            code = chunk[pos - 32] >> 4;

        if (code == 15) {
            uint64_t eidx;
            if (exc_cur >= n_exc)
                die("exception underflow");
            eidx = (uint64_t)exc[exc_cur * 5] | ((uint64_t)exc[exc_cur * 5 + 1] << 8) |
                   ((uint64_t)exc[exc_cur * 5 + 2] << 16) |
                   ((uint64_t)exc[exc_cur * 5 + 3] << 24);
            if (eidx != j)
                die("exception index out of order");
            e = exc[exc_cur * 5 + 4];
            exc_cur++;
        } else {
            e = table[code];
        }
        out[j] = (uint16_t)(((uint16_t)sign << 15) | ((uint16_t)e << 7) | mant);
    }
    if (exc_cur != n_exc)
        die("exception count mismatch (unused exceptions)");
}

int main(int argc, char **argv)
{
    FILE *pf, *mf, *of = NULL, *outf = NULL;
    char path[4096];
    Meta *meta;
    int n_tensors = 0, cap = 16, i;
    uint64_t total_n = 0, total_bad = 0;

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <packed_dir> <orig_weights.bin> [unpacked_out.bin]\n", argv[0]);
        return 2;
    }

    snprintf(path, sizeof path, "%s/packed.bin", argv[1]);
    pf = fopen(path, "rb");
    if (!pf)
        die("cannot open packed.bin");
    snprintf(path, sizeof path, "%s/packed_index.txt", argv[1]);
    mf = fopen(path, "r");
    if (!mf)
        die("cannot open packed_index.txt");
    of = fopen(argv[2], "rb");
    if (!of)
        die("cannot open original weights.bin");
    if (argc == 4) {
        outf = fopen(argv[3], "wb");
        if (!outf)
            die("cannot open unpacked output");
    }

    meta = malloc((size_t)cap * sizeof *meta);
    if (!meta)
        die("out of memory");
    for (;;) {
        char line[512];
        unsigned long long v[9];
        if (n_tensors >= cap) {
            cap *= 2;
            meta = realloc(meta, (size_t)cap * sizeof *meta);
            if (!meta)
                die("out of memory");
        }
        if (fgets(line, sizeof line, mf) == NULL)
            break;
        if (line[0] == '\n' || line[0] == '\0')
            continue;
        if (sscanf(line, "%127s %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                   meta[n_tensors].name, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6],
                   &v[7], &v[8]) != 10)
            die("bad index line");
        meta[n_tensors].n = v[0];
        meta[n_tensors].n_rows = v[1];
        meta[n_tensors].row_len = v[2];
        meta[n_tensors].tbl_off = v[3];
        meta[n_tensors].sa_off = v[4];
        meta[n_tensors].sb_off = v[5];
        meta[n_tensors].exc_off = v[6];
        meta[n_tensors].n_exc = v[7];
        meta[n_tensors].orig_off = v[8];
        n_tensors++;
    }
    fclose(mf);
    fprintf(stderr, "unpack_ref: %d tensors in index\n", n_tensors);

    for (i = 0; i < n_tensors; i++) {
        Meta *m = &meta[i];
        uint8_t table[15];
        uint8_t *sa, *sb, *exc = NULL;
        uint16_t *got;
        const uint16_t *want;
        uint64_t j, n_exc_stored = 0;
        uint64_t sb_bytes;

        read_at(pf, m->tbl_off, table, 15);

        sa = xmalloc((size_t)m->n);
        read_at(pf, m->sa_off, sa, (size_t)m->n);

        sb_bytes = m->n / 2;
        sb = xmalloc((size_t)sb_bytes);
        read_at(pf, m->sb_off, sb, (size_t)sb_bytes);

        /* exception count field lives at exc_off; records follow it */
        {
            uint8_t cnt[4];
            read_at(pf, m->exc_off, cnt, 4);
            n_exc_stored = (uint64_t)cnt[0] | ((uint64_t)cnt[1] << 8) |
                           ((uint64_t)cnt[2] << 16) | ((uint64_t)cnt[3] << 24);
        }
        if (n_exc_stored != m->n_exc)
            die("exception count disagrees with index");
        if (m->n_exc) {
            exc = xmalloc((size_t)m->n_exc * 5);
            read_at(pf, m->exc_off + 4, exc, (size_t)m->n_exc * 5);
        }

        got = xmalloc((size_t)m->n * sizeof(uint16_t));
        unpack_tensor(table, sa, sb, exc, m->n_exc, m->n, m->row_len, got);

        want = (const uint16_t *)xmalloc((size_t)m->n * sizeof(uint16_t));
        read_at(of, m->orig_off, (void *)want, (size_t)m->n * sizeof(uint16_t));

        for (j = 0; j < m->n; j++) {
            if (got[j] != want[j]) {
                if (total_bad < 10)
                    fprintf(stderr, "  MISMATCH %s at %llu: got %04x want %04x\n", m->name,
                            (unsigned long long)j, got[j], want[j]);
                total_bad++;
            }
        }
        if (outf && fwrite(got, sizeof(uint16_t), (size_t)m->n, outf) != (size_t)m->n)
            die("short write to unpacked output");

        total_n += m->n;
        if ((i + 1) % 100 == 0 || i == n_tensors - 1)
            fprintf(stderr, "  unpacked %d/%d tensors, %llu mismatched so far\n", i + 1,
                    n_tensors, (unsigned long long)total_bad);

        free(sa);
        free(sb);
        free(exc);
        free(got);
        free((void *)want);
    }

    fclose(pf);
    fclose(of);
    if (outf)
        fclose(outf);

    fprintf(stderr, "unpack_ref: tensors=%d weights=%llu mismatched_bits=%llu\n", n_tensors,
            (unsigned long long)total_n, (unsigned long long)total_bad);
    printf("tensors %d\nweights %llu\nmismatched_bits %llu\n", n_tensors,
           (unsigned long long)total_n, (unsigned long long)total_bad);
    return total_bad == 0 ? 0 : 1;
}
