/* race.c - A2: the kernel race on Kaggle.
 *
 * Three contestants compute y = W*x over the real Qwen2.5-3B gate/up/down
 * matrices whose total BF16 size reaches target_bytes:
 *
 *   FP32    - weights pre-expanded from BF16 in memory (exact: bits << 16)
 *   BF16    - weights loaded as BF16, converted to FP32 in registers
 *   PACKED  - weights decoded from the Phase-2 format in registers;
 *             unpacked weight values are NEVER written back to RAM
 *
 * All three share one loop skeleton: ROWS rows-at-once, VL-wide SIMD,
 * ascending column order in 64-weight groups, one fused (or mul+add)
 * step per chunk, identical horizontal reduction. Within one build the
 * three outputs must therefore be BIT-IDENTICAL (inputs are the same
 * FP32 values: both formats are lossless, Phase 2 proved it).
 *
 * Two builds (the old path is "forced" by being the only code in its
 * binary, per mission A2):
 *   gcc -std=c99 -Wall -Wextra -Werror -O2 -D_POSIX_C_SOURCE=200112L \
 *       -mavx2 -mfma  -fopenmp race.c -o race_modern
 *   gcc -std=c99 -Wall -Wextra -Werror -O2 -D_POSIX_C_SOURCE=200112L \
 *       -mssse3       -fopenmp race.c -o race_old
 * The old build uses only SSE2+SSSE3 (no SSE4.1, no AVX): byte widening
 * goes through PUNPCKLBW/PUNPCKLWD against zero, never PMOVZX (mission 3.4).
 *
 * Usage: race <export_dir> <packed_dir> <target_bytes> <threads> <out_file>
 *   threads: 1 = single thread, 0 = all cores
 *
 * Exit: 0 all checks passed | 1 a check failed | 2 usage
 *       3 load error | 4 CPU lacks instructions this build needs
 *
 * Decode per 64-weight group (mission A2, step 1-5): load code bytes and
 * split the nibbles, PSHUFB with the per-tensor 16-entry table, compare
 * codes to 15 and patch escape lanes from the exception list BEFORE the
 * multiply, rebuild FP32 bits as (sign<<31)|(exp<<23)|(mant7<<16), FMA.
 */
#include <cpuid.h>
#include <fcntl.h>
#include <omp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define BUILD_NAME "modern (AVX2+FMA)"
#define VL 8
#define ROWS 4
typedef __m256 vf;
typedef __m256i vi;
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#define BUILD_NAME "old (SSSE3, MULPS+ADDPS)"
#define VL 4
#define ROWS 4
typedef __m128 vf;
typedef __m128i vi;
#else
#error "build with -mavx2 -mfma or with -mssse3"
#endif

#if (VL == 8) && !defined(__AVX2__)
#error "VL=8 requires AVX2"
#endif
#if ((VL != 8) && !defined(__SSSE3__))
#error "VL=4 path requires SSSE3 (pshufb)"
#endif

#define NAME_MAX 128

static FILE *OUTF;

static void say(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    if (OUTF) {
        va_start(ap, fmt);
        vfprintf(OUTF, fmt, ap);
        va_end(ap);
        fputc('\n', OUTF);
        fflush(OUTF);
    }
}

static void die(int code, const char *fmt, ...)
{
    va_list ap;
    printf("FATAL: ");
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    if (OUTF) {
        fprintf(OUTF, "FATAL: ");
        va_start(ap, fmt);
        vfprintf(OUTF, fmt, ap);
        va_end(ap);
        fputc('\n', OUTF);
        fflush(OUTF);
    }
    exit(code);
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* ------------------------------------------------------------------ */
/* formats                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    char name[NAME_MAX];
    int nrows, cols;
    const uint8_t *bf16;      /* into mmap'd weights.bin            */
    float *fp32;              /* into the expansion buffer          */
    const uint8_t *tbl;       /* 16-byte padded local copy          */
    const uint8_t *sa, *sb;   /* into mmap'd packed.bin             */
    const uint8_t *exc;       /* exception records (5 bytes each)   */
    uint32_t n_exc;
    uint32_t *res;            /* nrows+1: first exception per row   */
    size_t yoff;              /* row offset into the y buffers      */
    uint64_t packed_bytes;
} Mat;

typedef struct {
    char name[NAME_MAX];
    uint64_t n, n_rows, row_len, tbl, sa, sb, exc, n_exc, orig;
} PIdx;

static inline vf vzero(void)
{
#if defined(__AVX2__)
    return _mm256_setzero_ps();
#else
    return _mm_setzero_ps();
#endif
}

static inline vf vload(const float *p)
{
#if defined(__AVX2__)
    return _mm256_loadu_ps(p);
#else
    return _mm_loadu_ps(p);
#endif
}

static inline void vstore(float *d, vf v)
{
#if defined(__AVX2__)
    _mm256_storeu_ps(d, v);
#else
    _mm_storeu_ps(d, v);
#endif
}

static inline vf vfma(vf w, vf x, vf a)
{
#if defined(__AVX2__)
    return _mm256_fmadd_ps(w, x, a);
#else
    /* old path: MULPS then ADDPS, two roundings (mission 3.4) */
    return _mm_add_ps(a, _mm_mul_ps(w, x));
#endif
}

static inline float hsum(vf v)
{
    float t[VL];
    float s;
    int i;
    vstore(t, v);
    s = t[0];
    for (i = 1; i < VL; i++)
        s += t[i];
    return s;
}

static inline vf load_fp32(const Mat *M, int row, int g, int k)
{
    return vload(M->fp32 + (size_t)row * (size_t)M->cols + (size_t)g + (size_t)k);
}

static inline vf load_bf16(const Mat *M, int row, int g, int k)
{
    const uint8_t *p = M->bf16 +
        ((size_t)row * (size_t)M->cols + (size_t)g + (size_t)k) * 2;
#if defined(__AVX2__)
    __m128i w = _mm_loadu_si128((const __m128i *)p);   /* 8 x bf16  */
    vi d = _mm256_cvtepu16_epi32(w);
    d = _mm256_slli_epi32(d, 16);
    return _mm256_castsi256_ps(d);
#else
    uint64_t r64;
    __m128i w, d;
    memcpy(&r64, p, 8);                                /* 4 x bf16  */
    w = _mm_cvtsi64_si128((long long)r64);
    d = _mm_unpacklo_epi16(w, _mm_setzero_si128());    /* no SSE4.1  */
    d = _mm_slli_epi32(d, 16);
    return _mm_castsi128_ps(d);
#endif
}

/* Decode VL weights of one row out of the packed format, in registers.
 * g = start of a 64-weight group (multiple of 64), k = offset in the
 * group (multiple of VL). *cur = this row's exception-list cursor. */
static inline vf load_packed(const Mat *M, int row, int g, int k, uint32_t *cur)
{
    const size_t grp = (size_t)g / 64;
    const uint8_t *sbrow = M->sb + (size_t)row * ((size_t)M->cols / 2) + grp * 32;
    const int hi = (k >= 32);
    const uint8_t *cb = sbrow + (hi ? k - 32 : k);
    const uint8_t *sap = M->sa + (size_t)row * (size_t)M->cols +
                         (size_t)g + (size_t)k;
    __m128i vb, codes, work, ab, aw, ew, w16, t, m, e;
#if VL == 8
    uint64_t raw;
    memcpy(&raw, cb, 8);
    vb = _mm_cvtsi64_si128((long long)raw);
#else
    uint32_t raw;
    memcpy(&raw, cb, 4);
    vb = _mm_cvtsi32_si128((int)raw);
#endif
    /* 1-2: split nibbles, PSHUFB with the 16-entry table */
    codes = hi ? _mm_srli_epi16(vb, 4) : vb;
    codes = _mm_and_si128(codes, _mm_set1_epi8(0x0F));
    work = _mm_loadu_si128((const __m128i *)M->tbl);
    work = _mm_shuffle_epi8(work, codes);
    /* 3: escape lanes (code 15) -> patch from exception list, before FMA */
    if (_mm_movemask_epi8(_mm_cmpeq_epi8(codes, _mm_set1_epi8(15))) != 0) {
        uint8_t eb[16];
        int i;
        _mm_storeu_si128((__m128i *)eb, work);
        for (i = 0; i < VL; i++) {
            uint8_t byt = (uint8_t)(raw >> (8 * i));
            int code = hi ? (byt >> 4) : (byt & 0x0F);
            if (code == 15) {
                const uint8_t *rec;
                uint32_t eidx;
                if (*cur >= M->n_exc)
                    die(4, "exception underflow in %s", M->name);
                rec = M->exc + (size_t)(*cur) * 5;
                eidx = (uint32_t)rec[0] | ((uint32_t)rec[1] << 8) |
                       ((uint32_t)rec[2] << 16) | ((uint32_t)rec[3] << 24);
                if (eidx != (uint32_t)((size_t)row * (size_t)M->cols +
                                       (size_t)g + (size_t)k + (size_t)i))
                    die(4, "exception out of order in %s", M->name);
                eb[i] = rec[4];
                (*cur)++;
            }
        }
        work = _mm_loadu_si128((const __m128i *)eb);
    }
    /* 4: rebuild bf16 bits (sign<<15)|(exp<<7)|mant, then <<16 -> fp32 */
#if VL == 8
    {
        uint64_t ra;
        memcpy(&ra, sap, 8);
        ab = _mm_cvtsi64_si128((long long)ra);
    }
#else
    {
        uint32_t ra;
        memcpy(&ra, sap, 4);
        ab = _mm_cvtsi32_si128((int)ra);
    }
#endif
#if defined(__SSE4_1__)
    aw = _mm_cvtepu8_epi16(ab);
    ew = _mm_cvtepu8_epi16(work);
#else
    aw = _mm_unpacklo_epi8(ab, _mm_setzero_si128());
    ew = _mm_unpacklo_epi8(work, _mm_setzero_si128());
#endif
    t = _mm_and_si128(_mm_slli_epi16(aw, 8), _mm_set1_epi16((short)0x8000));
    m = _mm_and_si128(aw, _mm_set1_epi16(0x007F));
    e = _mm_slli_epi16(ew, 7);
    w16 = _mm_or_si128(_mm_or_si128(t, m), e);
#if defined(__AVX2__)
    {
        vi d = _mm256_cvtepu16_epi32(w16);
        d = _mm256_slli_epi32(d, 16);
        return _mm256_castsi256_ps(d);
    }
#else
    {
        __m128i d = _mm_unpacklo_epi16(w16, _mm_setzero_si128());
        d = _mm_slli_epi32(d, 16);
        return _mm_castsi128_ps(d);
    }
#endif
}

/* ------------------------------------------------------------------ */
/* the three sweeps: identical skeleton, only the weight load differs  */
/* ------------------------------------------------------------------ */

static void sweep_fp32(const Mat *mats, int nm, const float *x, float *y)
{
    int mi;
#pragma omp parallel for schedule(static)
    for (mi = 0; mi < nm; mi++) {
        const Mat *M = &mats[mi];
        float *ym = y + M->yoff;
        int row0;
        for (row0 = 0; row0 < M->nrows; row0 += ROWS) {
            int nr = (M->nrows - row0 < ROWS) ? (M->nrows - row0) : ROWS;
            vf acc[ROWS];
            int r, g, k;
            for (r = 0; r < ROWS; r++)
                acc[r] = vzero();
            for (g = 0; g < M->cols; g += 64)
                for (k = 0; k < 64; k += VL) {
                    vf xv = vload(x + g + k);
                    for (r = 0; r < nr; r++)
                        acc[r] = vfma(load_fp32(M, row0 + r, g, k), xv, acc[r]);
                }
            for (r = 0; r < nr; r++)
                ym[row0 + r] = hsum(acc[r]);
        }
    }
}

static void sweep_bf16(const Mat *mats, int nm, const float *x, float *y)
{
    int mi;
#pragma omp parallel for schedule(static)
    for (mi = 0; mi < nm; mi++) {
        const Mat *M = &mats[mi];
        float *ym = y + M->yoff;
        int row0;
        for (row0 = 0; row0 < M->nrows; row0 += ROWS) {
            int nr = (M->nrows - row0 < ROWS) ? (M->nrows - row0) : ROWS;
            vf acc[ROWS];
            int r, g, k;
            for (r = 0; r < ROWS; r++)
                acc[r] = vzero();
            for (g = 0; g < M->cols; g += 64)
                for (k = 0; k < 64; k += VL) {
                    vf xv = vload(x + g + k);
                    for (r = 0; r < nr; r++)
                        acc[r] = vfma(load_bf16(M, row0 + r, g, k), xv, acc[r]);
                }
            for (r = 0; r < nr; r++)
                ym[row0 + r] = hsum(acc[r]);
        }
    }
}

static void sweep_packed(const Mat *mats, int nm, const float *x, float *y)
{
    int mi;
#pragma omp parallel for schedule(static)
    for (mi = 0; mi < nm; mi++) {
        const Mat *M = &mats[mi];
        float *ym = y + M->yoff;
        int row0;
        for (row0 = 0; row0 < M->nrows; row0 += ROWS) {
            int nr = (M->nrows - row0 < ROWS) ? (M->nrows - row0) : ROWS;
            uint32_t cur[ROWS];
            vf acc[ROWS];
            int r, g, k;
            for (r = 0; r < nr; r++) {
                acc[r] = vzero();
                cur[r] = M->res[row0 + r];
            }
            for (r = nr; r < ROWS; r++)
                acc[r] = vzero();
            for (g = 0; g < M->cols; g += 64)
                for (k = 0; k < 64; k += VL) {
                    vf xv = vload(x + g + k);
                    for (r = 0; r < nr; r++)
                        acc[r] = vfma(load_packed(M, row0 + r, g, k, &cur[r]),
                                      xv, acc[r]);
                }
            for (r = 0; r < nr; r++) {
                if (cur[r] != M->res[row0 + r + 1])
                    die(4, "unused exceptions in %s row %d", M->name, row0 + r);
                ym[row0 + r] = hsum(acc[r]);
            }
        }
    }
}

typedef void (*sweep_fn)(const Mat *, int, const float *, float *);

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static uint64_t xgetbv0(void)
{
    uint32_t lo, hi;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}

static void cpu_features(void)
{
    unsigned a = 0, b = 0, c = 0, d = 0;
    int ssse3, sse41, avx, avx2, fma, osxsave, os_ymm;
    __get_cpuid(1, &a, &b, &c, &d);
    ssse3 = (c & bit_SSSE3) != 0;
    sse41 = (c & bit_SSE4_1) != 0;
    osxsave = (c & bit_OSXSAVE) != 0;
    avx = (c & bit_AVX) != 0;
    fma = (c & bit_FMA) != 0;
    a = b = c = d = 0;
    __get_cpuid_count(7, 0, &a, &b, &c, &d);
    avx2 = (b & bit_AVX2) != 0;
    os_ymm = osxsave && avx && ((xgetbv0() & 6u) == 6u);
    say("cpuid: ssse3=%d sse4.1=%d avx=%d avx2=%d fma=%d osxsave=%d os_ymm=%d",
        ssse3, sse41, avx, avx2, fma, osxsave, os_ymm);
#if defined(__AVX2__)
    if (!(avx2 && fma && os_ymm))
        die(4, "this build needs AVX2+FMA with OS ymm state support");
#else
    if (!ssse3)
        die(4, "this build needs SSSE3");
#endif
}

static int name_in(const char *name, const char *needle)
{
    return strstr(name, needle) != NULL;
}

/* ------------------------------------------------------------------ */

typedef struct {
    char name[NAME_MAX];
    int nrows, cols;
    uint64_t off, nb;
} ManEnt;

int main(int argc, char **argv)
{
    const char *export_dir, *packed_dir, *out_path;
    uint64_t target;
    int threads_req, threads;
    char path[4096];
    FILE *f;
    struct stat st;
    const uint8_t *wmap = NULL, *pmap = NULL;
    int wfd = -1, pfd = -1;
    ManEnt *all = NULL;
    int cap = 16;
    PIdx *pidx = NULL;
    int n_pidx = 0;
    Mat *mats = NULL;
    int nm = 0;
    uint64_t total_params = 0, total_bf16 = 0, total_fp32b = 0;
    uint64_t total_packed = 0;
    float *fbuf = NULL;
    size_t ny = 0;
    float *x = NULL;
    int maxcols = 0;
    double *yr = NULL;
    float *yA, *yB, *yC;
    double tbest[3], tmed[3];
    sweep_fn fns[3];
    const char *labels[3];
    uint64_t tbytes[3];
    int ci, mi, row, r0;
    double mar = 0.0;
    size_t ndiff_ab, ndiff_ac, ndiff_bc;
    int failed = 0;

    if (argc != 6) {
        fprintf(stderr,
                "usage: %s <export_dir> <packed_dir> <target_bytes> "
                "<threads: 0=all 1=single> <out_file>\n", argv[0]);
        return 2;
    }
    export_dir = argv[1];
    packed_dir = argv[2];
    target = strtoull(argv[3], NULL, 10);
    threads_req = atoi(argv[4]);
    out_path = argv[5];

    OUTF = fopen(out_path, "w");
    if (!OUTF)
        return 3;
    threads = (threads_req <= 0) ? omp_get_max_threads() : threads_req;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    say("=== A2 kernel race: y = W*x over real Qwen2.5-3B matrices ===");
    say("build: %s  rows_at_once=%d  simd_width=%d floats  threads=%d",
        BUILD_NAME, ROWS, VL, threads);
    cpu_features();
    say("target_bf16_bytes: %llu", (unsigned long long)target);

    /* ---- manifest: pick gate/up/down until the target is reached ---- */
    snprintf(path, sizeof path, "%s/manifest.txt", export_dir);
    f = fopen(path, "r");
    if (!f)
        die(3, "cannot open %s", path);
    for (;;) {
        char line[512], nm_[NAME_MAX], dt[32];
        int idx, rows, cols;
        unsigned long long off, nb;
        if (!fgets(line, sizeof line, f))
            break;
        if (sscanf(line, "%d %127s %31s %d %d %llu %llu", &idx, nm_, dt,
                   &rows, &cols, &off, &nb) != 7)
            die(3, "bad manifest line: %s", line);
        if (strcmp(dt, "bf16") != 0)
            die(3, "manifest dtype not bf16: %s", dt);
        if ((uint64_t)rows * (uint64_t)cols * 2 != (uint64_t)nb)
            die(3, "manifest size mismatch for %s", nm_);
        /* manifest names are short: layers.<i>.gate.weight etc. (export_weights.py) */
        if (!(name_in(nm_, ".gate.weight") ||
              name_in(nm_, ".up.weight") ||
              name_in(nm_, ".down.weight")))
            continue;
        if (total_bf16 >= target)
            continue; /* target reached; skip the rest */
        if (!mats || nm >= cap) {
            /* first hit allocates: mats/all start NULL, cap=16 */
            int ncap = mats ? cap * 2 : 16;
            Mat *mt = realloc(mats, (size_t)ncap * sizeof *mats);
            ManEnt *at = realloc(all, (size_t)ncap * sizeof *all);
            if (!mt || !at)
                die(3, "out of memory");
            mats = mt;
            all = at;
            cap = ncap;
        }
        memset(&mats[nm], 0, sizeof mats[nm]);
        snprintf(mats[nm].name, NAME_MAX, "%s", nm_);
        mats[nm].nrows = rows;
        mats[nm].cols = cols;
        all[nm].off = (uint64_t)off;
        all[nm].nb = (uint64_t)nb;
        nm++;
        total_bf16 += (uint64_t)nb;
    }
    fclose(f);
    if (nm == 0)
        die(3, "no gate/up/down matrices found in manifest");
    if (total_bf16 < target)
        die(3, "only %llu BF16 bytes of gate/up/down available, target %llu",
            (unsigned long long)total_bf16, (unsigned long long)target);

    /* ---- packed index: match by name ---- */
    snprintf(path, sizeof path, "%s/packed_index.txt", packed_dir);
    f = fopen(path, "r");
    if (!f)
        die(3, "cannot open %s", path);
    {
        int cap_i = 16;
        pidx = malloc((size_t)cap_i * sizeof *pidx);
        if (!pidx)
            die(3, "out of memory");
        for (;;) {
            char line[512];
            unsigned long long v[9];
            if (n_pidx >= cap_i) {
                cap_i *= 2;
                pidx = realloc(pidx, (size_t)cap_i * sizeof *pidx);
                if (!pidx)
                    die(3, "out of memory");
            }
            if (!fgets(line, sizeof line, f))
                break;
            if (line[0] == '\n' || line[0] == '\0')
                continue;
            if (sscanf(line, "%127s %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                       pidx[n_pidx].name, &v[0], &v[1], &v[2], &v[3], &v[4],
                       &v[5], &v[6], &v[7], &v[8]) != 10)
                die(3, "bad packed_index line: %s", line);
            pidx[n_pidx].n = v[0];
            pidx[n_pidx].n_rows = v[1];
            pidx[n_pidx].row_len = v[2];
            pidx[n_pidx].tbl = v[3];
            pidx[n_pidx].sa = v[4];
            pidx[n_pidx].sb = v[5];
            pidx[n_pidx].exc = v[6];
            pidx[n_pidx].n_exc = v[7];
            pidx[n_pidx].orig = v[8];
            n_pidx++;
        }
        fclose(f);
    }

    /* ---- mmap both data files ---- */
    snprintf(path, sizeof path, "%s/weights.bin", export_dir);
    wfd = open(path, O_RDONLY);
    if (wfd < 0 || fstat(wfd, &st) != 0)
        die(3, "cannot open %s", path);
    if ((uint64_t)st.st_size != 6171877376ULL)
        die(3, "weights.bin is %llu bytes, expected 6171877376",
            (unsigned long long)st.st_size);
    wmap = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, wfd, 0);
    if (wmap == MAP_FAILED)
        die(3, "mmap weights.bin failed");
    snprintf(path, sizeof path, "%s/packed.bin", packed_dir);
    pfd = open(path, O_RDONLY);
    if (pfd < 0 || fstat(pfd, &st) != 0)
        die(3, "cannot open %s", path);
    pmap = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, pfd, 0);
    if (pmap == MAP_FAILED)
        die(3, "mmap packed.bin failed");

    /* ---- wire up each selected matrix ---- */
    for (mi = 0; mi < nm; mi++) {
        int j, found = -1;
        size_t n;
        const PIdx *P;
        if (mats[mi].cols % 64 != 0)
            die(3, "%s: cols %d not a multiple of 64", mats[mi].name,
                mats[mi].cols);
        for (j = 0; j < n_pidx; j++)
            if (strcmp(pidx[j].name, mats[mi].name) == 0) {
                found = j;
                break;
            }
        if (found < 0)
            die(3, "%s: no packed record", mats[mi].name);
        P = &pidx[found];
        n = (size_t)mats[mi].nrows * (size_t)mats[mi].cols;
        if ((uint64_t)n != P->n || (uint64_t)all[mi].off != P->orig)
            die(3, "%s: manifest/packed index disagree", mats[mi].name);
        mats[mi].bf16 = wmap + all[mi].off;
        mats[mi].tbl = pmap + P->tbl;
        mats[mi].sa = pmap + P->sa;
        mats[mi].sb = pmap + P->sb;
        mats[mi].exc = pmap + P->exc + 4;
        mats[mi].n_exc = (uint32_t)P->n_exc;
        mats[mi].packed_bytes = 15 + P->n + P->n / 2 + 4 + 5 * P->n_exc;
        mats[mi].yoff = ny;
        ny += (size_t)mats[mi].nrows;
        total_params += (uint64_t)n;
        total_fp32b += (uint64_t)n * 4;
        total_packed += mats[mi].packed_bytes;
        if (mats[mi].cols > maxcols)
            maxcols = mats[mi].cols;
        /* first exception index per row */
        mats[mi].res = malloc(((size_t)mats[mi].nrows + 1) * sizeof(uint32_t));
        if (!mats[mi].res)
            die(3, "out of memory");
        {
            uint32_t cur = 0;
            for (row = 0; row < mats[mi].nrows; row++) {
                uint64_t lim = (uint64_t)(row + 1) * (uint64_t)mats[mi].cols;
                mats[mi].res[row] = cur;
                while (cur < P->n_exc) {
                    const uint8_t *rec = pmap + P->exc + 4 + (size_t)cur * 5;
                    uint64_t eidx = (uint64_t)rec[0] | ((uint64_t)rec[1] << 8) |
                                    ((uint64_t)rec[2] << 16) |
                                    ((uint64_t)rec[3] << 24);
                    if (eidx >= lim)
                        break;
                    cur++;
                }
            }
            mats[mi].res[mats[mi].nrows] = cur;
            if (cur != P->n_exc)
                die(3, "%s: %llu of %llu exceptions placed", mats[mi].name,
                    (unsigned long long)cur, (unsigned long long)P->n_exc);
        }
        {
            /* 16-byte table copy: entries 0..14 from the file, 15 = 0
             * (code 15 = escape: always patched before use) */
            uint8_t *t16 = malloc(16);
            if (!t16)
                die(3, "out of memory");
            memcpy(t16, mats[mi].tbl, 15);
            t16[15] = 0;
            mats[mi].tbl = t16;
        }
    }
    say("selected: %d matrices  weights=%llu  bf16_bytes=%llu  "
        "fp32_bytes=%llu  packed_bytes=%llu",
        nm, (unsigned long long)total_params,
        (unsigned long long)(total_params * 2),
        (unsigned long long)total_fp32b, (unsigned long long)total_packed);
    say("first=%s last=%s", mats[0].name, mats[nm - 1].name);

    /* ---- exact FP32 expansion of the selected weights ---- */
    fbuf = malloc((size_t)total_fp32b);
    if (!fbuf)
        die(3, "cannot allocate %llu byte FP32 buffer",
            (unsigned long long)total_fp32b);
    {
        size_t fo = 0;
        for (mi = 0; mi < nm; mi++) {
            mats[mi].fp32 = fbuf + fo;
            fo += (size_t)mats[mi].nrows * (size_t)mats[mi].cols;
        }
#pragma omp parallel for schedule(static)
        for (mi = 0; mi < nm; mi++) {
            size_t i;
            size_t n = (size_t)mats[mi].nrows * (size_t)mats[mi].cols;
            const uint16_t *w = (const uint16_t *)mats[mi].bf16;
            float *dst = mats[mi].fp32;
            for (i = 0; i < n; i++) {
                uint32_t bits = (uint32_t)w[i] << 16;
                memcpy(&dst[i], &bits, 4);
            }
        }
    }

    /* ---- deterministic input vector ---- */
    x = malloc((size_t)maxcols * sizeof *x);
    if (!x)
        die(3, "out of memory");
    for (r0 = 0; r0 < maxcols; r0++) {
        uint32_t h = (uint32_t)r0 * 2654435761u;
        x[r0] = (float)((int)((h >> 8) % 2001u) - 1000) / 1000.0f;
    }

    yA = malloc(ny * sizeof *yA);
    yB = malloc(ny * sizeof *yB);
    yC = malloc(ny * sizeof *yC);
    yr = malloc(ny * sizeof *yr);
    if (!yA || !yB || !yC || !yr)
        die(3, "out of memory");

    /* ---- float64 reference (per row: sequential double accumulation) ---- */
    {
        double t0 = now_s();
#pragma omp parallel for schedule(static)
        for (mi = 0; mi < nm; mi++) {
            const Mat *M = &mats[mi];
            double *yrm = yr + M->yoff;
            int rr;
            for (rr = 0; rr < M->nrows; rr++) {
                const uint16_t *w = (const uint16_t *)M->bf16 +
                                    (size_t)rr * (size_t)M->cols;
                double s = 0.0;
                int c;
                for (c = 0; c < M->cols; c++) {
                    float fv;
                    uint32_t bits = (uint32_t)w[c] << 16;
                    memcpy(&fv, &bits, 4);
                    s += (double)fv * (double)x[c];
                }
                yrm[rr] = s;
            }
        }
        say("float64 reference computed in %.2fs", now_s() - t0);
        for (r0 = 0; r0 < (int)ny; r0++) {
            double a = yr[r0] < 0 ? -yr[r0] : yr[r0];
            if (a > mar)
                mar = a;
        }
        if (mar == 0.0)
            die(1, "reference is all zeros");
    }

    /* ---- correctness: three sweeps, pairwise bit-compare + vs ref ---- */
    fns[0] = sweep_fp32;
    fns[1] = sweep_bf16;
    fns[2] = sweep_packed;
    labels[0] = "fp32";
    labels[1] = "bf16";
    labels[2] = "packed";
    fns[0](mats, nm, x, yA);
    fns[1](mats, nm, x, yB);
    fns[2](mats, nm, x, yC);
    {
        size_t i, d;
        d = 0;
        for (i = 0; i < ny; i++)
            if (memcmp(&yA[i], &yB[i], 4) != 0)
                d++;
        ndiff_ab = d;
        d = 0;
        for (i = 0; i < ny; i++)
            if (memcmp(&yA[i], &yC[i], 4) != 0)
                d++;
        ndiff_ac = d;
        d = 0;
        for (i = 0; i < ny; i++)
            if (memcmp(&yB[i], &yC[i], 4) != 0)
                d++;
        ndiff_bc = d;
    }
    say("correctness: fp32_vs_bf16=%zu fp32_vs_packed=%zu bf16_vs_packed=%zu "
        "(differing elements of %zu; 0 = bit-identical)",
        ndiff_ab, ndiff_ac, ndiff_bc, ny);
    if (ndiff_ab || ndiff_ac || ndiff_bc)
        failed = 1;
    {
        const float *ys[3];
        int k;
        ys[0] = yA;
        ys[1] = yB;
        ys[2] = yC;
        for (k = 0; k < 3; k++) {
            size_t i;
            double local = 0.0;
            for (i = 0; i < ny; i++) {
                double d2 = (double)ys[k][i] - yr[i];
                if (d2 < 0)
                    d2 = -d2;
                if (d2 > local)
                    local = d2;
            }
            say("ref check %-6s: max_abs_diff=%.6e max_abs_y_ref=%.6e "
                "rel=%.6e (threshold 1e-4) %s",
                labels[k], local, mar, local / mar,
                local / mar < 1e-4 ? "PASS" : "FAIL");
            if (local / mar >= 1e-4)
                failed = 1;
        }
    }

    /* ---- measured RAM read bandwidth (fp32 buffer, >= 4xL3) ---- */
    {
        volatile uint64_t sink = 0;
        double t[10] = { 0 };
        int rep;
        size_t n = (size_t)total_fp32b / 4;
        for (rep = 0; rep < 3; rep++) {
            uint64_t s = 0;
            size_t i;
            for (i = 0; i < n; i += 16)
                s += ((const uint32_t *)fbuf)[i];
            sink += s;
        }
        for (rep = 0; rep < 10; rep++) {
            uint64_t s = 0;
            size_t i;
            double t0 = now_s();
            for (i = 0; i < n; i += 16)
                s += ((const uint32_t *)fbuf)[i];
            t[rep] = now_s() - t0;
            sink += s;
        }
        qsort(t, 10, sizeof t[0], cmp_dbl);
        say("ram_read_bandwidth: best=%.2f GB/s median=%.2f GB/s "
            "(one 64B line touched per 16 floats, %llu bytes)",
            (double)(n / 16 * 64) / 1e9 / t[0],
            (double)(n / 16 * 64) / 1e9 / ((t[4] + t[5]) / 2.0),
            (unsigned long long)(n / 16 * 64));
        (void)sink;
    }

    /* ---- timing: 3 warm-ups + 10 timed runs per contestant ---- */
    tbytes[0] = total_fp32b;
    tbytes[1] = total_params * 2;
    tbytes[2] = total_packed;
    for (ci = 0; ci < 3; ci++) {
        double t[10] = { 0 };
        int rep;
        for (rep = 0; rep < 3; rep++)
            fns[ci](mats, nm, x, yA);
        for (rep = 0; rep < 10; rep++) {
            double t0 = now_s();
            fns[ci](mats, nm, x, yA);
            t[rep] = now_s() - t0;
        }
        qsort(t, 10, sizeof t[0], cmp_dbl);
        tbest[ci] = t[0];
        tmed[ci] = (t[4] + t[5]) / 2.0;
        say("timing %-6s: best=%.4fs median=%.4fs  %.2f GB/s  "
            "%.3f G weights/s  (10 runs sorted: %.4f %.4f %.4f %.4f %.4f "
            "%.4f %.4f %.4f %.4f %.4f)",
            labels[ci], tbest[ci], tmed[ci],
            (double)tbytes[ci] / 1e9 / tbest[ci],
            (double)total_params / 1e9 / tbest[ci],
            t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9]);
    }
    say("speedup packed_vs_bf16: best=%.3f median=%.3f",
        tbest[1] / tbest[2], tmed[1] / tmed[2]);
    say("speedup bf16_vs_fp32  : best=%.3f median=%.3f",
        tbest[0] / tbest[1], tmed[0] / tmed[1]);

    /* checksum keeps every sweep live for the optimizer */
    {
        double cs = 0.0;
        size_t i;
        for (i = 0; i < ny; i++)
            cs += (double)yA[i];
        say("y_checksum: %.6e", cs);
    }

    say(failed ? "RUN FAIL" : "RUN PASS");
    if (OUTF)
        fclose(OUTF);
    return failed ? 1 : 0;
}
