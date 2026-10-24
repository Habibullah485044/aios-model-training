/* bench.c - A3: old-PC benchmark. FP32 vs BF16 vs PACKED matvec race,
 * exactly the A2 kernels (SSSE3, MULPS+ADDPS, ROWS=4, VL=4), ported to
 * bare-metal Windows: kernel32 imports only, no CRT, no SSE4.1, no AVX.
 *
 * Built with zig cc (see a3_pack_check.py):
 *   bench64.exe: -target x86_64-windows-gnu -mcpu=core2
 *   bench32.exe: -target x86-windows-gnu    -mcpu=core2
 *
 * Data file "a3bench.bin" (written by make_a3bench_data.py):
 *   "A3B1" | u32 nmat | per matrix:
 *     u32 rows, u32 cols, u32 n_exc,
 *     15B exp table, bf16[rows*cols], sa[rows*cols], sb[rows*cols/2],
 *     exc[n_exc * 5]   (u32 flat index, u8 true exponent, sorted)
 *
 * Output: console + results_64bit.txt / results_32bit.txt next to the exe.
 *
 * Exit codes: 0 all checks passed | 1 a check failed | 2 data file problem
 *             | 3 out of memory.
 *
 * No 64-bit integer division anywhere (i386 has no inline u64 div and
 * -nostdlib must not drag in runtime helpers): time/rate math uses SSE2
 * doubles, printed via own number-to-text.
 */
#include <stddef.h> /* wchar_t, size_t (no CRT needed) */
#include <tmmintrin.h>

/* ---------------- kernel32 surface (hand-declared, KERNEL32.dll only) */
typedef void *HANDLE;
typedef unsigned long DWORD;
typedef int BOOL;
typedef const wchar_t *LPCWSTR;
typedef void *LPVOID;

#define STD_OUTPUT_HANDLE ((DWORD)-11)
#define GENERIC_READ 0x80000000u
#define GENERIC_WRITE 0x40000000u
#define FILE_SHARE_READ 1u
#define OPEN_EXISTING 3u
#define CREATE_ALWAYS 2u
#define INVALID_HANDLE_VALUE ((HANDLE)(uptr)-1)
#define MEM_COMMIT_RESERVE 0x3000u
#define PAGE_READWRITE 4u
#define INFINITE ((DWORD)-1)

HANDLE GetStdHandle(DWORD n);
BOOL WriteFile(HANDLE h, const void *buf, DWORD n, DWORD *written, void *ovl);
HANDLE CreateFileW(LPCWSTR name, DWORD access, DWORD share, void *sec,
                   DWORD disp, DWORD flags, HANDLE tmpl);
BOOL GetFileSizeEx(HANDLE h, long long *sz);
BOOL ReadFile(HANDLE h, void *buf, DWORD n, DWORD *read, void *ovl);
LPVOID VirtualAlloc(LPVOID addr, unsigned long long size, DWORD type,
                    DWORD protect);
/* fn passed as void*: the callback carries WINAPI (stdcall) on i386 and an
 * attribute inside a parenthesized declarator typedef is not accepted by
 * clang; the definition below keeps the correct convention per target. */
HANDLE CreateThread(void *sec, unsigned long long stack,
                    void *fn, LPVOID arg, DWORD flags,
                    DWORD *tid);
DWORD WaitForSingleObject(HANDLE h, DWORD ms);
BOOL CloseHandle(HANDLE h);
BOOL QueryPerformanceCounter(long long *c);
BOOL QueryPerformanceFrequency(long long *f);
void ExitProcess(unsigned code);
void GetSystemInfo(void *info); /* SYSTEM_INFO: nproc at offset 20 (i386)
                                 * / 32 (x64); we only read that DWORD. */
#define WINAPI __attribute__((stdcall))
#ifdef _WIN64
typedef unsigned long long uptr; /* pointer-sized integer, both targets */
#else
typedef unsigned int uptr;
#endif

/* memcpy/memset: the compiler may emit calls; -nostdlib has none. */
void *memcpy(void *d, const void *s, unsigned long n)
{
    unsigned char *dd = (unsigned char *)d;
    const unsigned char *ss = (const unsigned char *)s;
    unsigned long i;
    for (i = 0; i < n; i++)
        dd[i] = ss[i];
    return d;
}
void *memset(void *d, int c, unsigned long n)
{
    unsigned char *dd = (unsigned char *)d;
    unsigned long i;
    for (i = 0; i < n; i++)
        dd[i] = (unsigned char)c;
    return d;
}
int memcmp(const void *a, const void *b, unsigned long n)
{
    const unsigned char *aa = (const unsigned char *)a;
    const unsigned char *bb = (const unsigned char *)b;
    unsigned long i;
    for (i = 0; i < n; i++)
        if (aa[i] != bb[i])
            return (int)aa[i] - (int)bb[i];
    return 0;
}

/* CPUID without any header (clang builtin needs intrin.h; none here) */
static void do_cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{
    __asm__ volatile("cpuid"
                     : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3])
                     : "a"(leaf), "c"(sub));
}

/* ---------------- output: console + results file ------------------- */
static HANDLE g_out; /* results file, may be 0 */

static void wboth(const char *s, DWORD n)
{
    DWORD w;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), s, n, &w, 0);
    if (g_out)
        WriteFile(g_out, s, n, &w, 0);
}

/* own number-to-text */
static DWORD fmt_u64(char *buf, unsigned long long v)
{
    char tmp[24];
    DWORD len = 0, i;
    if (v == 0) {
        buf[0] = '0';
        return 1;
    }
    while (v) {
        tmp[len++] = (char)('0' + (int)(v % 10u));
        v /= 10u;
    }
    for (i = 0; i < len; i++)
        buf[i] = tmp[len - 1 - i];
    return len;
}

/* fixed-point from a u32 scaled by 10^dec (dec in 0..4) */
static DWORD fmt_fixed(char *buf, unsigned long long x100, int dec)
{
    DWORD len = fmt_u64(buf, x100 / 100u);
    int i;
    unsigned long long scale = 100u;
    for (i = 1; i < dec; i++)
        scale *= 10u;
    buf[len++] = '.';
    {
        DWORD fl = fmt_u64(buf + len, (x100 % 100u));
        for (i = (int)fl; i < dec; i++)
            buf[len++] = '0';
        len += fl;
    }
    (void)scale;
    return len;
}

static void say(const char *s)
{
    DWORD n = 0;
    while (s[n])
        n++;
    wboth(s, n);
}
static void say_u64(const char *k, unsigned long long v)
{
    char buf[64];
    DWORD n = 0, kl = 0, vl;
    while (k[kl]) {
        buf[n++] = k[kl++];
    }
    vl = fmt_u64(buf + n, v);
    n += vl;
    buf[n++] = '\n';
    wboth(buf, n);
}
static void say_fix(const char *k, unsigned long long x100, const char *tail)
{
    char buf[96];
    DWORD n = 0, kl = 0, tl = 0;
    while (k[kl])
        buf[n++] = k[kl++];
    n += fmt_fixed(buf + n, x100, 2);
    while (tail[tl])
        buf[n++] = tail[tl++];
    buf[n++] = '\n';
    wboth(buf, n);
}

static void die(int code, const char *msg)
{
    say("FATAL: ");
    say(msg);
    say("\n");
    if (g_out)
        CloseHandle(g_out);
    ExitProcess((unsigned)code);
}

/* ---------------- time (SSE2 doubles; no u64 division) ------------- */
static long long g_qpc_freq;

static double now_s(void)
{
    long long c = 0;
    QueryPerformanceCounter(&c);
    return (double)c / (double)g_qpc_freq;
}

/* ---------------- matrices ----------------------------------------- */
#define NAME_MAX 8

typedef struct {
    int nrows, cols;
    const unsigned char *bf16; /* into the file mapping buffer  */
    float *fp32;               /* into the expansion buffer      */
    const unsigned char *tbl;  /* 16-byte padded table copy      */
    const unsigned char *sa, *sb, *exc;
    unsigned n_exc;
    unsigned *res;      /* nrows+1: first exception index per row */
    unsigned long yoff; /* row offset into the y buffers          */
    unsigned long long packed_bytes;
} Mat;

static Mat g_mats[64];
static int g_nm;
static unsigned long g_ny;
static float *g_x;
static float *g_yA, *g_yB, *g_yC;
static double *g_yr;
static unsigned g_nt; /* threads */
static volatile long g_go, g_done;

/* ---------------- the three kernels (A2 old path, verbatim) -------- */
static inline __m128 vfma(__m128 w, __m128 x, __m128 a)
{
    return _mm_add_ps(a, _mm_mul_ps(w, x)); /* MULPS+ADDPS, no FMA */
}
static inline float hsum(__m128 v)
{
    float t[4];
    _mm_storeu_ps(t, v);
    return t[0] + t[1] + t[2] + t[3];
}

static inline __m128 load_fp32(const Mat *M, int row, int g, int k)
{
    return _mm_loadu_ps(M->fp32 + (unsigned long)row * (unsigned)M->cols +
                        (unsigned)g + (unsigned)k);
}
static inline __m128 load_bf16(const Mat *M, int row, int g, int k)
{
    const unsigned char *p = M->bf16 +
        ((unsigned long)row * (unsigned)M->cols + (unsigned)g + (unsigned)k) * 2;
    unsigned long long r64;
    __m128i w, d;
    memcpy(&r64, p, 8);                             /* 4 x bf16 */
    w = _mm_cvtsi64_si128((long long)r64);
    d = _mm_unpacklo_epi16(w, _mm_setzero_si128()); /* no SSE4.1 */
    d = _mm_slli_epi32(d, 16);
    return _mm_castsi128_ps(d);
}

/* decode VL=4 weights of one row from the packed format, in registers */
static __m128 load_packed(const Mat *M, int row, int g, int k, unsigned *cur)
{
    const unsigned long grp = (unsigned long)g / 64;
    const unsigned char *sbrow =
        M->sb + (unsigned long)row * ((unsigned)M->cols / 2) + grp * 32;
    const int hi = (k >= 32);
    const unsigned char *cb = sbrow + (hi ? k - 32 : k);
    const unsigned char *sap =
        M->sa + (unsigned long)row * (unsigned)M->cols + (unsigned)g + (unsigned)k;
    __m128i vb, codes, work, ab, aw, ew, w16, t, m, e;
    unsigned raw;
    memcpy(&raw, cb, 4);
    vb = _mm_cvtsi32_si128((int)raw);
    /* nibble split + PSHUFB table lookup */
    codes = hi ? _mm_srli_epi16(vb, 4) : vb;
    codes = _mm_and_si128(codes, _mm_set1_epi8(0x0F));
    work = _mm_loadu_si128((const __m128i *)M->tbl);
    work = _mm_shuffle_epi8(work, codes);
    /* escape lanes (code 15) patched from the exception list */
    if (_mm_movemask_epi8(_mm_cmpeq_epi8(codes, _mm_set1_epi8(15))) != 0) {
        unsigned char eb[16];
        int i;
        _mm_storeu_si128((__m128i *)eb, work);
        for (i = 0; i < 4; i++) {
            unsigned char byt = (unsigned char)(raw >> (8 * i));
            int code = hi ? (byt >> 4) : (byt & 0x0F);
            if (code == 15) {
                const unsigned char *rec;
                unsigned eidx;
                if (*cur >= M->n_exc)
                    die(1, "exception underflow");
                rec = M->exc + (*cur) * 5;
                eidx = (unsigned)rec[0] | ((unsigned)rec[1] << 8) |
                       ((unsigned)rec[2] << 16) | ((unsigned)rec[3] << 24);
                if (eidx != (unsigned)((unsigned long)row * (unsigned)M->cols +
                                       (unsigned)g + (unsigned)k + (unsigned)i))
                    die(1, "exception out of order");
                eb[i] = rec[4];
                (*cur)++;
            }
        }
        work = _mm_loadu_si128((const __m128i *)eb);
    }
    /* rebuild bf16 bits -> fp32 bits */
    {
        unsigned ra;
        memcpy(&ra, sap, 4);
        ab = _mm_cvtsi32_si128((int)ra);
    }
    aw = _mm_unpacklo_epi8(ab, _mm_setzero_si128());
    ew = _mm_unpacklo_epi8(work, _mm_setzero_si128());
    t = _mm_and_si128(_mm_slli_epi16(aw, 8), _mm_set1_epi16((short)0x8000));
    m = _mm_and_si128(aw, _mm_set1_epi16(0x007F));
    e = _mm_slli_epi16(ew, 7);
    w16 = _mm_or_si128(_mm_or_si128(t, m), e);
    {
        __m128i d = _mm_unpacklo_epi16(w16, _mm_setzero_si128());
        d = _mm_slli_epi32(d, 16);
        return _mm_castsi128_ps(d);
    }
}

/* row ranges: global row space [lo,hi) of all matrices, static partition */
static void sweep_fp32(unsigned long lo, unsigned long hi, float *y)
{
    int mi;
    for (mi = 0; mi < g_nm; mi++) {
        const Mat *M = &g_mats[mi];
        unsigned long r0 = lo > M->yoff ? lo : M->yoff;
        unsigned long r1 = hi < M->yoff + (unsigned long)M->nrows
                             ? hi : M->yoff + (unsigned long)M->nrows;
        unsigned long rr;
        for (rr = r0; rr < r1; rr++) {
            int row = (int)(rr - M->yoff);
            __m128 acc = _mm_setzero_ps();
            int g, k;
            for (g = 0; g < M->cols; g += 64)
                for (k = 0; k < 64; k += 4) {
                    __m128 xv = _mm_loadu_ps(g_x + g + k);
                    acc = vfma(load_fp32(M, row, g, k), xv, acc);
                }
            y[rr] = hsum(acc);
        }
    }
}
static void sweep_bf16(unsigned long lo, unsigned long hi, float *y)
{
    int mi;
    for (mi = 0; mi < g_nm; mi++) {
        const Mat *M = &g_mats[mi];
        unsigned long r0 = lo > M->yoff ? lo : M->yoff;
        unsigned long r1 = hi < M->yoff + (unsigned long)M->nrows
                             ? hi : M->yoff + (unsigned long)M->nrows;
        unsigned long rr;
        for (rr = r0; rr < r1; rr++) {
            int row = (int)(rr - M->yoff);
            __m128 acc = _mm_setzero_ps();
            int g, k;
            for (g = 0; g < M->cols; g += 64)
                for (k = 0; k < 64; k += 4) {
                    __m128 xv = _mm_loadu_ps(g_x + g + k);
                    acc = vfma(load_bf16(M, row, g, k), xv, acc);
                }
            y[rr] = hsum(acc);
        }
    }
}
static void sweep_packed(unsigned long lo, unsigned long hi, float *y)
{
    int mi;
    for (mi = 0; mi < g_nm; mi++) {
        const Mat *M = &g_mats[mi];
        unsigned long r0 = lo > M->yoff ? lo : M->yoff;
        unsigned long r1 = hi < M->yoff + (unsigned long)M->nrows
                             ? hi : M->yoff + (unsigned long)M->nrows;
        unsigned long rr;
        for (rr = r0; rr < r1; rr++) {
            int row = (int)(rr - M->yoff);
            __m128 acc = _mm_setzero_ps();
            unsigned cur = M->res[row];
            int g, k;
            for (g = 0; g < M->cols; g += 64)
                for (k = 0; k < 64; k += 4) {
                    __m128 xv = _mm_loadu_ps(g_x + g + k);
                    acc = vfma(load_packed(M, row, g, k, &cur), xv, acc);
                }
            if (cur != M->res[row + 1])
                die(1, "unused exceptions");
            y[rr] = hsum(acc);
        }
    }
}

typedef void (*sweep_fn)(unsigned long, unsigned long, float *);
static sweep_fn g_fn;      /* set by worker mode */
static float *g_worker_y;

static DWORD WINAPI worker(LPVOID arg)
{
    unsigned t = (unsigned)(uptr)arg;
    unsigned long lo = g_ny * t / g_nt;
    unsigned long hi = g_ny * (t + 1) / g_nt;
    while (g_go == 0)
        _mm_pause();
    g_fn(lo, hi, g_worker_y);
    __sync_fetch_and_add(&g_done, 1);
    return 0;
}

/* run one sweep with g_nt threads (1 = direct call) */
static void run_mt(sweep_fn fn, float *y)
{
    g_fn = fn;
    g_worker_y = y;
    if (g_nt <= 1) {
        fn(0, g_ny, y);
        return;
    }
    {
        HANDLE th[16];
        unsigned i, n = g_nt;
        if (n > 16)
            n = 16;
        g_done = 0;
        for (i = 0; i < n; i++) {
            th[i] = CreateThread(0, 0, (void *)worker,
                                 (LPVOID)(uptr)(i + 1), 0, 0);
            if (!th[i])
                die(3, "CreateThread failed");
        }
        g_go = 1;
        while (g_done < (long)n)
            _mm_pause();
        g_go = 0;
        for (i = 0; i < n; i++)
            CloseHandle(th[i]);
    }
}

/* ---------------- main --------------------------------------------- */
void WINAPI start(void)
{
    HANDLE hfile;
    long long fsz = 0;
    unsigned char *buf;
    unsigned char *p;
    unsigned nmat, mi, i;
    unsigned maxcols = 0;
    unsigned long long total_bf16 = 0, total_params = 0, total_packed = 0;
    double tbest[3], tmed[3];
    const char *labels[3];
    sweep_fn fns[3];
    unsigned long long tbytes[3];
    int ci, failed = 0;
    char fname[] = "a3bench.bin";
    wchar_t wres[32];
    int wi;

    {
        DWORD w;
        WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), "A3 old-PC bench\n", 16, &w, 0);
    }

    /* results file name per bitness */
#ifdef _WIN64
    {
        const wchar_t *rn = L"results_64bit.txt";
        for (wi = 0; rn[wi]; wi++)
            wres[wi] = rn[wi];
        wres[wi] = 0;
    }
#else
    {
        const wchar_t *rn = L"results_32bit.txt";
        for (wi = 0; rn[wi]; wi++)
            wres[wi] = rn[wi];
        wres[wi] = 0;
    }
#endif
    g_out = CreateFileW(wres, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0);
    if (g_out == INVALID_HANDLE_VALUE)
        g_out = 0;

    QueryPerformanceFrequency(&g_qpc_freq);
    if (g_qpc_freq <= 0)
        die(3, "no QPC");

    /* CPU info */
    {
        unsigned r[4];
        char vendor[13];
        do_cpuid(0, 0, r);
        *(unsigned *)(vendor + 0) = r[1];
        *(unsigned *)(vendor + 4) = r[3];
        *(unsigned *)(vendor + 8) = r[2];
        vendor[12] = 0;
        say("cpu_vendor: ");
        say(vendor);
        say("\n");
        do_cpuid(1, 0, r);
        say_u64("cpu_family:", (unsigned)((r[0] >> 8) & 0xF));
        say_u64("cpu_model:", (unsigned)((r[0] >> 4) & 0xF));
        say_u64("cpu_stepping:", (unsigned)(r[0] & 0xF));
        say_u64("flag_ssse3:", (r[2] >> 9) & 1u);
        say_u64("flag_sse41:", (r[2] >> 19) & 1u);
        say_u64("flag_avx:", (r[2] >> 28) & 1u);
    }
    {
        unsigned char si[64];
        DWORD nproc;
        memset(si, 0, sizeof si);
        GetSystemInfo(si);
#ifdef _WIN64
        nproc = *(DWORD *)(si + 32);
#else
        nproc = *(DWORD *)(si + 20);
#endif
        if (nproc == 0)
            nproc = 1;
        g_nt = nproc;
        if (g_nt > 16)
            g_nt = 16;
        say_u64("cpu_threads_available:", nproc);
    }

    /* ---- load a3bench.bin ---- */
    {
        wchar_t wf[16];
        for (wi = 0; fname[wi]; wi++)
            wf[wi] = (wchar_t)fname[wi];
        wf[wi] = 0;
        hfile = CreateFileW(wf, GENERIC_READ, FILE_SHARE_READ, 0,
                            OPEN_EXISTING, 0, 0);
    }
    if (hfile == INVALID_HANDLE_VALUE)
        die(2, "cannot open a3bench.bin (keep it next to this exe)");
    if (!GetFileSizeEx(hfile, &fsz) || fsz < 16 || fsz > 4LL * 1024 * 1024 * 1024)
        die(2, "bad a3bench.bin size");
    buf = (unsigned char *)VirtualAlloc(0, (unsigned long long)fsz,
                                        MEM_COMMIT_RESERVE, PAGE_READWRITE);
    if (!buf)
        die(3, "out of memory (file buffer)");
    {
        DWORD got = 0;
        long long off = 0;
        while (off < fsz) {
            DWORD chunk = (DWORD)((fsz - off) > 0x40000000LL
                                      ? 0x40000000LL : (fsz - off));
            if (!ReadFile(hfile, buf + off, chunk, &got, 0) || got == 0)
                die(2, "short read on a3bench.bin");
            off += got;
        }
    }
    CloseHandle(hfile);
    if (buf[0] != 'A' || buf[1] != '3' || buf[2] != 'B' || buf[3] != '1')
        die(2, "bad a3bench.bin magic");
    nmat = *(const unsigned *)(buf + 4);
    if (nmat == 0 || nmat > 64)
        die(2, "bad matrix count in a3bench.bin");

    p = buf + 8;
    for (mi = 0; mi < nmat; mi++) {
        Mat *M = &g_mats[mi];
        unsigned n, r;
        unsigned cur = 0;
        if ((unsigned long long)(p - buf) + 12 > (unsigned long long)fsz)
            die(2, "truncated a3bench.bin header");
        M->nrows = (int)*(const unsigned *)p;
        M->cols = (int)*(const unsigned *)(p + 4);
        M->n_exc = *(const unsigned *)(p + 8);
        p += 12;
        if (M->nrows <= 0 || M->cols <= 0 || (M->cols % 64) != 0)
            die(2, "bad matrix shape (cols must be multiple of 64)");
        n = (unsigned)M->nrows * (unsigned)M->cols;
        {
            unsigned char *t16 = (unsigned char *)VirtualAlloc(
                0, 16, MEM_COMMIT_RESERVE, PAGE_READWRITE);
            if (!t16)
                die(3, "out of memory");
            memcpy(t16, p, 15);
            t16[15] = 0; /* code 15 = escape, always patched */
            M->tbl = t16;
            p += 15;
        }
        M->bf16 = p;
        p += (unsigned long)n * 2;
        M->sa = p;
        p += n;
        M->sb = p;
        p += n / 2;
        M->exc = p;
        p += (unsigned long)M->n_exc * 5;
        if ((unsigned long long)(p - buf) > (unsigned long long)fsz)
            die(2, "truncated a3bench.bin matrix");

        M->packed_bytes = 15 + n + n / 2 + 4u + 5ULL * M->n_exc;
        M->yoff = g_ny;
        g_ny += (unsigned long)M->nrows;
        total_params += n;
        total_bf16 += (unsigned long long)n * 2;
        total_packed += M->packed_bytes;
        if ((unsigned)M->cols > maxcols)
            maxcols = (unsigned)M->cols;

        /* per-row exception cursors (same as race.c) */
        M->res = (unsigned *)VirtualAlloc(0,
                                          ((unsigned long)M->nrows + 1) * sizeof(unsigned),
                                          MEM_COMMIT_RESERVE, PAGE_READWRITE);
        if (!M->res)
            die(3, "out of memory");
        for (r = 0; r < (unsigned)M->nrows; r++) {
            unsigned long long lim =
                (unsigned long long)(r + 1) * (unsigned long)M->cols;
            M->res[r] = cur;
            while (cur < M->n_exc) {
                const unsigned char *rec = M->exc + (unsigned long)cur * 5;
                unsigned long long eidx =
                    (unsigned long long)rec[0] |
                    ((unsigned long long)rec[1] << 8) |
                    ((unsigned long long)rec[2] << 16) |
                    ((unsigned long long)rec[3] << 24);
                if (eidx >= lim)
                    break;
                cur++;
            }
        }
        M->res[M->nrows] = cur;
        if (cur != M->n_exc)
            die(2, "exception records do not cover the matrix");
    }

    say_u64("matrices:", nmat);
    say_u64("weights:", total_params);
    say_u64("bf16_bytes:", total_bf16);
    say_u64("packed_bytes:", total_packed);

    /* ---- buffers ---- */
    {
        unsigned long long fp32bytes = total_params * 4;
        unsigned char *fb = (unsigned char *)VirtualAlloc(
            0, fp32bytes, MEM_COMMIT_RESERVE, PAGE_READWRITE);
        unsigned long long fo = 0;
        if (!fb)
            die(3, "out of memory (fp32 expansion)");
        for (mi = 0; mi < nmat; mi++) {
            g_mats[mi].fp32 = (float *)(fb + fo * 4);
            fo += (unsigned long)g_mats[mi].nrows * (unsigned)g_mats[mi].cols;
        }
        for (mi = 0; mi < nmat; mi++) {
            const Mat *M = &g_mats[mi];
            unsigned long n = (unsigned long)M->nrows * (unsigned)M->cols;
            unsigned long j;
            for (j = 0; j < n; j++) {
                unsigned bits = (unsigned)M->bf16[j * 2] |
                                ((unsigned)M->bf16[j * 2 + 1] << 8);
                bits <<= 16;
                memcpy(&M->fp32[j], &bits, 4);
            }
        }
    }
    g_x = (float *)VirtualAlloc(0, (unsigned long long)maxcols * 4,
                                MEM_COMMIT_RESERVE, PAGE_READWRITE);
    g_yA = (float *)VirtualAlloc(0, (unsigned long long)g_ny * 4,
                                 MEM_COMMIT_RESERVE, PAGE_READWRITE);
    g_yB = (float *)VirtualAlloc(0, (unsigned long long)g_ny * 4,
                                 MEM_COMMIT_RESERVE, PAGE_READWRITE);
    g_yC = (float *)VirtualAlloc(0, (unsigned long long)g_ny * 4,
                                 MEM_COMMIT_RESERVE, PAGE_READWRITE);
    g_yr = (double *)VirtualAlloc(0, (unsigned long long)g_ny * 8,
                                  MEM_COMMIT_RESERVE, PAGE_READWRITE);
    if (!g_x || !g_yA || !g_yB || !g_yC || !g_yr)
        die(3, "out of memory");
    for (i = 0; i < maxcols; i++) {
        unsigned h = i * 2654435761u;
        g_x[i] = (float)((int)((h >> 8) % 2001u) - 1000) / 1000.0f;
    }

    /* ---- float64 reference ---- */
    {
        double t0 = now_s();
        int m2;
        for (m2 = 0; m2 < g_nm; m2++) {
            const Mat *M = &g_mats[m2];
            int rr;
            for (rr = 0; rr < M->nrows; rr++) {
                const unsigned char *w =
                    M->bf16 + (unsigned long)rr * (unsigned)M->cols * 2;
                double s = 0.0;
                int c2;
                for (c2 = 0; c2 < M->cols; c2++) {
                    unsigned bits = (unsigned)w[c2 * 2] |
                                    ((unsigned)w[c2 * 2 + 1] << 8);
                    float fv;
                    bits <<= 16;
                    memcpy(&fv, &bits, 4);
                    s += (double)fv * (double)g_x[c2];
                }
                g_yr[M->yoff + (unsigned long)rr] = s;
            }
        }
        say_fix("float64_reference_seconds: ",
                (unsigned long long)((now_s() - t0) * 100.0 + 0.5), "");
    }

    /* ---- correctness: three sweeps, pairwise bit-compare + vs ref ---- */
    run_mt(sweep_fp32, g_yA); /* g_nt threads here too: identical rows */
    run_mt(sweep_bf16, g_yB);
    run_mt(sweep_packed, g_yC);
    {
        unsigned long nd_ab = 0, nd_ac = 0, nd_bc = 0, j;
        for (j = 0; j < g_ny; j++) {
            if (memcmp(&g_yA[j], &g_yB[j], 4) != 0)
                nd_ab++;
            if (memcmp(&g_yA[j], &g_yC[j], 4) != 0)
                nd_ac++;
            if (memcmp(&g_yB[j], &g_yC[j], 4) != 0)
                nd_bc++;
        }
        say_u64("bit_diff fp32_vs_bf16:", nd_ab);
        say_u64("bit_diff fp32_vs_packed:", nd_ac);
        say_u64("bit_diff bf16_vs_packed:", nd_bc);
        if (nd_ab || nd_ac || nd_bc)
            failed = 1;
    }
    {
        double mar = 0.0, mad[3];
        const float *ys[3];
        int k2;
        unsigned long j;
        ys[0] = g_yA;
        ys[1] = g_yB;
        ys[2] = g_yC;
        for (j = 0; j < g_ny; j++) {
            double a = g_yr[j] < 0 ? -g_yr[j] : g_yr[j];
            if (a > mar)
                mar = a;
        }
        if (mar == 0.0)
            die(1, "reference is all zeros");
        for (k2 = 0; k2 < 3; k2++) {
            double local = 0.0;
            for (j = 0; j < g_ny; j++) {
                double d2 = (double)ys[k2][j] - g_yr[j];
                if (d2 < 0)
                    d2 = -d2;
                if (d2 > local)
                    local = d2;
            }
            mad[k2] = local;
        }
        labels[0] = "fp32";
        labels[1] = "bf16";
        labels[2] = "packed";
        for (k2 = 0; k2 < 3; k2++) {
            char buf[96];
            DWORD n = 0;
            const char *kk = labels[k2];
            unsigned long long rel100 =
                (unsigned long long)(mad[k2] / mar * 1000000.0 + 0.5);
            while (*kk)
                buf[n++] = *kk++;
            buf[n++] = ' ';
            n += fmt_u64(buf + n, rel100 / 1000u); /* rel x1e-3 integer part */
            buf[n++] = '.';
            {
                DWORD fl = fmt_u64(buf + n, rel100 % 1000u);
                int zi;
                for (zi = (int)fl; zi < 3; zi++)
                    buf[n++] = '0';
                n += fl;
            }
            say("ref_rel_x1000 ");
            wboth(buf, n);
            say(" (threshold 100 = 1e-4)\n");
            if (mad[k2] / mar >= 1e-4)
                failed = 1;
        }
    }

    /* ---- RAM read bandwidth probe (fp32 buffer, single thread) ---- */
    {
        volatile unsigned long long sink = 0;
        double t[10];
        int rep;
        unsigned long n = (unsigned long)total_params; /* floats */
        unsigned long jj;
        for (rep = 0; rep < 3; rep++) {
            unsigned long long s = 0;
            const unsigned *fb32 = *(const unsigned *const *)&g_mats[0].fp32;
            for (jj = 0; jj < n; jj += 16)
                s += fb32[jj];
            sink += s;
        }
        for (rep = 0; rep < 10; rep++) {
            double t0 = now_s();
            unsigned long long s = 0;
            const unsigned *fb32 = *(const unsigned *const *)&g_mats[0].fp32;
            for (jj = 0; jj < n; jj += 16)
                s += fb32[jj];
            t[rep] = now_s() - t0;
            sink += s;
        }
        /* sort t[10] */
        {
            int a2, b2;
            for (a2 = 1; a2 < 10; a2++)
                for (b2 = a2; b2 > 0 && t[b2 - 1] > t[b2]; b2--) {
                    double tmp = t[b2];
                    t[b2] = t[b2 - 1];
                    t[b2 - 1] = tmp;
                }
        }
        {
            double bytes_touched = (double)(n / 16 * 64);
            say_fix("ram_read_bandwidth_best_GBps: ",
                    (unsigned long long)(bytes_touched / 1e9 / t[0] * 100.0 + 0.5),
                    "");
            say_fix("ram_read_bandwidth_med_GBps: ",
                    (unsigned long long)(bytes_touched / 1e9 /
                                             ((t[4] + t[5]) / 2.0) * 100.0 + 0.5),
                    "");
        }
        (void)sink;
    }

    /* ---- timing: 1 thread, then all threads; 3 warmups + 10 runs ---- */
    fns[0] = sweep_fp32;
    fns[1] = sweep_bf16;
    fns[2] = sweep_packed;
    tbytes[0] = total_bf16 * 2; /* fp32 expansion traffic */
    tbytes[1] = total_bf16;
    tbytes[2] = total_packed;
    {
        int mode;
        for (mode = 0; mode < 2; mode++) {
            unsigned saved = g_nt;
            if (mode == 0)
                g_nt = 1;
            for (ci = 0; ci < 3; ci++) {
                double t[10];
                int rep;
                for (rep = 0; rep < 3; rep++)
                    run_mt(fns[ci], g_yA);
                for (rep = 0; rep < 10; rep++) {
                    double t0 = now_s();
                    run_mt(fns[ci], g_yA);
                    t[rep] = now_s() - t0;
                }
                {
                    int a2, b2;
                    for (a2 = 1; a2 < 10; a2++)
                        for (b2 = a2; b2 > 0 && t[b2 - 1] > t[b2]; b2--) {
                            double tmp = t[b2];
                            t[b2] = t[b2 - 1];
                            t[b2 - 1] = tmp;
                        }
                }
                tbest[ci] = t[0];
                tmed[ci] = (t[4] + t[5]) / 2.0;
                {
                    char buf[128];
                    DWORD n = 0;
                    const char *kk = fns[ci] == sweep_fp32
                                         ? "fp32"
                                         : (fns[ci] == sweep_bf16 ? "bf16"
                                                                  : "packed");
                    while (*kk)
                        buf[n++] = *kk++;
                    buf[n++] = ' ';
                    buf[n++] = mode ? 'a' : '1';
                    buf[n++] = 'T';
                    buf[n++] = ':';
                    buf[n++] = ' ';
                    say(mode ? "timing all-threads " : "timing 1-thread ");
                    wboth(buf, n);
                }
                say_fix("  best_ms: ",
                        (unsigned long long)(tbest[ci] * 1000.0 * 100.0 + 0.5),
                        "");
                say_fix("  med_ms: ",
                        (unsigned long long)(tmed[ci] * 1000.0 * 100.0 + 0.5),
                        "");
                say_fix("  GBps: ",
                        (unsigned long long)((double)tbytes[ci] / tbest[ci] /
                                                 1e9 * 100.0 + 0.5),
                        "");
                say_fix("  Gweights_per_s: ",
                        (unsigned long long)((double)total_params / tbest[ci] /
                                                 1e9 * 1000.0 + 0.5),
                        "");
            }
            say_fix(mode ? "speedup_packed_vs_bf16_all: "
                         : "speedup_packed_vs_bf16_1t: ",
                    (unsigned long long)(tmed[1] / tmed[2] * 100.0 + 0.5), "");
            say_fix(mode ? "speedup_bf16_vs_fp32_all: "
                         : "speedup_bf16_vs_fp32_1t: ",
                    (unsigned long long)(tmed[0] / tmed[1] * 100.0 + 0.5), "");
            g_nt = saved;
        }
    }

    /* checksum keeps results live */
    {
        double cs = 0.0;
        unsigned long j;
        for (j = 0; j < g_ny; j++)
            cs += (double)g_yA[j];
        say_fix("y_checksum_e6: ",
                (unsigned long long)(cs * 1.0e6 < 0 ? 0 : cs * 1.0e6), "");
    }

    say(failed ? "BENCH FAIL\n" : "BENCH PASS\n");
    if (g_out)
        CloseHandle(g_out);
    ExitProcess(failed ? 1u : 0u);
}
