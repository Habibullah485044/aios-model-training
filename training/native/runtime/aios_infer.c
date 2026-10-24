/* aios_infer.c - AIOS native runtime oracle (test-only, never shipped).
 *
 * Plain C99. Usage: aios_infer <model_dir> <tokens.txt> <logits_out.bin>
 *
 * Reads config.txt, manifest.txt and weights.bin (BF16) from model_dir,
 * converts every weight to FP32 on load, runs one forward pass over the
 * token sequence (RMSNorm, rotate-half RoPE, causal GQA attention with QKV
 * bias, SwiGLU, tied or untied head), and writes FP32 logits
 * [n_tokens x n_vocab] to logits_out.bin.
 *
 * Single-threaded and unoptimized on purpose: this is the correctness
 * oracle, not the production kernel.
 *
 * Compile: gcc -std=c99 -Wall -Wextra -Werror -O2 aios_infer.c -o aios_infer -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NAME_MAX_LEN 96
#define CHUNK_SHORTS (1 << 22) /* 8 MiB bf16 read chunks */

typedef struct {
    char name[NAME_MAX_LEN];
    int rows;
    int cols;
    float *data; /* rows*cols floats, converted from bf16 */
} Tensor;

typedef struct {
    int n_layer;
    int d_model;
    int n_head;
    int n_kv_head;
    int d_head;
    int d_ff;
    int n_vocab;
    int tied;
    double rms_eps;
    double rope_theta;
} Config;

static void die(const char *msg)
{
    fprintf(stderr, "aios_infer: fatal: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p)
        die("out of memory");
    return p;
}

/* ---- config.txt --------------------------------------------------------- */

static int parse_config(const char *path, Config *cfg)
{
    FILE *f = fopen(path, "r");
    char key[64], val[256];
    if (!f)
        return -1;
    while (fscanf(f, "%63s %255s", key, val) == 2) {
        if (strcmp(key, "n_layer") == 0)
            cfg->n_layer = atoi(val);
        else if (strcmp(key, "d_model") == 0)
            cfg->d_model = atoi(val);
        else if (strcmp(key, "n_head") == 0)
            cfg->n_head = atoi(val);
        else if (strcmp(key, "n_kv_head") == 0)
            cfg->n_kv_head = atoi(val);
        else if (strcmp(key, "d_head") == 0)
            cfg->d_head = atoi(val);
        else if (strcmp(key, "d_ff") == 0)
            cfg->d_ff = atoi(val);
        else if (strcmp(key, "n_vocab") == 0)
            cfg->n_vocab = atoi(val);
        else if (strcmp(key, "tied") == 0)
            cfg->tied = atoi(val);
        else if (strcmp(key, "rms_eps") == 0)
            cfg->rms_eps = atof(val);
        else if (strcmp(key, "rope_theta") == 0)
            cfg->rope_theta = atof(val);
        /* model_type / n_params: informational, ignored */
    }
    fclose(f);
    if (cfg->n_layer <= 0 || cfg->d_model <= 0 || cfg->n_head <= 0 || cfg->n_kv_head <= 0 ||
        cfg->d_head <= 0 || cfg->d_ff <= 0 || cfg->n_vocab <= 0 || cfg->rms_eps <= 0.0 ||
        cfg->rope_theta <= 0.0)
        return -1;
    if (cfg->n_head % cfg->n_kv_head != 0)
        return -1;
    if (cfg->d_head * cfg->n_head != cfg->d_model && cfg->d_model <= 0)
        return -1;
    return 0;
}

/* ---- manifest + weights ------------------------------------------------- */

static Tensor *load_tensors(const char *dir, int *n_out)
{
    char path[4096];
    FILE *mf, *wf;
    Tensor *t;
    int n = 0, cap = 16, i;
    long chunk_elems;

    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    mf = fopen(path, "r");
    if (!mf)
        die("cannot open manifest.txt");
    snprintf(path, sizeof path, "%s/weights.bin", dir);
    wf = fopen(path, "rb");
    if (!wf)
        die("cannot open weights.bin");

    t = malloc((size_t)cap * sizeof *t);
    if (!t)
        die("out of memory");

    for (;;) {
        char dtype[16];
        int rows, cols;
        long long nbytes, off;
        long want, got, done;
        uint16_t *src;
        float *dst;

        if (n >= cap) {
            cap *= 2;
            t = realloc(t, (size_t)cap * sizeof *t);
            if (!t)
                die("out of memory");
        }
        if (fscanf(mf, "%d %95s %15s %d %d %lld %lld", &i, t[n].name, dtype, &rows, &cols, &off,
                   &nbytes) != 7)
            break;
        if (strcmp(dtype, "bf16") != 0)
            die("manifest dtype must be bf16");
        t[n].rows = rows;
        t[n].cols = cols;
        want = (long long)rows * cols;
        if (nbytes != want * 2)
            die("manifest nbytes mismatch");
        if (fseek(wf, (long)off, SEEK_SET) != 0)
            die("seek into weights.bin failed");
        t[n].data = xmalloc((size_t)want * sizeof(float));
        src = xmalloc((size_t)CHUNK_SHORTS * sizeof(uint16_t));
        dst = t[n].data;
        done = 0;
        while (done < want) {
            long j;
            chunk_elems = want - done;
            if (chunk_elems > CHUNK_SHORTS)
                chunk_elems = CHUNK_SHORTS;
            got = (long)fread(src, sizeof(uint16_t), (size_t)chunk_elems, wf);
            if (got != chunk_elems)
                die("short read from weights.bin");
            for (j = 0; j < got; j++) {
                uint32_t u = ((uint32_t)src[j]) << 16; /* bf16 -> fp32, lossless */
                memcpy(&dst[done + j], &u, sizeof(float));
            }
            done += got;
        }
        free(src);
        n++;
    }
    fclose(mf);
    fclose(wf);
    *n_out = n;
    return t;
}

static Tensor *find_tensor(Tensor *t, int n, const char *name)
{
    int i;
    for (i = 0; i < n; i++)
        if (strcmp(t[i].name, name) == 0)
            return &t[i];
    return NULL;
}

static Tensor *need_tensor(Tensor *t, int n, const char *name)
{
    Tensor *r = find_tensor(t, n, name);
    if (!r) {
        fprintf(stderr, "aios_infer: missing tensor %s\n", name);
        exit(1);
    }
    return r;
}

/* ---- ops ----------------------------------------------------------------- */

static void rmsnorm(float *out, const float *x, const float *w, int n, double eps)
{
    double ss = 0.0;
    float scale;
    int i;
    for (i = 0; i < n; i++)
        ss += (double)x[i] * (double)x[i];
    scale = (float)(1.0 / sqrt(ss / (double)n + eps));
    for (i = 0; i < n; i++)
        out[i] = x[i] * scale * w[i];
}

/* y[o] = sum_i W[o*in_n + i] * x[i] ; W is [out_n, in_n] row-major */
static void matmul(float *y, const float *W, const float *x, int out_n, int in_n)
{
    int o, i;
    for (o = 0; o < out_n; o++) {
        const float *row = W + (size_t)o * in_n;
        double s = 0.0;
        for (i = 0; i < in_n; i++)
            s += (double)row[i] * (double)x[i];
        y[o] = (float)s;
    }
}

static void add_bias(float *y, const float *b, int n)
{
    int i;
    if (!b)
        return;
    for (i = 0; i < n; i++)
        y[i] += b[i];
}

/* rotate-half RoPE as in HF transformers */
static void apply_rope(float *v, const float *cos_t, const float *sin_t, int n_heads, int d_head)
{
    int h, i;
    int half = d_head / 2;
    for (h = 0; h < n_heads; h++) {
        float *q = v + (size_t)h * d_head;
        for (i = 0; i < half; i++) {
            float x1 = q[i], x2 = q[i + half];
            float c = cos_t[i], s = sin_t[i];
            q[i] = x1 * c - x2 * s;
            q[i + half] = x2 * c + x1 * s;
        }
    }
}

/* ---- main ---------------------------------------------------------------- */

int main(int argc, char **argv)
{
    Config cfg;
    Tensor *t;
    int nt = 0;
    char path[4096], name[NAME_MAX_LEN + 32];
    FILE *tf, *of;
    int *tokens;
    int n_tokens = 0, cap_tokens = 16;
    int pos, l, hd, kvh, i, g;
    float *x, *hbuf, *q, *k, *v, *att, *scores, *g1, *g2, *g3, *logits;
    float *kcache, *vcache, *cos_t, *sin_t;
    Tensor **ln1, **qw, **qb, **kw, **kb, **vw, **vb, **ow, **ln2, **gw, **uw, **dw;
    Tensor *embed, *norm, *head;
    long head_elems;

    if (argc != 4) {
        fprintf(stderr, "usage: %s <model_dir> <tokens.txt> <logits_out.bin>\n", argv[0]);
        return 2;
    }

    snprintf(path, sizeof path, "%s/config.txt", argv[1]);
    memset(&cfg, 0, sizeof cfg);
    if (parse_config(path, &cfg) != 0)
        die("bad or missing config.txt");
    if (cfg.d_head * cfg.n_head != cfg.d_model)
        die("d_head * n_head != d_model");

    t = load_tensors(argv[1], &nt);

    /* tokens */
    tf = fopen(argv[2], "r");
    if (!tf)
        die("cannot open tokens.txt");
    tokens = malloc((size_t)cap_tokens * sizeof(int));
    if (!tokens)
        die("out of memory");
    {
        long long v_;
        while (fscanf(tf, "%lld", &v_) == 1) {
            if (v_ < 0 || v_ >= cfg.n_vocab)
                die("token id out of range");
            if (n_tokens >= cap_tokens) {
                cap_tokens *= 2;
                tokens = realloc(tokens, (size_t)cap_tokens * sizeof(int));
                if (!tokens)
                    die("out of memory");
            }
            tokens[n_tokens++] = (int)v_;
        }
    }
    fclose(tf);
    if (n_tokens == 0)
        die("no tokens");

    /* resolve tensors (contract order from export_weights.py) */
    embed = need_tensor(t, nt, "embed");
    norm = need_tensor(t, nt, "norm");
    head = cfg.tied ? embed : need_tensor(t, nt, "lm_head");
    if (embed->cols != cfg.d_model || head->rows != cfg.n_vocab)
        die("embed/lm_head shape mismatch");

    ln1 = xmalloc((size_t)cfg.n_layer * sizeof *ln1);
    qw = xmalloc((size_t)cfg.n_layer * sizeof *qw);
    qb = xmalloc((size_t)cfg.n_layer * sizeof *qb);
    kw = xmalloc((size_t)cfg.n_layer * sizeof *kw);
    kb = xmalloc((size_t)cfg.n_layer * sizeof *kb);
    vw = xmalloc((size_t)cfg.n_layer * sizeof *vw);
    vb = xmalloc((size_t)cfg.n_layer * sizeof *vb);
    ow = xmalloc((size_t)cfg.n_layer * sizeof *ow);
    ln2 = xmalloc((size_t)cfg.n_layer * sizeof *ln2);
    gw = xmalloc((size_t)cfg.n_layer * sizeof *gw);
    uw = xmalloc((size_t)cfg.n_layer * sizeof *uw);
    dw = xmalloc((size_t)cfg.n_layer * sizeof *dw);
    for (l = 0; l < cfg.n_layer; l++) {
        snprintf(name, sizeof name, "layers.%d.input_layernorm.weight", l);
        ln1[l] = need_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.q.weight", l);
        qw[l] = need_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.q.bias", l);
        qb[l] = find_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.k.weight", l);
        kw[l] = need_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.k.bias", l);
        kb[l] = find_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.v.weight", l);
        vw[l] = need_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.v.bias", l);
        vb[l] = find_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.o.weight", l);
        ow[l] = need_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.post_attention_layernorm.weight", l);
        ln2[l] = need_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.gate.weight", l);
        gw[l] = need_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.up.weight", l);
        uw[l] = need_tensor(t, nt, name);
        snprintf(name, sizeof name, "layers.%d.down.weight", l);
        dw[l] = need_tensor(t, nt, name);
    }

    /* buffers */
    x = xmalloc((size_t)cfg.d_model * sizeof(float));
    hbuf = xmalloc((size_t)cfg.d_model * sizeof(float));
    q = xmalloc((size_t)cfg.n_head * cfg.d_head * sizeof(float));
    k = xmalloc((size_t)cfg.n_kv_head * cfg.d_head * sizeof(float));
    v = xmalloc((size_t)cfg.n_kv_head * cfg.d_head * sizeof(float));
    att = xmalloc((size_t)cfg.n_head * cfg.d_head * sizeof(float));
    scores = xmalloc((size_t)n_tokens * sizeof(float));
    g1 = xmalloc((size_t)cfg.d_ff * sizeof(float));
    g2 = xmalloc((size_t)cfg.d_ff * sizeof(float));
    g3 = xmalloc((size_t)cfg.d_ff * sizeof(float));
    logits = xmalloc((size_t)cfg.n_vocab * sizeof(float));
    kcache = xmalloc((size_t)cfg.n_layer * cfg.n_kv_head * n_tokens * cfg.d_head * sizeof(float));
    vcache = xmalloc((size_t)cfg.n_layer * cfg.n_kv_head * n_tokens * cfg.d_head * sizeof(float));
    cos_t = xmalloc((size_t)cfg.d_head / 2 * sizeof(float));
    sin_t = xmalloc((size_t)cfg.d_head / 2 * sizeof(float));

    g = cfg.n_head / cfg.n_kv_head;

    of = fopen(argv[3], "wb");
    if (!of)
        die("cannot open logits_out.bin");

    for (pos = 0; pos < n_tokens; pos++) {
        const float *krow;
        /* embedding */
        memcpy(x, embed->data + (size_t)tokens[pos] * cfg.d_model, (size_t)cfg.d_model * sizeof(float));

        /* rope tables for this position */
        for (i = 0; i < cfg.d_head / 2; i++) {
            double freq = pow(cfg.rope_theta, -2.0 * i / cfg.d_head);
            double ang = (double)pos * freq;
            cos_t[i] = (float)cos(ang);
            sin_t[i] = (float)sin(ang);
        }

        for (l = 0; l < cfg.n_layer; l++) {
            /* attention block */
            rmsnorm(hbuf, x, ln1[l]->data, cfg.d_model, cfg.rms_eps);
            matmul(q, qw[l]->data, hbuf, cfg.n_head * cfg.d_head, cfg.d_model);
            add_bias(q, qb[l] ? qb[l]->data : NULL, cfg.n_head * cfg.d_head);
            matmul(k, kw[l]->data, hbuf, cfg.n_kv_head * cfg.d_head, cfg.d_model);
            add_bias(k, kb[l] ? kb[l]->data : NULL, cfg.n_kv_head * cfg.d_head);
            matmul(v, vw[l]->data, hbuf, cfg.n_kv_head * cfg.d_head, cfg.d_model);
            add_bias(v, vb[l] ? vb[l]->data : NULL, cfg.n_kv_head * cfg.d_head);
            apply_rope(q, cos_t, sin_t, cfg.n_head, cfg.d_head);
            apply_rope(k, cos_t, sin_t, cfg.n_kv_head, cfg.d_head);

            for (i = 0; i < cfg.n_kv_head; i++) {
                memcpy(kcache + (((size_t)l * cfg.n_kv_head + i) * n_tokens + pos) * cfg.d_head,
                       k + (size_t)i * cfg.d_head, (size_t)cfg.d_head * sizeof(float));
                memcpy(vcache + (((size_t)l * cfg.n_kv_head + i) * n_tokens + pos) * cfg.d_head,
                       v + (size_t)i * cfg.d_head, (size_t)cfg.d_head * sizeof(float));
            }

            for (hd = 0; hd < cfg.n_head; hd++) {
                float mx, sum, scale = 1.0f / sqrtf((float)cfg.d_head);
                float *qh = q + (size_t)hd * cfg.d_head;
                float *ah = att + (size_t)hd * cfg.d_head;
                kvh = hd / g;
                mx = -1e30f;
                for (i = 0; i <= pos; i++) {
                    float s = 0.0f;
                    krow = kcache + (((size_t)l * cfg.n_kv_head + kvh) * n_tokens + i) * cfg.d_head;
                    {
                        int d;
                        for (d = 0; d < cfg.d_head; d++)
                            s += qh[d] * krow[d];
                    }
                    s *= scale;
                    scores[i] = s;
                    if (s > mx)
                        mx = s;
                }
                sum = 0.0f;
                for (i = 0; i <= pos; i++) {
                    scores[i] = expf(scores[i] - mx);
                    sum += scores[i];
                }
                memset(ah, 0, (size_t)cfg.d_head * sizeof(float));
                for (i = 0; i <= pos; i++) {
                    const float *vrow = vcache + (((size_t)l * cfg.n_kv_head + kvh) * n_tokens + i) * cfg.d_head;
                    float w = scores[i] / sum;
                    int d;
                    for (d = 0; d < cfg.d_head; d++)
                        ah[d] += w * vrow[d];
                }
            }

            matmul(hbuf, ow[l]->data, att, cfg.d_model, cfg.n_head * cfg.d_head);
            for (i = 0; i < cfg.d_model; i++)
                x[i] += hbuf[i];

            /* mlp block */
            rmsnorm(hbuf, x, ln2[l]->data, cfg.d_model, cfg.rms_eps);
            matmul(g1, gw[l]->data, hbuf, cfg.d_ff, cfg.d_model);
            matmul(g2, uw[l]->data, hbuf, cfg.d_ff, cfg.d_model);
            for (i = 0; i < cfg.d_ff; i++) {
                float e = g1[i];
                g3[i] = (e / (1.0f + expf(-e))) * g2[i];
            }
            matmul(hbuf, dw[l]->data, g3, cfg.d_model, cfg.d_ff);
            for (i = 0; i < cfg.d_model; i++)
                x[i] += hbuf[i];
        }

        /* final norm + head */
        rmsnorm(hbuf, x, norm->data, cfg.d_model, cfg.rms_eps);
        matmul(logits, head->data, hbuf, cfg.n_vocab, cfg.d_model);
        if ((long)fwrite(logits, sizeof(float), (size_t)cfg.n_vocab, of) != cfg.n_vocab)
            die("short write to logits_out.bin");
    }
    fclose(of);

    head_elems = (long)n_tokens * cfg.n_vocab;
    fprintf(stderr, "aios_infer: %d tokens, %d layers, %d vocab -> %ld logits written\n",
            n_tokens, cfg.n_layer, cfg.n_vocab, head_elems);
    return 0;
}
