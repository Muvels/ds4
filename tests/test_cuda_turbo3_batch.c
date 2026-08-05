/* Device-level oracle for packed Turbo3 batched decode attention.
 *
 * Runs on a single CUDA device with no model file. Four synthetic sessions
 * (two dense, two indexer-selected) are pushed through
 * ds4_gpu_attention_decode_rows_rope_tensor twice: once with float
 * compressor rows dequantized from the packed cache, once with the packed
 * Turbo3 rows consumed directly in-kernel. Both calls must agree, packed
 * caches must round-trip within codec tolerance, sessions must stay
 * isolated, and malformed packed descriptors must be rejected.
 *
 * Run with:
 *   make test-cuda-turbo3-batch CUDA_ARCH=...
 * It is also part of `make cuda-regression`.
 */

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ROWS 4u
#define TEST_HEADS 8u
#define TEST_HEAD_DIM 512u
#define TEST_ROT 64u
#define TEST_RAW_CAP 64u
#define TEST_N_RAW 48u
#define TEST_N_COMP 600u
#define TEST_TOP_K 200u
#define TURBO3_ROW_BYTES (TEST_HEAD_DIM * 3u / 8u + TEST_HEAD_DIM / 64u)

static uint64_t rng_state = 0x243f6a8885a308d3ull;

static float rng_float(void) {
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t bits = (uint32_t)(rng_state >> 32);
    return ((float)bits / 4294967296.0f) * 2.0f - 1.0f;
}

typedef struct {
    ds4_gpu_tensor *raw;
    ds4_gpu_tensor *packed;
    ds4_gpu_tensor *comp_ref;   /* dequant(packed): what the F32 mode reads */
    ds4_gpu_tensor *topk;
    float          *comp_src_host;
    uint32_t        pos;
    uint32_t        indexed;
} test_session;

static int fill_row_common(ds4_gpu_attention_decode_row *r,
                           const test_session *s,
                           int packed_mode) {
    memset(r, 0, sizeof(*r));
    r->raw_kv = (uint64_t)(uintptr_t)s->raw->ptr;
    r->comp_kv = packed_mode
        ? (uint64_t)(uintptr_t)s->packed->ptr
        : (uint64_t)(uintptr_t)s->comp_ref->ptr;
    r->pos = s->pos;
    r->n_raw = TEST_N_RAW;
    r->raw_cap = TEST_RAW_CAP;
    r->raw_start = 5u;
    r->n_comp = TEST_N_COMP;
    if (s->indexed) {
        r->topk = (uint64_t)(uintptr_t)s->topk->ptr;
        r->top_k = TEST_TOP_K;
        r->window = 32u;
        r->ratio = 4u;
        r->indexed = 1u;
    }
    r->comp_fmt = packed_mode ? DS4_GPU_COMP_CACHE_FMT_TURBO3
                              : DS4_GPU_COMP_CACHE_FMT_F32;
    r->comp_row_bytes = packed_mode ? (uint32_t)TURBO3_ROW_BYTES : 0u;
    return 0;
}

static int session_create(test_session *s, uint32_t index, int indexed) {
    const uint64_t raw_count = (uint64_t)TEST_RAW_CAP * TEST_HEAD_DIM;
    const uint64_t comp_count = (uint64_t)TEST_N_COMP * TEST_HEAD_DIM;
    memset(s, 0, sizeof(*s));
    s->indexed = indexed ? 1u : 0u;
    /* Indexed rows expose visible_comp = (pos + 1) / ratio rows; choose a
     * position that makes the whole synthetic compressor cache visible. */
    s->pos = indexed ? TEST_N_COMP * 4u - 1u : 1000u + index * 17u;

    float *raw_host = (float *)malloc(raw_count * sizeof(float));
    s->comp_src_host = (float *)malloc(comp_count * sizeof(float));
    int32_t *topk_host = (int32_t *)malloc(TEST_TOP_K * sizeof(int32_t));
    if (!raw_host || !s->comp_src_host || !topk_host) {
        free(topk_host);
        free(raw_host);
        return 1;
    }
    for (uint64_t i = 0; i < raw_count; i++) raw_host[i] = rng_float() * 0.5f;
    for (uint64_t i = 0; i < comp_count; i++) {
        s->comp_src_host[i] = rng_float();
    }
    for (uint32_t i = 0; i < TEST_TOP_K; i++) {
        /* 7 is coprime with TEST_N_COMP, so the selections stay distinct. */
        topk_host[i] = (int32_t)((i * 7u) % TEST_N_COMP);
    }

    s->raw = ds4_gpu_tensor_alloc(raw_count * sizeof(float));
    s->packed = ds4_gpu_tensor_alloc((uint64_t)TEST_N_COMP * TURBO3_ROW_BYTES);
    s->comp_ref = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    s->topk = ds4_gpu_tensor_alloc((uint64_t)TEST_TOP_K * sizeof(int32_t));
    ds4_gpu_tensor *comp_src = ds4_gpu_tensor_alloc(comp_count * sizeof(float));
    int rc = 1;
    if (s->raw && s->packed && s->comp_ref && s->topk && comp_src &&
        ds4_gpu_tensor_write(s->raw, 0, raw_host,
                             raw_count * sizeof(float)) &&
        ds4_gpu_tensor_write(comp_src, 0, s->comp_src_host,
                             comp_count * sizeof(float)) &&
        ds4_gpu_tensor_write(s->topk, 0, topk_host,
                             (uint64_t)TEST_TOP_K * sizeof(int32_t)) &&
        ds4_gpu_dsv4_turbo3_comp_pack_tensor(comp_src, s->packed,
                                             TEST_N_COMP, 0,
                                             TEST_HEAD_DIM,
                                             TURBO3_ROW_BYTES) &&
        ds4_gpu_dsv4_turbo3_comp_dequant_to_scratch_tensor(
                s->packed, s->comp_ref, 0, TEST_N_COMP,
                TEST_HEAD_DIM, TURBO3_ROW_BYTES) &&
        ds4_gpu_synchronize()) {
        rc = 0;
    }
    ds4_gpu_tensor_free(comp_src);
    free(topk_host);
    free(raw_host);
    return rc;
}

static void session_destroy(test_session *s) {
    ds4_gpu_tensor_free(s->topk);
    ds4_gpu_tensor_free(s->comp_ref);
    ds4_gpu_tensor_free(s->packed);
    ds4_gpu_tensor_free(s->raw);
    free(s->comp_src_host);
    memset(s, 0, sizeof(*s));
}

/* Codec sanity: dequant(pack(x)) must stay within the lossy-codec envelope
 * of x, and the packed representation must actually be consulted (a zeroed
 * packed row cannot reproduce the source). */
static int check_pack_roundtrip(const test_session *s) {
    const uint64_t comp_count = (uint64_t)TEST_N_COMP * TEST_HEAD_DIM;
    float *ref_host = (float *)malloc(comp_count * sizeof(float));
    if (!ref_host) return 1;
    int rc = 0;
    if (!ds4_gpu_tensor_read(s->comp_ref, 0, ref_host,
                             comp_count * sizeof(float))) {
        rc = 1;
    }
    double worst_rel = 0.0;
    double rel_sum = 0.0;
    for (uint32_t row = 0; rc == 0 && row < TEST_N_COMP; row++) {
        const float *src = s->comp_src_host + (uint64_t)row * TEST_HEAD_DIM;
        const float *ref = ref_host + (uint64_t)row * TEST_HEAD_DIM;
        double err_sq = 0.0;
        double src_sq = 0.0;
        for (uint32_t d = 0; d < TEST_HEAD_DIM; d++) {
            const double delta = (double)ref[d] - (double)src[d];
            err_sq += delta * delta;
            src_sq += (double)src[d] * (double)src[d];
        }
        const double rel = src_sq > 0.0 ? sqrt(err_sq / src_sq) : 0.0;
        rel_sum += rel;
        if (rel > worst_rel) worst_rel = rel;
        if (rel > 0.5) {
            fprintf(stderr,
                    "FAIL: turbo3 round-trip row=%u relative error %.3f\n",
                    row, rel);
            rc = 1;
        }
    }
    if (rc == 0 && rel_sum / TEST_N_COMP > 0.35) {
        fprintf(stderr,
                "FAIL: turbo3 round-trip mean relative error %.3f\n",
                rel_sum / TEST_N_COMP);
        rc = 1;
    }
    if (rc == 0) {
        fprintf(stderr,
                "turbo3 round-trip: rows=%u mean_rel=%.3f worst_rel=%.3f\n",
                TEST_N_COMP, rel_sum / TEST_N_COMP, worst_rel);
    }
    free(ref_host);
    return rc;
}

static int run_rows(float *heads_host,
                    test_session *sessions,
                    ds4_gpu_tensor *heads,
                    ds4_gpu_tensor *q,
                    const float *sinks,
                    int packed_mode) {
    const uint64_t head_count =
        (uint64_t)TEST_ROWS * TEST_HEADS * TEST_HEAD_DIM;
    ds4_gpu_attention_decode_row rows[TEST_ROWS];
    for (uint32_t i = 0; i < TEST_ROWS; i++) {
        fill_row_common(&rows[i], &sessions[i], packed_mode);
    }
    if (!ds4_gpu_attention_decode_rows_rope_tensor(
                heads, sinks, (uint64_t)TEST_HEADS * sizeof(float), 0,
                q, rows, TEST_ROWS, TEST_HEADS, TEST_HEAD_DIM, TEST_ROT,
                0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) ||
        !ds4_gpu_synchronize() ||
        !ds4_gpu_tensor_read(heads, 0, heads_host,
                             head_count * sizeof(float))) {
        return 1;
    }
    return 0;
}

static int compare_heads(const float *a, const float *b,
                         uint32_t row_first, uint32_t row_last,
                         const char *what, int expect_equal) {
    const uint64_t per_row = (uint64_t)TEST_HEADS * TEST_HEAD_DIM;
    double worst = 0.0;
    double worst_mag = 0.0;
    uint64_t nonexact = 0;
    for (uint32_t row = row_first; row <= row_last; row++) {
        const float *pa = a + (uint64_t)row * per_row;
        const float *pb = b + (uint64_t)row * per_row;
        for (uint64_t i = 0; i < per_row; i++) {
            const double delta = fabs((double)pa[i] - (double)pb[i]);
            const double mag = fabs((double)pa[i]);
            if (pa[i] != pb[i]) nonexact++;
            if (delta > worst) worst = delta;
            if (mag > worst_mag) worst_mag = mag;
        }
    }
    const double tolerance = 1e-5 * (worst_mag > 1.0 ? worst_mag : 1.0);
    if (expect_equal) {
        if (worst > tolerance) {
            fprintf(stderr,
                    "FAIL: %s rows=%u..%u worst_abs=%g nonexact=%llu\n",
                    what, row_first, row_last, worst,
                    (unsigned long long)nonexact);
            return 1;
        }
        fprintf(stderr, "%s rows=%u..%u worst_abs=%g nonexact=%llu\n",
                what, row_first, row_last, worst,
                (unsigned long long)nonexact);
        return 0;
    }
    if (nonexact == 0) {
        fprintf(stderr, "FAIL: %s rows=%u..%u unexpectedly identical\n",
                what, row_first, row_last);
        return 1;
    }
    return 0;
}

static int check_packed_rejects(test_session *sessions,
                                ds4_gpu_tensor *heads,
                                ds4_gpu_tensor *q,
                                const float *sinks) {
    ds4_gpu_attention_decode_row rows[TEST_ROWS];
    for (uint32_t i = 0; i < TEST_ROWS; i++) {
        fill_row_common(&rows[i], &sessions[i], 1);
    }
    rows[1].comp_row_bytes = (uint32_t)TURBO3_ROW_BYTES - 1u;
    if (ds4_gpu_attention_decode_rows_rope_tensor(
                heads, sinks, (uint64_t)TEST_HEADS * sizeof(float), 0,
                q, rows, TEST_ROWS, TEST_HEADS, TEST_HEAD_DIM, TEST_ROT,
                0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)) {
        fprintf(stderr, "FAIL: wrong turbo3 row stride was accepted\n");
        return 1;
    }
    fill_row_common(&rows[1], &sessions[1], 1);
    rows[2].comp_fmt = 2u;
    if (ds4_gpu_attention_decode_rows_rope_tensor(
                heads, sinks, (uint64_t)TEST_HEADS * sizeof(float), 0,
                q, rows, TEST_ROWS, TEST_HEADS, TEST_HEAD_DIM, TEST_ROT,
                0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)) {
        fprintf(stderr, "FAIL: unknown comp cache format was accepted\n");
        return 1;
    }
    fill_row_common(&rows[2], &sessions[2], 1);
    /* A dense n_comp large enough to wrap raw_count + n_comp around 2^32
     * must be rejected before the addition. */
    rows[0].n_comp = UINT32_MAX;
    if (ds4_gpu_attention_decode_rows_rope_tensor(
                heads, sinks, (uint64_t)TEST_HEADS * sizeof(float), 0,
                q, rows, TEST_ROWS, TEST_HEADS, TEST_HEAD_DIM, TEST_ROT,
                0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)) {
        fprintf(stderr, "FAIL: overflowing dense n_comp was accepted\n");
        return 1;
    }
    return 0;
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "FAIL: ds4_gpu_init\n");
        return 1;
    }

    test_session sessions[TEST_ROWS];
    memset(sessions, 0, sizeof(sessions));
    const uint64_t head_count =
        (uint64_t)TEST_ROWS * TEST_HEADS * TEST_HEAD_DIM;
    float sinks[TEST_HEADS];
    float *q_host = (float *)malloc(head_count * sizeof(float));
    float *heads_f32 = (float *)malloc(head_count * sizeof(float));
    float *heads_packed = (float *)malloc(head_count * sizeof(float));
    float *heads_mutated = (float *)malloc(head_count * sizeof(float));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(head_count * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(head_count * sizeof(float));
    int rc = 1;
    if (!q_host || !heads_f32 || !heads_packed || !heads_mutated ||
        !heads || !q) {
        fprintf(stderr, "FAIL: allocation\n");
        goto done;
    }
    for (uint32_t h = 0; h < TEST_HEADS; h++) sinks[h] = 0.05f * (float)h;
    for (uint64_t i = 0; i < head_count; i++) q_host[i] = rng_float();
    if (!ds4_gpu_tensor_write(q, 0, q_host, head_count * sizeof(float))) {
        fprintf(stderr, "FAIL: q upload\n");
        goto done;
    }
    for (uint32_t i = 0; i < TEST_ROWS; i++) {
        if (session_create(&sessions[i], i, i >= 2) != 0) {
            fprintf(stderr, "FAIL: session %u setup\n", i);
            goto done;
        }
    }

    if (check_pack_roundtrip(&sessions[0]) != 0) goto done;

    if (run_rows(heads_f32, sessions, heads, q, sinks, 0) != 0) {
        fprintf(stderr, "FAIL: float-mode rows call\n");
        goto done;
    }
    if (run_rows(heads_packed, sessions, heads, q, sinks, 1) != 0) {
        fprintf(stderr, "FAIL: packed-mode rows call\n");
        goto done;
    }
    if (compare_heads(heads_f32, heads_packed, 0, 1,
                      "packed dense parity", 1) != 0 ||
        compare_heads(heads_f32, heads_packed, 2, 3,
                      "packed indexed parity", 1) != 0) {
        goto done;
    }

    /* Session isolation: perturbing one packed cache must change only that
     * session's output. */
    {
        const uint64_t comp_count = (uint64_t)TEST_N_COMP * TEST_HEAD_DIM;
        ds4_gpu_tensor *mutated_src =
            ds4_gpu_tensor_alloc(comp_count * sizeof(float));
        float *mutated_host = (float *)malloc(comp_count * sizeof(float));
        if (!mutated_src || !mutated_host) {
            fprintf(stderr, "FAIL: mutation allocation\n");
            ds4_gpu_tensor_free(mutated_src);
            free(mutated_host);
            goto done;
        }
        for (uint64_t i = 0; i < comp_count; i++) {
            mutated_host[i] = -sessions[1].comp_src_host[i] + 0.25f;
        }
        int mutated_ok =
            ds4_gpu_tensor_write(mutated_src, 0, mutated_host,
                                 comp_count * sizeof(float)) &&
            ds4_gpu_dsv4_turbo3_comp_pack_tensor(
                    mutated_src, sessions[1].packed, TEST_N_COMP, 0,
                    TEST_HEAD_DIM, TURBO3_ROW_BYTES) &&
            ds4_gpu_synchronize();
        ds4_gpu_tensor_free(mutated_src);
        free(mutated_host);
        if (!mutated_ok) {
            fprintf(stderr, "FAIL: packed cache mutation\n");
            goto done;
        }
    }
    if (run_rows(heads_mutated, sessions, heads, q, sinks, 1) != 0) {
        fprintf(stderr, "FAIL: mutated packed rows call\n");
        goto done;
    }
    if (compare_heads(heads_packed, heads_mutated, 1, 1,
                      "isolation (mutated session)", 0) != 0 ||
        compare_heads(heads_packed, heads_mutated, 0, 0,
                      "isolation (untouched dense session)", 1) != 0 ||
        compare_heads(heads_packed, heads_mutated, 2, 3,
                      "isolation (untouched indexed sessions)", 1) != 0) {
        goto done;
    }

    if (check_packed_rejects(sessions, heads, q, sinks) != 0) goto done;

    rc = 0;
    puts("test_cuda_turbo3_batch PASS");

done:
    ds4_gpu_tensor_free(q);
    ds4_gpu_tensor_free(heads);
    for (uint32_t i = 0; i < TEST_ROWS; i++) session_destroy(&sessions[i]);
    free(heads_mutated);
    free(heads_packed);
    free(heads_f32);
    free(q_host);
    ds4_gpu_cleanup();
    return rc;
}
