/* sim/native/rng.c: the keyed RNG (sim/engine.py) and the integer and double helpers.
 * Draws are pure functions of their keys, so call order does not matter; only the key tuples must match the oracle's.
 * All arithmetic on the hash is uint32 with wraparound, which is Python's `& 0xffffffff` (negative keys included:
 * (k * 0x9e3779b9) mod 2^32 equals (uint32_t)k * 0x9e3779b9u mod 2^32 for any int64 k). A world's draws go through
 * rs_wm / rs_wrng, which log each key tuple into the world's trace when the harness set one. */
#include <math.h>

#include "rs.h"

uint32_t rs_mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

uint32_t rs_fold(uint32_t h, int64_t k) {
    return rs_mix32(h ^ ((uint32_t)k * 0x9e3779b9u));
}

uint32_t rs_seed(int64_t seed) {
    return rs_mix32((uint32_t)seed);
}

uint32_t rs_prefix(uint32_t h, int n, const int64_t *keys) {
    for (int i = 0; i < n; i++) h = rs_fold(h, keys[i]);
    return h;
}

uint32_t rs_m(uint32_t h, int n, const int64_t *keys) {
    return rs_prefix(h, n, keys) >> 8;
}

double rs_draw(uint32_t h, int n, const int64_t *keys) {
    return (double)rs_m(h, n, keys) / 16777216.0;
}

/* int(d * n): the double product truncated toward zero, as Python's int() of a float (exact for every m and the n the
   simulator uses; computed the Python way anyway, so negative n and large n agree too). */
int64_t rs_pick(uint32_t m, int64_t n) {
    return (int64_t)((double)m / 16777216.0 * (double)n);
}

int64_t rs_js_round(double v) {
    return (int64_t)floor(v + 0.5);
}

int64_t rs_floordiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}

int64_t rs_mod(int64_t a, int64_t b) {
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) r += b;
    return r;
}

/* Python's a >> s: floor(a / 2^s) for either sign (C leaves >> of a negative value implementation-defined). */
int64_t rs_asr(int64_t a, int s) {
    if (s <= 0) return a;
    if (s >= 63) return a < 0 ? -1 : 0;
    return a >= 0 ? a >> s : -(int64_t)((~(uint64_t)a) >> s) - 1;
}

/* ---- draws from a world's Rng (rs.h RNG): the one call site where a traced world logs its key tuples ------------ */
uint32_t rs_wm(const World *w, int n, const int64_t *keys) {
    if (n < 0 || n > RS_DRAW_KEYS) rs_trap(__func__);
    uint32_t m = rs_m(w->seed, n, keys);
    RsTrace *t = w->trace;
    if (t) {
        if (t->n_draws < t->draws_cap) {
            RsDraw *d = &t->draws[t->n_draws++];
            d->tick = w->tick;
            d->n = (uint8_t)n;
            d->m = m;
            for (int i = 0; i < RS_DRAW_KEYS; i++) d->key[i] = i < n ? keys[i] : 0;
        } else {
            t->truncated = 1;
        }
    }
    return m;
}

double rs_wrng(const World *w, int n, const int64_t *keys) {
    return (double)rs_wm(w, n, keys) / 16777216.0;
}
