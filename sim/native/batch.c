/* sim/native/batch.c: the batch over world slots, CpuBatch (sim/vec.py).
 *
 * reset and step run the worlds in index order, as CpuBatch's loops do, and write every output row of every world
 * they reach straight into the caller's arrays (RsOut, sim/workers.py fields): CpuBatch.step builds its outputs
 * zeroed, so a row this step does not set is written 0. A world whose step reaches an Unsupported is flagged with its
 * evidence (vec.py: its goal, unsupported 1, its last projected observation, every other field 0) and its next
 * step fails the call (vec.py); any other error fails the call at once, as the exception propagates out of
 * CpuBatch.step. A done world auto-resets after its info row is written (vec.py), outside that handling, so an
 * Unsupported in the new episode's setup fails the call (the port keeps the oracle's raise).
 *
 * Threads (RsConfig.threads > 1): the worlds split into contiguous slices, one per thread, the caller running the
 * first. A world reads only its slot and the immutable pack and log1p tables, and writes only its slot and its output
 * rows, so no result depends on the thread count. Each slice stops at its first error, as the sequential loop does;
 * the call reports the lowest failing world's error, the one the sequential loop raises (worlds after it in other
 * slices have been stepped: a failed call's outputs and later state are not CpuBatch's, which nobody reads). */
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "rs.h"

#define RS_POOL_MAX 64
enum { JOB_RESET = 0, JOB_STEP };

typedef struct RsPool RsPool;
typedef struct { RsPool *pool; int k; } PoolArg;
struct RsPool {
    int n;                         /* slices; slice 0 runs on the calling thread */
    pthread_t thread[RS_POOL_MAX];
    PoolArg arg[RS_POOL_MAX];
    pthread_mutex_t mu;
    pthread_cond_t go, idle;
    uint64_t gen;                  /* job generation */
    int busy, quit;
    /* the job */
    int kind;
    RsBatch *b;
    const int32_t *actions;
    const RsOut *out;
    int failed[RS_POOL_MAX];
    RsErr err[RS_POOL_MAX];
    int32_t err_world[RS_POOL_MAX];
};

/* ---- output rows ------------------------------------------------------------------------------------------------ */
static void write_obs(const RsOut *out, int32_t i, const uint8_t *row) {
    for (int k = 0; k < RS_OBS_N_KEYS; k++)
        memcpy((uint8_t *)out->obs[k] + (size_t)i * (size_t)RS_OBS_NBYTES[k], row + RS_OBS_OFFSET[k],
               (size_t)RS_OBS_NBYTES[k]);
}

static void write_info(const RsOut *out, int32_t i, const RsStepInfo *f) {
    ((int32_t *)out->info[RS_INFO_GOAL])[i] = f->goal;
    ((uint8_t *)out->info[RS_INFO_UNSUPPORTED])[i] = f->unsupported;
    ((int32_t *)out->info[RS_INFO_TERMINAL])[i] = f->terminal;
    ((float *)out->info[RS_INFO_EPISODE_RETURN])[i] = f->episode_return;
    ((int32_t *)out->info[RS_INFO_EPISODE_DECISIONS])[i] = f->episode_decisions;
    ((int32_t *)out->info[RS_INFO_EPISODE_TICKS])[i] = f->episode_ticks;
    ((int32_t *)out->info[RS_INFO_OUTCOMES])[i] = f->outcomes;
    ((int32_t *)out->info[RS_INFO_DECISION_TICKS])[i] = f->decision_ticks;
    memcpy(out->made + (size_t)i * 3, f->made, sizeof f->made);
    out->kills[i] = f->kills;
    memcpy(out->facts + (size_t)i * RS_N_FACTS, f->facts, sizeof f->facts);
    out->reward[i] = f->reward;
    out->done[i] = f->done;
}

/* The info fields' types as write_info stores them (sim/spec.py INFO_FIELDS, through rs_gen.h). */
static int info_types_ok(void) {
    static const struct { int k; char c; } TYPES[] = {
        {RS_INFO_GOAL, 'i'}, {RS_INFO_UNSUPPORTED, '?'}, {RS_INFO_TERMINAL, 'i'}, {RS_INFO_EPISODE_RETURN, 'f'},
        {RS_INFO_EPISODE_DECISIONS, 'i'}, {RS_INFO_EPISODE_TICKS, 'i'}, {RS_INFO_OUTCOMES, 'i'},
        {RS_INFO_DECISION_TICKS, 'i'}};
    if (RS_INFO_N != (int)(sizeof TYPES / sizeof *TYPES)) return 0;
    for (size_t k = 0; k < sizeof TYPES / sizeof *TYPES; k++)
        if (RS_INFO_FORMAT[TYPES[k].k] != TYPES[k].c) return 0;
    return 1;
}

/* ---- one world -------------------------------------------------------------------------------------------------- */
/* CpuBatch.reset's _episode(b) for world i (vec.py): episode 0, not failed, its first episode. */
static int reset_world(RsBatch *b, int32_t i, const RsOut *out, RsErr *err) {
    RsSlot *s = &b->slots[i];
    s->episode = 0;
    s->failed = 0;
    memset(&s->fail, 0, sizeof s->fail);
    s->fail_decision = s->fail_tick = s->fail_action = 0;
    s->n_late = 0;                     /* no row of this world is written until its episode begins */
    if (b->external_starts) return 0;  /* the caller begins it (rs_batch_begin) */
    if (rs_episode_new(s, b->key)) {
        *err = s->w.err;
        return 1;
    }
    write_obs(out, i, s->obs);
    return 0;
}

/* CpuBatch.step's loop body for world i (vec.py). */
static int step_world(RsBatch *b, int32_t i, int32_t action, const RsOut *out, RsErr *err) {
    RsSlot *s = &b->slots[i];
    if (s->failed) return RS_FAIL(err, RS_E_WORLD_FAILED, i);
    RsStepInfo info;
    if (rs_episode_step(s, action, &info)) {
        if (s->w.err.code != RS_UNSUPPORTED) {
            *err = s->w.err;
            return 1;
        }
        /* the evidence (vec.py): the reason, task.decisions, world.tick mid-cycle, the action */
        s->failed = 1;
        s->fail = s->w.err;
        s->fail_decision = s->ep.decisions, s->fail_tick = s->w.tick, s->fail_action = action;
        memset(&info, 0, sizeof info);
        info.goal = s->ep.goal;
        info.unsupported = 1;
        write_info(out, i, &info);
        write_obs(out, i, s->obs);
        return 0;
    }
    write_info(out, i, &info);
    if (info.done) {
        s->episode++;
        if (!b->external_starts && rs_episode_new(s, b->key)) {
            *err = s->w.err;
            return 1;
        }
    }
    write_obs(out, i, s->obs);
    return 0;
}

static int run_range(RsBatch *b, int kind, int32_t lo, int32_t hi, const int32_t *actions, const RsOut *out,
                     RsErr *err, int32_t *err_world) {
    for (int32_t i = lo; i < hi; i++) {
        memset(err, 0, sizeof *err);
        int bad = kind == JOB_RESET ? reset_world(b, i, out, err) : step_world(b, i, actions[i], out, err);
        if (bad) {
            *err_world = i;
            return 1;
        }
    }
    return 0;
}

/* ---- the pool ----------------------------------------------------------------------------------------------------- */
static void slice_of(const RsPool *pool, int k, int32_t n, int32_t *lo, int32_t *hi) {
    *lo = (int32_t)((int64_t)n * k / pool->n);
    *hi = (int32_t)((int64_t)n * (k + 1) / pool->n);
}

static void run_slice(RsPool *pool, int k) {
    int32_t lo, hi;
    slice_of(pool, k, pool->b->n, &lo, &hi);
    pool->failed[k] = run_range(pool->b, pool->kind, lo, hi, pool->actions, pool->out, &pool->err[k],
                                &pool->err_world[k]);
}

static void *pool_main(void *arg) {
    RsPool *pool = ((PoolArg *)arg)->pool;
    int k = ((PoolArg *)arg)->k;
    uint64_t seen = 0;
    pthread_mutex_lock(&pool->mu);
    for (;;) {
        while (!pool->quit && pool->gen == seen) pthread_cond_wait(&pool->go, &pool->mu);
        if (pool->quit) break;
        seen = pool->gen;
        pthread_mutex_unlock(&pool->mu);
        run_slice(pool, k);
        pthread_mutex_lock(&pool->mu);
        if (--pool->busy == 0) pthread_cond_signal(&pool->idle);
    }
    pthread_mutex_unlock(&pool->mu);
    return NULL;
}

static void pool_free(RsPool *pool, int started) {
    pthread_mutex_lock(&pool->mu);
    pool->quit = 1;
    pthread_cond_broadcast(&pool->go);
    pthread_mutex_unlock(&pool->mu);
    for (int k = 1; k < started; k++) pthread_join(pool->thread[k], NULL);
    pthread_cond_destroy(&pool->go);
    pthread_cond_destroy(&pool->idle);
    pthread_mutex_destroy(&pool->mu);
    free(pool);
}

static RsPool *pool_new(int n) {
    RsPool *pool = calloc(1, sizeof *pool);
    if (!pool) return NULL;
    pool->n = n;
    if (pthread_mutex_init(&pool->mu, NULL)) {
        free(pool);
        return NULL;
    }
    pthread_cond_init(&pool->go, NULL);
    pthread_cond_init(&pool->idle, NULL);
    for (int k = 1; k < n; k++) {
        pool->arg[k] = (PoolArg){pool, k};
        if (pthread_create(&pool->thread[k], NULL, pool_main, &pool->arg[k])) {
            pool_free(pool, k);
            return NULL;
        }
    }
    return pool;
}

/* A job over every world: the slices in parallel, then the lowest failing world's error. */
static int run_all(RsBatch *b, int kind, const int32_t *actions, const RsOut *out) {
    memset(&b->err, 0, sizeof b->err);
    b->err_world = -1;
    RsPool *pool = b->pool;
    if (!pool) return run_range(b, kind, 0, b->n, actions, out, &b->err, &b->err_world);
    pthread_mutex_lock(&pool->mu);
    pool->kind = kind, pool->b = b, pool->actions = actions, pool->out = out;
    pool->busy = pool->n - 1;
    pool->gen++;
    pthread_cond_broadcast(&pool->go);
    pthread_mutex_unlock(&pool->mu);
    run_slice(pool, 0);
    pthread_mutex_lock(&pool->mu);
    while (pool->busy > 0) pthread_cond_wait(&pool->idle, &pool->mu);
    pthread_mutex_unlock(&pool->mu);
    for (int k = 0; k < pool->n; k++)
        if (pool->failed[k]) {
            b->err = pool->err[k];
            b->err_world = pool->err_world[k];
            return 1;
        }
    return 0;
}

/* ---- the batch ---------------------------------------------------------------------------------------------------- */
int rs_batch_new(const RsPack *P, const RsConfig *cfg, const float *log1p, RsBatch **out, RsErr *err) {
    *out = NULL;
    memset(err, 0, sizeof *err);
    if (!log1p) return RS_FAIL_TEXT(err, RS_E_CONTRACT, "a batch needs the log1p feature tables");
    if (!P->str_row) return RS_FAIL_TEXT(err, RS_E_PACK, "the pack has no STR_ROW section (names-v1 rows)");
    if (cfg->task != RS_TASK_V2 && cfg->task != RS_TASK_V3 && cfg->task != RS_TASK_V5)
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "the native engine runs goal-activity-v2, v3 and v5");
    if (cfg->n < 1 || cfg->first < 0)
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "a batch needs n >= 1 worlds, first >= 0");
    if (!(cfg->n_goals == RS_N_GOALS || (cfg->n_goals == 1 && cfg->goal >= 0 && cfg->goal < RS_N_GOALS)))
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "a goal set is all four goals in order or one of them");
    if (cfg->task != RS_TASK_V2 && cfg->horizon != RS_HORIZON_TICKS)
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "a combat task keeps the gathering goals' horizon");
    if (cfg->threads < 0 || cfg->threads > RS_POOL_MAX)
        return RS_FAIL_TEXT(err, RS_E_CONTRACT, "a batch runs on 0 to 64 threads");
    if (!info_types_ok()) return RS_FAIL_TEXT(err, RS_E_INTERNAL, "sim/spec.py INFO_FIELDS changed type or order");
    RsBatch *b = calloc(1, sizeof *b);
    RsSlot *slots = b ? calloc((size_t)cfg->n, sizeof *slots) : NULL;
    if (!slots) {
        free(b);
        return RS_OVERFLOW(err, "memory for the batch's world slots");
    }
    b->P = P;
    b->cfg = *cfg;
    b->log1p = log1p;
    b->n = cfg->n;
    b->slots = slots;
    b->err_world = -1;
    for (int32_t i = 0; i < cfg->n; i++) {
        slots[i].P = P;
        slots[i].cfg = &b->cfg;
        slots[i].log1p = log1p;
        slots[i].slot = cfg->first + i;
        slots[i].goal = cfg->n_goals == 1 ? cfg->goal : (cfg->first + i) % RS_N_GOALS;  /* spec.slot_goal */
    }
    int threads = cfg->threads < cfg->n ? cfg->threads : cfg->n;
    if (threads > 1 && !(b->pool = pool_new(threads))) {
        rs_batch_free(b);
        return RS_OVERFLOW(err, "threads for the batch's pool");
    }
    *out = b;
    return 0;
}

void rs_batch_free(RsBatch *b) {
    if (!b) return;
    if (b->pool) pool_free(b->pool, b->pool->n);
    free(b->slots);
    free(b);
}

int rs_batch_reset(RsBatch *b, int64_t key, const RsOut *out) {
    b->key = key;
    b->has_key = 1;
    return run_all(b, JOB_RESET, NULL, out);
}

int rs_batch_step(RsBatch *b, const int32_t *actions, const RsOut *out) {
    if (!b->has_key) {
        memset(&b->err, 0, sizeof b->err);
        b->err_world = -1;
        return RS_FAIL_TEXT(&b->err, RS_E_CONTRACT, "a step before the first reset");
    }
    return run_all(b, JOB_STEP, actions, out);
}

int rs_batch_check(const RsBatch *b, const RsOut *out) {
    size_t n = (size_t)b->n;
    for (int k = 0; k < RS_OBS_N_KEYS; k++) {
        if (RS_OBS_FORMAT[k] != 'f') continue;
        const float *v = out->obs[k];   /* np.isfinite: |x| <= FLT_MAX, which NaN and infinities fail */
        size_t count = n * (size_t)RS_OBS_NELEM[k];
        int bad = 0;
        for (size_t j = 0; j < count; j++) bad |= !(fabsf(v[j]) <= FLT_MAX);
        if (bad) return k;
    }
    const int8_t *mask = out->obs[RS_OBS_MASK];
    for (size_t i = 0; i < n; i++) {
        const int8_t *row = mask + i * RS_N_ACTIONS;
        int any = 0, other = 0;
        for (int a = 0; a < RS_N_ACTIONS; a++) any |= row[a] == 1, other |= row[a] != 0 && row[a] != 1;
        if (other || !any) return RS_OBS_N_KEYS;
    }
    return -1;
}

int rs_world_step(RsBatch *b, int32_t world, int32_t action, const RsOut *out) {
    memset(&b->err, 0, sizeof b->err);
    b->err_world = world;
    if (world < 0 || world >= b->n || !b->has_key)
        return RS_FAIL_TEXT(&b->err, RS_E_CONTRACT, "a world step outside the batch or before its reset");
    if (step_world(b, world, action, out, &b->err)) return 1;
    b->err_world = -1;
    return 0;
}

int rs_batch_begin(RsBatch *b, int32_t world, const RsStart *start, const RsOut *out) {
    memset(&b->err, 0, sizeof b->err);
    b->err_world = world;
    if (world < 0 || world >= b->n || !b->has_key)
        return RS_FAIL_TEXT(&b->err, RS_E_CONTRACT, "an episode begun outside the batch or before its reset");
    RsSlot *s = &b->slots[world];
    s->failed = 0;
    memset(&s->fail, 0, sizeof s->fail);
    if (start ? rs_episode_begin(s, start) : rs_episode_new(s, b->key)) {
        b->err = s->w.err;
        return 1;
    }
    write_obs(out, world, s->obs);
    b->err_world = -1;
    return 0;
}
