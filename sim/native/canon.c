/* sim/native/canon.c: canonical state records for the per-tick equivalence harness and the world
 * traces they feed. rs_canon is the World's part (world.c rs_world_canon) followed by the agent's and the episode's
 * fields as sim/native_harness.py Canon writes them (its docstring states each variable field's layout; Canon.agent
 * and the RS_CF_EPISODE row of Canon.__call__ are the Python this mirrors). The tick hash is zlib.crc32 over the
 * record's little-endian int64 bytes. A trace is the harness's buffer: a world logs its draws into it (rng.c rs_wm),
 * the agent's tick hook appends one row per cycle (rs_trace_tick), and the harness takes the rows after each batch
 * call (rs_trace_clear); a full buffer or a failed record sets truncated and keeps no later row. */
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

#include "rs.h"

/* ---- the record ------------------------------------------------------------------------------------------------- */
typedef struct { int64_t *buf; int cap, n, at; } Emit;
static void em_v(Emit *e, int64_t v) {
    if (e->n < e->cap) e->buf[e->n] = v;
    e->n++;
}
static void em_begin(Emit *e, int id) { em_v(e, id); e->at = e->n; em_v(e, 0); }
static void em_end(Emit *e) { if (e->at < e->cap) e->buf[e->at] = e->n - e->at - 1; }

static int popcount64(uint64_t v) {
    int n = 0;
    for (; v; v &= v - 1) n++;
    return n;
}

/* A message's text string id (agent texts keep message ids). */
static int64_t msg_text(const RsPack *P, int32_t msg) { return msg >= 0 && msg < P->n_msgs ? P->msgs[msg].text : -1; }

/* Canon.key: an affordance key as (kind, id, x, z, level, option string id). */
static void em_key(Emit *e, const RsAffKey *k) {
    em_v(e, k->kind), em_v(e, k->id), em_v(e, k->x), em_v(e, k->z), em_v(e, k->level), em_v(e, k->option_lc);
}

/* An inventory as listed(): n, then slot, id, count per occupied slot (Canon.agent's item_state parse). */
static void em_items(Emit *e, const Slot *items, int cap) {
    int n = 0;
    for (int i = 0; i < cap; i++) n += items[i].id >= 0;
    em_v(e, n);
    for (int i = 0; i < cap; i++)
        if (items[i].id >= 0) em_v(e, i), em_v(e, items[i].id), em_v(e, items[i].count);
}

/* Lexicographic order of two int64 rows (Python's tuple order). */
static int row_less(const int64_t *a, const int64_t *b, int k) {
    for (int i = 0; i < k; i++)
        if (a[i] != b[i]) return a[i] < b[i];
    return 0;
}

/* Rows of k values sorted by their tuples (insertion sort: a handful of rows), then emitted. */
static void em_sorted(Emit *e, int64_t *rows, int n, int k) {
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && row_less(&rows[j * k], &rows[(j - 1) * k], k); j--)
            for (int v = 0; v < k; v++) {
                int64_t t = rows[j * k + v];
                rows[j * k + v] = rows[(j - 1) * k + v];
                rows[(j - 1) * k + v] = t;
            }
    for (int i = 0; i < n * k; i++) em_v(e, rows[i]);
}

/* Canon.agent (sim/native_harness.py): RS_CF_AGENT .. RS_CF_AGENT_TEXTS. */
static void em_agent(Emit *e, const RsSlot *s) {
    const Agent *a = &s->a;
    const World *w = &s->w;
    const RsOption *o = &a->option;
    em_begin(e, RS_CF_AGENT);
    em_v(e, a->decision), em_v(e, a->charged), em_v(e, a->episode_ticks), em_v(e, a->last_mask_sum), em_v(e, a->pace);
    em_v(e, a->inv_page), em_v(e, a->bank_page), em_v(e, a->bank_pages), em_v(e, a->style_stale);
    em_v(e, a->has_using ? a->using_slot : -1), em_v(e, a->has_using ? a->using_id : -1);
    em_v(e, a->death_tick);
    em_v(e, popcount64(a->visited) + a->n_visited_off);   /* len(agent.visited) */
    em_v(e, a->chat_seq), em_v(e, a->chat_covered), em_v(e, a->n_chat);
    em_end(e);
    /* the option: (has, label, k, start, last_pos, last_progress, action, done kind, done_start); done_start is
       act_done's start only */
    em_begin(e, RS_CF_AGENT_OPTION);
    em_v(e, o->has);
    if (o->has) {
        em_v(e, o->label), em_v(e, o->label_k), em_v(e, o->start), em_v(e, o->last_x), em_v(e, o->last_z);
        em_v(e, o->last_progress), em_v(e, o->has_action ? o->action : -1), em_v(e, o->done);
        em_v(e, o->done == RS_DONE_ACT ? o->done_start : 0);
    }
    em_end(e);
    em_begin(e, RS_CF_AGENT_ACT);
    em_v(e, o->has && o->has_act);
    if (o->has && o->has_act) {
        const RsAct *act = &o->act;
        em_key(e, &act->key);
        em_v(e, act->xp), em_v(e, act->last_xp);
        em_v(e, act->has_loc);
        if (act->has_loc) em_v(e, act->loc_type), em_v(e, act->loc_x), em_v(e, act->loc_z), em_v(e, act->loc_level);
        em_items(e, act->items, RS_INV_CAP);
        em_items(e, act->last_items, RS_INV_CAP);
    }
    em_end(e);
    em_begin(e, RS_CF_AGENT_OPTION_END);
    em_v(e, a->option_end.has);
    if (a->option_end.has)
        em_v(e, a->option_end.action), em_v(e, a->option_end.reason), em_v(e, a->option_end.refused),
            em_v(e, a->option_end.reported);
    em_end(e);
    /* refusals still in force (until > tick), sorted by their tuple */
    em_begin(e, RS_CF_AGENT_REFUSED);
    {
        int64_t rows[RS_REFUSED_CAP * 7];
        int n = 0;
        for (int i = 0; i < a->n_refused && i < RS_REFUSED_CAP; i++) {
            const RsRefused *r = &a->refused[i];
            if (r->until <= w->tick) continue;
            int64_t *row = &rows[n++ * 7];
            row[0] = r->key.kind, row[1] = r->key.id, row[2] = r->key.x, row[3] = r->key.z, row[4] = r->key.level;
            row[5] = r->key.option_lc, row[6] = r->until;
        }
        em_sorted(e, rows, n, 7);
    }
    em_end(e);
    /* texts sorted by (iface string, com): the string's code-point rank, then com */
    em_begin(e, RS_CF_AGENT_TEXTS);
    {
        int64_t rows[RS_TEXTS_CAP * 6];   /* rank, com, then the emitted iface, com, text, tick */
        int n = 0;
        for (int i = 0; i < a->n_texts && i < RS_TEXTS_CAP; i++, n++) {
            const RsTextEntry *t = &a->texts[i];
            int64_t *row = &rows[n * 6];
            row[0] = t->iface >= 0 && t->iface < s->P->n_strings ? s->P->str_rank[t->iface] : -1;
            row[1] = t->com, row[2] = t->iface, row[3] = t->com, row[4] = msg_text(s->P, t->msg), row[5] = t->tick;
        }
        for (int i = 1; i < n; i++)
            for (int j = i; j > 0 && row_less(&rows[j * 6], &rows[(j - 1) * 6], 2); j--)
                for (int v = 0; v < 6; v++) {
                    int64_t t = rows[j * 6 + v];
                    rows[j * 6 + v] = rows[(j - 1) * 6 + v];
                    rows[(j - 1) * 6 + v] = t;
                }
        for (int i = 0; i < n; i++)
            for (int v = 2; v < 6; v++) em_v(e, rows[i * 6 + v]);
    }
    em_end(e);
}

/* Canon's RS_CF_EPISODE: GoalEpisode's values, 0 for those not set before arm, paid as its double's bits. */
static void em_episode(Emit *e, const Episode *ep) {
    int64_t paid;
    memcpy(&paid, &ep->paid, sizeof paid);
    em_begin(e, RS_CF_EPISODE);
    em_v(e, ep->armed), em_v(e, ep->active), em_v(e, ep->start_tick), em_v(e, ep->end), em_v(e, ep->event_start);
    em_v(e, ep->terminal), em_v(e, ep->terminal ? ep->term_reason : 0), em_v(e, ep->terminal ? ep->term_tick : 0);
    em_v(e, ep->decisions), em_v(e, ep->carried_before), em_v(e, ep->banked), em_v(e, paid);
    em_end(e);
}

int rs_canon(const RsSlot *s, int64_t *buf, int cap, int *n, RsErr *err) {
    int nw = 0;
    if (rs_world_canon(&s->w, buf, cap, &nw, err) && err->reason != RS_E_OVERFLOW) return 1;
    Emit e = {buf, cap, nw, 0};
    em_agent(&e, s);
    em_episode(&e, &s->ep);
    *n = e.n;
    if (e.n > cap) return RS_OVERFLOW(err, "the canonical record's buffer");
    return 0;
}

uint32_t rs_canon_hash(const int64_t *buf, int n) {
    uLong crc = crc32(0L, Z_NULL, 0);
    const Bytef *bytes = (const Bytef *)buf;
    size_t left = (size_t)n * sizeof *buf;  /* the hosts are little-endian (arm64, x86-64) */
    while (left > 0) {
        uInt chunk = left > (1u << 30) ? (1u << 30) : (uInt)left;
        crc = crc32(crc, bytes, chunk);
        bytes += chunk;
        left -= chunk;
    }
    return (uint32_t)crc;
}

/* ---- traces ----------------------------------------------------------------------------------------------------- */
RsTrace *rs_trace_new(int32_t draws_cap, int32_t rows_cap, int64_t data_cap) {
    RsTrace *t = calloc(1, sizeof *t);
    if (!t || draws_cap < 0 || rows_cap < 0 || data_cap < 0) {
        free(t);
        return NULL;
    }
    t->draws = malloc((size_t)(draws_cap ? draws_cap : 1) * sizeof *t->draws);
    t->rows = malloc((size_t)(rows_cap ? rows_cap : 1) * sizeof *t->rows);
    t->data = malloc((size_t)(data_cap ? data_cap : 1) * sizeof *t->data);
    if (!t->draws || !t->rows || !t->data) {
        rs_trace_free(t);
        return NULL;
    }
    t->draws_cap = draws_cap, t->rows_cap = rows_cap, t->data_cap = data_cap;
    return t;
}

void rs_trace_free(RsTrace *t) {
    if (!t) return;
    free(t->draws);
    free(t->rows);
    free(t->data);
    free(t);
}

/* The harness took the rows: keep the draws made after the last one (they belong to the next row). */
void rs_trace_clear(RsTrace *t) {
    int32_t pending = t->n_draws - t->draw_mark;
    if (pending > 0 && t->draw_mark > 0) memmove(t->draws, &t->draws[t->draw_mark], (size_t)pending * sizeof *t->draws);
    t->n_draws = pending > 0 ? pending : 0;
    t->draw_mark = 0;
    t->n_rows = 0;
    t->n_data = 0;
    t->truncated = 0;
}

/* WorldTrace.bind (sim/native_harness.py): the previous world's draws after its last cycle go to that cycle's row
   when this call holds it, else nowhere; the new world's cursors start at its empty logs. */
void rs_trace_begin(RsTrace *t, int32_t episode) {
    if (t->n_rows > 0) {
        RsTraceRow *last = &t->rows[t->n_rows - 1];
        last->n_draw = t->n_draws - last->draw0;
    } else {
        t->n_draws = t->draw_mark;
    }
    t->draw_mark = t->n_draws;
    t->ev_seen = 0;
    t->chat_seen = 0;
    t->episode = episode;
}

void rs_trace(RsBatch *b, int32_t world, RsTrace *trace) {
    if (world < 0 || world >= b->n) return;
    RsSlot *s = &b->slots[world];
    s->trace = trace;
    s->w.trace = trace;
    if (trace) {
        trace->ev_seen = s->w.n_events;
        trace->chat_seen = s->w.chat.n;
        trace->episode = s->episode;
    }
}

/* WorldTrace.tick (sim/native_harness.py): the record and its hash, the draws since the last row, then the events,
   chat lines (since the cursors) and texts (the text log) the world holds. */
void rs_trace_tick(RsSlot *s) {
    RsTrace *t = s->w.trace;
    const World *w = &s->w;
    if (!t || t->truncated) return;
    if (t->n_rows >= t->rows_cap) {
        t->truncated = 1;
        return;
    }
    RsTraceRow *row = &t->rows[t->n_rows];
    int64_t room = t->data_cap - t->n_data;
    int n = 0;
    RsErr err;
    memset(&err, 0, sizeof err);
    if (rs_canon(s, t->data + t->n_data, room > INT32_MAX ? INT32_MAX : (int)room, &n, &err)) {
        t->truncated = 1;
        return;
    }
    int32_t n_ev = w->n_events - t->ev_seen;
    int64_t n_chat = w->chat.n - t->chat_seen;
    if (n_ev < 0 || n_chat < 0 || n_chat > RS_CHAT_RING
        || (int64_t)n + (int64_t)n_ev * RS_TRACE_EVENT + n_chat * RS_TRACE_CHAT + (int64_t)w->text.n * RS_TRACE_TEXT > room) {
        t->truncated = 1;
        return;
    }
    *row = (RsTraceRow){.tick = w->tick, .episode = t->episode, .crc = rs_canon_hash(t->data + t->n_data, n),
                        .draw0 = t->draw_mark, .n_draw = t->n_draws - t->draw_mark, .rec = t->n_data, .n_rec = n};
    t->n_data += n;
    t->draw_mark = t->n_draws;
    row->ev = t->n_data, row->n_ev = n_ev;
    for (int32_t i = t->ev_seen; i < w->n_events; i++) {
        const RsEvent *v = &w->ev[i];
        int64_t *d = t->data + t->n_data;
        d[0] = v->kind, d[1] = v->tick, d[2] = v->a8, d[3] = v->b8, d[4] = v->has_source, d[5] = v->obj;
        d[6] = v->count, d[7] = v->v0, d[8] = v->v1, d[9] = v->v2, d[10] = v->v3;
        t->n_data += RS_TRACE_EVENT;
    }
    t->ev_seen = w->n_events;
    row->chat = t->n_data, row->n_chat = (int32_t)n_chat;
    for (int64_t k = t->chat_seen; k < w->chat.n; k++) {
        const RsChatLine *line = &w->chat.ring[k % RS_CHAT_RING];
        t->data[t->n_data++] = line->tick;
        t->data[t->n_data++] = msg_text(s->P, line->msg);
    }
    t->chat_seen = w->chat.n;
    row->text = t->n_data, row->n_text = w->text.n;
    for (int32_t k = 0; k < w->text.n; k++) {
        const RsTextLine *line = &w->text.line[k];
        t->data[t->n_data++] = line->tick, t->data[t->n_data++] = line->iface, t->data[t->n_data++] = line->com;
        t->data[t->n_data++] = msg_text(s->P, line->msg);
    }
    t->n_rows++;
}
