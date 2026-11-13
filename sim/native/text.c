/* sim/native/text.c: text capture (sim/agent.py capture_text: the kept chat lines and the component texts)
 * and text vectors (rl/feedback_contract.py text_vector) over pack messages.
 *
 * No string is built here. A message carries its tokens (feedback_contract.words, numbers already as number()
 * tokens) as crc32(tok), crc32(tok + ' ') and len(tok), and its own chat counts in 64 buckets. A pair feature
 * crc32(a + ' ' + b) is crc32_combine(crc32(a + ' '), crc32(b), len(b)). The exporter asserts every message is ASCII
 * with no '<' or '>', so the tokens of ' '.join(parts) are the parts' tokens in order and a dialog's pairs cross
 * its parts, as Python's joined text makes them. */
#include <math.h>
#include <string.h>
#include <zlib.h>

#include "rs.h"

/* Agent.capture_text (agent.py): the chat lines since chat_seen get sequence numbers (CHAT_KEEP kept, oldest
   first), the text log's lines replace their component's text, and the text log empties. chat_seen and the ring's
   read mark move together (rs.h ChatLog). */
int rs_capture_text(RsSlot *s) {
    World *w = &s->w;
    Agent *a = &s->a;
    ChatLog *c = &w->chat;
    if (a->chat_seen < c->n - RS_CHAT_RING) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "chat lines lost before capture");
    for (int64_t k = a->chat_seen; k < c->n; k++) {
        const RsChatLine *line = &c->ring[k % RS_CHAT_RING];
        if (a->n_chat == RS_CHAT_KEEP) memmove(a->chat, a->chat + 1, (RS_CHAT_KEEP - 1) * sizeof *a->chat), a->n_chat--;
        a->chat[a->n_chat++] = (RsChatKept){line->msg, line->tick, ++a->chat_seq};
    }
    a->chat_seen = c->captured = c->n;
    for (int i = 0; i < w->text.n; i++) {
        const RsTextLine *t = &w->text.line[i];
        int k = 0;
        while (k < a->n_texts && !(a->texts[k].iface == t->iface && a->texts[k].com == t->com)) k++;
        if (k == a->n_texts) {
            if (a->n_texts >= RS_TEXTS_CAP) return RS_OVERFLOW(&w->err, "RS_TEXTS_CAP component texts");
            a->n_texts++;
        }
        a->texts[k] = (RsTextEntry){t->iface, t->com, t->msg, t->tick};
    }
    w->text.n = 0;
    return 0;
}

/* One feature's signed bucket: crc % dim, +1 when bit 16 is set (feedback_contract.py). */
static inline void feature(int32_t *count, int dim, uint32_t h) { count[h % (uint32_t)dim] += (h >> 16) & 1 ? 1 : -1; }

/* v / float32(norm) with norm = sqrt(v . v) in float32 (np.linalg.norm of a float32 vector, then a float32 divide:
   NEP 50 makes the Python float divisor weak); the integer counts are exact in float32, so is their dot. */
static void unit(const int32_t *count, int dim, float *out) {
    int64_t sumsq = 0;
    for (int i = 0; i < dim; i++) sumsq += (int64_t)count[i] * count[i];
    float norm = sqrtf((float)sumsq);
    for (int i = 0; i < dim; i++) out[i] = norm != 0.0f ? (float)count[i] / norm : (float)count[i];
}

/* A token run's features: every token, then every adjacent pair (the order is irrelevant to the counts). */
static void tokens(const RsPack *P, const int32_t *tok, int n, int dim, int32_t *count) {
    for (int i = 0; i < n; i++) {
        const RsToken *a = &P->tokens[tok[i]];
        feature(count, dim, a->crc);
        if (i + 1 < n) {
            const RsToken *b = &P->tokens[tok[i + 1]];
            feature(count, dim, (uint32_t)crc32_combine(a->crc_sp, b->crc, (z_off_t)b->len));
        }
    }
}

/* text_vector(texts, dim) over chat lines, each its own text (no pair crosses two lines): the messages' own counts at
   dim 64, else their tokens. */
void rs_text_vector_chat(const RsPack *P, const int32_t *msgs, int n, int dim, float *out) {
    int32_t count[64] = {0};
    if (dim < 1 || dim > 64) RS_TRAP();   /* the contracts' dims are 64 (TEXT_DIM) and 16 (CHOICE_DIM) */
    for (int k = 0; k < n; k++) {
        const RsMsg *m = &P->msgs[msgs[k]];
        if (dim == 64)
            for (int i = 0; i < 64; i++) count[i] += m->chat[i];
        else
            tokens(P, P->msg_tokens + m->tok0, m->n_tok, dim, count);
    }
    unit(count, dim, out);
}

/* text_vector([' '.join(parts)], dim): one text whose tokens are the parts' tokens in order. */
void rs_text_vector_parts(const RsPack *P, const int32_t *msgs, int n, int dim, float *out) {
    int32_t count[64] = {0};
    if (dim < 1 || dim > 64) RS_TRAP();
    int32_t last = -1;   /* the previous part's last token, for the pair across the join */
    for (int k = 0; k < n; k++) {
        const RsMsg *m = &P->msgs[msgs[k]];
        if (!m->n_tok) continue;
        const int32_t *tok = P->msg_tokens + m->tok0;
        if (last >= 0) {
            const RsToken *a = &P->tokens[last], *b = &P->tokens[tok[0]];
            feature(count, dim, (uint32_t)crc32_combine(a->crc_sp, b->crc, (z_off_t)b->len));
        }
        tokens(P, tok, m->n_tok, dim, count);
        last = tok[m->n_tok - 1];
    }
    unit(count, dim, out);
}
