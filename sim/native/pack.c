/* sim/native/pack.c: the read-only pack (format v1, rs.h) mapped once per process and shared by every world and thread,
 * its typed views and layouts, message lookup, the load-time declared-world guard, and the error helpers and tables.
 * Every bound is checked at open, so the rest of the engine indexes the views without checks of its own. */
#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rs.h"

/* ---- tables and errors ---------------------------------------------------------------------------------------- */
#define RS_REASON_ROW(id, exc, fmt, kinds) {id, exc, fmt, kinds},
const RsReasonInfo RS_REASON_INFO[RS_N_REASONS] = { RS_REASONS(RS_REASON_ROW) };
#undef RS_REASON_ROW
#define RS_SCRIPT_ROW(id, qualname, status) {#id, qualname, status},
const RsScriptInfo RS_SCRIPT_INFO[RS_N_SCRIPTS] = { RS_SCRIPTS(RS_SCRIPT_ROW) };
#undef RS_SCRIPT_ROW

static uint8_t code_of(uint8_t exc) {
    return exc == EXC_NONE ? RS_OK : exc == EXC_UNSUPPORTED ? RS_UNSUPPORTED : exc == EXC_NOT_PORTED ? RS_NOT_PORTED
        : RS_FATAL;
}

int rs_fail(RsErr *e, int reason, int n, const int32_t *args) {
    memset(e, 0, sizeof *e);
    if (reason <= 0 || reason >= RS_N_REASONS) reason = RS_E_INTERNAL;
    e->reason = (uint16_t)reason;
    e->exc = RS_REASON_INFO[reason].exc;
    e->code = code_of(e->exc);
    for (int i = 0; i < n && i < RS_ERR_ARGS; i++) e->arg[i] = args[i];
    return 1;
}

int rs_fail_text(RsErr *e, int reason, const char *text) {
    rs_fail(e, reason, 0, NULL);
    e->text = text;
    return 1;
}

int rs_wrap_outside(RsErr *e) {
    if (e->exc == EXC_OUTSIDE) {
        e->exc = EXC_UNSUPPORTED;
        e->code = RS_UNSUPPORTED;
    }
    return 1;
}

/* A stub of a function with no error channel was called: the port is incomplete there. */
void rs_trap(const char *fn) {
    fprintf(stderr, "native: %s is not ported\n", fn);
    abort();
}

/* ---- open ------------------------------------------------------------------------------------------------------- */
static const struct { uint32_t id; uint32_t size; } RECORDS[] = {
    {RS_SEC_CONSTS, sizeof(RsConsts)}, {RS_SEC_LOC_TYPES, sizeof(RsLocType)}, {RS_SEC_NPC_TYPES, sizeof(RsNpcType)},
    {RS_SEC_OBJ_TYPES, sizeof(RsObjType)}, {RS_SEC_TREES, sizeof(RsTreeRow)}, {RS_SEC_ROCKS, sizeof(RsRockRow)},
    {RS_SEC_GEMS, sizeof(RsGemRow)}, {RS_SEC_FISH, sizeof(RsFishRow)}, {RS_SEC_EAT_TRIGGERS, sizeof(RsEatTrigger)},
    {RS_SEC_EAT_ROWS, sizeof(RsEatRow)}, {RS_SEC_WIELD_RULES, sizeof(RsWieldRule)},
    {RS_SEC_STYLES, sizeof(RsStyleTable)}, {RS_SEC_LEVELUP, sizeof(RsLevelupRow)}, {RS_SEC_UNLOCKS, sizeof(RsUnlock)},
    {RS_SEC_MSGS, sizeof(RsMsg)}, {RS_SEC_MSG_KEYS, sizeof(RsMsgKeyRow)}, {RS_SEC_TOKENS, sizeof(RsToken)},
    {RS_SEC_DROP_ROOTS, sizeof(RsDropRoot)}, {RS_SEC_DROP_OPS, sizeof(RsDropOp)}, {RS_SEC_ENUMS, sizeof(RsEnumRow)},
    {RS_SEC_LAYOUTS, sizeof(RsLayoutRec)}, {RS_SEC_VARIANTS, sizeof(RsVariantKey)},
};
static const struct { uint32_t id; uint32_t dtype; } SCALARS[] = {
    {RS_SEC_STRINGS, RS_DT_BYTES}, {RS_SEC_STR_OFF, RS_DT_U32}, {RS_SEC_STR_RANK, RS_DT_I32},
    {RS_SEC_STR_LOWER, RS_DT_I32}, {RS_SEC_STR_NAME_ID, RS_DT_I32}, {RS_SEC_STR_ROW, RS_DT_I32},
    {RS_SEC_MSG_TOKENS, RS_DT_I32}, {RS_SEC_HEAP_I32, RS_DT_I32}, {RS_SEC_HEAP_I16, RS_DT_I16},
    {RS_SEC_HEAP_U16, RS_DT_U16}, {RS_SEC_HEAP_U8, RS_DT_U8}, {RS_SEC_HEAP_F64, RS_DT_F64},
};
/* The heaps may hold more than INT32_MAX elements (a derived 48x48 layout's state-0 trees are 3 x 2304^2 bytes); every
   other section's count fits the int32 views. */
static int is_heap(uint32_t id) {
    return id == RS_SEC_HEAP_I32 || id == RS_SEC_HEAP_I16 || id == RS_SEC_HEAP_U16 || id == RS_SEC_HEAP_U8
        || id == RS_SEC_HEAP_F64;
}
static const uint32_t DT_SIZE[] = {1, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8, 0};

static int bad(RsErr *err, const char *what) { return rs_fail_text(err, RS_E_PACK, what); }

const void *rs_pack_section(const RsPack *P, uint32_t id, uint64_t *count, uint32_t *elem) {
    for (uint32_t i = 0; i < P->n_sections; i++) {
        const RsSection *s = &P->sections[i];
        if (s->id != id) continue;
        if (count) *count = s->count;
        if (elem) *elem = s->elem;
        return P->base + s->off;
    }
    return NULL;
}

/* A typed view: the section's data and count, or NULL and 0 when absent (int32 counts: rs_pack_open refused larger
   ones, heaps apart, which are read with their int64 count). */
#define VIEW(P, id, type, ptr, n) do { uint64_t c_ = 0; (ptr) = (type)rs_pack_section((P), (id), &c_, NULL); \
    (n) = (int32_t)c_; } while (0)

typedef struct { const void *base; int64_t count; } Heap;

static int resolve(const Heap *h, int64_t off, int64_t count, size_t elem, const void **out, RsErr *err) {
    *out = NULL;
    if (off < 0) return 0;
    if (count < 0 || off > h->count || count > h->count - off) return bad(err, "a layout array lies outside its heap");
    *out = (const uint8_t *)h->base + (size_t)off * elem;
    return 0;
}

static int resolve_layout(RsPack *P, int32_t index, const Heap heap[5], RsErr *err) {
    const RsLayoutRec *r = &((const RsLayoutRec *)rs_pack_section(P, RS_SEC_LAYOUTS, NULL, NULL))[index];
    RsLayout *L = &P->layouts[index];
    const Heap *i32 = &heap[0], *i16 = &heap[1], *u16 = &heap[2], *u8 = &heap[3];
    if (r->W < 1 || r->H < 1 || r->W > RS_MAX_W || r->H > RS_MAX_W || r->T != r->W * r->H)
        return bad(err, "a layout's size is outside 1..64 or T != W * H");
    if (r->x0 % 8 || r->z0 % 8) return bad(err, "a layout's origin is not on the 8x8 zone grid");
    if (r->n_src < 0 || r->n_start < 0 || r->n_locs < 0 || r->n_locs > RS_MAX_LOCS || r->n_open < 0
        || r->n_npcs < 0 || r->n_npcs > RS_MAX_NPCS || r->n_fish < 0 || r->n_camp < 0 || r->n_outside < 0
        || r->n_groves < 0 || r->n_sites < 0)
        return bad(err, "a layout count is negative or beyond the world's capacity");
    *L = (RsLayout){.rec = r, .index = index, .kind = r->kind, .W = r->W, .H = r->H, .T = r->T, .x0 = r->x0,
                    .z0 = r->z0, .n_src = r->n_src, .n_start = r->n_start, .n_locs = r->n_locs, .n_open = r->n_open,
                    .n_npcs = r->n_npcs, .n_fish = r->n_fish, .n_camp = r->n_camp, .n_outside = r->n_outside,
                    .n_groves = r->n_groves, .n_sites = r->n_sites};
    int64_t T = r->T;
    const void *p;
#define R(heap_, field, count, type) do { RS_TRY(resolve((heap_), r->field, (count), sizeof(*L->field), &p, err)); \
        L->field = (type)p; } while (0)
    R(i32, flags_boot, T, const int32_t *);
    R(u8, steps_boot, T, const uint8_t *);
    R(u8, steps_snap, T, const uint8_t *);
    R(u8, sides, T, const uint8_t *);
    R(u16, entity_reach, T, const uint16_t *);
    R(i16, tile_src, T, const int16_t *);
    R(i16, src_tile, r->n_src, const int16_t *);
    R(i16, start_tile, r->n_start, const int16_t *);
    R(i32, locs, (int64_t)r->n_locs * 5, const int32_t *);
    R(i16, loc_reach, (int64_t)r->n_locs * 24, const int16_t *);
    R(i32, open_locs, (int64_t)r->n_open * 5, const int32_t *);
    R(i32, npcs, (int64_t)r->n_npcs * 4, const int32_t *);
    R(i32, fish, (int64_t)r->n_fish * 2, const int32_t *);
    R(u16, order0, T * T, const uint16_t *);
    R(u8, dist0, T * T, const uint8_t *);
    R(u8, parent0, T * T, const uint8_t *);
    R(u8, snap_dist, T * T, const uint8_t *);
    R(i16, camp, r->n_camp, const int16_t *);
    R(i16, outside, r->n_outside, const int16_t *);
    R(i32, grove_off, r->n_groves > 0 ? r->n_groves + 1 : 0, const int32_t *);
    R(i32, site_grove, r->n_sites, const int32_t *);
    R(i32, site_off, r->n_sites > 0 ? r->n_sites + 1 : 0, const int32_t *);
    int64_t rows = L->grove_off ? L->grove_off[r->n_groves] : 0, tiles = L->site_off ? L->site_off[r->n_sites] : 0;
    if (L->grove_off) for (int g = 0; g < r->n_groves; g++)
        if (L->grove_off[g] < 0 || L->grove_off[g] > L->grove_off[g + 1]) return bad(err, "grove offsets decrease");
    if (L->site_off) for (int s = 0; s < r->n_sites; s++)
        if (L->site_off[s] < 0 || L->site_off[s] > L->site_off[s + 1]) return bad(err, "site offsets decrease");
    R(i32, grove_row, rows, const int32_t *);
    R(u8, grove_normal, rows, const uint8_t *);
    R(i16, site_tile, tiles, const int16_t *);
#undef R
    if (!L->flags_boot || !L->steps_boot || !L->steps_snap || !L->sides || !L->entity_reach || !L->tile_src)
        return bad(err, "a layout lacks a per-tile array");
    for (int64_t t = 0; t < T; t++)
        if (L->tile_src[t] < -1 || L->tile_src[t] >= r->n_src) return bad(err, "a tile's source index is out of range");
    return 0;
}

int rs_pack_open(const char *path, const RsPack **out, RsErr *err) {
    *out = NULL;
    memset(err, 0, sizeof *err);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return bad(err, "cannot open the file");
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(RsPackHeader)) {
        close(fd);
        return bad(err, "the file is shorter than a pack header");
    }
    size_t size = (size_t)st.st_size;
    void *map = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);  /* the mapping keeps the file */
    if (map == MAP_FAILED) return bad(err, "mmap failed");
    RsPack *P = calloc(1, sizeof *P);
    if (!P) {
        munmap(map, size);
        return bad(err, "out of memory");
    }
    P->base = map;
    P->size = size;
    P->fd = -1;
    P->hdr = (const RsPackHeader *)map;
    const RsPackHeader *h = P->hdr;
#define FAIL(what) do { bad(err, (what)); rs_pack_close(P); return 1; } while (0)
    if (memcmp(h->magic, RS_PACK_MAGIC, 8) != 0) FAIL("bad magic");
    if (h->version != RS_PACK_VERSION) FAIL("unsupported format version");
    if (h->file_size != size) FAIL("the file size differs from the header's (truncated?)");
    if (h->manifest_off > size || h->manifest_len > size - h->manifest_off) FAIL("the manifest lies outside the file");
    if (h->table_off % 8 || h->table_off > size || (uint64_t)h->n_sections * sizeof(RsSection) > size - h->table_off)
        FAIL("the section table lies outside the file");
    P->manifest = (const char *)P->base + h->manifest_off;
    P->manifest_len = h->manifest_len;
    P->n_sections = h->n_sections;
    P->sections = (const RsSection *)(P->base + h->table_off);
    for (uint32_t i = 0; i < P->n_sections; i++) {
        const RsSection *s = &P->sections[i];
        if (s->off % RS_PACK_ALIGN) FAIL("a section is not 64-byte aligned");
        if (s->off > size || s->nbytes > size - s->off) FAIL("a section lies outside the file");
        if (s->dtype > RS_DT_RECORD) FAIL("a section has an unknown dtype");
        if (s->dtype != RS_DT_RECORD && s->elem != DT_SIZE[s->dtype])
            FAIL("a section's element size is not its dtype's");
        if (s->elem == 0 || s->count * s->elem != s->nbytes) FAIL("a section's size is not count x element size");
        if (s->count > (uint64_t)INT32_MAX && !is_heap(s->id)) FAIL("a section other than a heap has 2^31 entries");
        for (uint32_t j = 0; j < i; j++)
            if (P->sections[j].id == s->id) FAIL("a section id appears twice");
        for (size_t k = 0; k < sizeof RECORDS / sizeof *RECORDS; k++)
            if (RECORDS[k].id == s->id && (s->dtype != RS_DT_RECORD || s->elem != RECORDS[k].size))
                FAIL("a record section was written with another struct layout");
        for (size_t k = 0; k < sizeof SCALARS / sizeof *SCALARS; k++)
            if (SCALARS[k].id == s->id && s->dtype != SCALARS[k].dtype) FAIL("a section has the wrong dtype");
    }
    /* strings */
    int32_t n_blob;
    VIEW(P, RS_SEC_STRINGS, const char *, P->str_blob, n_blob);
    VIEW(P, RS_SEC_STR_OFF, const uint32_t *, P->str_off, P->n_strings);
    if (P->n_strings > 0) {
        if (!P->str_blob || n_blob < 1 || P->str_blob[n_blob - 1] != '\0')
            FAIL("the string blob is not NUL-terminated");
        for (int32_t i = 0; i < P->n_strings; i++)
            if (P->str_off[i] >= (uint32_t)n_blob) FAIL("a string offset lies outside the blob");
    }
    int32_t n;
    VIEW(P, RS_SEC_STR_RANK, const int32_t *, P->str_rank, n);
    if (P->str_rank && n != P->n_strings) FAIL("string ranks and strings differ in number");
    VIEW(P, RS_SEC_STR_LOWER, const int32_t *, P->str_lower, n);
    if (P->str_lower && n != P->n_strings) FAIL("lowercase ids and strings differ in number");
    VIEW(P, RS_SEC_STR_NAME_ID, const int32_t *, P->str_name_id, n);
    if (P->str_name_id && n != P->n_strings) FAIL("name ids and strings differ in number");
    VIEW(P, RS_SEC_STR_ROW, const int32_t *, P->str_row, n);
    if (P->str_row && (int64_t)n != 3 * (int64_t)P->n_strings) FAIL("name rows are not [3][n_strings]");
    for (int32_t i = 0; P->str_row && i < n; i++)
        if (P->str_row[i] < 0) FAIL("a name row is negative");
    /* records */
    VIEW(P, RS_SEC_CONSTS, const RsConsts *, P->consts, n);
    if (P->consts && n != 1) FAIL("CONSTS holds other than one record");
    VIEW(P, RS_SEC_LOC_TYPES, const RsLocType *, P->loc_types, P->n_loc_types);
    VIEW(P, RS_SEC_NPC_TYPES, const RsNpcType *, P->npc_types, P->n_npc_types);
    VIEW(P, RS_SEC_OBJ_TYPES, const RsObjType *, P->obj_types, P->n_obj_types);
    VIEW(P, RS_SEC_TREES, const RsTreeRow *, P->trees, P->n_trees);
    VIEW(P, RS_SEC_ROCKS, const RsRockRow *, P->rocks, P->n_rocks);
    VIEW(P, RS_SEC_GEMS, const RsGemRow *, P->gems, P->n_gems);
    VIEW(P, RS_SEC_FISH, const RsFishRow *, P->fish, P->n_fish);
    VIEW(P, RS_SEC_EAT_TRIGGERS, const RsEatTrigger *, P->eat_triggers, P->n_eat_triggers);
    VIEW(P, RS_SEC_EAT_ROWS, const RsEatRow *, P->eat_rows, P->n_eat_rows);
    VIEW(P, RS_SEC_WIELD_RULES, const RsWieldRule *, P->wield, P->n_wield);
    VIEW(P, RS_SEC_STYLES, const RsStyleTable *, P->styles, P->n_styles);
    VIEW(P, RS_SEC_LEVELUP, const RsLevelupRow *, P->levelup, P->n_levelup);
    VIEW(P, RS_SEC_UNLOCKS, const RsUnlock *, P->unlocks, P->n_unlocks);
    VIEW(P, RS_SEC_MSGS, const RsMsg *, P->msgs, P->n_msgs);
    VIEW(P, RS_SEC_MSG_KEYS, const RsMsgKeyRow *, P->msg_keys, P->n_msg_keys);
    VIEW(P, RS_SEC_TOKENS, const RsToken *, P->tokens, P->n_tokens);
    VIEW(P, RS_SEC_MSG_TOKENS, const int32_t *, P->msg_tokens, P->n_msg_tokens);
    VIEW(P, RS_SEC_DROP_ROOTS, const RsDropRoot *, P->drop_roots, P->n_drop_roots);
    VIEW(P, RS_SEC_DROP_OPS, const RsDropOp *, P->drop_ops, P->n_drop_ops);
    VIEW(P, RS_SEC_ENUMS, const RsEnumRow *, P->enums, P->n_enums);
    VIEW(P, RS_SEC_VARIANTS, const RsVariantKey *, P->variants, P->n_variants);
    for (int32_t i = 0; i < P->n_msgs; i++) {
        const RsMsg *m = &P->msgs[i];
        if (m->n_tok < 0 || m->tok0 < 0 || m->tok0 > P->n_msg_tokens - m->n_tok) FAIL("a message's tokens lie outside");
    }
    for (int32_t i = 0; i < P->n_msg_tokens; i++)
        if (P->msg_tokens[i] < 0 || P->msg_tokens[i] >= P->n_tokens) FAIL("a message token id is out of range");
    for (int32_t i = 1; i < P->n_msg_keys; i++) {
        const RsMsgKeyRow *a = &P->msg_keys[i - 1], *b = &P->msg_keys[i];
        if (a->key > b->key || (a->key == b->key && a->fill >= b->fill)) FAIL("message keys are not sorted and unique");
    }
    for (int32_t i = 0; i < P->n_msg_keys; i++)
        if (P->msg_keys[i].msg >= P->n_msgs) FAIL("a message key names no message");
    for (int32_t i = 0; i < P->n_drop_roots; i++) {
        const RsDropRoot *r = &P->drop_roots[i];
        if (r->first < 0 || r->n_ops < 0 || r->first > P->n_drop_ops - r->n_ops) FAIL("a drop program lies outside");
    }
    /* layouts */
    const RsLayoutRec *recs;
    VIEW(P, RS_SEC_LAYOUTS, const RsLayoutRec *, recs, P->n_layouts);
    if (P->n_layouts > 0) {
        Heap heap[5];
        static const uint32_t HEAPS[5] = {RS_SEC_HEAP_I32, RS_SEC_HEAP_I16, RS_SEC_HEAP_U16, RS_SEC_HEAP_U8,
                                          RS_SEC_HEAP_F64};
        for (int k = 0; k < 5; k++) {
            uint64_t c = 0;
            const void *base = rs_pack_section(P, HEAPS[k], &c, NULL);
            heap[k] = (Heap){base, (int64_t)c};
        }
        P->layouts = calloc((size_t)P->n_layouts, sizeof *P->layouts);
        if (!P->layouts) FAIL("out of memory");
        int32_t last = RS_LAYOUT_COMPACT;
        for (int32_t i = 0; i < P->n_layouts; i++) {
            if (recs[i].kind < last || recs[i].kind > RS_LAYOUT_VARIANT)
                FAIL("layouts are not compact, supply, variants");
            last = recs[i].kind;
            if (recs[i].kind == RS_LAYOUT_COMPACT) P->n_compact++;
            else if (recs[i].kind == RS_LAYOUT_SUPPLY) P->n_supply++;
            if (resolve_layout(P, i, heap, err)) { rs_pack_close(P); return 1; }
            if (!P->layouts[i].snap_dist) rs_snap_reach(&P->layouts[i]);   /* walk rows are searched at run time */
        }
        P->compact0 = 0;
        P->supply0 = P->n_compact;
        for (int32_t i = P->n_compact + P->n_supply; i < P->n_layouts; i++)
            if (recs[i].pool_index < 0 || recs[i].pool_index >= P->n_supply) FAIL("a variant names no supply layout");
    }
    for (int32_t i = 0; i < P->n_variants; i++) {
        const RsVariantKey *v = &P->variants[i];
        if (v->rec != RS_VARIANT_FAILS && (v->rec < P->n_compact + P->n_supply || v->rec >= P->n_layouts))
            FAIL("a variant key names no variant");
        if (i > 0) {
            const RsVariantKey *u = &P->variants[i - 1];
            if (u->layout > v->layout || (u->layout == v->layout && (u->grove > v->grove
                || (u->grove == v->grove && u->keep >= v->keep)))) FAIL("variant keys are not sorted and unique");
        }
    }
#undef FAIL
    *out = P;
    return 0;
}

void rs_pack_close(const RsPack *P) {
    if (!P) return;
    RsPack *p = (RsPack *)P;
    free(p->layouts);
    if (p->base) munmap((void *)p->base, p->size);
    if (p->fd >= 0) close(p->fd);
    free(p);
}

/* ---- lookups ---------------------------------------------------------------------------------------------------- */
const char *rs_str(const RsPack *P, int32_t id) {
    if (id < 0 || id >= P->n_strings) return "";
    return P->str_blob + P->str_off[id];
}

int rs_msg(World *w, int32_t key, int32_t fill, int32_t *msg) {
    const RsPack *P = w->P;
    int32_t lo = 0, hi = P->n_msg_keys - 1;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        const RsMsgKeyRow *r = &P->msg_keys[mid];
        if (r->key == key && r->fill == fill) {
            if (r->msg < 0) return RS_FAIL(&w->err, RS_E_GAME_TEXT, -r->msg - 1);
            *msg = r->msg;
            return 0;
        }
        if (r->key < key || (r->key == key && r->fill < fill)) lo = mid + 1;
        else hi = mid - 1;
    }
    return RS_FAIL(&w->err, RS_E_MSG_MISSING, key, fill);
}

/* A found key's record, or RS_VARIANT_FAILS for a key the exporter recorded as failing the check; RS_VARIANT_ABSENT
   when no row names the key. */
int32_t rs_variant_find(const RsPack *P, int32_t layout, int32_t grove, int32_t keep) {
    int32_t lo = 0, hi = P->n_variants - 1;
    while (lo <= hi) {
        int32_t mid = lo + (hi - lo) / 2;
        const RsVariantKey *v = &P->variants[mid];
        int cmp = v->layout != layout ? (v->layout < layout ? -1 : 1) : v->grove != grove ? (v->grove < grove ? -1 : 1)
            : v->keep != keep ? (v->keep < keep ? -1 : 1) : 0;
        if (cmp == 0) return v->rec;
        if (cmp < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return RS_VARIANT_ABSENT;
}

static int check_triggers(const int16_t *script, const int32_t *name, int n, RsErr *err) {
    for (int k = 0; k < n; k++) {
        if (script[k] < 0 || script[k] >= RS_N_SCRIPTS) return bad(err, "a trigger names an unknown script id");
        if (RS_SCRIPT_INFO[script[k]].status == RS_NP) {
            int32_t arg = name[k];
            return rs_fail(err, RS_E_NOT_PORTED_SCRIPT, 1, &arg);
        }
    }
    return 0;
}

static int check_npc(const RsPack *P, int32_t type, RsErr *err) {
    const RsNpcType *t = rs_npc_type(P, type);
    if (!t) return bad(err, "a layout names an npc type the pack lacks");
    return check_triggers(t->trig_script, t->trig_name, RS_NTRIG_N, err);
}

int rs_pack_check_declared(const RsPack *P, RsErr *err) {
    memset(err, 0, sizeof *err);
    for (int32_t i = 0; i < P->n_layouts; i++) {
        const RsLayout *L = &P->layouts[i];
        for (int pass = 0; pass < 2; pass++) {
            const int32_t *rows = pass ? L->open_locs : L->locs;
            int32_t n = pass ? L->n_open : L->n_locs;
            for (int32_t k = 0; rows && k < n; k++) {
                const RsLocType *t = rs_loc_type(P, rows[k * 5]);
                if (!t) return bad(err, "a layout names a loc type the pack lacks");
                RS_TRY(check_triggers(t->trig_script, t->trig_name, RS_LTRIG_N, err));
            }
        }
        for (int32_t k = 0; L->npcs && k < L->n_npcs; k++) RS_TRY(check_npc(P, L->npcs[k * 4 + 1], err));
    }
    if (P->consts) {
        if (P->consts->npc_goblin >= 0) RS_TRY(check_npc(P, P->consts->npc_goblin, err));
        if (P->consts->npc_mugger >= 0) RS_TRY(check_npc(P, P->consts->npc_mugger, err));
    }
    return 0;
}
