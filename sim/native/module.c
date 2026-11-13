/* sim/native/module.c: the CPython extension sim._native (built and loaded by sim/native.py).
 *
 * Pack(path) maps a pack (rs_pack_open). Batch(pack, cfg, log1p) holds an RsBatch and the log1p feature tables
 * (float32 [RS_LOG1P_N][RS_LOG1P_COUNTS], built by sim/native.py in this process; the batch keeps the buffer);
 * bind(arrays) takes sim/workers.py fields(n) arrays (obs/<key>, info/<field>, info/made, info/kills, info/facts,
 * reward, done; 'action' is ignored), checks each against sim/spec.py's dtypes and shapes (rs_gen.h) and keeps their
 * buffers, so reset and step write straight into them, with the GIL released (cfg 'threads' > 1 steps the worlds on
 * batch.c's pool). start(b) and failure(b) return a world's RsStart and its unsupported evidence as raw ints, which
 * sim/native.py turns into CpuBatch's starts and evidence dicts. For the equivalence harness: trace(b, on) attaches a
 * world's trace (canon.c) and trace_rows(b) takes its rows; begin(b, start) begins world b's next episode from an
 * injected start (a dict as start(b) gives it; None draws it); set_episode(b, k) sets its episode counter (repros);
 * external_starts(on) leaves done worlds for begin (scenario starts). A failed C
 * call raises Error(code, exc, reason, args, text, world), which sim/native.py renders into the Python exception the
 * oracle raises. The tables (reasons, scripts, canonical fields, sections, struct layouts) are exported from the
 * same X-macros the engine compiles, and the keyed RNG and numeric helpers are exported for the equivalence tests. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "rs.h"

static PyObject *Error;

/* Error's args: (code, exc, reason, args, text, world). */
static PyObject *err_tuple(const RsErr *e, int world) {
    PyObject *args = Py_BuildValue("(iiiiii)", e->arg[0], e->arg[1], e->arg[2], e->arg[3], e->arg[4], e->arg[5]);
    if (!args) return NULL;
    return Py_BuildValue("(iiiNzi)", e->code, e->exc, e->reason, args, e->text, world);
}

static PyObject *raise_err(const RsErr *e, int world) {
    PyObject *value = err_tuple(e, world);
    if (value) PyErr_SetObject(Error, value);
    Py_XDECREF(value);
    return NULL;
}

/* ---- tables ------------------------------------------------------------------------------------------------- */
typedef struct { const char *name, *ctype; size_t offset; int count; } FieldDesc;
#define DESC_S(t, n) {#n, #t, offsetof(STRUCT, n), 1},
#define DESC_A(t, n, k) {#n, #t, offsetof(STRUCT, n), (k)},
#define FIELDS_OF(type, FIELDS) static const FieldDesc F_##type[] = { FIELDS(DESC_S, DESC_A) {NULL, NULL, 0, 0} };
#define STRUCT RsLocType
FIELDS_OF(RsLocType, RS_LOC_TYPE_FIELDS)
#undef STRUCT
#define STRUCT RsNpcType
FIELDS_OF(RsNpcType, RS_NPC_TYPE_FIELDS)
#undef STRUCT
#define STRUCT RsObjType
FIELDS_OF(RsObjType, RS_OBJ_TYPE_FIELDS)
#undef STRUCT
#define STRUCT RsTreeRow
FIELDS_OF(RsTreeRow, RS_TREE_ROW_FIELDS)
#undef STRUCT
#define STRUCT RsRockRow
FIELDS_OF(RsRockRow, RS_ROCK_ROW_FIELDS)
#undef STRUCT
#define STRUCT RsGemRow
FIELDS_OF(RsGemRow, RS_GEM_ROW_FIELDS)
#undef STRUCT
#define STRUCT RsFishRow
FIELDS_OF(RsFishRow, RS_FISH_ROW_FIELDS)
#undef STRUCT
#define STRUCT RsEatTrigger
FIELDS_OF(RsEatTrigger, RS_EAT_TRIGGER_FIELDS)
#undef STRUCT
#define STRUCT RsEatRow
FIELDS_OF(RsEatRow, RS_EAT_ROW_FIELDS)
#undef STRUCT
#define STRUCT RsWieldRule
FIELDS_OF(RsWieldRule, RS_WIELD_RULE_FIELDS)
#undef STRUCT
#define STRUCT RsStyleTable
FIELDS_OF(RsStyleTable, RS_STYLE_TABLE_FIELDS)
#undef STRUCT
#define STRUCT RsLevelupRow
FIELDS_OF(RsLevelupRow, RS_LEVELUP_ROW_FIELDS)
#undef STRUCT
#define STRUCT RsUnlock
FIELDS_OF(RsUnlock, RS_UNLOCK_FIELDS)
#undef STRUCT
#define STRUCT RsMsg
FIELDS_OF(RsMsg, RS_MSG_FIELDS)
#undef STRUCT
#define STRUCT RsToken
FIELDS_OF(RsToken, RS_TOKEN_FIELDS)
#undef STRUCT
#define STRUCT RsMsgKeyRow
FIELDS_OF(RsMsgKeyRow, RS_MSG_KEY_FIELDS)
#undef STRUCT
#define STRUCT RsDropOp
FIELDS_OF(RsDropOp, RS_DROP_OP_FIELDS)
#undef STRUCT
#define STRUCT RsDropRoot
FIELDS_OF(RsDropRoot, RS_DROP_ROOT_FIELDS)
#undef STRUCT
#define STRUCT RsEnumRow
FIELDS_OF(RsEnumRow, RS_ENUM_ROW_FIELDS)
#undef STRUCT
#define STRUCT RsLayoutRec
FIELDS_OF(RsLayoutRec, RS_LAYOUT_REC_FIELDS)
#undef STRUCT
#define STRUCT RsVariantKey
FIELDS_OF(RsVariantKey, RS_VARIANT_KEY_FIELDS)
#undef STRUCT
#define STRUCT RsConsts
FIELDS_OF(RsConsts, RS_CONSTS_FIELDS)
#undef STRUCT

static const struct { const char *name; size_t size; const FieldDesc *fields; } STRUCTS[] = {
#define ROW(type) {#type, sizeof(type), F_##type},
    ROW(RsLocType) ROW(RsNpcType) ROW(RsObjType) ROW(RsTreeRow) ROW(RsRockRow) ROW(RsGemRow) ROW(RsFishRow)
    ROW(RsEatTrigger) ROW(RsEatRow) ROW(RsWieldRule) ROW(RsStyleTable) ROW(RsLevelupRow) ROW(RsUnlock) ROW(RsMsg)
    ROW(RsToken) ROW(RsMsgKeyRow) ROW(RsDropOp) ROW(RsDropRoot) ROW(RsEnumRow) ROW(RsLayoutRec) ROW(RsVariantKey)
    ROW(RsConsts)
#undef ROW
};

static PyObject *struct_layouts(PyObject *self, PyObject *unused) {
    (void)self, (void)unused;
    PyObject *out = PyDict_New();
    for (size_t i = 0; out && i < sizeof STRUCTS / sizeof *STRUCTS; i++) {
        PyObject *fields = PyList_New(0);
        for (const FieldDesc *f = STRUCTS[i].fields; fields && f->name; f++) {
            PyObject *row = Py_BuildValue("(ssni)", f->name, f->ctype, (Py_ssize_t)f->offset, f->count);
            if (!row || PyList_Append(fields, row)) Py_CLEAR(fields);
            Py_XDECREF(row);
        }
        PyObject *entry = fields ? Py_BuildValue("(nN)", (Py_ssize_t)STRUCTS[i].size, fields) : NULL;
        if (!entry || PyDict_SetItemString(out, STRUCTS[i].name, entry)) Py_CLEAR(out);
        Py_XDECREF(entry);
    }
    return out;
}

#define REASON_NAME(id, exc, fmt, kinds) #id,
static const char *const REASON_NAMES[RS_N_REASONS] = { RS_REASONS(REASON_NAME) };
#undef REASON_NAME
static const char *const EXC_NAMES[] = {"EXC_NONE", "EXC_UNSUPPORTED", "EXC_KEYERROR", "EXC_VALUEERROR", "EXC_OUTSIDE",
                                        "EXC_AUDIT", "EXC_RUNTIME", "EXC_INDEX", "EXC_NOT_PORTED",
                                        "EXC_ZERODIV"};  /* rs.h RsExc's last (rs2.py arithmetic) */
#define CANON_NAME(id) #id,
static const char *const CANON_NAMES[RS_N_CANON_FIELDS] = { RS_CANON_FIELDS(CANON_NAME) };
#undef CANON_NAME
#define SECTION_ROW(name) {#name, RS_SEC_##name},
static const struct { const char *name; int id; } SECTIONS[] = { RS_SECTIONS(SECTION_ROW) };
#undef SECTION_ROW

static PyObject *reasons(PyObject *self, PyObject *unused) {
    (void)self, (void)unused;
    PyObject *out = PyList_New(RS_N_REASONS);
    for (int i = 0; out && i < RS_N_REASONS; i++) {
        PyObject *row = Py_BuildValue("(siss)", REASON_NAMES[i], RS_REASON_INFO[i].exc, RS_REASON_INFO[i].format,
                                      RS_REASON_INFO[i].kinds);
        if (!row) Py_CLEAR(out);
        else PyList_SET_ITEM(out, i, row);
    }
    return out;
}

static PyObject *string_list(const char *const *names, int n) {
    PyObject *out = PyList_New(n);
    for (int i = 0; out && i < n; i++) {
        PyObject *s = PyUnicode_FromString(names[i]);
        if (!s) Py_CLEAR(out);
        else PyList_SET_ITEM(out, i, s);
    }
    return out;
}

static PyObject *exceptions(PyObject *self, PyObject *unused) {
    (void)self, (void)unused;
    return string_list(EXC_NAMES, (int)(sizeof EXC_NAMES / sizeof *EXC_NAMES));
}

static PyObject *canon_fields(PyObject *self, PyObject *unused) {
    (void)self, (void)unused;
    return string_list(CANON_NAMES, RS_N_CANON_FIELDS);
}

static PyObject *scripts(PyObject *self, PyObject *unused) {
    (void)self, (void)unused;
    PyObject *out = PyList_New(RS_N_SCRIPTS);
    for (int i = 0; out && i < RS_N_SCRIPTS; i++) {
        PyObject *row = Py_BuildValue("(ssi)", RS_SCRIPT_INFO[i].name, RS_SCRIPT_INFO[i].qualname,
                                      RS_SCRIPT_INFO[i].status);
        if (!row) Py_CLEAR(out);
        else PyList_SET_ITEM(out, i, row);
    }
    return out;
}

static int set_int(PyObject *d, const char *key, long long v) {
    PyObject *o = PyLong_FromLongLong(v);
    int r = o ? PyDict_SetItemString(d, key, o) : -1;
    Py_XDECREF(o);
    return r;
}

static PyObject *constants(PyObject *self, PyObject *unused) {
    (void)self, (void)unused;
    PyObject *out = PyDict_New(), *sections = PyDict_New(), *obs = PyList_New(RS_OBS_N_KEYS);
    PyObject *info = PyList_New(RS_INFO_N);
    if (!out || !sections || !obs || !info) goto fail;
    for (size_t i = 0; i < sizeof SECTIONS / sizeof *SECTIONS; i++)
        if (set_int(sections, SECTIONS[i].name, SECTIONS[i].id)) goto fail;
    if (set_int(sections, "USER", RS_SEC_USER) || PyDict_SetItemString(out, "sections", sections)) goto fail;
    for (int k = 0; k < RS_OBS_N_KEYS; k++) {
        PyObject *row = Py_BuildValue("(sCi(ii)iii)", RS_OBS_NAMES[k], RS_OBS_FORMAT[k], RS_OBS_NDIM[k],
                                      RS_OBS_SHAPE[2 * k], RS_OBS_SHAPE[2 * k + 1], RS_OBS_NELEM[k], RS_OBS_OFFSET[k],
                                      RS_OBS_NBYTES[k]);
        if (!row) goto fail;
        PyList_SET_ITEM(obs, k, row);
    }
    for (int k = 0; k < RS_INFO_N; k++) {
        PyObject *row = Py_BuildValue("(sC)", RS_INFO_NAMES[k], RS_INFO_FORMAT[k]);
        if (!row) goto fail;
        PyList_SET_ITEM(info, k, row);
    }
    if (PyDict_SetItemString(out, "obs", obs) || PyDict_SetItemString(out, "info", info)) goto fail;
    static const double LOG1P_K[RS_LOG1P_N] = RS_LOG1P_K;
    PyObject *k = Py_BuildValue("(dddd)", LOG1P_K[0], LOG1P_K[1], LOG1P_K[2], LOG1P_K[3]);
    int failed_k = !k || PyDict_SetItemString(out, "RS_LOG1P_K", k);
    Py_XDECREF(k);
    if (failed_k || set_int(out, "RS_LOG1P_COUNTS", RS_LOG1P_COUNTS)) goto fail;
    if (set_int(out, "RS_OBS_BYTES", RS_OBS_BYTES) || set_int(out, "RS_PACK_VERSION", RS_PACK_VERSION)
        || set_int(out, "RS_PACK_ALIGN", RS_PACK_ALIGN) || set_int(out, "RS_MAX_LOCS", RS_MAX_LOCS)
        || set_int(out, "RS_MAX_NPCS", RS_MAX_NPCS) || set_int(out, "RS_MAX_OBJS", RS_MAX_OBJS)
        || set_int(out, "RS_MAX_REQS", RS_MAX_REQS) || set_int(out, "RS_MAX_EVENTS", RS_MAX_EVENTS)
        || set_int(out, "RS_FRAME_DEPTH", RS_FRAME_DEPTH) || set_int(out, "RS_N_FACTS", RS_N_FACTS)
        || set_int(out, "sizeof_World", sizeof(World)) || set_int(out, "sizeof_Agent", sizeof(Agent))
        || set_int(out, "sizeof_Episode", sizeof(Episode)) || set_int(out, "sizeof_RsSlot", sizeof(RsSlot)))
        goto fail;
    Py_DECREF(sections);
    Py_DECREF(obs);
    Py_DECREF(info);
    return out;
fail:
    Py_XDECREF(out);
    Py_XDECREF(sections);
    Py_XDECREF(obs);
    Py_XDECREF(info);
    return NULL;
}

/* ---- the keyed RNG and helpers, for the equivalence tests --------------------------------------------------- */
static int keys_of(PyObject *seq, int64_t *keys, int cap, int *n) {
    PyObject *fast = PySequence_Fast(seq, "keys must be a sequence of ints");
    if (!fast) return -1;
    Py_ssize_t len = PySequence_Fast_GET_SIZE(fast);
    if (len > cap) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "too many keys");
        return -1;
    }
    for (Py_ssize_t i = 0; i < len; i++) {
        keys[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(fast, i));
        if (keys[i] == -1 && PyErr_Occurred()) {
            Py_DECREF(fast);
            return -1;
        }
    }
    *n = (int)len;
    Py_DECREF(fast);
    return 0;
}

static PyObject *rng_seed(PyObject *self, PyObject *args) {
    (void)self;
    long long seed;
    if (!PyArg_ParseTuple(args, "L", &seed)) return NULL;
    return PyLong_FromUnsignedLong(rs_seed(seed));
}

static PyObject *rng_chain(PyObject *args, int what) {
    unsigned long h;
    PyObject *seq;
    int64_t keys[64];
    int n;
    if (!PyArg_ParseTuple(args, "kO", &h, &seq) || keys_of(seq, keys, 64, &n)) return NULL;
    uint32_t h32 = (uint32_t)h;
    if (what == 0) return PyLong_FromUnsignedLong(rs_prefix(h32, n, keys));
    if (what == 1) return PyLong_FromUnsignedLong(rs_m(h32, n, keys));
    return PyFloat_FromDouble(rs_draw(h32, n, keys));
}
static PyObject *rng_prefix(PyObject *self, PyObject *args) { (void)self; return rng_chain(args, 0); }
static PyObject *rng_m(PyObject *self, PyObject *args) { (void)self; return rng_chain(args, 1); }
static PyObject *rng_draw(PyObject *self, PyObject *args) { (void)self; return rng_chain(args, 2); }

static PyObject *rng_pick(PyObject *self, PyObject *args) {
    (void)self;
    unsigned long m;
    long long n;
    if (!PyArg_ParseTuple(args, "kL", &m, &n)) return NULL;
    return PyLong_FromLongLong(rs_pick((uint32_t)m, n));
}

static PyObject *js_round(PyObject *self, PyObject *args) {
    (void)self;
    double v;
    if (!PyArg_ParseTuple(args, "d", &v)) return NULL;
    return PyLong_FromLongLong(rs_js_round(v));
}

static PyObject *int_op(PyObject *args, int what) {
    long long a, b;
    if (!PyArg_ParseTuple(args, "LL", &a, &b)) return NULL;
    if (what < 2 && (b == 0 || (a == INT64_MIN && b == -1))) {
        PyErr_SetString(PyExc_ValueError, "division undefined in int64");
        return NULL;
    }
    return PyLong_FromLongLong(what == 0 ? rs_floordiv(a, b) : what == 1 ? rs_mod(a, b) : rs_asr(a, (int)b));
}
static PyObject *floordiv(PyObject *self, PyObject *args) { (void)self; return int_op(args, 0); }
static PyObject *mod(PyObject *self, PyObject *args) { (void)self; return int_op(args, 1); }
static PyObject *asr(PyObject *self, PyObject *args) { (void)self; return int_op(args, 2); }

/* ---- start records ------------------------------------------------------------------------------------------- */
static int set_list(PyObject *d, const char *key, const int64_t *v, int n) {
    PyObject *list = PyList_New(n);
    for (int i = 0; list && i < n; i++) {
        PyObject *o = PyLong_FromLongLong(v[i]);
        if (!o) Py_CLEAR(list);
        else PyList_SET_ITEM(list, i, o);
    }
    int r = list ? PyDict_SetItemString(d, key, list) : -1;
    Py_XDECREF(list);
    return r;
}

/* An RsStart as raw ints (sim/native.py decode_start), plus 'seed': its pool layout's seed (None before a start). */
static PyObject *start_dict(const RsPack *P, const RsStart *s) {
    PyObject *d = PyDict_New();
    int64_t levels[7], food[4], mx[1], mz[1];
    for (int i = 0; i < 7; i++) levels[i] = s->levels[i];
    for (int i = 0; i < 4; i++) food[i] = s->food[i];
    mx[0] = s->mugger_x[0];
    mz[0] = s->mugger_z[0];
    int n_food = s->n_food < 0 ? 0 : s->n_food > 4 ? 4 : s->n_food;
    int n_muggers = s->n_muggers < 0 ? 0 : s->n_muggers > 1 ? 1 : s->n_muggers;
    if (!d || set_int(d, "kind", s->kind) || set_int(d, "layout", s->layout) || set_int(d, "tile_x", s->tile_x)
        || set_int(d, "tile_z", s->tile_z) || set_int(d, "weapon", s->weapon) || set_list(d, "levels", levels, 7)
        || set_int(d, "world_seed", s->world_seed) || set_int(d, "group", s->group) || set_int(d, "hp", s->hp)
        || set_list(d, "food", food, n_food) || set_int(d, "worn", s->worn) || set_list(d, "mugger_x", mx, n_muggers)
        || set_list(d, "mugger_z", mz, n_muggers) || set_int(d, "grove", s->grove) || set_int(d, "variant", s->variant)
        || set_int(d, "rec", s->rec)) {
        Py_XDECREF(d);
        return NULL;
    }
    int32_t pool = (s->kind == RS_START_SUPPLY ? P->supply0 : P->compact0) + s->layout;
    int ok = s->layout >= 0 && pool >= 0 && pool < P->n_layouts
        && P->layouts[pool].kind == (s->kind == RS_START_SUPPLY ? RS_LAYOUT_SUPPLY : RS_LAYOUT_COMPACT);
    if (ok ? set_int(d, "seed", P->layouts[pool].rec->seed) : PyDict_SetItemString(d, "seed", Py_None)) {
        Py_DECREF(d);
        return NULL;
    }
    return d;
}

/* ---- Pack ------------------------------------------------------------------------------------------------------ */
typedef struct { PyObject_HEAD const RsPack *pack; } PackObject;

static void Pack_dealloc(PackObject *self) {
    rs_pack_close(self->pack);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Pack_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    static char *names[] = {"path", NULL};
    const char *path;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", names, &path)) return NULL;
    const RsPack *pack;
    RsErr err;
    int failed;
    Py_BEGIN_ALLOW_THREADS
    failed = rs_pack_open(path, &pack, &err);
    Py_END_ALLOW_THREADS
    if (failed) return raise_err(&err, -1);
    PackObject *self = (PackObject *)type->tp_alloc(type, 0);
    if (!self) {
        rs_pack_close(pack);
        return NULL;
    }
    self->pack = pack;
    return (PyObject *)self;
}

static PyObject *Pack_manifest(PackObject *self, void *closure) {
    (void)closure;
    return PyBytes_FromStringAndSize(self->pack->manifest, (Py_ssize_t)self->pack->manifest_len);
}

static PyObject *Pack_n_strings(PackObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->pack->n_strings);
}

static PyObject *Pack_string(PackObject *self, PyObject *args) {
    int id;
    if (!PyArg_ParseTuple(args, "i", &id)) return NULL;
    return PyUnicode_FromString(rs_str(self->pack, id));
}

static PyObject *Pack_sections(PackObject *self, PyObject *unused) {
    (void)unused;
    const RsPack *P = self->pack;
    PyObject *out = PyList_New((Py_ssize_t)P->n_sections);
    for (uint32_t i = 0; out && i < P->n_sections; i++) {
        const RsSection *s = &P->sections[i];
        PyObject *row = Py_BuildValue("(IIKIKK)", s->id, s->dtype, (unsigned long long)s->count, s->elem,
                                      (unsigned long long)s->nbytes, (unsigned long long)s->off);
        if (!row) Py_CLEAR(out);
        else PyList_SET_ITEM(out, i, row);
    }
    return out;
}

/* A copy of a section's bytes (debugging and tests; the engine reads the mapping). */
static PyObject *Pack_section(PackObject *self, PyObject *args) {
    unsigned int id;
    if (!PyArg_ParseTuple(args, "I", &id)) return NULL;
    uint64_t count;
    uint32_t elem;
    const void *data = rs_pack_section(self->pack, id, &count, &elem);
    if (!data) {
        PyErr_Format(PyExc_KeyError, "no section %u", id);
        return NULL;
    }
    return PyBytes_FromStringAndSize(data, (Py_ssize_t)(count * elem));
}

static PyObject *Pack_check_declared(PackObject *self, PyObject *unused) {
    (void)unused;
    RsErr err;
    if (rs_pack_check_declared(self->pack, &err)) return raise_err(&err, -1);
    Py_RETURN_NONE;
}

/* ---- TEST HOOKS: the task draws and ledgers for sim/tests/test_native_draws.py ----------------------- */
/* Pack.draw_start(pool, key, slot, episode): rs_draw_start's record (start_dict), raising Error. */
static PyObject *Pack_draw_start(PackObject *self, PyObject *args) {
    int pool, slot, episode;
    long long key;
    if (!PyArg_ParseTuple(args, "iLii", &pool, &key, &slot, &episode)) return NULL;
    RsStart start;
    RsErr err;
    memset(&err, 0, sizeof err);
    if (rs_draw_start(self->pack, pool, key, slot, episode, &start, &err)) return raise_err(&err, -1);
    return start_dict(self->pack, &start);
}

/* Pack.episode_draw(task, goal, key, slot, episode): rs_episode_draw for a slot of that task and goal, with the
   start's kit and, for a combat start, its npc spawns: (start_dict, [(obj, count)], [(nid, type, x, z, level, size)]
   or None). Raises Error. */
static PyObject *Pack_episode_draw(PackObject *self, PyObject *args) {
    int task, goal, slot, episode;
    long long key;
    if (!PyArg_ParseTuple(args, "iiLii", &task, &goal, &key, &slot, &episode)) return NULL;
    RsSlot *s = calloc(1, sizeof *s);
    if (!s) return PyErr_NoMemory();
    RsConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.task = task;
    s->P = self->pack, s->cfg = &cfg, s->goal = goal, s->slot = slot, s->episode = episode;
    RsStart start;
    RsErr err;
    memset(&err, 0, sizeof err);
    RsNpcSpawn npcs[RS_MAX_NPCS];
    int n_npcs = 0, failed = rs_episode_draw(s, key, &start, &err);
    free(s);
    if (!failed && start.kind == RS_START_COMBAT) failed = rs_combat_npcs(self->pack, &start, npcs, &n_npcs, &err);
    if (failed) return raise_err(&err, -1);
    int32_t obj[RS_START_KIT_MAX], count[RS_START_KIT_MAX];
    int n_kit = rs_start_kit(self->pack, &start, obj, count);
    PyObject *d = start_dict(self->pack, &start), *kit = PyList_New(n_kit);
    PyObject *spawns = start.kind == RS_START_COMBAT ? PyList_New(n_npcs) : (Py_INCREF(Py_None), Py_None);
    for (int i = 0; kit && i < n_kit; i++) {
        PyObject *row = Py_BuildValue("(ii)", obj[i], count[i]);
        if (!row) Py_CLEAR(kit);
        else PyList_SET_ITEM(kit, i, row);
    }
    for (int i = 0; spawns && spawns != Py_None && i < n_npcs; i++) {
        const RsNpcSpawn *n = &npcs[i];
        PyObject *row = Py_BuildValue("(iiiiii)", n->nid, n->type, n->x, n->z, n->level, n->size);
        if (!row) Py_CLEAR(spawns);
        else PyList_SET_ITEM(spawns, i, row);
    }
    if (!d || !kit || !spawns) {
        Py_XDECREF(d), Py_XDECREF(kit), Py_XDECREF(spawns);
        return NULL;
    }
    return Py_BuildValue("(NNN)", d, kit, spawns);
}

static int int_items(PyObject *seq, int64_t *out, int n, const char *what) {
    PyObject *fast = PySequence_Fast(seq, what);
    if (!fast) return -1;
    if (PySequence_Fast_GET_SIZE(fast) != n) {
        Py_DECREF(fast);
        PyErr_Format(PyExc_ValueError, "%s: %d ints", what, n);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        out[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(fast, i));
        if (out[i] == -1 && PyErr_Occurred()) {
            Py_DECREF(fast);
            return -1;
        }
    }
    Py_DECREF(fast);
    return 0;
}

/* Each item of seq as k ints, through fn(row, ctx); -1 with a Python error. */
static int each_row(PyObject *seq, int k, int cap, const char *what, int (*fn)(const int64_t *, int, void *), void *ctx) {
    PyObject *fast = PySequence_Fast(seq, what);
    if (!fast) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n > cap) {
        Py_DECREF(fast);
        PyErr_Format(PyExc_ValueError, "%s: more than %d rows", what, cap);
        return -1;
    }
    int64_t row[16];
    for (Py_ssize_t i = 0; i < n; i++)
        if (int_items(PySequence_Fast_GET_ITEM(fast, i), row, k, what) || fn(row, (int)i, ctx)) {
            Py_DECREF(fast);
            return -1;
        }
    Py_DECREF(fast);
    return 0;
}

static int put_event(const int64_t *v, int i, void *ctx) {
    RsEvent *e = &((RsEvent *)ctx)[i];
    *e = (RsEvent){.tick = (int32_t)v[0], .kind = (uint8_t)v[1], .a8 = (uint8_t)v[2], .b8 = (uint8_t)v[3],
                   .has_source = (uint8_t)v[4], .obj = (int32_t)v[5], .count = (int32_t)v[6], .v0 = (int32_t)v[7],
                   .v1 = (int32_t)v[8], .v2 = (int32_t)v[9], .v3 = (int32_t)v[10]};
    return 0;
}

typedef struct { Slot *slots; int cap; } SlotRows;
static int put_slot(const int64_t *v, int i, void *ctx) {
    (void)i;
    SlotRows *r = ctx;
    if (v[0] < 0 || v[0] >= r->cap) {
        PyErr_SetString(PyExc_ValueError, "holdings: a slot outside its inventory");
        return -1;
    }
    r->slots[v[0]] = (Slot){.id = (int32_t)v[1], .count = (int32_t)v[2]};
    return 0;
}

/* holdings() as (inventory, bank, worn) lists of (slot, obj, count) into an RsHoldings of empty slots. */
static int put_holdings(PyObject *rows, RsHoldings *h) {
    for (int i = 0; i < RS_INV_CAP; i++) h->inv[i] = (Slot){-1, 0};
    for (int i = 0; i < RS_BANK_CAP; i++) h->bank[i] = (Slot){-1, 0};
    for (int i = 0; i < RS_WORN_CAP; i++) h->worn[i] = (Slot){-1, 0};
    PyObject *fast = PySequence_Fast(rows, "holdings: (inventory, bank, worn)");
    if (!fast) return -1;
    SlotRows parts[3] = {{h->inv, RS_INV_CAP}, {h->bank, RS_BANK_CAP}, {h->worn, RS_WORN_CAP}};
    int failed = PySequence_Fast_GET_SIZE(fast) != 3;
    if (failed) PyErr_SetString(PyExc_ValueError, "holdings: (inventory, bank, worn)");
    for (int k = 0; !failed && k < 3; k++)
        failed = each_row(PySequence_Fast_GET_ITEM(fast, k), 3, parts[k].cap, "holdings rows", put_slot, &parts[k]) != 0;
    Py_DECREF(fast);
    return failed ? -1 : 0;
}

static int put_log(const int64_t *v, int i, void *ctx) {
    RsSupply *t = ctx;
    t->log_tick[i] = (int32_t)v[0], t->log_local[i] = (uint8_t)v[1], t->n_logs = i + 1;
    return 0;
}

static PyObject *int_list(const int32_t *v, int n) {
    PyObject *out = PyList_New(n);
    for (int i = 0; out && i < n; i++) {
        PyObject *o = PyLong_FromLong(v[i]);
        if (!o) Py_CLEAR(out);
        else PyList_SET_ITEM(out, i, o);
    }
    return out;
}

/* Pack.ledgers(goal, enemies, target, events, cuts, terminal, initial, final, kind, group, variant, visited,
   exhausted, logs): an episode's ledgers over its events since arm (each an 11-int RsEvent row: tick, kind, a8, b8,
   has_source, obj, count, v0, v1, v2, v3). rs_ledger_update runs once per cut (the event count after each step);
   with a terminal (reason, tick), rs_audit, rs_fights and rs_facts follow (kind, group, variant: the RsStart's;
   visited: the agent's cell count; exhausted and logs [(tick, local)]: the SupplyTracker's). Returns dict(steps=[(made
   wood, fish, mine, kills)], error, audit, fights, facts), a failed call's error args in place of its result. */
static PyObject *Pack_ledgers(PackObject *self, PyObject *args, PyObject *kwds) {
    static char *names[] = {"goal", "enemies", "target", "events", "cuts", "terminal", "initial", "final", "kind",
                            "group", "variant", "visited", "exhausted", "logs", NULL};
    int goal, target, kind, group, variant, visited, exhausted;
    PyObject *enemies, *events, *cuts, *terminal, *initial, *final, *logs;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iOiOOOOOiiiiiO", names, &goal, &enemies, &target, &events, &cuts,
                                     &terminal, &initial, &final, &kind, &group, &variant, &visited, &exhausted, &logs))
        return NULL;
    RsSlot *s = calloc(1, sizeof *s);
    if (!s) return PyErr_NoMemory();
    PyObject *out = NULL, *steps = NULL, *error = NULL, *audit = NULL, *fights = NULL, *facts = NULL, *fast = NULL;
    Episode *ep = &s->ep;
    s->P = s->w.P = self->pack;
    ep->goal = (int8_t)goal, ep->target = target;
    int64_t row[4];
    Py_ssize_t n_enemies = PySequence_Size(enemies);
    if (n_enemies > 4) PyErr_SetString(PyExc_ValueError, "enemies: at most 4 npc types");
    if (n_enemies < 0 || n_enemies > 4 || int_items(enemies, row, (int)n_enemies, "enemies")) goto done;
    ep->n_enemies = (int32_t)n_enemies;
    for (int i = 0; i < n_enemies; i++) ep->enemies[i] = (int32_t)row[i];
    Py_ssize_t n_events = PySequence_Size(events);
    if (n_events < 0 || each_row(events, 11, RS_MAX_EVENTS, "events", put_event, s->w.ev)) goto done;
    if (put_holdings(initial, &ep->initial) || put_holdings(final, &ep->term_holdings)) goto done;
    if (terminal != Py_None) {
        if (int_items(terminal, row, 2, "terminal: (reason, tick)")) goto done;
        ep->terminal = 1, ep->term_reason = (uint8_t)row[0], ep->term_tick = (int32_t)row[1];
    }
    s->start.kind = kind, s->start.group = group, s->start.variant = variant;
    s->a.visited = visited >= 64 ? ~(uint64_t)0 : ((uint64_t)1 << visited) - 1;
    s->supply.exhausted = exhausted;
    if (each_row(logs, 2, RS_SUPPLY_LOGS, "logs", put_log, &s->supply)) goto done;
    steps = PyList_New(0);
    if (!steps) goto done;
    fast = PySequence_Fast(cuts, "cuts");
    if (!fast) goto done;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast) && !error; i++) {
        long cut = PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, i));
        if (cut < 0 || cut > n_events) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "a cut outside the events");
            goto done;
        }
        s->w.n_events = (int32_t)cut;
        if (rs_ledger_update(s)) {
            error = err_tuple(&s->w.err, -1);
        } else {
            PyObject *step = Py_BuildValue("(iiii)", ep->made[0], ep->made[1], ep->made[2], ep->kills);
            if (!step || PyList_Append(steps, step)) {
                Py_XDECREF(step);
                goto done;
            }
            Py_DECREF(step);
        }
    }
    s->w.n_events = (int32_t)n_events;
    if (!error) error = (Py_INCREF(Py_None), Py_None);
    if (error == Py_None && ep->terminal) {
        RsAudit report;
        RsErr err;
        memset(&err, 0, sizeof err);
        int bad = rs_audit(s, &report, &err);
        audit = bad ? err_tuple(&err, -1) : Py_BuildValue(
            "{s:i,s:i,s:i,s:i,s:i,s:i,s:i,s:i}", "survived", report.survived, "reason", report.reason, "tick",
            report.tick, "produced", report.produced, "retained", report.retained, "retained_upper",
            report.retained_upper, "picked", report.picked, "kills", report.kills);
        int32_t f[6], all[RS_N_FACTS];
        memset(&err, 0, sizeof err);
        fights = rs_fights(s, f, &err) ? err_tuple(&err, -1) : int_list(f, 6);
        if (bad) {
            facts = (Py_INCREF(Py_None), Py_None);
        } else {
            memset(&err, 0, sizeof err);
            facts = rs_facts(s, &report, all, &err) ? err_tuple(&err, -1) : int_list(all, RS_N_FACTS);
        }
    } else {
        audit = (Py_INCREF(Py_None), Py_None);
        fights = (Py_INCREF(Py_None), Py_None);
        facts = (Py_INCREF(Py_None), Py_None);
    }
    if (error && audit && fights && facts)
        out = Py_BuildValue("{s:O,s:O,s:O,s:O,s:O}", "steps", steps, "error", error, "audit", audit, "fights", fights,
                            "facts", facts);
done:
    free(s);
    Py_XDECREF(fast), Py_XDECREF(steps), Py_XDECREF(error), Py_XDECREF(audit), Py_XDECREF(fights), Py_XDECREF(facts);
    return out;
}

static PyMethodDef Pack_methods[] = {
    {"string", (PyCFunction)Pack_string, METH_VARARGS, "the string of a string id ('' out of range)"},
    {"sections", (PyCFunction)Pack_sections, METH_NOARGS, "[(id, dtype, count, elem, nbytes, offset)]"},
    {"section", (PyCFunction)Pack_section, METH_VARARGS, "a copy of a section's bytes"},
    {"check_declared", (PyCFunction)Pack_check_declared, METH_NOARGS, "rs_pack_check_declared (raises Error)"},
    /* TEST HOOKS */
    {"draw_start", (PyCFunction)Pack_draw_start, METH_VARARGS, "rs_draw_start(pool, key, slot, episode): the start"},
    {"episode_draw", (PyCFunction)Pack_episode_draw, METH_VARARGS,
     "rs_episode_draw(task, goal, key, slot, episode): (start, kit, combat npcs or None)"},
    {"ledgers", (PyCFunction)(void (*)(void))Pack_ledgers, METH_VARARGS | METH_KEYWORDS,
     "an episode's step ledgers and terminal accounts over recorded events"},
    {NULL, NULL, 0, NULL},
};
static PyGetSetDef Pack_getset[] = {
    {"manifest", (getter)Pack_manifest, NULL, "the manifest JSON (bytes)", NULL},
    {"n_strings", (getter)Pack_n_strings, NULL, "strings in the pack", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};
static PyTypeObject PackType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sim._native.Pack",
    .tp_basicsize = sizeof(PackObject),
    .tp_dealloc = (destructor)Pack_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "A pack file mapped read-only (rs_pack_open).",
    .tp_methods = Pack_methods,
    .tp_getset = Pack_getset,
    .tp_new = Pack_new,
};

/* ---- Batch ------------------------------------------------------------------------------------------------------ */
enum { OUT_OBS = 0, OUT_INFO = OUT_OBS + RS_OBS_N_KEYS, OUT_MADE = OUT_INFO + RS_INFO_N, OUT_KILLS, OUT_FACTS,
       OUT_REWARD, OUT_DONE, OUT_N };

typedef struct {
    PyObject_HEAD
    RsBatch *batch;
    PyObject *pack;
    Py_buffer log1p;           /* the tables every slot reads, held for the batch's life */
    Py_buffer views[OUT_N];
    int bound;
    RsOut out;
    RsTrace **traces;          /* [n]: a world's trace (canon.c) or NULL */
} BatchObject;

static void release_views(Py_buffer *views, int n) {
    for (int i = 0; i < n; i++)
        if (views[i].obj) PyBuffer_Release(&views[i]);
}

static void Batch_dealloc(BatchObject *self) {
    if (self->bound) release_views(self->views, OUT_N);
    for (int32_t b = 0; self->traces && self->batch && b < self->batch->n; b++) rs_trace_free(self->traces[b]);
    free(self->traces);
    rs_batch_free(self->batch);
    if (self->log1p.obj) PyBuffer_Release(&self->log1p);
    Py_XDECREF(self->pack);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int cfg_int(PyObject *cfg, const char *key, int32_t *out) {
    PyObject *v = PyDict_GetItemString(cfg, key);  /* borrowed */
    if (!v) {
        PyErr_Format(PyExc_KeyError, "batch config lacks %s", key);
        return -1;
    }
    long x = PyLong_AsLong(v);
    if (x == -1 && PyErr_Occurred()) return -1;
    *out = (int32_t)x;
    return 0;
}

/* The log1p tables: float32, C-contiguous, [RS_LOG1P_N][RS_LOG1P_COUNTS]. */
static int take_log1p(PyObject *array, Py_buffer *view) {
    if (PyObject_GetBuffer(array, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) return -1;
    const char *f = view->format ? view->format : "B";
    if (*f == '<' || *f == '=' || *f == '@') f++;
    if (f[0] != 'f' || f[1] != '\0' || view->ndim != 2 || view->shape[0] != RS_LOG1P_N
        || view->shape[1] != RS_LOG1P_COUNTS) {
        PyBuffer_Release(view);
        PyErr_Format(PyExc_ValueError, "log1p: float32 [%d][%d] (sim/native.py log1p_table)", RS_LOG1P_N,
                     RS_LOG1P_COUNTS);
        return -1;
    }
    return 0;
}

static PyObject *Batch_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    static char *names[] = {"pack", "cfg", "log1p", NULL};
    PyObject *pack, *cfg, *log1p;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O", names, &PackType, &pack, &PyDict_Type, &cfg, &log1p))
        return NULL;
    RsConfig c;
    memset(&c, 0, sizeof c);
    if (cfg_int(cfg, "task", &c.task) || cfg_int(cfg, "n", &c.n) || cfg_int(cfg, "first", &c.first)
        || cfg_int(cfg, "n_goals", &c.n_goals) || cfg_int(cfg, "goal", &c.goal) || cfg_int(cfg, "horizon", &c.horizon)
        || cfg_int(cfg, "pace_jitter", &c.pace_jitter) || cfg_int(cfg, "decision_points", &c.decision_points)
        || cfg_int(cfg, "max_decisions", &c.max_decisions) || cfg_int(cfg, "max_episode_ticks", &c.max_episode_ticks))
        return NULL;
    c.threads = 1;
    if (PyDict_GetItemString(cfg, "threads") && cfg_int(cfg, "threads", &c.threads)) return NULL;
    Py_buffer table;
    if (take_log1p(log1p, &table)) return NULL;
    RsBatch *batch;
    RsErr err;
    if (rs_batch_new(((PackObject *)pack)->pack, &c, table.buf, &batch, &err)) {
        PyBuffer_Release(&table);
        return raise_err(&err, -1);
    }
    BatchObject *self = (BatchObject *)type->tp_alloc(type, 0);
    if (!self) {
        rs_batch_free(batch);
        PyBuffer_Release(&table);
        return NULL;
    }
    self->batch = batch;
    self->log1p = table;
    Py_INCREF(pack);
    self->pack = pack;
    self->traces = calloc((size_t)batch->n, sizeof *self->traces);
    if (!self->traces) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    return (PyObject *)self;
}

/* Acquire one bound array: C-contiguous, writable, of this buffer format, shape (n, d0[, d1]). */
static int take(PyObject *arrays, const char *name, char format, int ndim, int d0, int d1, Py_ssize_t n,
                Py_buffer *view) {
    PyObject *array = PyDict_GetItemString(arrays, name);  /* borrowed */
    if (!array) {
        PyErr_Format(PyExc_ValueError, "bind: %s missing", name);
        return -1;
    }
    if (PyObject_GetBuffer(array, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE)) {
        PyErr_Clear();
        PyErr_Format(PyExc_ValueError, "bind: %s is not a writable C-contiguous array", name);
        return -1;
    }
    const char *f = view->format ? view->format : "B";
    if (*f == '<' || *f == '=' || *f == '@') f++;
    int ok = f[0] == format && f[1] == '\0' && view->ndim == 1 + ndim && view->shape[0] == n
             && (ndim < 1 || view->shape[1] == d0) && (ndim < 2 || view->shape[2] == d1);
    if (!ok) {
        PyErr_Format(PyExc_ValueError, "bind: %s is not %c (%zd, ...) as sim/spec.py declares it", name, format, n);
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}

static PyObject *Batch_bind(BatchObject *self, PyObject *args) {
    PyObject *arrays;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &arrays)) return NULL;
    Py_buffer views[OUT_N];
    memset(views, 0, sizeof views);
    Py_ssize_t n = self->batch->n;
    char name[64];
    int failed = 0;
    for (int k = 0; k < RS_OBS_N_KEYS && !failed; k++) {
        snprintf(name, sizeof name, "obs/%s", RS_OBS_NAMES[k]);
        int ndim = RS_OBS_NDIM[k];
        failed = take(arrays, name, RS_OBS_FORMAT[k], ndim, RS_OBS_SHAPE[2 * k], RS_OBS_SHAPE[2 * k + 1], n,
                      &views[OUT_OBS + k]);
    }
    for (int k = 0; k < RS_INFO_N && !failed; k++) {
        snprintf(name, sizeof name, "info/%s", RS_INFO_NAMES[k]);
        failed = take(arrays, name, RS_INFO_FORMAT[k], 0, 0, 0, n, &views[OUT_INFO + k]);
    }
    if (!failed) failed = take(arrays, "info/made", 'i', 1, 3, 0, n, &views[OUT_MADE]);
    if (!failed) failed = take(arrays, "info/kills", 'i', 0, 0, 0, n, &views[OUT_KILLS]);
    if (!failed) failed = take(arrays, "info/facts", 'i', 1, RS_N_FACTS, 0, n, &views[OUT_FACTS]);
    if (!failed) failed = take(arrays, "reward", 'f', 0, 0, 0, n, &views[OUT_REWARD]);
    if (!failed) failed = take(arrays, "done", '?', 0, 0, 0, n, &views[OUT_DONE]);
    if (failed) {
        release_views(views, OUT_N);
        return NULL;
    }
    if (self->bound) release_views(self->views, OUT_N);
    memcpy(self->views, views, sizeof views);
    self->bound = 1;
    RsOut *o = &self->out;
    for (int k = 0; k < RS_OBS_N_KEYS; k++) o->obs[k] = views[OUT_OBS + k].buf;
    for (int k = 0; k < RS_INFO_N; k++) o->info[k] = views[OUT_INFO + k].buf;
    o->made = views[OUT_MADE].buf;
    o->kills = views[OUT_KILLS].buf;
    o->facts = views[OUT_FACTS].buf;
    o->reward = views[OUT_REWARD].buf;
    o->done = views[OUT_DONE].buf;
    Py_RETURN_NONE;
}

static PyObject *batch_failed(BatchObject *self) {
    return raise_err(&self->batch->err, self->batch->err_world);
}

static PyObject *Batch_reset(BatchObject *self, PyObject *args) {
    long long key;
    if (!PyArg_ParseTuple(args, "L", &key)) return NULL;
    if (!self->bound) {
        PyErr_SetString(PyExc_RuntimeError, "bind the output arrays first");
        return NULL;
    }
    int failed;
    Py_BEGIN_ALLOW_THREADS
    failed = rs_batch_reset(self->batch, key, &self->out);
    Py_END_ALLOW_THREADS
    if (failed) return batch_failed(self);
    Py_RETURN_NONE;
}

static PyObject *Batch_step(BatchObject *self, PyObject *args) {
    PyObject *actions;
    if (!PyArg_ParseTuple(args, "O", &actions)) return NULL;
    if (!self->bound) {
        PyErr_SetString(PyExc_RuntimeError, "bind the output arrays first");
        return NULL;
    }
    Py_buffer view;
    if (PyObject_GetBuffer(actions, &view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)) return NULL;
    const char *f = view.format ? view.format : "B";
    if (*f == '<' || *f == '=' || *f == '@') f++;
    if (f[0] != 'i' || f[1] != '\0' || view.ndim != 1 || view.shape[0] != self->batch->n) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "actions: int32 [n]");
        return NULL;
    }
    int failed;
    Py_BEGIN_ALLOW_THREADS
    failed = rs_batch_step(self->batch, view.buf, &self->out);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (failed) return batch_failed(self);
    Py_RETURN_NONE;
}

static PyObject *Batch_episodes(BatchObject *self, PyObject *unused) {
    (void)unused;
    PyObject *out = PyList_New(self->batch->n);
    for (int32_t b = 0; out && b < self->batch->n; b++) {
        PyObject *v = PyLong_FromLong(self->batch->slots[b].episode);
        if (!v) Py_CLEAR(out);
        else PyList_SET_ITEM(out, b, v);
    }
    return out;
}

static int world_arg(BatchObject *self, PyObject *args, int32_t *b) {
    int k;
    if (!PyArg_ParseTuple(args, "i", &k)) return -1;
    if (k < 0 || k >= self->batch->n) {
        PyErr_Format(PyExc_IndexError, "world %d of a batch of %d", k, (int)self->batch->n);
        return -1;
    }
    *b = k;
    return 0;
}

/* World b's RsStart as raw ints, plus 'seed': its pool layout's seed (None before a start). */
static PyObject *Batch_start(BatchObject *self, PyObject *args) {
    int32_t b;
    if (world_arg(self, args, &b)) return NULL;
    return start_dict(self->batch->P, &self->batch->slots[b].start);
}

/* The late log1p features (rs.h RsLate) of every world's last observation, the row each call writes for it:
   [(b, byte offset in the packed row, RS_LOG1P_* row, count)], for NativeBatch to fill. */
static PyObject *Batch_late(BatchObject *self, PyObject *unused) {
    (void)unused;
    PyObject *out = PyList_New(0);
    for (int32_t b = 0; out && b < self->batch->n; b++) {
        const RsSlot *s = &self->batch->slots[b];
        for (int32_t i = 0; i < s->n_late; i++) {
            PyObject *v = Py_BuildValue("(iiiL)", (int)b, (int)s->late[i].off, (int)s->late[i].k,
                                        (long long)s->late[i].count);
            if (!v || PyList_Append(out, v)) Py_CLEAR(out);
            Py_XDECREF(v);
        }
    }
    return out;
}

/* World b's unsupported evidence: None while it has not failed, else (error args as Error carries them, the decision
   (task.decisions), the tick, the action, the episode). */
static PyObject *Batch_failure(BatchObject *self, PyObject *args) {
    int32_t b;
    if (world_arg(self, args, &b)) return NULL;
    const RsSlot *s = &self->batch->slots[b];
    if (!s->failed) Py_RETURN_NONE;
    PyObject *error = err_tuple(&s->fail, s->slot);
    if (!error) return NULL;
    return Py_BuildValue("(Niiii)", error, s->fail_decision, s->fail_tick, s->fail_action, s->episode);
}


/* ---- the equivalence harness's calls (sim/native_harness.py NativeSide) ---------------------------------------- */
static int get_list(PyObject *d, const char *key, int64_t *out, int cap, int *n) {
    PyObject *v = PyDict_GetItemString(d, key);  /* borrowed */
    if (!v) {
        PyErr_Format(PyExc_KeyError, "start lacks %s", key);
        return -1;
    }
    PyObject *fast = PySequence_Fast(v, "a start list");
    if (!fast) return -1;
    Py_ssize_t len = PySequence_Fast_GET_SIZE(fast);
    if (len > cap) {
        Py_DECREF(fast);
        PyErr_Format(PyExc_ValueError, "start %s: more than %d values", key, cap);
        return -1;
    }
    for (Py_ssize_t i = 0; i < len; i++) {
        out[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(fast, i));
        if (out[i] == -1 && PyErr_Occurred()) {
            Py_DECREF(fast);
            return -1;
        }
    }
    *n = (int)len;
    Py_DECREF(fast);
    return 0;
}

static int get_int(PyObject *d, const char *key, int32_t *out, int required, int32_t otherwise) {
    PyObject *v = PyDict_GetItemString(d, key);  /* borrowed */
    if (!v) {
        if (required) PyErr_Format(PyExc_KeyError, "start lacks %s", key);
        *out = otherwise;
        return required ? -1 : 0;
    }
    long x = PyLong_AsLong(v);
    if (x == -1 && PyErr_Occurred()) return -1;
    *out = (int32_t)x;
    return 0;
}

/* An RsStart from start_dict's keys (sim/native.py encode_start); rec absent or -1 is rs_start_rec's. */
static int start_of(const RsPack *P, PyObject *d, RsStart *s, RsErr *err) {
    memset(s, 0, sizeof *s);
    int32_t tx, tz, worn;
    int64_t levels[7], food[4], mx[1], mz[1];
    int n_levels, n_food, nmx, nmz;
    if (get_int(d, "kind", &s->kind, 1, 0) || get_int(d, "layout", &s->layout, 1, 0)
        || get_int(d, "tile_x", &tx, 1, 0) || get_int(d, "tile_z", &tz, 1, 0) || get_int(d, "weapon", &s->weapon, 1, 0)
        || get_int(d, "world_seed", &s->world_seed, 1, 0) || get_list(d, "levels", levels, 7, &n_levels)
        || get_int(d, "group", &s->group, 0, 0) || get_int(d, "hp", &s->hp, 0, 0) || get_int(d, "worn", &worn, 0, 0)
        || get_int(d, "grove", &s->grove, 0, -1) || get_int(d, "variant", &s->variant, 0, -1)
        || get_int(d, "rec", &s->rec, 0, -1))
        return -1;
    n_food = nmx = nmz = 0;
    if ((PyDict_GetItemString(d, "food") && get_list(d, "food", food, 4, &n_food))
        || (PyDict_GetItemString(d, "mugger_x") && get_list(d, "mugger_x", mx, 1, &nmx))
        || (PyDict_GetItemString(d, "mugger_z") && get_list(d, "mugger_z", mz, 1, &nmz)))
        return -1;
    if (n_levels != 7 || nmx != nmz) {
        PyErr_SetString(PyExc_ValueError, "start: 7 levels and one z per mugger x");
        return -1;
    }
    s->tile_x = (int16_t)tx, s->tile_z = (int16_t)tz, s->worn = (uint8_t)(worn != 0);
    for (int i = 0; i < 7; i++) s->levels[i] = (int16_t)levels[i];
    s->n_food = n_food;
    for (int i = 0; i < n_food; i++) s->food[i] = (int8_t)food[i];
    s->n_muggers = nmx;
    for (int i = 0; i < nmx; i++) s->mugger_x[i] = (int16_t)mx[i], s->mugger_z[i] = (int16_t)mz[i];
    if (s->rec < 0 && rs_start_rec(P, s, err)) return 1;
    return 0;
}

/* Batch.begin(b, start=None): rs_batch_begin with the injected start (or the drawn one), into the bound arrays. */
static PyObject *Batch_begin(BatchObject *self, PyObject *args) {
    int k;
    PyObject *start = Py_None;
    if (!PyArg_ParseTuple(args, "i|O", &k, &start)) return NULL;
    if (k < 0 || k >= self->batch->n) {
        PyErr_Format(PyExc_IndexError, "world %d of a batch of %d", k, (int)self->batch->n);
        return NULL;
    }
    if (!self->bound) {
        PyErr_SetString(PyExc_RuntimeError, "bind the output arrays first");
        return NULL;
    }
    RsStart s;
    RsErr err;
    memset(&err, 0, sizeof err);
    if (start != Py_None) {
        if (!PyDict_Check(start)) {
            PyErr_SetString(PyExc_TypeError, "begin: a start dict (Batch.start's keys) or None");
            return NULL;
        }
        int bad = start_of(self->batch->P, start, &s, &err);
        if (bad < 0) return NULL;
        if (bad) return raise_err(&err, k);
    }
    int failed;
    Py_BEGIN_ALLOW_THREADS
    failed = rs_batch_begin(self->batch, k, start == Py_None ? NULL : &s, &self->out);
    Py_END_ALLOW_THREADS
    if (failed) return batch_failed(self);
    Py_RETURN_NONE;
}

/* Batch.check(): rs_batch_check over the bound arrays, None when they pass, else the failing key index (RS_OBS_N_KEYS
   for the mask). */
static PyObject *Batch_check(BatchObject *self, PyObject *unused) {
    (void)unused;
    if (!self->bound) {
        PyErr_SetString(PyExc_RuntimeError, "bind the output arrays first");
        return NULL;
    }
    int bad = rs_batch_check(self->batch, &self->out);
    if (bad < 0) Py_RETURN_NONE;
    return PyLong_FromLong(bad);
}

/* Batch.cycles(): the world cycles every slot has run so far (RsSlot.cycles summed). */
static PyObject *Batch_cycles(BatchObject *self, PyObject *unused) {
    (void)unused;
    long long total = 0;
    for (int32_t i = 0; i < self->batch->n; i++) total += self->batch->slots[i].cycles;
    return PyLong_FromLongLong(total);
}

static PyObject *Batch_set_episode(BatchObject *self, PyObject *args) {
    int k, episode;
    if (!PyArg_ParseTuple(args, "ii", &k, &episode)) return NULL;
    if (k < 0 || k >= self->batch->n || episode < 0) {
        PyErr_Format(PyExc_IndexError, "world %d of a batch of %d, episode %d", k, (int)self->batch->n, episode);
        return NULL;
    }
    self->batch->slots[k].episode = episode;
    Py_RETURN_NONE;
}

static PyObject *Batch_external_starts(BatchObject *self, PyObject *args) {
    int on;
    if (!PyArg_ParseTuple(args, "p", &on)) return NULL;
    self->batch->external_starts = (uint8_t)on;
    Py_RETURN_NONE;
}

/* Batch.trace(b, on, draws_cap, rows_cap, data_cap): attach a fresh trace to world b, or detach and free it. */
static PyObject *Batch_trace(BatchObject *self, PyObject *args) {
    int k, on, draws_cap = 1 << 16, rows_cap = 1 << 12;
    long long data_cap = 1 << 22;
    if (!PyArg_ParseTuple(args, "ip|iiL", &k, &on, &draws_cap, &rows_cap, &data_cap)) return NULL;
    if (k < 0 || k >= self->batch->n) {
        PyErr_Format(PyExc_IndexError, "world %d of a batch of %d", k, (int)self->batch->n);
        return NULL;
    }
    RsTrace *fresh = NULL;
    if (on && !(fresh = rs_trace_new(draws_cap, rows_cap, data_cap))) return PyErr_NoMemory();
    rs_trace(self->batch, k, fresh);
    rs_trace_free(self->traces[k]);
    self->traces[k] = fresh;
    Py_RETURN_NONE;
}

/* Batch.trace_rows(b): world b's rows since the last call, each (tick, crc, record bytes (little-endian int64),
   draws [(tick, keys, m)], events [11-int lists], chat [(tick, text)], texts [(tick, iface, com, text)], episode),
   the Row fields sim/native_harness.py compares; then the trace drops them. A truncated trace raises. */
static PyObject *Batch_trace_rows(BatchObject *self, PyObject *args) {
    int k;
    if (!PyArg_ParseTuple(args, "i", &k)) return NULL;
    if (k < 0 || k >= self->batch->n || !self->traces[k]) {
        PyErr_Format(PyExc_IndexError, "world %d has no trace", k);
        return NULL;
    }
    RsTrace *t = self->traces[k];
    if (t->truncated) {
        rs_trace_clear(t);
        PyErr_SetString(PyExc_RuntimeError, "a native trace filled its buffers (larger caps in Batch.trace)");
        return NULL;
    }
    PyObject *out = PyList_New(t->n_rows);
    for (int32_t r = 0; out && r < t->n_rows; r++) {
        const RsTraceRow *row = &t->rows[r];
        const int64_t *d = t->data;
        PyObject *record = PyBytes_FromStringAndSize((const char *)(d + row->rec), (Py_ssize_t)row->n_rec * 8);
        PyObject *draws = PyList_New(row->n_draw), *events = PyList_New(row->n_ev);
        PyObject *chat = PyList_New(row->n_chat), *texts = PyList_New(row->n_text);
        int ok = record && draws && events && chat && texts;
        for (int32_t i = 0; ok && i < row->n_draw; i++) {
            const RsDraw *draw = &t->draws[row->draw0 + i];
            PyObject *keys = PyTuple_New(draw->n);
            for (int j = 0; keys && j < draw->n; j++) {
                PyObject *v = PyLong_FromLongLong(draw->key[j]);
                if (!v) Py_CLEAR(keys);
                else PyTuple_SET_ITEM(keys, j, v);
            }
            PyObject *item = keys ? Py_BuildValue("(iNk)", draw->tick, keys, (unsigned long)draw->m) : NULL;
            if (!item) ok = 0;
            else PyList_SET_ITEM(draws, i, item);
        }
        for (int32_t i = 0; ok && i < row->n_ev; i++) {
            const int64_t *v = d + row->ev + (int64_t)i * RS_TRACE_EVENT;
            PyObject *item = Py_BuildValue("[LLLLLLLLLLL]", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9],
                                           v[10]);
            if (!item) ok = 0;
            else PyList_SET_ITEM(events, i, item);
        }
        for (int32_t i = 0; ok && i < row->n_chat; i++) {
            const int64_t *v = d + row->chat + (int64_t)i * RS_TRACE_CHAT;
            PyObject *item = Py_BuildValue("(LL)", v[0], v[1]);
            if (!item) ok = 0;
            else PyList_SET_ITEM(chat, i, item);
        }
        for (int32_t i = 0; ok && i < row->n_text; i++) {
            const int64_t *v = d + row->text + (int64_t)i * RS_TRACE_TEXT;
            PyObject *item = Py_BuildValue("(LLLL)", v[0], v[1], v[2], v[3]);
            if (!item) ok = 0;
            else PyList_SET_ITEM(texts, i, item);
        }
        PyObject *tuple = ok ? Py_BuildValue("(ikNNNNNi)", row->tick, (unsigned long)row->crc, record, draws, events,
                                             chat, texts, row->episode) : NULL;
        if (!tuple) {
            if (!ok) Py_XDECREF(record), Py_XDECREF(draws), Py_XDECREF(events), Py_XDECREF(chat), Py_XDECREF(texts);
            Py_CLEAR(out);
        } else {
            PyList_SET_ITEM(out, r, tuple);
        }
    }
    rs_trace_clear(t);
    return out;
}

static PyMethodDef Batch_methods[] = {
    {"bind", (PyCFunction)Batch_bind, METH_VARARGS, "bind sim/workers.py fields(n) arrays as the outputs"},
    {"reset", (PyCFunction)Batch_reset, METH_VARARGS, "rs_batch_reset(key) into the bound arrays"},
    {"step", (PyCFunction)Batch_step, METH_VARARGS, "rs_batch_step(actions: int32 [n]) into the bound arrays"},
    {"episodes", (PyCFunction)Batch_episodes, METH_NOARGS, "each world's episode counter"},
    {"late", (PyCFunction)Batch_late, METH_NOARGS, "the late log1p features of each world's last observation"},
    {"cycles", (PyCFunction)Batch_cycles, METH_NOARGS, "world cycles run by every slot so far (setup included)"},
    {"check", (PyCFunction)Batch_check, METH_NOARGS,
     "check_batch's value checks on the bound observations: None, or the failing key (the mask's: N_KEYS)"},
    {"start", (PyCFunction)Batch_start, METH_VARARGS, "world b's RsStart as raw ints, with its pool layout's seed"},
    {"failure", (PyCFunction)Batch_failure, METH_VARARGS,
     "world b's unsupported evidence: None, or (error args, decision, tick, action, episode)"},
    {"begin", (PyCFunction)Batch_begin, METH_VARARGS, "rs_batch_begin(b, start dict or None) into the bound arrays"},
    {"set_episode", (PyCFunction)Batch_set_episode, METH_VARARGS, "set world b's episode counter"},
    {"external_starts", (PyCFunction)Batch_external_starts, METH_VARARGS, "leave done worlds for begin (scenarios)"},
    {"trace", (PyCFunction)Batch_trace, METH_VARARGS, "trace(b, on[, draws_cap, rows_cap, data_cap])"},
    {"trace_rows", (PyCFunction)Batch_trace_rows, METH_VARARGS, "world b's trace rows since the last call"},
    {NULL, NULL, 0, NULL},
};
static PyTypeObject BatchType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sim._native.Batch",
    .tp_basicsize = sizeof(BatchObject),
    .tp_dealloc = (destructor)Batch_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "An RsBatch over a Pack (rs_batch_new).",
    .tp_methods = Batch_methods,
    .tp_new = Batch_new,
};

/* ---- module ---------------------------------------------------------------------------------------------------- */
static PyMethodDef methods[] = {
    {"struct_layouts", struct_layouts, METH_NOARGS, "{struct: (size, [(field, C type, offset, count)])}"},
    {"reasons", reasons, METH_NOARGS, "[(name, exception index, format, arg kinds)] by reason id"},
    {"exceptions", exceptions, METH_NOARGS, "RsExc names by index"},
    {"canon_fields", canon_fields, METH_NOARGS, "RsCanonField names by id"},
    {"scripts", scripts, METH_NOARGS, "[(name, Python qualname, status)] by RsScript id"},
    {"constants", constants, METH_NOARGS, "section ids, the observation and info layouts, capacities"},
    {"rng_seed", rng_seed, METH_VARARGS, "Rng(seed).seed"},
    {"rng_prefix", rng_prefix, METH_VARARGS, "Rng.prefix from state h over keys"},
    {"rng_m", rng_m, METH_VARARGS, "the 24-bit m of Rng.draw(h, *keys)"},
    {"rng_draw", rng_draw, METH_VARARGS, "Rng.draw(h, *keys)"},
    {"rng_pick", rng_pick, METH_VARARGS, "int(m / 2**24 * n)"},
    {"js_round", js_round, METH_VARARGS, "Math.round"},
    {"floordiv", floordiv, METH_VARARGS, "Python //"},
    {"mod", mod, METH_VARARGS, "Python %"},
    {"asr", asr, METH_VARARGS, "Python >>"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "sim._native", "The native simulator (sim/native/rs.h).",
                                    -1, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__native(void) {
    if (PyType_Ready(&PackType) < 0 || PyType_Ready(&BatchType) < 0) return NULL;
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;
    Error = PyErr_NewExceptionWithDoc("sim._native.Error",
                                      "A failed native call: (code, exc, reason, args, text, world).", NULL, NULL);
    if (!Error) goto fail;
    Py_INCREF(Error);  /* the module's reference; the static one stays */
    if (PyModule_AddObject(m, "Error", Error)) goto fail;
    Py_INCREF(&PackType);
    if (PyModule_AddObject(m, "Pack", (PyObject *)&PackType)) goto fail;
    Py_INCREF(&BatchType);
    if (PyModule_AddObject(m, "Batch", (PyObject *)&BatchType)) goto fail;
    return m;
fail:
    Py_DECREF(m);
    return NULL;
}
