/* The C port of sim/paths.py's hot path.
 *
 * Each function returns what its Python reference in sim/paths.py returns and raises OutsideTables where it does:
 * WalkMap.steps and .approach are walk_steps and walk_approach on a layout's frozen snapshot distances (sim/world.py
 * LayoutTables), bfs_tree is search's breadth-first fill, end_tile and backtrack are end_tile and backtrack on a
 * Tree's arrays. The Python functions stay the reference (sim/paths.py use('python')); sim/tests/test_paths.py checks
 * these against them call by call and over whole rollouts. Built and loaded by sim/cpaths.py; sim/paths.py passes the
 * constants it shares with the Python code to configure(), so they have one definition. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>

static PyObject *OutsideTables;            /* sim.paths.OutsideTables */
static long HALF = -1, SEARCH, MAX_WAYPOINTS;
static long STEP_DX[8], STEP_DZ[8];        /* sim.paths.STEPS, in the path finder's order */
static unsigned char BACK[8];              /* sim.paths.BACK: the way back of each step */
static long D_NORTH, D_EAST, D_SOUTH, D_WEST;

/* ---- a C-contiguous numeric buffer (a numpy array) read as long ---- */
typedef struct {
    Py_buffer view;
    char kind;      /* 'B' u8, 'H' u16, 'h' i16, 'i' i32, 'I' u32, 'q' i64 */
} Grid;

static int grid_get(PyObject *obj, Grid *g, int writable)
{
    if (PyObject_GetBuffer(obj, &g->view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0)) < 0)
        return -1;
    const char *f = g->view.format ? g->view.format : "B";
    while (*f == '<' || *f == '=' || *f == '@' || *f == '|')
        f++;
    char k = *f;
    Py_ssize_t size = g->view.itemsize;
    if ((k == 'l' || k == 'q') && size == 8) k = 'q';
    else if ((k == 'l' || k == 'i') && size == 4) k = 'i';
    else if ((k == 'L' || k == 'I') && size == 4) k = 'I';
    else if (!((k == 'B' && size == 1) || (k == 'H' && size == 2) || (k == 'h' && size == 2))) {
        PyErr_Format(PyExc_TypeError, "unsupported array format %s", g->view.format ? g->view.format : "?");
        PyBuffer_Release(&g->view);
        return -1;
    }
    g->kind = k;
    return 0;
}

static inline long grid_at(const Grid *g, Py_ssize_t i)
{
    switch (g->kind) {
    case 'B': return ((const uint8_t *)g->view.buf)[i];
    case 'H': return ((const uint16_t *)g->view.buf)[i];
    case 'h': return ((const int16_t *)g->view.buf)[i];
    case 'i': return ((const int32_t *)g->view.buf)[i];
    case 'I': return (long)((const uint32_t *)g->view.buf)[i];
    default: return (long)((const int64_t *)g->view.buf)[i];
    }
}

static inline void grid_set(Grid *g, Py_ssize_t i, long v)
{
    switch (g->kind) {
    case 'B': ((uint8_t *)g->view.buf)[i] = (uint8_t)v; break;
    case 'H': ((uint16_t *)g->view.buf)[i] = (uint16_t)v; break;
    case 'h': ((int16_t *)g->view.buf)[i] = (int16_t)v; break;
    case 'i': ((int32_t *)g->view.buf)[i] = (int32_t)v; break;
    case 'I': ((uint32_t *)g->view.buf)[i] = (uint32_t)v; break;
    default: ((int64_t *)g->view.buf)[i] = (int64_t)v; break;
    }
}

static inline Py_ssize_t grid_rows(const Grid *g) { return g->view.ndim >= 1 ? g->view.shape[0] : g->view.len / g->view.itemsize; }
static inline Py_ssize_t grid_cols(const Grid *g) { return g->view.ndim >= 2 ? g->view.shape[1] : 1; }

/* numpy's a[i] on one axis: a negative index counts from the end; beyond the axis is an IndexError */
static int axis_index(long i, Py_ssize_t n, Py_ssize_t *out)
{
    if (i < 0) i += n;
    if (i < 0 || i >= n) {
        PyErr_SetString(PyExc_IndexError, "index out of bounds");
        return -1;
    }
    *out = i;
    return 0;
}

static inline long lmax(long a, long b) { return a > b ? a : b; }

static int configured(void)
{
    if (HALF < 0) {
        PyErr_SetString(PyExc_RuntimeError, "sim._paths is not configured (sim/paths.py calls configure)");
        return 0;
    }
    return 1;
}

/* ---- configure(OutsideTables, half, search, max_waypoints, steps, back, north, east, south, west) ---- */
static PyObject *configure(PyObject *self, PyObject *args)
{
    PyObject *error, *steps, *back;
    if (!PyArg_ParseTuple(args, "OlllOOllll", &error, &HALF, &SEARCH, &MAX_WAYPOINTS, &steps, &back,
                          &D_NORTH, &D_EAST, &D_SOUTH, &D_WEST))
        return NULL;
    PyObject *fast = PySequence_Fast(steps, "steps must be a sequence");
    if (!fast) return NULL;
    if (PySequence_Fast_GET_SIZE(fast) != 8) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "eight steps");
        return NULL;
    }
    for (int k = 0; k < 8; k++) {
        if (!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(fast, k), "ll", &STEP_DX[k], &STEP_DZ[k])) {
            Py_DECREF(fast);
            return NULL;
        }
    }
    Py_DECREF(fast);
    Grid b;
    if (grid_get(back, &b, 0) < 0) return NULL;
    if (grid_rows(&b) * grid_cols(&b) != 8) {
        PyBuffer_Release(&b.view);
        PyErr_SetString(PyExc_ValueError, "eight backs");
        return NULL;
    }
    for (int k = 0; k < 8; k++) BACK[k] = (unsigned char)grid_at(&b, k);
    PyBuffer_Release(&b.view);
    Py_XDECREF(OutsideTables);
    Py_INCREF(error);
    OutsideTables = error;
    Py_RETURN_NONE;
}

/* ---- WalkMap: walk_steps and walk_approach on a layout's frozen snapshot distances ---- */
typedef struct {
    PyObject_HEAD
    Grid snap;      /* (T, T): row x * H + z of a source's local tile, column x * H + z of the target */
    Grid sources;   /* (n, 2): each source's local x, z */
    long W, H, r;   /* layout size, the walk map's radius (meta walk_r) */
    int ready;
} WalkMap;

static void walkmap_dealloc(WalkMap *m)
{
    if (m->ready) {
        PyBuffer_Release(&m->snap.view);
        PyBuffer_Release(&m->sources.view);
    }
    Py_TYPE(m)->tp_free((PyObject *)m);
}

static int walkmap_init(WalkMap *m, PyObject *args, PyObject *kwds)
{
    PyObject *snap, *sources;
    if (!PyArg_ParseTuple(args, "OOlll", &snap, &sources, &m->W, &m->H, &m->r)) return -1;
    if (!configured()) return -1;
    if (grid_get(snap, &m->snap, 0) < 0) return -1;
    if (grid_get(sources, &m->sources, 0) < 0) {
        PyBuffer_Release(&m->snap.view);
        return -1;
    }
    if (grid_rows(&m->snap) * grid_cols(&m->snap) != m->W * m->H * m->W * m->H || grid_cols(&m->sources) != 2) {
        PyBuffer_Release(&m->snap.view);
        PyBuffer_Release(&m->sources.view);
        PyErr_SetString(PyExc_ValueError, "snapshot distances must be (W*H, W*H) and sources (n, 2)");
        return -1;
    }
    m->ready = 1;
    return 0;
}

/* walk_steps: -1 unreached; raises beyond the walk map's radius. The window's distance of a tile is the layout row's
 * at the source's local tile plus the offset, 255 (unreached) outside the layout or the 2*HALF window (sim/world.py
 * _window, paths.Tree.dist). */
static int walk_one(WalkMap *m, long src, long sx, long sz, long x, long z, long *out)
{
    long dx = x - sx, dz = z - sz;
    if (lmax(labs(dx), labs(dz)) > m->r) {
        PyErr_Format(OutsideTables, "walk steps %ld,%ld -> %ld,%ld beyond the stored walk map", sx, sz, x, z);
        return -1;
    }
    Py_ssize_t i;
    if (axis_index(src, grid_rows(&m->sources), &i) < 0) return -1;
    long ox = grid_at(&m->sources, 2 * i), oz = grid_at(&m->sources, 2 * i + 1);
    long d = 255;
    if (-HALF <= dx && dx < HALF && -HALF <= dz && dz < HALF) {
        long tx = ox + dx, tz = oz + dz;
        if (0 <= tx && tx < m->W && 0 <= tz && tz < m->H)
            d = grid_at(&m->snap, (ox * m->H + oz) * m->W * m->H + tx * m->H + tz);
    }
    *out = d == 255 ? -1 : d;
    return 0;
}

static PyObject *walkmap_steps(WalkMap *m, PyObject *const *args, Py_ssize_t n)
{
    long v[5], out;
    if (n != 5) {
        PyErr_SetString(PyExc_TypeError, "steps(src, sx, sz, x, z)");
        return NULL;
    }
    for (int k = 0; k < 5; k++) {
        v[k] = PyLong_AsLong(args[k]);
        if (v[k] == -1 && PyErr_Occurred()) return NULL;
    }
    if (walk_one(m, v[0], v[1], v[2], v[3], v[4], &out) < 0) return NULL;
    return PyLong_FromLong(out);
}

/* walk_approach: the fewest steps to a tile beside or on (x, z), -1 if none; the first raise in its loop order */
static PyObject *walkmap_approach(WalkMap *m, PyObject *const *args, Py_ssize_t n)
{
    long v[5];
    if (n != 5) {
        PyErr_SetString(PyExc_TypeError, "approach(src, sx, sz, x, z)");
        return NULL;
    }
    for (int k = 0; k < 5; k++) {
        v[k] = PyLong_AsLong(args[k]);
        if (v[k] == -1 && PyErr_Occurred()) return NULL;
    }
    long best = -1;
    for (long dx = -1; dx <= 1; dx++) {
        for (long dz = -1; dz <= 1; dz++) {
            long s;
            if (walk_one(m, v[0], v[1], v[2], v[3] + dx, v[4] + dz, &s) < 0) return NULL;
            if (s >= 0 && (best < 0 || s < best)) best = s;
        }
    }
    return PyLong_FromLong(best);
}

static PyMethodDef walkmap_methods[] = {
    {"steps", (PyCFunction)(void (*)(void))walkmap_steps, METH_FASTCALL, "paths.walk_steps(t, src, sx, sz, x, z)"},
    {"approach", (PyCFunction)(void (*)(void))walkmap_approach, METH_FASTCALL, "paths.walk_approach(t, src, sx, sz, x, z)"},
    {NULL}
};

static PyTypeObject WalkMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sim._paths.WalkMap",
    .tp_basicsize = sizeof(WalkMap),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "WalkMap(snapshot_dist, sources, W, H, r): a layout's walk steps (sim/paths.py walk_steps)",
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)walkmap_init,
    .tp_dealloc = (destructor)walkmap_dealloc,
    .tp_methods = walkmap_methods,
};

/* ---- bfs_tree(steps, height, start, lo0, lo1, hi0, hi1, lx, lz, orders, dists, parents) ----
 * paths._bfs in findPath1's queue order over a flat step-mask grid (index x * height + z) within lo..hi, written as
 * paths.search writes its Tree: each reached tile's queue position, its level (at most 254) and its step back (the
 * source's 99), at window coordinates (x - lx, z - lz). A queue that marks a tile when first discovered visits the
 * tiles in the order the level-by-level search does: the first discovery fixes a tile's parent and position. */
static PyObject *bfs_tree(PyObject *self, PyObject *args)
{
    PyObject *steps_o, *orders_o, *dists_o, *parents_o;
    long height, start, lo0, lo1, hi0, hi1, lx, lz;
    if (!configured()) return NULL;
    if (!PyArg_ParseTuple(args, "OllllllllOOO", &steps_o, &height, &start, &lo0, &lo1, &hi0, &hi1, &lx, &lz,
                          &orders_o, &dists_o, &parents_o))
        return NULL;
    Grid steps, orders, dists, parents;
    if (grid_get(steps_o, &steps, 0) < 0) return NULL;
    if (grid_get(orders_o, &orders, 1) < 0) goto fail0;
    if (grid_get(dists_o, &dists, 1) < 0) goto fail1;
    if (grid_get(parents_o, &parents, 1) < 0) goto fail2;
    Py_ssize_t total = steps.view.len / steps.view.itemsize;
    long ww = hi0 - lo0 + 1, wh = hi1 - lo1 + 1;
    Py_ssize_t ocols = grid_cols(&orders), dcols = grid_cols(&dists), pcols = grid_cols(&parents);
    Py_ssize_t orows = grid_rows(&orders), drows = grid_rows(&dists), prows = grid_rows(&parents);
    if (ww <= 0 || wh <= 0 || start < 0 || start >= total) {
        PyErr_SetString(PyExc_ValueError, "empty window or start outside the grid");
        goto fail3;
    }
    size_t cap = (size_t)ww * (size_t)wh + 1;  /* every window tile once, and a start outside it */
    long *queue = malloc(sizeof(long) * cap), *level = malloc(sizeof(long) * cap);
    unsigned char *seen = calloc(cap, 1), *back = malloc(cap);
    if (!queue || !level || !seen || !back) {
        free(queue); free(level); free(seen); free(back);
        PyErr_NoMemory();
        goto fail3;
    }
    long sx0 = start / height - lo0, sz0 = start % height - lo1;
    Py_ssize_t head = 0, tail = 0;
    if (0 <= sx0 && sx0 < ww && 0 <= sz0 && sz0 < wh) seen[sx0 * wh + sz0] = 1;
    queue[tail] = start; level[tail] = 0; back[tail] = 99; tail++;
    while (head < tail) {
        long t = queue[head], lev = level[head];
        head++;
        long bits = grid_at(&steps, t), fx = t / height, fz = t % height;
        for (int k = 0; k < 8; k++) {
            if (!((bits >> k) & 1)) continue;
            long nx = fx + STEP_DX[k], nz = fz + STEP_DZ[k];
            if (nx < lo0 || nx > hi0 || nz < lo1 || nz > hi1) continue;
            long w = (nx - lo0) * wh + (nz - lo1);
            if (seen[w]) continue;
            seen[w] = 1;
            queue[tail] = t + STEP_DX[k] * height + STEP_DZ[k];
            level[tail] = lev + 1;
            back[tail] = BACK[k];
            tail++;
        }
    }
    for (Py_ssize_t q = 0; q < tail; q++) {
        long gx = queue[q] / height - lx, gz = queue[q] % height - lz;
        Py_ssize_t ox, oz, dx, dz, px, pz;
        if (axis_index(gx, orows, &ox) < 0 || axis_index(gz, ocols, &oz) < 0 || axis_index(gx, drows, &dx) < 0 ||
            axis_index(gz, dcols, &dz) < 0 || axis_index(gx, prows, &px) < 0 || axis_index(gz, pcols, &pz) < 0) {
            free(queue); free(level); free(seen); free(back);
            goto fail3;
        }
        grid_set(&orders, ox * ocols + oz, (long)q);
        grid_set(&dists, dx * dcols + dz, level[q] < 254 ? level[q] : 254);
        grid_set(&parents, px * pcols + pz, back[q]);
    }
    free(queue); free(level); free(seen); free(back);
    PyBuffer_Release(&parents.view);
    PyBuffer_Release(&dists.view);
    PyBuffer_Release(&orders.view);
    PyBuffer_Release(&steps.view);
    return PyLong_FromSsize_t(tail);
fail3:
    PyBuffer_Release(&parents.view);
fail2:
    PyBuffer_Release(&dists.view);
fail1:
    PyBuffer_Release(&orders.view);
fail0:
    PyBuffer_Release(&steps.view);
    return NULL;
}

/* ---- a Tree's order and dist (paths.Tree.order, .dist): 0 ok, 1 none / unreached, -1 raised ---- */
static int tree_order(const Grid *orders, long r, long none, long dx, long dz, long *out)
{
    if (lmax(labs(dx), labs(dz)) > r) {
        if (-HALF <= dx && dx < HALF && -HALF <= dz && dz < HALF) {
            PyErr_Format(OutsideTables, "tile %ld,%ld from the source beyond the stored queue order", dx, dz);
            return -1;
        }
        return 1;
    }
    Py_ssize_t i, j;
    if (axis_index(dx + r, grid_rows(orders), &i) < 0 || axis_index(dz + r, grid_cols(orders), &j) < 0) return -1;
    long o = grid_at(orders, i * grid_cols(orders) + j);
    if (o == none) return 1;
    *out = o;
    return 0;
}

static int tree_dist(const Grid *dists, long r, long dx, long dz, long *out)
{
    if (!(-HALF <= dx && dx < HALF && -HALF <= dz && dz < HALF)) {
        *out = 255;
        return 0;
    }
    if (lmax(labs(dx), labs(dz)) > r) {
        PyErr_Format(OutsideTables, "tile %ld,%ld from the source beyond the stored distances", dx, dz);
        return -1;
    }
    Py_ssize_t i, j;
    if (axis_index(dx + r, grid_rows(dists), &i) < 0 || axis_index(dz + r, grid_cols(dists), &j) < 0) return -1;
    *out = grid_at(dists, i * grid_cols(dists) + j);
    return 0;
}

/* an (x, z) pair of any sequence of two integers */
static int pair(PyObject *item, long *x, long *z)
{
    PyObject *fast = PySequence_Fast(item, "a tile must be a pair");
    if (!fast) return -1;
    if (PySequence_Fast_GET_SIZE(fast) != 2) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "a tile must be a pair");
        return -1;
    }
    *x = PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, 0));
    *z = PyLong_AsLong(PySequence_Fast_GET_ITEM(fast, 1));
    Py_DECREF(fast);
    return PyErr_Occurred() ? -1 : 0;
}

/* ---- end_tile(orders, order_r, none, dists, dist_r, sx, sz, reaching, dest_x, dest_z, w, h): paths.end_tile ---- */
static PyObject *end_tile(PyObject *self, PyObject *args)
{
    PyObject *orders_o, *dists_o, *reaching;
    long order_r, none, dist_r, sx, sz, ddx0, ddz0, w, h;
    if (!configured()) return NULL;
    if (!PyArg_ParseTuple(args, "OllOlllOllll", &orders_o, &order_r, &none, &dists_o, &dist_r, &sx, &sz, &reaching,
                          &ddx0, &ddz0, &w, &h))
        return NULL;
    Grid orders, dists;
    PyObject *result = NULL;
    if (grid_get(orders_o, &orders, 0) < 0) return NULL;
    if (grid_get(dists_o, &dists, 0) < 0) {
        PyBuffer_Release(&orders.view);
        return NULL;
    }
    PyObject *fast = PySequence_Fast(reaching, "reaching must be a sequence of tiles");
    if (!fast) goto done;
    long best = -1, ex = 0, ez = 0;
    int found = 0;
    for (Py_ssize_t k = 0; k < PySequence_Fast_GET_SIZE(fast); k++) {
        long x, z, o;
        if (pair(PySequence_Fast_GET_ITEM(fast, k), &x, &z) < 0) goto done_fast;
        int r = tree_order(&orders, order_r, none, x - sx, z - sz, &o);
        if (r < 0) goto done_fast;
        if (r == 0 && (!found || o < best)) {
            best = o; ex = x; ez = z; found = 1;
        }
    }
    if (!found) {
        /* PathFinder.findClosestApproachPoint: x outer, z inner; a lower cost, or the same cost with a shorter path */
        long ddx = ddx0 - sx, ddz = ddz0 - sz, lowest = 1000, shortest = 100;
        for (long x = ddx - 10; x <= ddx + 10; x++) {
            for (long z = ddz - 10; z <= ddz + 10; z++) {
                long d;
                if (tree_dist(&dists, dist_r, x, z, &d) < 0) goto done_fast;
                if (d >= 100) continue;
                long cx = x < ddx ? ddx - x : x > ddx + w - 1 ? x - (w + ddx - 1) : 0;
                long cz = z < ddz ? ddz - z : z > ddz + h - 1 ? z - (h + ddz - 1) : 0;
                long cost = cx * cx + cz * cz;
                if (cost < lowest || (cost == lowest && shortest > d)) {
                    ex = sx + x; ez = sz + z; lowest = cost; shortest = d; found = 1;
                }
            }
        }
    }
    result = found ? Py_BuildValue("(ll)", ex, ez) : (Py_INCREF(Py_None), Py_None);
done_fast:
    Py_DECREF(fast);
done:
    PyBuffer_Release(&dists.view);
    PyBuffer_Release(&orders.view);
    return result;
}

/* ---- backtrack(parents, sx, sz, end_x, end_z): paths.backtrack's waypoints, the MAX_WAYPOINTS nearest the source ---- */
static PyObject *backtrack(PyObject *self, PyObject *args)
{
    PyObject *parents_o;
    long sx, sz, ex, ez;
    if (!configured()) return NULL;
    if (!PyArg_ParseTuple(args, "Ollll", &parents_o, &sx, &sz, &ex, &ez)) return NULL;
    Grid parents;
    if (grid_get(parents_o, &parents, 0) < 0) return NULL;
    Py_ssize_t rows = grid_rows(&parents), cols = grid_cols(&parents), limit = SEARCH * SEARCH;
    long *turns = malloc(sizeof(long) * 2 * (size_t)(limit + 1));
    PyObject *out = NULL;
    if (!turns) {
        PyErr_NoMemory();
        goto done;
    }
    Py_ssize_t n = 0, i, j;
    long cx = ex - sx + HALF, cz = ez - sz + HALF, curr = -1, nxt;
    if (axis_index(cx, rows, &i) < 0 || axis_index(cz, cols, &j) < 0) goto done;
    nxt = grid_at(&parents, i * cols + j);
    for (Py_ssize_t step = 0; step < limit; step++) {
        if (cx == HALF && cz == HALF) break;
        if (curr != nxt) {
            curr = nxt;
            turns[2 * n] = sx + cx - HALF;
            turns[2 * n + 1] = sz + cz - HALF;
            n++;
        }
        if (curr & D_EAST) cx += 1;
        else if (curr & D_WEST) cx -= 1;
        if (curr & D_NORTH) cz += 1;
        else if (curr & D_SOUTH) cz -= 1;
        if (axis_index(cx, rows, &i) < 0 || axis_index(cz, cols, &j) < 0) goto done;
        nxt = grid_at(&parents, i * cols + j);
    }
    /* each turn went in front and the list kept its first MAX_WAYPOINTS: the last turns found, nearest-source first */
    Py_ssize_t keep = n < MAX_WAYPOINTS ? n : MAX_WAYPOINTS;
    out = PyList_New(keep);
    if (!out) goto done;
    for (Py_ssize_t k = 0; k < keep; k++) {
        Py_ssize_t t = n - 1 - k;
        PyObject *p = Py_BuildValue("(ll)", turns[2 * t], turns[2 * t + 1]);
        if (!p) {
            Py_CLEAR(out);
            goto done;
        }
        PyList_SET_ITEM(out, k, p);
    }
done:
    free(turns);
    PyBuffer_Release(&parents.view);
    return out;
}

static PyMethodDef methods[] = {
    {"configure", configure, METH_VARARGS, "the constants sim/paths.py shares with this port"},
    {"bfs_tree", bfs_tree, METH_VARARGS, "paths.search's breadth-first fill of a Tree's arrays"},
    {"end_tile", end_tile, METH_VARARGS, "paths.end_tile on a Tree's arrays"},
    {"backtrack", backtrack, METH_VARARGS, "paths.backtrack on a Tree's parents"},
    {NULL}
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_paths", "sim/paths.py's hot path in C", -1, methods};

PyMODINIT_FUNC PyInit__paths(void)
{
    if (PyType_Ready(&WalkMapType) < 0) return NULL;
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;
    Py_INCREF(&WalkMapType);
    if (PyModule_AddObject(m, "WalkMap", (PyObject *)&WalkMapType) < 0) {
        Py_DECREF(&WalkMapType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
