/* sim/native/paths.c: the path finder of sim/paths.py and the World methods around it (sim/engine.py,
 * 957-1014): collision and step masks, the findPath1-order search, end tiles and waypoints, reach for every shape, line
 * of sight, walk steps on the snapshot, the client's tryMove (act-v3's clicks and ACT routes), changed-tile tracking
 * and the path caches.
 *
 * Everything works in layout coordinates: a layout tile is t = (x - x0) * H + (z - z0) (rs.h COORDINATES). A world's
 * layout is bounded (W, H <= 64, walled in by step_masks' FLAG_NULL padding), so the 128-tile search window of any
 * layout tile covers the whole layout: a tree is a search over the layout, every layout tile lies inside the window
 * (|dx|, |dz| <= 63 < HALF), and a tile off the layout is unreached (Tree.order None, Tree.dist 255). Neither
 * OutsideTables of Tree.order/dist can be raised then (their radius is HALF), so they are not ported.
 *
 * Stored or live (World.tree, engine.py): with no changed tile in the window the oracle reads the boot map's
 * stored tree, refusing a tile that is not a source; that tree is the pack's derived state-0 rows where the pool has
 * them (sim/world.py derive), else this search on the boot steps (LayoutTables.rows -> tree_rows: the same search).
 * With a changed tile it searches the live steps. Reach (entity, loc) and line of sight follow the same split: their
 * stored tables are the live functions on the boot flags, so they are recomputed live, except where the stored branch
 * answers from a list the live one would not give (a map loc's stored reach, filtered to the layout) or refuses (the
 * entity tile off the data area, a sight line beyond los_r).
 *
 * Caches (rs.h RsPathCache) are pure memo, keyed by the layout and the steps they were built on: one tree (the last
 * source's, resumable: a path query stops at its first reach tile), the live step masks, the client's steps and its
 * last scene search, and the most recent snapshot distance rows (a bitset search: walk steps need depths only, and a depth does not depend on the queue order).
 * Nothing here allocates. */
#include <string.h>

#include "rs.h"

#define ORDER_NONE 65535u   /* Tree.none of a stored tree (uint16 rows) */
#define DIST_NONE 255u
#define PARENT_SOURCE 99u
#define BACKTRACK_LIMIT (RS_SEARCH * RS_SEARCH)  /* paths.backtrack's loop bound */

static inline int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }
static inline int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t iabs(int32_t a) { return a < 0 ? -a : a; }

/* ---- flags ------------------------------------------------------------------------------------------------------ */
/* A flag grid over a layout's area: World.live_flags (engine.py, World.flags with npc_bits False) on a world, or a
   bare grid (sim/world.py _flags_fn) in the tests; FLAG_NULL off it. */
typedef struct { const int32_t *flags; int32_t x0, z0, W, H; } Grid;

static inline uint32_t grid_flag(const Grid *g, int32_t x, int32_t z) {
    int32_t lx = x - g->x0, lz = z - g->z0;
    return lx >= 0 && lx < g->W && lz >= 0 && lz < g->H ? (uint32_t)g->flags[lx * g->H + lz] : (uint32_t)RS_FLAG_NULL;
}

static inline Grid world_grid(const World *w) {
    Grid g = {w->flags, w->L->x0, w->L->z0, w->L->W, w->L->H};
    return g;
}

/* paths.step_masks (paths.py): the steps canTravel (size 1, NORMAL, no extra flag) allows from each tile, as
   STEPS bits W, E, S, N, SW, SE, NW, NE; a diagonal needs both of its orthogonal steps. Tiles off the grid read
   FLAG_NULL, so no step leaves it: the flags are read from a copy inside a FLAG_NULL border. */
static void step_masks(const int32_t *flags, int32_t W, int32_t H, uint8_t *out) {
    uint32_t pad[(RS_MAX_W + 2) * (RS_MAX_W + 2)];   /* tile (x, z) at (x + 1) * P + z + 1 */
    const int32_t P = H + 2;
    for (int32_t z = 0; z < P; z++) pad[z] = pad[(W + 1) * P + z] = (uint32_t)RS_FLAG_NULL;
    for (int32_t x = 0; x < W; x++) {
        uint32_t *col = pad + (x + 1) * P;
        col[0] = col[H + 1] = (uint32_t)RS_FLAG_NULL;
        memcpy(col + 1, flags + x * H, (size_t)H * sizeof *flags);
    }
    for (int32_t x = 0; x < W; x++) {
        const uint32_t *c = pad + (x + 1) * P + 1;
        uint8_t *o = out + x * H;
        for (int32_t z = 0; z < H; z++) {
            unsigned w = !(c[z - P] & RS_BLOCK_W), e = !(c[z + P] & RS_BLOCK_E);
            unsigned s = !(c[z - 1] & RS_BLOCK_S), n = !(c[z + 1] & RS_BLOCK_N);
            unsigned sw = !(c[z - P - 1] & RS_BLOCK_SW) & w & s;
            unsigned se = !(c[z + P - 1] & RS_BLOCK_SE) & e & s;
            unsigned nw = !(c[z - P + 1] & RS_BLOCK_NW) & w & n;
            unsigned ne = !(c[z + P + 1] & RS_BLOCK_NE) & e & n;
            o[z] = (uint8_t)(w | e << 1 | s << 2 | n << 3 | sw << 4 | se << 5 | nw << 6 | ne << 7);
        }
    }
}

/* The live step masks (World._steps, engine.py), rebuilt after each collision change; only read while a tile
   differs from the boot map (otherwise they are the layout's steps_boot). */
static const uint8_t *live_steps(World *w) {
    RsPathCache *pc = &w->pc;
    if (pc->steps_L != w->L || pc->steps_ver != w->coll_ver) {
        step_masks(w->flags, w->L->W, w->L->H, pc->steps);
        pc->steps_L = w->L;
        pc->steps_ver = w->coll_ver;
    }
    return pc->steps;
}

/* ---- collision masks and changed tiles -------------------------------------------------------------------------- */
/* paths.shape_layer (paths.py). */
static inline int shape_layer(int32_t shape) { return shape <= 3 ? 0 : shape <= 8 ? 1 : shape == 22 ? 3 : 2; }

/* CollisionEngine wall flags by angle (W, N, E, S): (dx, dz, flag), paths.py. */
static const int16_t STRAIGHT[4][2][3] = {{{0, 0, 0x80}, {-1, 0, 0x8}}, {{0, 0, 0x2}, {0, 1, 0x20}},
                                          {{0, 0, 0x8}, {1, 0, 0x80}}, {{0, 0, 0x20}, {0, -1, 0x2}}};
static const int16_t CORNER[4][2][3] = {{{0, 0, 0x1}, {-1, 1, 0x10}}, {{0, 0, 0x4}, {1, 1, 0x40}},
                                        {{0, 0, 0x10}, {1, -1, 0x1}}, {{0, 0, 0x40}, {-1, -1, 0x4}}};
static const int16_t LWALL[4][3][3] = {{{0, 0, 0x82}, {-1, 0, 0x8}, {0, 1, 0x20}}, {{0, 0, 0xa}, {0, 1, 0x20}, {1, 0, 0x80}},
                                       {{0, 0, 0x28}, {1, 0, 0x80}, {0, -1, 0x2}}, {{0, 0, 0xa0}, {0, -1, 0x2}, {-1, 0, 0x8}}};

int rs_collision_masks(const RsPack *P, int32_t x, int32_t z, int32_t shape, int32_t angle, int32_t loc_type,
                       int32_t mx[16], int32_t mz[16], int32_t mask[16]) {
    /* paths.collision_masks (paths.py) */
    const RsLocType *t = rs_loc_type(P, loc_type);
    if (!t) RS_TRAP();  /* a KeyError in the oracle (engine.py): every loc's type is in the pack */
    if (!t->blockwalk) return 0;
    int layer = shape_layer(shape), n = 0;
    if (layer == 0) {
        if (angle < 0 || angle > 3) RS_TRAP();  /* table[angle]: angles are 0..3 */
        int rows = shape == 2 ? 3 : 2;
        for (int k = 0; k < rows; k++) {
            int32_t dx, dz, flag;
            if (shape == 0) dx = STRAIGHT[angle][k][0], dz = STRAIGHT[angle][k][1], flag = STRAIGHT[angle][k][2];
            else if (shape == 2) dx = LWALL[angle][k][0], dz = LWALL[angle][k][1], flag = LWALL[angle][k][2];
            else dx = CORNER[angle][k][0], dz = CORNER[angle][k][1], flag = CORNER[angle][k][2];
            mx[n] = x + dx, mz[n] = z + dz, mask[n] = flag | (t->blockrange ? flag << 9 : 0);  /* + projectile flags */
            n++;
        }
        return n;
    }
    if (layer == 2) {
        int32_t w = angle & 1 ? t->length : t->width, l = angle & 1 ? t->width : t->length;
        if (w * l > 16) RS_TRAP();  /* the archive's largest blockwalk loc is 4x4 */
        int32_t m = RS_FLAG_LOC | (t->blockrange ? RS_FLAG_LOC_PROJ : 0);
        for (int32_t j = 0; j < l; j++)
            for (int32_t i = 0; i < w; i++) mx[n] = x + i, mz[n] = z + j, mask[n] = m, n++;
        return n;
    }
    if (layer == 3 && t->active == 1) {
        mx[0] = x, mz[0] = z, mask[0] = 0x200000;  /* changeFloor: CollisionFlag.FLOOR */
        return 1;
    }
    return 0;
}

int rs_loc_collision(World *w, const Loc *loc, int add) {
    /* World.loc_collision (engine.py): each mask in or out of the live flags, its tile added to or discarded from
       the changed set against the boot flags; any mask invalidates the step and tree caches (coll_ver, never 0 once
       bumped: 0 keys the boot steps). A mask off the data area raises with the earlier masks already applied. */
    const RsLayout *L = w->L;
    int32_t mx[16], mz[16], mask[16];
    int n = rs_collision_masks(w->P, loc->x, loc->z, loc->shape, loc->angle, loc->type, mx, mz, mask);
    for (int k = 0; k < n; k++) {
        int32_t lx = mx[k] - L->x0, lz = mz[k] - L->z0;
        if (lx < 0 || lx >= L->W || lz < 0 || lz >= L->H) return RS_FAIL(&w->err, RS_E_COLLISION_OUTSIDE, mx[k], mz[k]);
        if (loc->level != 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a loc collision on another level");
        int32_t t = lx * L->H + lz;
        uint32_t f = (uint32_t)w->flags[t], m = (uint32_t)mask[k];
        f = add ? f | m : f & ~m;
        w->flags[t] = (int32_t)f;
        uint64_t bit = 1ull << (t & 63), *word = &w->changed[t >> 6];
        if ((int32_t)f != L->flags_boot[t]) {
            if (!(*word & bit)) *word |= bit, w->n_changed++;
        } else if (*word & bit) {
            *word &= ~bit, w->n_changed--;
        }
    }
    if (n && ++w->coll_ver == 0) w->coll_ver = 1;
    return 0;
}

/* Whether any changed bit lies in tiles lo..hi of one column (both inside the layout). */
static inline int changed_range(const uint64_t *changed, int32_t lo, int32_t hi) {
    int32_t a = lo >> 6, b = hi >> 6;
    uint64_t first = ~0ull << (lo & 63), last = ~0ull >> (63 - (hi & 63));
    if (a == b) return (changed[a] & first & last) != 0;
    if (changed[a] & first) return 1;
    for (int32_t k = a + 1; k < b; k++)
        if (changed[k]) return 1;
    return (changed[b] & last) != 0;
}

int rs_changed_in(const World *w, int32_t level, int32_t x0, int32_t x1, int32_t z0, int32_t z1) {
    /* World._changed_in (engine.py): every changed tile is a layout tile of level 0 */
    if (level != 0 || w->n_changed == 0) return 0;
    const RsLayout *L = w->L;
    int64_t lx0 = (int64_t)x0 - L->x0, lx1 = (int64_t)x1 - L->x0, lz0 = (int64_t)z0 - L->z0, lz1 = (int64_t)z1 - L->z0;
    if (lx0 < 0) lx0 = 0;
    if (lz0 < 0) lz0 = 0;
    if (lx1 > L->W - 1) lx1 = L->W - 1;
    if (lz1 > L->H - 1) lz1 = L->H - 1;
    if (lx0 > lx1 || lz0 > lz1) return 0;
    if (lx0 == 0 && lz0 == 0 && lx1 == L->W - 1 && lz1 == L->H - 1) return 1;
    for (int64_t x = lx0; x <= lx1; x++)
        if (changed_range(w->changed, (int32_t)(x * L->H + lz0), (int32_t)(x * L->H + lz1))) return 1;
    return 0;
}

int rs_window_changed(const World *w, int32_t level, int32_t sx, int32_t sz) {
    /* World.window_changed (engine.py) */
    return rs_changed_in(w, level, sx - RS_HALF, sx + RS_HALF - 1, sz - RS_HALF, sz + RS_HALF - 1);
}

/* ---- the search tree ---------------------------------------------------------------------------------------------- */
/* PathFinder.findPath1's queue order (paths._bfs, paths.py): a FIFO queue that marks a tile when it is first
   discovered and expands each tile's steps in STEPS order visits the tiles as the level-by-level search does, and the
   first discovery fixes a tile's queue position (order), depth and step back. The tree grows on demand: tree_grow
   expands queued tiles until the queue is empty or, given a mark set, until it has expanded the tile that discovered
   the first marked one; it stops between tiles, so a later call resumes the same search. Steps never leave the grid,
   so a step's tile is t + offset with no bounds test (rs_paths_reset checks the pack's border steps). */
static void tree_start(RsTree *tr, const RsLayout *L, uint32_t key, int32_t s) {
    int32_t T = L->T;
    memset(tr->order, 0xff, (size_t)T * sizeof tr->order[0]);
    memset(tr->dist, DIST_NONE, (size_t)T);
    memset(tr->parent, 0, (size_t)T);
    tr->L = L, tr->key = key, tr->src = s;
    tr->order[s] = 0, tr->dist[s] = 0, tr->parent[s] = PARENT_SOURCE, tr->queue[0] = (uint16_t)s;
    tr->head = 0, tr->tail = 1;
}

/* A fixed 8-step loop: on the supply layouts it builds a whole tree in 27-38 us, a ctz loop over the step bits in 49-58
   us and a seen-bitset test of all eight neighbours at once in 57-60 us (M2, all 128 train-128 layouts). */
static inline int32_t tree_grow(RsTree *tr, const uint8_t *steps, int32_t H, const uint64_t *mark) {
    const int32_t off[8] = {-H, H, -1, 1, -H - 1, H - 1, -H + 1, H + 1};  /* STEPS as tile offsets */
    uint16_t *order = tr->order, *queue = tr->queue;
    uint8_t *dist = tr->dist, *parent = tr->parent;
    int32_t head = tr->head, tail = tr->tail, found = -1;
    while (head < tail) {
        int32_t t = queue[head++];
        unsigned bits = steps[t];
        uint8_t d = dist[t], nd = (uint8_t)(d < 254 ? d + 1 : 254);
        for (int k = 0; k < 8; k++) {
            if (!(bits >> k & 1)) continue;
            int32_t n = t + off[k];
            if (order[n] != ORDER_NONE) continue;
            order[n] = (uint16_t)tail, dist[n] = nd, parent[n] = RS_STEP_BACK[k], queue[tail++] = (uint16_t)n;
            if (mark && found < 0 && (mark[n >> 6] >> (n & 63) & 1)) found = n;
        }
        if (found >= 0) break;
    }
    tr->head = head, tr->tail = tail;
    return found;
}

/* A tree as the queries read it: a stored tree's rows (complete) or the cache's growing tree. */
typedef struct {
    const RsLayout *L;
    const uint16_t *order;
    const uint8_t *dist, *parent;
    RsTree *grow;              /* NULL: complete rows */
    const uint8_t *steps;      /* the steps `grow` searches */
} View;

static void view_complete(View *v) {
    if (v->grow && v->grow->head < v->grow->tail) tree_grow(v->grow, v->steps, v->L->H, NULL);
}

/* World.tree (engine.py): the stored tree from (sx, sz) unless a changed tile lies in its window, then the
   live one. The stored branch refuses a tile that is not a source (paths.source_index, paths.py). */
static int tree_view(World *w, int32_t sx, int32_t sz, View *v) {
    const RsLayout *L = w->L;
    int32_t s = rs_tile(L, 0, sx, sz);
    uint32_t key;
    const uint8_t *steps;
    if (!rs_window_changed(w, 0, sx, sz)) {
        if (s < 0 || L->tile_src[s] < 0) return RS_FAIL(&w->err, RS_E_NO_PATH_TABLE, 0, sx, sz);
        if (L->order0 && L->dist0 && L->parent0) {
            size_t row = (size_t)s * (size_t)L->T;
            *v = (View){L, L->order0 + row, L->dist0 + row, L->parent0 + row, NULL, NULL};
            return 0;
        }
        key = 0, steps = L->steps_boot;
    } else {
        /* the live search has no source check; its source is always the player's tile, a source */
        if (s < 0) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a live path search from off the layout");
        key = w->coll_ver, steps = live_steps(w);
    }
    RsTree *tr = &w->pc.live;
    if (tr->L != L || tr->key != key || tr->src != s) tree_start(tr, L, key, s);
    *v = (View){L, tr->order, tr->dist, tr->parent, tr, steps};
    return 0;
}

static inline uint8_t view_dist(const View *v, int32_t x, int32_t z) {
    int32_t t = rs_tile(v->L, 0, x, z);
    return t < 0 ? DIST_NONE : v->dist[t];
}

static inline uint8_t view_parent(const View *v, int32_t x, int32_t z) {
    int32_t t = rs_tile(v->L, 0, x, z);
    return t < 0 ? 0 : v->parent[t];
}

/* paths.end_tile (paths.py) over layout tiles `reach` (the reaching tiles on the layout; the others have no
   queue position): the one first in queue order, else PathFinder.findClosestApproachPoint within 10 tiles of the
   destination rectangle (x outer, z inner; a lower cost, or the same cost with a shorter path). Returns 1 with the end
   tile, 0 for None. A growing tree stops at the first reach tile it discovers: later discoveries come later in the
   queue. */
static int end_tile(World *w, View *v, int32_t sx, int32_t sz, const int32_t *reach, int nr, int32_t dest_x,
                    int32_t dest_z, int32_t rw, int32_t rh, int32_t *ex, int32_t *ez) {
    const RsLayout *L = v->L;
    int32_t best = -1;
    for (int k = 0; k < nr; k++) {
        uint16_t o = v->order[reach[k]];
        if (o != ORDER_NONE && (best < 0 || o < v->order[best])) best = reach[k];
    }
    if (best < 0 && nr && v->grow && v->grow->head < v->grow->tail) {
        uint64_t *mark = w->pc.mark;
        for (int k = 0; k < nr; k++) mark[reach[k] >> 6] |= 1ull << (reach[k] & 63);
        best = tree_grow(v->grow, v->steps, L->H, mark);
        for (int k = 0; k < nr; k++) mark[reach[k] >> 6] = 0;
    }
    if (best >= 0) {
        *ex = L->x0 + best / L->H, *ez = L->z0 + best % L->H;
        return 1;
    }
    view_complete(v);
    int32_t ddx = dest_x - sx, ddz = dest_z - sz, lowest = 1000, shortest = 100, found = 0;
    for (int32_t x = ddx - 10; x <= ddx + 10; x++) {
        for (int32_t z = ddz - 10; z <= ddz + 10; z++) {
            int32_t d = view_dist(v, sx + x, sz + z);
            if (d >= 100) continue;  /* 255 marks unreached */
            int32_t cx = x < ddx ? ddx - x : x > ddx + rw - 1 ? x - (rw + ddx - 1) : 0;
            int32_t cz = z < ddz ? ddz - z : z > ddz + rh - 1 ? z - (rh + ddz - 1) : 0;
            int32_t cost = cx * cx + cz * cz;
            if (cost < lowest || (cost == lowest && shortest > d)) *ex = sx + x, *ez = sz + z, lowest = cost, shortest = d, found = 1;
        }
    }
    return found;
}

/* paths.backtrack (paths.py): the turning points from the end back to the source; the list keeps the 25
   nearest the source (each turn went in front and the 26th dropped the last), so a ring of the last 25 found does. */
static void backtrack(const View *v, int32_t sx, int32_t sz, int32_t ex, int32_t ez, int16_t out[RS_PATH_MAX][2], int *n) {
    int32_t cx = ex - sx + RS_HALF, cz = ez - sz + RS_HALF, curr = -1, found = 0;
    int16_t ring[RS_MAX_WAYPOINTS][2];
    int32_t nxt = view_parent(v, ex, ez);
    for (int32_t step = 0; step < BACKTRACK_LIMIT; step++) {
        if (cx == RS_HALF && cz == RS_HALF) break;
        if (curr != nxt) {
            curr = nxt;
            ring[found % RS_MAX_WAYPOINTS][0] = (int16_t)(sx + cx - RS_HALF);
            ring[found % RS_MAX_WAYPOINTS][1] = (int16_t)(sz + cz - RS_HALF);
            found++;
        }
        if (curr & RS_D_EAST) cx += 1;
        else if (curr & RS_D_WEST) cx -= 1;
        if (curr & RS_D_NORTH) cz += 1;
        else if (curr & RS_D_SOUTH) cz -= 1;
        nxt = view_parent(v, sx + cx - RS_HALF, sz + cz - RS_HALF);
    }
    int keep = found < RS_MAX_WAYPOINTS ? found : RS_MAX_WAYPOINTS;
    for (int k = 0; k < keep; k++) {
        int i = (found - 1 - k) % RS_MAX_WAYPOINTS;
        out[k][0] = ring[i][0], out[k][1] = ring[i][1];
    }
    *n = keep;
}

static void route(World *w, View *v, int32_t sx, int32_t sz, const int32_t *reach, int nr, int32_t dest_x,
                  int32_t dest_z, int32_t rw, int32_t rh, int16_t out[RS_PATH_MAX][2], int *n) {
    int32_t ex, ez;
    *n = 0;
    if (end_tile(w, v, sx, sz, reach, nr, dest_x, dest_z, rw, rh, &ex, &ez)) backtrack(v, sx, sz, ex, ez, out, n);
}

int rs_path_to_tile(World *w, int32_t sx, int32_t sz, int32_t x, int32_t z, int16_t out[RS_PATH_MAX][2], int *n) {
    /* World.path_to_tile (engine.py) = paths.path_to_tile (paths.py) */
    View v;
    RS_TRY(tree_view(w, sx, sz, &v));
    int32_t t = rs_tile(w->L, 0, x, z);
    route(w, &v, sx, sz, &t, t >= 0, x, z, 1, 1, out, n);
    return 0;
}

/* ---- ReachStrategy for a 1x1 source (paths.py) --------------------------------------------------------------- */
static inline int32_t rotate_access(int32_t angle, int32_t flags) {
    /* paths._rotate_access (paths.py); angles are 0..3 */
    return angle == 0 ? flags : ((flags << angle) & 0xf) | (flags >> (4 - angle));
}

static int reach_rectangle(const Grid *g, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t w, int32_t h,
                           int32_t access) {
    /* paths._reach_rectangle (paths.py) */
    int32_t east = dx + w - 1, north = dz + h - 1;
    if (sx == dx - 1 && dz <= sz && sz <= north && !(grid_flag(g, sx, sz) & RS_WALL_EAST) && !(access & 0x8)) return 1;
    if (sx == east + 1 && dz <= sz && sz <= north && !(grid_flag(g, sx, sz) & RS_WALL_WEST) && !(access & 0x2)) return 1;
    if (sz + 1 == dz && dx <= sx && sx <= east && !(grid_flag(g, sx, sz) & RS_WALL_NORTH) && !(access & 0x4)) return 1;
    return sz == north + 1 && dx <= sx && sx <= east && !(grid_flag(g, sx, sz) & RS_WALL_SOUTH) && !(access & 0x1);
}

#define AT(ox, oz) (sx == dx + (ox) && sz == dz + (oz))

static int reach_wall(uint32_t f, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t shape, int32_t angle) {
    /* paths._reach_wall (paths.py) */
    if (shape == 0) {
        if (angle == 0) return AT(-1, 0) || (AT(0, 1) && !(f & RS_BLOCK_N)) || (AT(0, -1) && !(f & RS_BLOCK_S));
        if (angle == 1) return AT(0, 1) || (AT(-1, 0) && !(f & RS_BLOCK_W)) || (AT(1, 0) && !(f & RS_BLOCK_E));
        if (angle == 2) return AT(1, 0) || (AT(0, 1) && !(f & RS_BLOCK_N)) || (AT(0, -1) && !(f & RS_BLOCK_S));
        return AT(0, -1) || (AT(-1, 0) && !(f & RS_BLOCK_W)) || (AT(1, 0) && !(f & RS_BLOCK_E));
    }
    if (shape == 2) {
        if (angle == 0)
            return AT(-1, 0) || AT(0, 1) || (AT(1, 0) && !(f & RS_BLOCK_E)) || (AT(0, -1) && !(f & RS_BLOCK_S));
        if (angle == 1)
            return (AT(-1, 0) && !(f & RS_BLOCK_W)) || AT(0, 1) || AT(1, 0) || (AT(0, -1) && !(f & RS_BLOCK_S));
        if (angle == 2)
            return (AT(-1, 0) && !(f & RS_BLOCK_W)) || (AT(0, 1) && !(f & RS_BLOCK_N)) || AT(1, 0) || AT(0, -1);
        return AT(-1, 0) || (AT(0, 1) && !(f & RS_BLOCK_N)) || (AT(1, 0) && !(f & RS_BLOCK_E)) || AT(0, -1);
    }
    if (shape == 9)
        return (AT(0, 1) && !(f & RS_WALL_SOUTH)) || (AT(0, -1) && !(f & RS_WALL_NORTH)) ||
               (AT(-1, 0) && !(f & RS_WALL_EAST)) || (AT(1, 0) && !(f & RS_WALL_WEST));
    return 0;
}

static int reach_wall_decor(uint32_t f, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t shape, int32_t angle) {
    /* paths._reach_wall_decor (paths.py) */
    if (shape == 6 || shape == 7) {
        int32_t rotation = shape == 7 ? (angle + 2) & 3 : angle;
        if (rotation == 0) return (AT(1, 0) && !(f & RS_WALL_WEST)) || (AT(0, -1) && !(f & RS_WALL_NORTH));
        if (rotation == 1) return (AT(-1, 0) && !(f & RS_WALL_EAST)) || (AT(0, -1) && !(f & RS_WALL_NORTH));
        if (rotation == 2) return (AT(-1, 0) && !(f & RS_WALL_EAST)) || (AT(0, 1) && !(f & RS_WALL_SOUTH));
        return (AT(1, 0) && !(f & RS_WALL_WEST)) || (AT(0, 1) && !(f & RS_WALL_SOUTH));
    }
    if (shape == 8)
        return (AT(0, 1) && !(f & RS_WALL_SOUTH)) || (AT(0, -1) && !(f & RS_WALL_NORTH)) ||
               (AT(-1, 0) && !(f & RS_WALL_EAST)) || (AT(1, 0) && !(f & RS_WALL_WEST));
    return 0;
}

#undef AT

static int reached(const Grid *g, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t width, int32_t length,
                   int32_t angle, int32_t shape, int32_t block_access) {
    /* paths.reached (paths.py): ReachStrategy.reached for srcSize 1 */
    if (shape == -2) {
        int32_t rw = angle & 1 ? length : width, rh = angle & 1 ? width : length;
        int inside = dx <= sx && sx < dx + rw && dz <= sz && sz < dz + rh;
        return !inside && reach_rectangle(g, sx, sz, dx, dz, rw, rh, rotate_access(angle, block_access));
    }
    if (sx == dx && sz == dz) return 1;
    if ((0 <= shape && shape <= 3) || shape == 9) return reach_wall(grid_flag(g, sx, sz), sx, sz, dx, dz, shape, angle);
    if (shape < 0) return 0;
    if (shape < 9) return reach_wall_decor(grid_flag(g, sx, sz), sx, sz, dx, dz, shape, angle);
    if (shape == 10 || shape == 11 || shape == 22) {
        int32_t rw = angle & 1 ? length : width, rh = angle & 1 ? width : length;
        return (dx <= sx && sx < dx + rw && dz <= sz && sz < dz + rh) ||
               reach_rectangle(g, sx, sz, dx, dz, rw, rh, rotate_access(angle, block_access));
    }
    return 0;
}

/* paths.loc_reaching (paths.py): the tiles of the reach box that reach the loc, x outer, z inner. */
static int loc_reaching_live(const Grid *g, int32_t x, int32_t z, int32_t width, int32_t length, int32_t angle,
                             int32_t shape, int32_t forceapproach, int16_t out[][2], int *n, int cap) {
    int k = 0;
    for (int32_t sx = x - RS_REACH_BOX; sx < x + RS_REACH_BOX + width + 1; sx++) {
        for (int32_t sz = z - RS_REACH_BOX; sz < z + RS_REACH_BOX + length + 1; sz++) {
            if (!reached(g, sx, sz, x, z, width, length, angle, shape, forceapproach)) continue;
            if (k == cap) return 1;
            out[k][0] = (int16_t)sx, out[k][1] = (int16_t)sz, k++;
        }
    }
    *n = k;
    return 0;
}

/* paths.entity_reaching_live (paths.py) as NEIGHBOURS bits. */
static uint16_t entity_reaching_live(const Grid *g, int32_t x, int32_t z) {
    uint16_t mask = 0;
    for (int b = 0; b < 9; b++)
        if (reached(g, x + RS_NEIGHBOUR_DX[b], z + RS_NEIGHBOUR_DZ[b], x, z, 1, 1, 0, -2, 0)) mask |= (uint16_t)(1u << b);
    return mask;
}

int rs_entity_reaching(World *w, int32_t x, int32_t z, uint16_t *mask) {
    /* World.entity_reaching (engine.py): live when a tile of the 3x3 changed, else the stored bits
       (paths.entity_reaching, paths.py), which refuse a tile off the data area */
    if (rs_changed_in(w, 0, x - 1, x + 1, z - 1, z + 1)) {
        Grid g = world_grid(w);
        *mask = entity_reaching_live(&g, x, z);
        return 0;
    }
    int32_t t = rs_tile(w->L, 0, x, z);
    if (t < 0) return RS_FAIL(&w->err, RS_E_ENTITY_OUTSIDE, 0, x, z);
    if (w->L->entity_reach) {
        *mask = w->L->entity_reach[t];
    } else {
        Grid g = world_grid(w);  /* the stored bits are the live ones on the boot flags, which these are */
        *mask = entity_reaching_live(&g, x, z);
    }
    return 0;
}

int rs_loc_reaching(World *w, const Loc *loc, int16_t out[][2], int *n, int cap) {
    /* World.loc_reaching (engine.py): a map loc in its base shape and angle, whose type keeps the base's
       forceapproach and whose reach box saw no change, reads its stored list (on the layout only, sim/world.py
       live_tables); any other loc is searched live */
    const RsLocType *t = rs_loc_type(w->P, loc->type), *base = rs_loc_type(w->P, loc->base);
    if (!t || !base) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a loc of a type the pack lacks");
    const RsLayout *L = w->L;
    int box = RS_REACH_BOX;
    if (loc->index < w->n_stored_locs && !L->loc_reach) return RS_FAIL_TEXT(&w->err, RS_E_PACK, "a layout without loc reach");
    if (loc->index < w->n_stored_locs && loc->shape == loc->base_shape && loc->angle == loc->base_angle &&
        t->forceapproach == base->forceapproach &&
        !rs_changed_in(w, loc->level, loc->x - box, loc->x + box + loc->width, loc->z - box, loc->z + box + loc->length)) {
        const int16_t *row = L->loc_reach + (size_t)loc->index * 24;
        int k = 0;
        for (int i = 0; i < 24 && row[i] >= 0; i++) {
            if (k == cap) return RS_OVERFLOW(&w->err, "loc reach tiles");
            out[k][0] = (int16_t)(L->x0 + row[i] / L->H), out[k][1] = (int16_t)(L->z0 + row[i] % L->H), k++;
        }
        *n = k;
        return 0;
    }
    Grid g = world_grid(w);
    if (loc->level != 0) g.W = 0;  /* live_flags(level) reads FLAG_NULL off level 0 */
    if (loc_reaching_live(&g, loc->x, loc->z, loc->width, loc->length, loc->angle, loc->shape, t->forceapproach, out, n,
                          cap))
        return RS_OVERFLOW(&w->err, "loc reach tiles");
    return 0;
}

/* ---- LineValidator.hasLineOfSight, 1x1 to 1x1, no extra flag (paths.py) ------------------------------------ */
static inline int64_t shr16(int64_t a) { return a >= 0 ? a >> 16 : -(int64_t)((~(uint64_t)a) >> 16) - 1; }  /* a >> 16 */

static int line_of_sight(const Grid *g, int32_t sx, int32_t sz, int32_t ex, int32_t ez) {
    if (sx == ex && sz == ez) return 1;
    if (grid_flag(g, sx, sz) & RS_FLAG_LOC) return 0;
    int32_t dx = ex - sx, dz = ez - sz;
    int east = dx >= 0, north = dz >= 0;
    uint32_t x_flags = RS_FLAG_LOC_PROJ | (east ? RS_PROJ_WEST : RS_PROJ_EAST);
    uint32_t z_flags = RS_FLAG_LOC_PROJ | (north ? RS_PROJ_SOUTH : RS_PROJ_NORTH);
    const int64_t half = 1 << 15;
    if (iabs(dx) > iabs(dz)) {
        int32_t ox = east ? 1 : -1, oz = north ? 0 : -1;
        int64_t scaled = (int64_t)sz * 65536 + half + oz;
        int64_t tangent = (int64_t)dz * 65536 / iabs(dx);  /* int((dz << 16) / abs(dx)): exact for these magnitudes */
        int32_t x = sx;
        while (x != ex) {
            x += ox;
            int32_t z = (int32_t)shr16(scaled);
            if (x == ex && z == ez) x_flags &= ~(uint32_t)RS_FLAG_LOC_PROJ;
            if (grid_flag(g, x, z) & x_flags) return 0;
            scaled += tangent;
            int32_t nz = (int32_t)shr16(scaled);
            if (x == ex && nz == ez) z_flags &= ~(uint32_t)RS_FLAG_LOC_PROJ;
            if (nz != z && (grid_flag(g, x, nz) & z_flags)) return 0;
        }
    } else {
        int32_t ox = east ? 0 : -1, oz = north ? 1 : -1;
        int64_t scaled = (int64_t)sx * 65536 + half + ox;
        int64_t tangent = (int64_t)dx * 65536 / iabs(dz);
        int32_t z = sz;
        while (z != ez) {
            z += oz;
            int32_t x = (int32_t)shr16(scaled);
            if (x == ex && z == ez) z_flags &= ~(uint32_t)RS_FLAG_LOC_PROJ;
            if (grid_flag(g, x, z) & z_flags) return 0;
            scaled += tangent;
            int32_t nx = (int32_t)shr16(scaled);
            if (nx == ex && z == ez) x_flags &= ~(uint32_t)RS_FLAG_LOC_PROJ;
            if (nx != x && (grid_flag(g, nx, z) & x_flags)) return 0;
        }
    }
    return 1;
}

int rs_line_of_sight(World *w, int32_t ax, int32_t az, int32_t bx, int32_t bz, int *visible) {
    /* World.line_of_sight (engine.py): live when a changed tile lies in the box around both ends; else the
       stored table (Tables.line_of_sight, data.py), which is the ray on the boot flags (sim/world.py
       live_tables: the conjunction of the ray's checks, translation invariant), and which raises KeyError for a first
       tile off the data area or a second beyond los_r. The ray reads only tiles of that box. */
    Grid g = world_grid(w);
    if (!rs_changed_in(w, 0, imin(ax, bx) - 1, imax(ax, bx) + 1, imin(az, bz) - 1, imax(az, bz) + 1)) {
        if (rs_tile(w->L, 0, ax, az) < 0 || imax(iabs(bx - ax), iabs(bz - az)) > RS_LOS_R)
            return RS_FAIL(&w->err, RS_E_LOS_OUTSIDE, 0, ax, az, bx, bz);
    }
    *visible = line_of_sight(&g, ax, az, bx, bz);
    return 0;
}

/* ---- walk steps on the snapshot (paths.py) -------------------------------------------------------------------- */
/* Step masks as column bitsets: M[x + 1][k] holds bit z where tile (x, z) has step k (columns 0 and W + 1 stay empty;
   a column's eight words share a cache line). */
static void column_masks(const uint8_t *steps, int32_t W, int32_t H, uint64_t M[RS_MAX_W + 2][8]) {
    memset(M, 0, sizeof(uint64_t) * (RS_MAX_W + 2) * 8);
    for (int32_t x = 0; x < W; x++)
        for (int32_t z = 0; z < H; z++) {
            unsigned bits = steps[x * H + z];
            for (int k = 0; k < 8; k++)
                if (bits >> k & 1) M[x + 1][k] |= 1ull << z;
        }
}

/* The search's depths from tile s (min(depth, 254), 255 unreached), level by level over column bitsets: a depth is the
   same in any visiting order, so this gives the dists of paths.search on the same steps (14-21 us a whole supply
   layout, against 27-38 us for the queue-order search). A row's search starts with only its source written and runs
   a level at a time, as far as the queries of the row need: a tile's depth is written at its own level, so a row read
   where its search found the tile already equals the whole search's. */
static void grow_start(RsSnapGrow *g, const RsLayout *L, int32_t s, uint8_t *dist) {
    int32_t H = L->H, c = s / H + 1;
    memset(dist, DIST_NONE, (size_t)L->T);
    memset(g->seen, 0, sizeof g->seen);
    memset(g->front, 0, sizeof g->front);
    g->front[c] = g->seen[c] = 1ull << (s % H);
    dist[s] = 0;
    g->lo = g->hi = c, g->level = 0, g->done = 0;
    g->reach = L->has_reach && (L->snap_core[c] >> (s % H) & 1);
}

/* One level of the search: the tiles one step from the last level's that it had not found, written at level + 1. */
static void grow_level(RsSnapGrow *g, const uint64_t M[RS_MAX_W + 2][8], int32_t W, int32_t H, uint8_t *dist) {
    uint64_t next[RS_MAX_W + 2];
    int32_t lo = g->lo, hi = g->hi, level = ++g->level;
    int32_t a = imax(lo - 1, 1), b = imin(hi + 1, W);
    for (int32_t k = a - 1; k <= b + 1; k++) next[k] = 0;
    for (int32_t k = lo; k <= hi; k++) {
        uint64_t f = g->front[k];
        const uint64_t *m = M[k];
        next[k - 1] |= (f & m[0]) | ((f & m[4]) >> 1) | ((f & m[6]) << 1);  /* W, SW, NW */
        next[k + 1] |= (f & m[1]) | ((f & m[5]) >> 1) | ((f & m[7]) << 1);  /* E, SE, NE */
        next[k] |= ((f & m[2]) >> 1) | ((f & m[3]) << 1);                   /* S, N */
    }
    uint8_t d = (uint8_t)(level < 254 ? level : 254);
    int32_t nlo = b + 1, nhi = a - 1;
    for (int32_t k = a; k <= b; k++) {
        uint64_t fresh = next[k] & ~g->seen[k];
        g->front[k] = fresh;
        if (!fresh) continue;
        g->seen[k] |= fresh;
        nlo = imin(nlo, k), nhi = imax(nhi, k);
        uint8_t *col = dist + (size_t)(k - 1) * H;
        while (fresh) {
            col[__builtin_ctzll(fresh)] = d;
            fresh &= fresh - 1;
        }
    }
    g->lo = nlo, g->hi = nhi;  /* [a, b] covers the old front's columns, so every one was rewritten */
    g->done = nlo > nhi;
}

/* The walk row's depth of local tile (tx, tz) once its search has run far enough to know it: a tile the search has not
   found is unreached once the search is complete, or at once when the source reaches only snap_reach and the tile
   lies outside it; otherwise the search runs on until it finds the tile, which it then must. Kept out of line, so
   the reads of a stored or already searched row stay small enough to inline. */
__attribute__((noinline)) static uint8_t grow_to(RsPathCache *pc, const RsLayout *L, int32_t tx, int32_t tz) {
    RsSnapGrow *g = pc->walk_grow;
    uint8_t *dist = (uint8_t *)pc->walk_row;
    int32_t t = tx * L->H + tz;
    if (g->reach && !(L->snap_reach[tx + 1] >> tz & 1)) return DIST_NONE;
    while (!g->done && dist[t] == DIST_NONE)
        grow_level(g, (const uint64_t (*)[8])pc->snap_masks, L->W, L->H, dist);
    return dist[t];
}

/* The snapshot distance row of source tile s: the pack's derived row (complete, *grow NULL), else a cached row and its
   search (RS_SNAP_ROWS rows, the least recently used replaced). */
static const uint8_t *snap_row(World *w, int32_t s, RsSnapGrow **grow) {
    const RsLayout *L = w->L;
    *grow = NULL;
    if (L->snap_dist) return L->snap_dist + (size_t)s * (size_t)L->T;
    RsPathCache *pc = &w->pc;
    int slot = 0;
    for (int k = 0; k < RS_SNAP_ROWS; k++) {
        if (pc->snap_L[k] == L && pc->snap_src[k] == s) {
            pc->snap_used[k] = ++pc->snap_clock;
            *grow = &pc->snap_grow[k];
            return pc->snap_dist[k];
        }
        if (pc->snap_used[k] < pc->snap_used[slot]) slot = k;
    }
    if (pc->mask_L != L) column_masks(L->steps_snap, L->W, L->H, pc->snap_masks), pc->mask_L = L;
    grow_start(&pc->snap_grow[slot], L, s, pc->snap_dist[slot]);
    pc->snap_L[slot] = L, pc->snap_src[slot] = s, pc->snap_used[slot] = ++pc->snap_clock;
    *grow = &pc->snap_grow[slot];
    return pc->snap_dist[slot];
}

/* The walk row of source index src (meta['sources'][src], a list: a negative index counts from the end) into
   pc.walk_*; the last one resolved is kept, so the queries of one observation resolve it once. */
static int walk_resolve(World *w, int32_t src) {
    const RsLayout *L = w->L;
    RsPathCache *pc = &w->pc;
    int32_t k = src < 0 ? src + L->n_src : src;
    if (k < 0 || k >= L->n_src) return RS_FAIL_TEXT(&w->err, RS_E_INTERNAL, "a walk from a source index out of range");
    int32_t s = L->src_tile[k];
    pc->walk_row = snap_row(w, s, &pc->walk_grow);
    pc->walk_L = L, pc->walk_src = src, pc->walk_ox = s / L->H, pc->walk_oz = s % L->H;
    return 0;
}

/* The resolved row's walk steps at offset (dx, dz) from its source: 255 off the layout or the window, -1 unreached. */
static inline int32_t walk_read(RsPathCache *pc, const RsLayout *L, int32_t dx, int32_t dz) {
    int32_t tx = pc->walk_ox + dx, tz = pc->walk_oz + dz;
    if (dx < -RS_HALF || dx >= RS_HALF || dz < -RS_HALF || dz >= RS_HALF || tx < 0 || tx >= L->W || tz < 0 || tz >= L->H)
        return -1;
    uint8_t d = pc->walk_row[tx * L->H + tz];
    if (d == DIST_NONE && pc->walk_grow) d = grow_to(pc, L, tx, tz);
    return d == DIST_NONE ? -1 : d;
}

int rs_walk_steps(World *w, int32_t src, int32_t sx, int32_t sz, int32_t x, int32_t z, int32_t *steps) {
    /* paths.walk_steps (paths.py) through LayoutTables' walks (sim/world.py _PerSource: the snapshot window of
       source src, read at the offset (x - sx, z - sz); 255 off the layout or the window), -1 unreached */
    int32_t dx = x - sx, dz = z - sz;
    if (imax(iabs(dx), iabs(dz)) > RS_WALK_R) return RS_FAIL(&w->err, RS_E_WALK_BEYOND, sx, sz, x, z);
    if (w->pc.walk_L != w->L || w->pc.walk_src != src) RS_TRY(walk_resolve(w, src));
    *steps = walk_read(&w->pc, w->L, dx, dz);
    return 0;
}

int rs_walk_approach(World *w, int32_t src, int32_t sx, int32_t sz, int32_t x, int32_t z, int32_t *steps) {
    /* paths.walk_approach (paths.py): dx outer, dz inner; the first refusal in that order */
    int32_t best = -1, cx = x - sx, cz = z - sz;
    if (imax(iabs(cx), iabs(cz)) + 1 > RS_WALK_R) {   /* a tile of the 3x3 may be refused: query each in order */
        for (int32_t dx = -1; dx <= 1; dx++) {
            for (int32_t dz = -1; dz <= 1; dz++) {
                int32_t s;
                RS_TRY(rs_walk_steps(w, src, sx, sz, x + dx, z + dz, &s));
                if (s >= 0 && (best < 0 || s < best)) best = s;
            }
        }
        *steps = best;
        return 0;
    }
    if (w->pc.walk_L != w->L || w->pc.walk_src != src) RS_TRY(walk_resolve(w, src));
    for (int32_t dx = -1; dx <= 1; dx++) {
        for (int32_t dz = -1; dz <= 1; dz++) {
            int32_t s = walk_read(&w->pc, w->L, cx + dx, cz + dz);
            if (s >= 0 && (best < 0 || s < best)) best = s;
        }
    }
    *steps = best;
    return 0;
}

/* ---- the client's tryMove (paths.py client_*, act-v3) -------------------------------------------------------------- */
/* paths.client_flags of one live flag: walls and scenery (0x1ff), and the floor where the server has FLOOR or
   FLOOR_DECORATION (0x240000) as C_GRND; no npc or player bit. */
static inline uint32_t client_flag(uint32_t f) { return (f & 0x1ffu) | (f & 0x240000u ? (uint32_t)RS_C_GRND : 0u); }

/* The client's collision of a tile (SceneTree.flags): its live flags, C_BOUNDS off the layout (client_steps' border). */
static inline uint32_t client_at(const World *w, int32_t x, int32_t z) {
    const RsLayout *L = w->L;
    int32_t lx = x - L->x0, lz = z - L->z0;
    return lx >= 0 && lx < L->W && lz >= 0 && lz < L->H ? client_flag((uint32_t)w->flags[lx * L->H + lz])
                                                         : (uint32_t)RS_C_BOUNDS;
}

/* World.client_grid's steps (paths.client_steps): a step's destination may hold no PL_WALK bit of the side it is
   entered from; a diagonal needs both of its orthogonal steps. Rebuilt after each collision change. */
static const uint8_t *client_steps(World *w) {
    RsPathCache *pc = &w->pc;
    const RsLayout *L = w->L;
    if (pc->client_L == L && pc->client_ver == w->coll_ver) return pc->client_steps;
    for (int32_t x = 0; x < L->W; x++)
        for (int32_t z = 0; z < L->H; z++) {
            int32_t gx = L->x0 + x, gz = L->z0 + z;
#define FREE(dx, dz, m) (!(client_at(w, gx + (dx), gz + (dz)) & (uint32_t)(m)))
            unsigned we = FREE(-1, 0, RS_PL_WALK_E), ea = FREE(1, 0, RS_PL_WALK_W);
            unsigned so = FREE(0, -1, RS_PL_WALK_N), no = FREE(0, 1, RS_PL_WALK_S);
            unsigned sw = FREE(-1, -1, RS_PL_WALK_NE) & we & so, se = FREE(1, -1, RS_PL_WALK_NW) & ea & so;
            unsigned nw = FREE(-1, 1, RS_PL_WALK_SE) & we & no, ne = FREE(1, 1, RS_PL_WALK_SW) & ea & no;
#undef FREE
            pc->client_steps[x * L->H + z] = (uint8_t)(we | ea << 1 | so << 2 | no << 3 | sw << 4 | se << 5 | nw << 6
                                                       | ne << 7);
        }
    pc->client_L = L, pc->client_ver = w->coll_ver;
    return pc->client_steps;
}

/* World.scene_tree (paths.client_search, bounded): tryMove's search from (sx, sz) over the scene at (bx, bz), kept to
   the scene's inner tiles (base + 1 .. base + SCENE - 2) that lie on the layout, in findPath1's queue order (STEPS
   order, first discovery fixes a tile's position, steps and step back). A source outside that window is
   RS_E_ROUTE_SOURCE (OutsideTables). Kept until the collision, the source or the scene changes. */
static int scene_tree(World *w, int32_t sx, int32_t sz, int32_t bx, int32_t bz, const RsSceneTree **out) {
    RsSceneTree *tr = &w->pc.scene;
    const RsLayout *L = w->L;
    int32_t H = L->H;
    int32_t lo_x = imax(bx + 1 - L->x0, 0), lo_z = imax(bz + 1 - L->z0, 0);
    int32_t hi_x = imin(bx + RS_SCENE - 2 - L->x0, L->W - 1), hi_z = imin(bz + RS_SCENE - 2 - L->z0, H - 1);
    int32_t lx = sx - L->x0, lz = sz - L->z0;
    if (!(lo_x <= lx && lx <= hi_x && lo_z <= lz && lz <= hi_z)) return RS_FAIL(&w->err, RS_E_ROUTE_SOURCE, sx, sz);
    int32_t s = lx * H + lz;
    *out = tr;
    if (tr->L == L && tr->ver == w->coll_ver && tr->src == s && tr->bx == bx && tr->bz == bz) return 0;
    const uint8_t *steps = client_steps(w);
    memset(tr->order, 0xff, (size_t)L->T * sizeof tr->order[0]);
    memset(tr->dist, 0xff, (size_t)L->T * sizeof tr->dist[0]);
    memset(tr->back, 0, (size_t)L->T);
    const int32_t off[8] = {-H, H, -1, 1, -H - 1, H - 1, -H + 1, H + 1};  /* STEPS as tile offsets */
    tr->order[s] = 0, tr->dist[s] = 0, tr->back[s] = PARENT_SOURCE, tr->queue[0] = (uint16_t)s;
    int32_t head = 0, tail = 1;
    while (head < tail) {
        int32_t t = tr->queue[head++], tx = t / H, tz = t % H;
        unsigned bits = steps[t];
        for (int k = 0; k < 8; k++) {
            int32_t nx = tx + RS_STEP_DX[k], nz = tz + RS_STEP_DZ[k];
            if (!(bits >> k & 1) || nx < lo_x || nx > hi_x || nz < lo_z || nz > hi_z) continue;
            int32_t n = t + off[k];
            if (tr->order[n] != ORDER_NONE) continue;
            tr->order[n] = (uint16_t)tail, tr->dist[n] = (uint16_t)(tr->dist[t] + 1), tr->back[n] = RS_STEP_BACK[k];
            tr->queue[tail++] = (uint16_t)n;
        }
    }
    tr->L = L, tr->ver = w->coll_ver, tr->src = s, tr->bx = bx, tr->bz = bz;
    return 0;
}

/* SceneTree.at for the queue position, the steps and the step back of a tile: off the layout, as unreached. */
static inline int32_t scene_order(const RsSceneTree *tr, const RsLayout *L, int32_t x, int32_t z) {
    int32_t t = rs_tile(L, 0, x, z);
    return t < 0 || tr->order[t] == ORDER_NONE ? -1 : tr->order[t];
}
static inline int32_t scene_dist(const RsSceneTree *tr, const RsLayout *L, int32_t x, int32_t z) {
    int32_t t = rs_tile(L, 0, x, z);
    return t < 0 || tr->dist[t] == 0xffff ? RS_SCENE_UNREACHED : tr->dist[t];
}
static inline uint8_t scene_back(const RsSceneTree *tr, const RsLayout *L, int32_t x, int32_t z) {
    int32_t t = rs_tile(L, 0, x, z);
    return t < 0 ? 0 : tr->back[t];
}

/* CollisionMap.testWall (paths.client_test_wall): (sx, sz), of client collision f, reaches a wall of shape, angle. */
static int test_wall(uint32_t f, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t shape, int32_t angle) {
    if (sx == dx && sz == dz) return 1;
#define WALK(m) (!(f & (uint32_t)(m)))
    if (shape == 0) {
        if (angle == 0) return (sx == dx - 1 && sz == dz) || (sx == dx && sz == dz + 1 && WALK(RS_PL_WALK_S))
                               || (sx == dx && sz == dz - 1 && WALK(RS_PL_WALK_N));
        if (angle == 1) return (sx == dx && sz == dz + 1) || (sx == dx - 1 && sz == dz && WALK(RS_PL_WALK_E))
                               || (sx == dx + 1 && sz == dz && WALK(RS_PL_WALK_W));
        if (angle == 2) return (sx == dx + 1 && sz == dz) || (sx == dx && sz == dz + 1 && WALK(RS_PL_WALK_S))
                               || (sx == dx && sz == dz - 1 && WALK(RS_PL_WALK_N));
        if (angle == 3) return (sx == dx && sz == dz - 1) || (sx == dx - 1 && sz == dz && WALK(RS_PL_WALK_E))
                               || (sx == dx + 1 && sz == dz && WALK(RS_PL_WALK_W));
    } else if (shape == 2) {
        if (angle == 0) return (sx == dx - 1 && sz == dz) || (sx == dx && sz == dz + 1)
                               || (sx == dx + 1 && sz == dz && WALK(RS_PL_WALK_W)) || (sx == dx && sz == dz - 1 && WALK(RS_PL_WALK_N));
        if (angle == 1) return (sx == dx - 1 && sz == dz && WALK(RS_PL_WALK_E)) || (sx == dx && sz == dz + 1)
                               || (sx == dx + 1 && sz == dz) || (sx == dx && sz == dz - 1 && WALK(RS_PL_WALK_N));
        if (angle == 2) return (sx == dx - 1 && sz == dz && WALK(RS_PL_WALK_E)) || (sx == dx && sz == dz + 1 && WALK(RS_PL_WALK_S))
                               || (sx == dx + 1 && sz == dz) || (sx == dx && sz == dz - 1);
        if (angle == 3) return (sx == dx - 1 && sz == dz) || (sx == dx && sz == dz + 1 && WALK(RS_PL_WALK_S))
                               || (sx == dx + 1 && sz == dz && WALK(RS_PL_WALK_W)) || (sx == dx && sz == dz - 1);
    } else if (shape == 9) {
        return (sx == dx && sz == dz + 1 && WALK(RS_C_W_S)) || (sx == dx && sz == dz - 1 && WALK(RS_C_W_N))
               || (sx == dx - 1 && sz == dz && WALK(RS_C_W_E)) || (sx == dx + 1 && sz == dz && WALK(RS_C_W_W));
    }
    return 0;
}

/* CollisionMap.testWDecor (paths.client_test_wdecor). */
static int test_wdecor(uint32_t f, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t shape, int32_t angle) {
    if (sx == dx && sz == dz) return 1;
    if (shape == 6 || shape == 7) {
        if (shape == 7) angle = (angle + 2) & 3;
        if (angle == 0) return (sx == dx + 1 && sz == dz && WALK(RS_C_W_W)) || (sx == dx && sz == dz - 1 && WALK(RS_C_W_N));
        if (angle == 1) return (sx == dx - 1 && sz == dz && WALK(RS_C_W_E)) || (sx == dx && sz == dz - 1 && WALK(RS_C_W_N));
        if (angle == 2) return (sx == dx - 1 && sz == dz && WALK(RS_C_W_E)) || (sx == dx && sz == dz + 1 && WALK(RS_C_W_S));
        if (angle == 3) return (sx == dx + 1 && sz == dz && WALK(RS_C_W_W)) || (sx == dx && sz == dz + 1 && WALK(RS_C_W_S));
    } else if (shape == 8) {
        return (sx == dx && sz == dz + 1 && WALK(RS_C_W_S)) || (sx == dx && sz == dz - 1 && WALK(RS_C_W_N))
               || (sx == dx - 1 && sz == dz && WALK(RS_C_W_E)) || (sx == dx + 1 && sz == dz && WALK(RS_C_W_W));
    }
    return 0;
}

/* CollisionMap.testLoc (paths.client_test_loc): inside the width x length rectangle at (dx, dz), or beside a side that
   forceapproach leaves open with no wall between. */
static int test_loc(uint32_t f, int32_t sx, int32_t sz, int32_t dx, int32_t dz, int32_t width, int32_t length,
                    int32_t forceapproach) {
    int32_t mx = dx + width - 1, mz = dz + length - 1;
    if (dx <= sx && sx <= mx && dz <= sz && sz <= mz) return 1;
    return (sx == dx - 1 && dz <= sz && sz <= mz && WALK(RS_C_W_E) && !(forceapproach & RS_D_WEST))
           || (sx == mx + 1 && dz <= sz && sz <= mz && WALK(RS_C_W_W) && !(forceapproach & RS_D_EAST))
           || (sz == dz - 1 && dx <= sx && sx <= mx && WALK(RS_C_W_N) && !(forceapproach & RS_D_SOUTH))
           || (sz == mz + 1 && dx <= sx && sx <= mx && WALK(RS_C_W_S) && !(forceapproach & RS_D_NORTH));
#undef WALK
}

/* client_arrival's test (paths.py): a wall test when locShape - 1 is a wall, a wall-decoration test when it is below
   CENTREPIECE_STRAIGHT, a rectangle test with a size. */
static int arrives(const RsArrival *a, uint32_t f, int32_t sx, int32_t sz, int32_t dx, int32_t dz) {
    int32_t shape = a->loc_shape - 1;
    int wall = a->loc_shape != 0 && (a->loc_shape < 5 || a->loc_shape == 10), decor = a->loc_shape != 0 && a->loc_shape < 10;
    int rect = a->width != 0 && a->length != 0;
    return (wall && test_wall(f, sx, sz, dx, dz, shape, a->angle)) || (decor && test_wdecor(f, sx, sz, dx, dz, shape, a->angle))
           || (rect && test_loc(f, sx, sz, dx, dz, a->width, a->length, a->forceapproach));
}

const RsArrival RS_ENTITY_ARRIVAL = {1, 1, 0, 0, 0};

RsArrival rs_loc_arrival(const RsPack *P, const Loc *loc) {
    /* paths.loc_arrival (Client.ts 7920-7939) */
    const RsLocType *t = &P->loc_types[loc->type];
    int32_t shape = loc->shape, angle = loc->angle;
    if (shape == 10 || shape == 11 || shape == 22) {
        int32_t w = angle == 0 || angle == 2 ? t->width : t->length, h = angle == 0 || angle == 2 ? t->length : t->width;
        int32_t fa = t->forceapproach;
        if (angle != 0) fa = ((fa << angle) & 0xf) + (fa >> (4 - angle));
        return (RsArrival){(int16_t)w, (int16_t)h, 0, 0, (int16_t)fa};
    }
    return (RsArrival){0, 0, (int16_t)angle, (int16_t)(shape + 1), 0};
}

int rs_client_route(World *w, int32_t x, int32_t z, const RsArrival *arrival, int nearest, int16_t out[RS_PATH_MAX][2],
                    int *n, uint8_t *via, int *found) {
    /* Agent.route: paths.client_route over World.player_scene (scene_base of the build area's origin) */
    const Player *p = &w->p;
    const RsLayout *L = w->L;
    const RsSceneTree *tr;
    int32_t bx = ((p->origin_x >> 3) - 6) << 3, bz = ((p->origin_z >> 3) - 6) << 3;
    if (scene_tree(w, p->x, p->z, bx, bz, &tr)) return rs_wrap_outside(&w->err);
    int32_t rw = arrival ? imax(arrival->width, 1) : 1, rh = arrival ? imax(arrival->length, 1) : 1;
    int32_t ex = 0, ez = 0, best = -1;
    *found = 0, *n = 0;
    for (int32_t cx = x - 1; cx < x + rw + 1; cx++)
        for (int32_t cz = z - 1; cz < z + rh + 1; cz++) {
            int32_t o = scene_order(tr, L, cx, cz);
            if (o >= 0 && (best < 0 || o < best)
                && ((cx == x && cz == z) || (arrival && arrives(arrival, client_at(w, cx, cz), cx, cz, x, z))))
                ex = cx, ez = cz, best = o, *found = 1;
        }
    *via = ex == x && ez == z && *found ? RS_VIA_TARGET : RS_VIA_REACH;
    if (!*found && nearest) {
        int32_t least = RS_NEAREST_STEPS;
        *via = RS_VIA_FALLBACK;
        for (int32_t cx = x - 1; cx < x + 2; cx++)
            for (int32_t cz = z - 1; cz < z + 2; cz++) {
                int32_t d = scene_dist(tr, L, cx, cz);
                if (d < least) ex = cx, ez = cz, least = d, *found = 1;
            }
    }
    if (!*found) return 0;
    /* the turn points from the end back to the source, then the RS_PATH_MAX nearest the source */
    int16_t turns[RS_MAX_T][2];
    int32_t nt = 0, cx = ex, cz = ez;
    uint8_t curr = scene_back(tr, L, cx, cz), nxt = curr;
    turns[nt][0] = (int16_t)cx, turns[nt++][1] = (int16_t)cz;
    while (cx != p->x || cz != p->z) {
        if (nxt != curr) {
            curr = nxt;
            turns[nt][0] = (int16_t)cx, turns[nt++][1] = (int16_t)cz;
        }
        cx += nxt & RS_D_EAST ? 1 : nxt & RS_D_WEST ? -1 : 0;
        cz += nxt & RS_D_NORTH ? 1 : nxt & RS_D_SOUTH ? -1 : 0;
        nxt = scene_back(tr, L, cx, cz);
    }
    int keep = nt < RS_PATH_MAX ? nt : RS_PATH_MAX;
    for (int k = 0; k < keep; k++) out[k][0] = turns[nt - 1 - k][0], out[k][1] = turns[nt - 1 - k][1];
    *n = keep;
    return 0;
}

/* ---- reset ----------------------------------------------------------------------------------------------------------- */
/* A layout's steps must stay on it (the searches add tile offsets without bounds tests): no step leaves the border. */
static int border_ok(const uint8_t *steps, int32_t W, int32_t H) {
    const unsigned west = 1u | 1u << 4 | 1u << 6, east = 1u << 1 | 1u << 5 | 1u << 7;
    const unsigned south = 1u << 2 | 1u << 4 | 1u << 5, north = 1u << 3 | 1u << 6 | 1u << 7;
    for (int32_t z = 0; z < H; z++)
        if ((steps[z] & west) || (steps[(W - 1) * H + z] & east)) return 0;
    for (int32_t x = 0; x < W; x++)
        if ((steps[x * H] & south) || (steps[x * H + H - 1] & north)) return 0;
    return 1;
}

/* World init: the live steps and a tree of live steps are keyed by World.coll_ver, which restarts with the world, so
   they go; a tree of the boot steps, the snapshot rows and their masks are functions of their layout and stay (a
   World zeroed by memset has none). */
void rs_paths_reset(World *w) {
    RsPathCache *pc = &w->pc;
    pc->steps_L = NULL, pc->steps_ver = 0;
    pc->client_L = NULL, pc->scene.L = NULL;   /* keyed by World.coll_ver */
    if (pc->live.key != 0) pc->live.L = NULL;
    memset(pc->mark, 0, sizeof pc->mark);
    const RsLayout *L = w->L;
    if (L && ((L->steps_boot && !border_ok(L->steps_boot, L->W, L->H)) ||
              (L->steps_snap && !border_ok(L->steps_snap, L->W, L->H))))
        RS_TRAP();  /* a pack whose step masks leave the layout */
    pc->snap_boot = (uint8_t)(L && L->steps_boot && L->steps_snap
                              && memcmp(L->steps_boot, L->steps_snap, (size_t)L->T) == 0);
}

int rs_snap_is_boot(const World *w) { return w->pc.snap_boot; }

/* ---- the snapshot reach (pack open) ---------------------------------------------------------------------------------- */
/* Close `set` (column bitsets) under the steps of M forward (the tiles the set reaches) or backward (the tiles reaching
   it): a tile (x, z) in column x + 1 with step k reaches its neighbour in that direction. Each round expands the last
   round's new tiles, whose columns lie in [lo, hi]. */
static void reach_close(const uint64_t M[RS_MAX_W + 2][8], int32_t W, int backward, uint64_t set[RS_MAX_W + 2]) {
    uint64_t front[RS_MAX_W + 2] = {0}, next[RS_MAX_W + 2];
    int32_t lo = W + 1, hi = 0;
    for (int32_t k = 1; k <= W; k++)
        if ((front[k] = set[k])) lo = imin(lo, k), hi = imax(hi, k);
    while (lo <= hi) {
        int32_t a = imax(lo - 1, 1), b = imin(hi + 1, W);
        for (int32_t k = a - 1; k <= b + 1; k++) next[k] = 0;
        if (backward) {   /* the tiles of columns a..b with a step into the front */
            for (int32_t k = a; k <= b; k++) {
                const uint64_t *m = M[k];
                uint64_t w = front[k - 1], f = front[k], e = front[k + 1];
                next[k] = (m[0] & w) | (m[1] & e) | (m[2] & (f << 1)) | (m[3] & (f >> 1)) | (m[4] & (w << 1))
                          | (m[5] & (e << 1)) | (m[6] & (w >> 1)) | (m[7] & (e >> 1));
            }
        } else {
            for (int32_t k = lo; k <= hi; k++) {
                const uint64_t *m = M[k];
                uint64_t f = front[k];
                next[k - 1] |= (f & m[0]) | ((f & m[4]) >> 1) | ((f & m[6]) << 1);  /* W, SW, NW */
                next[k + 1] |= (f & m[1]) | ((f & m[5]) >> 1) | ((f & m[7]) << 1);  /* E, SE, NE */
                next[k] |= ((f & m[2]) >> 1) | ((f & m[3]) << 1);                   /* S, N */
            }
        }
        for (int32_t k = lo; k <= hi; k++) front[k] = 0;
        int32_t nlo = W + 1, nhi = 0;
        for (int32_t k = a; k <= b; k++) {
            front[k] = next[k] & ~set[k];
            if (!front[k]) continue;
            set[k] |= front[k];
            nlo = imin(nlo, k), nhi = imax(nhi, k);
        }
        lo = nlo, hi = nhi;
    }
}

void rs_snap_reach(RsLayout *L) {
    uint64_t M[RS_MAX_W + 2][8], back[RS_MAX_W + 2] = {0};
    L->has_reach = 0;
    memset(L->snap_reach, 0, sizeof L->snap_reach);
    memset(L->snap_core, 0, sizeof L->snap_core);
    int32_t r = L->n_start > 0 ? L->start_tile[0] : L->n_src > 0 ? L->src_tile[0] : -1;
    if (r < 0 || r >= L->T || !L->steps_snap || !border_ok(L->steps_snap, L->W, L->H)) return;
    column_masks(L->steps_snap, L->W, L->H, M);
    L->snap_reach[r / L->H + 1] = back[r / L->H + 1] = 1ull << (r % L->H);
    reach_close((const uint64_t (*)[8])M, L->W, 0, L->snap_reach);
    reach_close((const uint64_t (*)[8])M, L->W, 1, back);
    for (int32_t k = 0; k < RS_MAX_W + 2; k++) L->snap_core[k] = L->snap_reach[k] & back[k];
    L->has_reach = 1;
}
