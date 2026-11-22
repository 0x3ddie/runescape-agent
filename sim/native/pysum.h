/* sim/native/pysum.h: the oracle's sum() of floats, as CPython 3.12 and later compute it (Python/bltinmodule.c
 * builtin_sum_impl), for the C twin's sums that mirror a sum() in the oracle (task.c free_settle and free_report: the
 * fractional level gains of free-play-v1 over S, sim/task.py FreePlayEpisode._account and report).
 *
 * sum(terms) starts from the int 0. Its first term, a float, leaves the int branch through int + float, 0.0 + x[0];
 * every later float term goes through Neumaier's compensation (the improved Kahan-Babuska sum); at the end the
 * compensation is added only when it is non-zero and finite. Every term of those sums is a float (a difference of two
 * fractional levels, sim/audit.py fractional_level), so sum()'s int branch adds no term here. A Python whose sum() adds
 * floats plainly (before 3.12) gives other bits: sim/native.py check_interpreter refuses it.
 *
 * Only additions and subtractions of doubles: no product that -ffp-contract=off could have fused, no reassociation
 * without -ffast-math (FLAGS: -fno-fast-math), and every double evaluated as a double (FLT_EVAL_METHOD 0: x86-64 with
 * SSE2, arm64), checked here so that a host that evaluates wider fails to build instead of summing otherwise. */
#ifndef RS_PYSUM_H
#define RS_PYSUM_H

#include <float.h>
#include <math.h>

#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "rs_py_sum needs every double evaluated as a double (FLT_EVAL_METHOD 0) to equal CPython's sum()"
#endif

static inline double rs_py_sum(const double *x, int n) {
    if (n <= 0) return 0.0;  /* sum([]): the int 0 */
    double f = 0.0 + x[0], c = 0.0;
    for (int i = 1; i < n; i++) {
        double t = f + x[i];
        if (fabs(f) >= fabs(x[i]))
            c += (f - t) + x[i];
        else
            c += (x[i] - t) + f;
        f = t;
    }
    if (c != 0.0 && isfinite(c)) f += c;
    return f;
}

#endif
