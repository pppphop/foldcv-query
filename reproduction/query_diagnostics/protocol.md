# Search and fallback diagnostics (2026-10-06)

This study addresses baseline identity, replacement-sampling ties, and query
iteration sensitivity. Existing timings, inputs and libraries remain frozen.
No production search algorithm is changed. CPU only; one numerical thread.

1. Read all stored Table 4 fallback flags and Table 7 native selector flags.
2. Generate an instrumented copy of the 0.2.3 C++ source: counters and reason
   labels only, with the same pivots, arithmetic, margins and 80-iteration cap.
3. Replay the six saved main-domain states (512 queries each). Reconstruct
   the 21 frozen Housing/Ridge scaling states using their original row IDs,
   fold assignments, default Ridge and weight construction. Record per-tail
   iterations, index fallback reasons and tie-group row IDs. Compare the
   returned endpoints with the saved endpoints, as part of this experiment.
4. As a sampling control, draw 16,000 distinct Housing rows for each of the
   same three seeds and K=2,5,10,20, retain each seed's 512 original query IDs,
   and refit the same models and weights. This changes fitted models as well
   as duplicates; attribute a fallback to duplicates only when its actual
   crossing tie group maps to repeated identical source IDs within one fold.
5. Predetermined distribution diagnostics: N=16,000, K=5, Q=128, seed 20261006;
   raw exponential, raw Pareto(shape 0.5), narrowly clustered residuals,
   and a mostly narrow distribution with a 300-point powers-of-two ladder
   from 2^1 to 2^300, distributed across folds. Residuals
   are nonnegative; use positive unit base masses, changing fold factors and
   centers, test mass 1, and the existing 0.1/0.9 tail convention. Compare with
   the materialized lexicographic CDF reference. Report all cases, including
   iteration-cap fallback. These are numerical diagnostic distributions, not
   new application datasets or evidence of coverage under distribution shift.

Save the numerical states, per-query trace arrays, aggregate JSON and a
readable report in a new run directory. Report p50/p95/max iterations, fast
and fallback counts separately, and both value and rank backend flags.
Instrumentation timings are not performance claims. No result-based tuning.

## Query-timing extension, fixed before execution

The four diagnostic distributions above are frozen in run_v1. Measure each
with the unmodified 0.2.3 public value and rank batch interfaces and the
existing validated flat batch implementation. All four cases are included;
three query repetitions, rotated method order, one numerical thread, no
fitting or preparation within the query timer. No instrumentation is used
inside these timers, and no search parameters change. Report medians and
raw repetitions as local query measurements, not application speedups or
timing confidence intervals.
