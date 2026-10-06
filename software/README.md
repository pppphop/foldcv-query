# foldcv-query 0.2.3

Python/C++ indexed weighted CV+ queries with two searches over shared fold
residual ordering and mass prefixes. Value bisection is the default;
rank search remains available through `backend="rank"`.

## Build and install

With C++17 g++ and NumPy available:

```powershell
python build_native.py
python -m pip wheel --no-deps --no-build-isolation . -w dist
python -m pip install --no-deps path/to/platform_wheel.whl
```

Build on the target platform. The build disables fast-math and floating
contraction; its ctypes library is independent of the Python C ABI.
NumPy >=2.4 x86 wheels require x86-v2; older virtual CPUs can use 2.3.5.

## Select a backend

```python
import numpy as np
from foldcv_query import Backend, FoldIndex, Policy

residual = [.2, .5, .8]
fold = np.array([0, 1, 0])
base_mass = [1., 2., 3.]
default_index = FoldIndex(residual, fold, base_mass, 2)
rank_index = FoldIndex(residual, fold, base_mass, 2, backend=Backend.RANK)
lower, upper = default_index.interval(
    [0., 1.], [1., 1.], 1., policy=Policy.CALIBRATE_GE)
print(default_index.backend, lower.value, upper.value)
```

## Query a batch

```python
result = default_index.interval_batch(
    centers=np.array([[0., 1.], [.2, 1.1]]),
    a=np.ones((2, 2)), t=np.ones(2), alpha=.1,
    policy=Policy.CALIBRATE_GE)
print(result.bounds)                  # immutable (Q, 2): lower, upper
print(result.used_fallback)           # immutable (Q, 2) Boolean flags
print(result.secondary_fallback)      # immutable (Q, 2) Boolean flags
```

`interval_batch` validates and copies the Q-by-K centers/factors and Q test
masses once, then computes both tails through one native call. It supports
both backends and both explicit policies. Ambiguous endpoints retain the
scalar policy's normalization and tie behavior. `interval` remains available.
`UniformFinitePoolFCS.predict_indices` uses this batch path automatically.

Owned input copies use `16*Q*K + 8*Q` bytes; returned numeric bounds use
`16*Q` bytes, with two `2*Q`-byte Boolean flag arrays. Native status and
validation temporaries add transient allocations. This is numeric-array
accounting, not measured process peak memory. The calibration index is
unchanged. Chunk a very large query pool into separate batches to bound
query-buffer memory; batching does not change the quantile algorithm.

Both searches use the same prefix arrays, tail rules and fallback. Value
search chooses a numerical midpoint; rank search chooses an endpoint from
the active fold midpoints. `index.backend` records the selected search.
`Endpoint.path` remains `index` or `fallback`, preserving the earlier result
interface. The native `weighted_index` entrypoint retains rank semantics;
the additional `weighted_value_index` entrypoint provides value search.

## Explicit tail policies

`CALIBRATE_GE` normalizes `concatenate(a[fold]*b, [t])` with NumPy's sum.
The first cumulative mass >=q determines the crossing; the lower tail
returns the current endpoint on exact equality and its predecessor otherwise.
`UTILS_STRICT` normalizes using `np.sum(a[fold]*b)+t` and takes the first
cumulative mass >q; the lower tail returns the predecessor when available.
`scs_interval` uses unit fold factors with the utilities policy.

Reference: JAWS-X commit a5c347be096af38832c427d0bf4b702429dc92ba.
The indexed fast path requires positive b,a,t in [2^-100,2^100], binary64
round-to-nearest and gradual underflow. Close cumulative comparisons and
relevant endpoint ties use native flat selection. Both searches have an
80-iteration cap followed by the same flat fallback.

Tail queries use 0<q<.5 for sign=-1 and .5<q<1 for sign=+1. Inputs must be
finite; residuals and masses must be nonnegative. Zero masses route to flat
selection when needed. Overflow or failure to obtain a CDF crossing raises
an exception. Inputs are copied into immutable owned arrays.

## Finite-pool model interface

`UniformFinitePoolFCS.prepare(pool, train_ids, ytrain, fold, models)` prepares
K supplied fold models followed by the full model. Its optional `fit` method
constructs these models with scikit-learn. Both methods accept `backend="value"`
(default) or `backend="rank"`; `app.backend` exposes the selected backend.
Query through `predict_indices(pool_ids)` or sample with
`sample_query_indices(count, seed=...)`.

The pool is the declared uniform finite support. Training labels determine
clipped preferences; pool observations determine their normalizers. Candidate
labels are not preparation/query inputs. Models are snapshotted from trusted
caller-owned estimators; `model_snapshot_bytes` permits optional model replay.
Preparation records configuration and normalizers without generating file,
array or model fingerprints. Rebuilding is required when models, calibration
observations or preferences change.

## Measurements

The paper records controlled native comparisons separately from calls through
the public Python interface. Earlier six-setting results are preserved and
refer to version 0.2.1's rank implementation and experimental value comparator.
The 0.2.2 integration replay invokes the public default value and explicit rank
interfaces on a complete 100-round Housing acquisition task. Its results are
stored in `cvplus_public_backends_2026-10-05`, alongside its protocol.

Version 0.2.3 adds the batch interface. Its three-seed, 100-round comparison
uses a native flat batch comparator with matching query input validation;
both methods receive the whole batch in one native call. Results and the
predeclared protocol are in `cvplus_batch_interface_2026-10-05`. The portable
reproduction directory contains saved-state existing-software comparisons
and the complete application runner. Older version-specific measurements
remain separate.

## License and citation

GPL-2.0-only. See LICENSE and the release-level LICENSE_SCOPE.md and
THIRD_PARTY_NOTICES.md. Statistical constructions are credited to JAWS-X.
Release: https://github.com/pppphop/foldcv-query/releases/tag/v0.2.3
