# FoldCV-Query 0.2.3 independent reproduction

This directory is self-contained: move or extract it to a new location. It
does not import the original research project, saved estimator pickles, or
absolute project paths. A Python environment, C++17 compiler and Rscript
are external runtime requirements. No GPU is needed.

## Build and reproduce

Use Python 3.12, R 4.5 or later and g++ with C++17 support. From this directory:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe build.py --cxx g++
.venv/Scripts/python.exe run.py --rscript Rscript --out results/endpoints
.venv/Scripts/python.exe fetch_application_data.py
.venv/Scripts/python.exe run_application.py --out results/application
```

On Linux replace `.venv/Scripts/python.exe` with `.venv/bin/python`.
Pass full executable paths for `--cxx` and `--rscript` when not on PATH.
Output directories must be new: the scripts preserve previous runs rather
than replacing them. Compiler flags disable fast-math and floating contraction.
The package library and both comparators are built locally. Only NumPy and
ipdb are required for the endpoint runner; scikit-learn is also required for
the application. R uses its standard packages and the supplied author sources.

`run.py` executes all six saved-state cases at Q=32 and 512, with three timing
repetitions, and all six changing-factor extensions at Q=512. It records raw
timings, output arrays, infinity comparisons and finite differences. The
common-weight reference executes unmodified cfcausal prediction source;
the changing-factor reference separately materializes all weighted atoms.
The `1e-10` absolute comparison tolerance allows R CSV serialization rounding.
Endpoint timing excludes model fitting/prediction, IO and runtime startup.
This runner reproduces the query study, not original model training or the
entire cfcausal causal-inference pipeline.

`run_application.py` executes all nine independently fitted 100-round Housing
jobs, rotating flat-batch, scalar-value and batch-value order over three seeds.
It includes model fitting, density estimation, index preparation, prediction,
intervals and selection; logging and final serialization are outside timing.
Every round's bounds and selected observation IDs are retained and compared.
For one complete trajectory, useful when deploying the supplied software:

```powershell
# The parent output directory must already exist for a single child job.
mkdir results/one_trajectory
.venv/Scripts/python.exe run_application.py --child --seed 20261011 --method value_batch --out results/one_trajectory
```

The existing relocation execution ran all twelve saved-state cases and one
full application trajectory. Historical results accompany the release.

## Inputs and semantics

`cases/*.npz` store residuals, fold IDs, calibration masses, fold predictions,
test masses, changing fold factors and original query IDs. They reuse the six
frozen model/query states from SGEMM, California Housing and Bike Sharing.
The complete application downloads the public California Housing data and refits
Ridge and calibrated density models each round, with no candidate labels used
to prepare intervals. Data references: UCI SGEMM DOI
10.24432/C5MK70; California Housing, Pace and Barry (1997), DOI
10.1016/S0167-7152(96)00140-X; UCI Bike Sharing DOI 10.24432/C5W894.

The existing-software study uses cfcausal's strict CDF cutoff with its epsilon
and infinite-interval rule. Its public index adapter uses scalar upper-tail
calls on signed centers. This is distinct from the fixed-alpha JAWS-X batch
API used in the application. The package exposes both explicit JAWS-X tail
policies; it does not expose a cfcausal-tail batch API.

## Sources and distribution

The unmodified sources under `upstream/` retain their original MIT licenses:

- FavMac / QuantileTree, Zhen Lin, commit
  `0130e63b86ac0092b4359bb9993113b4956c87ea`,
  https://github.com/zlin7/FavMac ; Lin et al., ICML 2023,
  https://proceedings.mlr.press/v202/lin23j.html .
- cfcausal version 0.2.1, commit
  `e4ca6706599873505e1cce2d8cf0078ceff5d2e6`,
  https://github.com/lihualei71/cfcausal . Original prediction and helpers are
  loaded unchanged into a supplied saved-state object.

Shifted-fold tree/prefix adapters and native flat comparators are our experiment
code. The tree retains its insert/delete capabilities; the package index is
immutable. Weighted CV+ statistical construction is credited to JAWS-X and
related prior work, rather than presented as a new quantile algorithm.

Original experiment code: GPL-2.0-only. Documentation and original
measurement records: CC BY 4.0. Bundled author code retains MIT terms.
See the release-level license scope and third-party notices.
