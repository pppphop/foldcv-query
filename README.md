# FoldCV-Query

Python/C++ software for repeated weighted CV+ interval queries. Version 0.2.3
reuses sorted fold residuals and cumulative masses with query-dependent
centers, fold factors and test masses. Value bisection is the default;
rank search is also available. Scalar and batch calls preserve explicit
discrete-tail policies.

## Install from source

Requires Python 3.10+, NumPy and a C++17 g++ compiler. From software/:

```
python -m pip install numpy setuptools
python build_native.py
python -m pip install --no-build-isolation .
```

Build on the target platform; precompiled native libraries are not distributed.
See software/README.md for the API and its numerical requirements.

## Reproduce

- reproduction/standalone: six original-software states at Q=32/512,
  six changing-factor cases, and the complete three-seed acquisition study.
  The original cfcausal R code and FavMac QuantileTree are included with MIT
  notices. Download Housing using fetch_application_data.py before running
  the acquisition study. Python, g++, and Rscript are external requirements.
- reproduction/query_diagnostics: the 43-state size, fold, duplicate-row and
  numerical-range study; saved states, iteration/fallback traces and timings.
  Its diagnostic build instructions describe the measured Windows setup.
- The measured-results release asset contains original timings, endpoints
  and selected IDs, including the nine complete application trajectories.

No GPU is required. Run outputs are written to new directories.

## Release and citation

Versioned source and accompanying measured results:
https://github.com/pppphop/foldcv-query/releases/tag/v0.2.3

Zhang, H. (2026). FoldCV-Query (Version 0.2.3) [Software and reproduction materials].
Use the versioned release; CITATION.cff contains machine-readable metadata.

## Licenses and attribution

Original code: GPL-2.0-only. Original documentation and measurements: CC BY 4.0.
Bundled third-party code retains its original MIT license. Raw California
Housing data are downloaded separately. See LICENSE_SCOPE.md,
THIRD_PARTY_NOTICES.md and LICENSES/. These grants apply only to the indicated
materials and do not relicense third-party datasets or implementations.

JAWS-X supplies the statistical construction and source tail conventions.
QOOB, sorted-column selection and weighted-mixture CDF inversion provide
prior algorithmic building blocks. This release provides reusable interfaces
and computational evaluation; the accompanying manuscript is prepared
separately for journal review.
