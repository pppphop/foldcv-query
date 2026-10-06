# v14 saved-state diagnostic supplement

GPL-2.0-only reproduction code with CC BY 4.0 original records.
This supplements the full application and original-author-software reproduction.

The folder contains 43 numeric states, per-query counts/reasons, crossing
row-ID groups, uninstrumented raw timings, the original 0.2.3 source, and
counter-only trace-source generation. No old project imports, estimator
pickles, downloaded dataset or GPU is needed for saved-state replay.
Windows x64, Python 3.12, NumPy 2.5.3 and scikit-learn 1.9.1; tested compiler
GCC 11.1.0. Compiler flags here are the project's Windows flags.

```
python -m pip install -r requirements.txt
python build_trace.py --cxx C:/path/to/g++.exe
python replay_saved.py --out reproduction_v1
```

Use --case synthetic_wide_ladder to replay that one specified case. Every
output path must be new. replay_saved recomputes all endpoints and indexed
attempts; it compares endpoints, iteration counts and reason labels with
the frozen recorded values. It does not regenerate calibration draws or
refit models, and provides no new application performance result.

synthetic_timings.json contains three rotated repetitions of the unmodified
public batch interfaces. Those measurements never use the trace library.
Reported iteration counts apply to one indexed attempt. Public batch
fallback dispatch repeats that attempt before the scalar flat-policy path.
Both indexed attempts and materialization are included in public timing.
