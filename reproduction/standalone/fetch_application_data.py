"""Fetch the public application data through scikit-learn."""
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_california_housing

target = Path(__file__).resolve().parent / 'data/housing.npz'
if target.exists():
    raise FileExistsError(target)
target.parent.mkdir(parents=True, exist_ok=True)
data = fetch_california_housing()
np.savez_compressed(target, X=np.asarray(data.data, dtype=np.float64),
                    y=np.asarray(data.target, dtype=np.float64))
print('California Housing application input written:', target)
