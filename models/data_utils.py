import os
from typing import Tuple

import numpy as np
from gift_eval.data import Dataset


def load_boom_dataset(name: str, term: str, to_univariate: bool = False, storage_env_var: str = "BOOM") -> Dataset:
    """Load BOOM dataset using gift_eval Dataset wrapper.

    Returns a Dataset instance with attributes: train_data, test_data, freq, prediction_length, etc.
    """
    return Dataset(name=name, term=term, to_univariate=to_univariate, storage_env_var=storage_env_var)


def prepare_series(entry: dict, max_length: int = None) -> np.ndarray:
    target = np.asarray(entry["target"], dtype=np.float32)
    if max_length is not None and len(target) > max_length:
        target = target[-max_length:]
    return target
