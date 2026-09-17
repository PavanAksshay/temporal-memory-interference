"""Random seed and determinism utilities."""
import random
import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Set global seeds and return an isolated NumPy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
