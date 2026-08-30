"""
Random seed setting module
Ensures experiment reproducibility
"""

import random
import os
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set the random seed of all libraries

    Args:
        seed: random seed value
    """
    # Set the Python random seed
    random.seed(seed)

    # Set the NumPy random seed
    np.random.seed(seed)

    # Set the PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Make CuDNN use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Disable certain non-deterministic operations in nn.Utils
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_seed(config: dict) -> int:
    """Get the random seed from the config

    Args:
        config: configuration dict

    Returns:
        random seed value
    """
    return config.get('seed', 42)


class SeedContext:
    """Seed context manager; temporarily sets the seed and restores it on exit"""

    def __init__(self, seed: int):
        """
        Args:
            seed: the random seed to set
        """
        self.seed = seed
        self.original_state = None

    def __enter__(self):
        # Save the current random states
        self.original_state = {
            'random': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state()
        }
        if torch.cuda.is_available():
            self.original_state['cuda'] = torch.cuda.get_rng_state_all()

        # Set the new seed
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original random states
        if self.original_state:
            random.setstate(self.original_state['random'])
            np.random.set_state(self.original_state['numpy'])
            torch.set_rng_state(self.original_state['torch'])
            if torch.cuda.is_available() and 'cuda' in self.original_state:
                torch.cuda.set_rng_state_all(self.original_state['cuda'])
        return False
