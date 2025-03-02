import random
import numpy as np
import torch
from contextlib import contextmanager


@contextmanager
def temp_rng(new_seed=None):
	"""
    https://github.com/fpgaminer/bigasp-training/blob/main/utils.py#L73
	Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.
	If new_seed is not None, the RNG state is set to this value before the context is entered.
	"""

	# Save RNG state
	old_torch_rng_state = torch.get_rng_state()
	old_torch_cuda_rng_state = torch.cuda.get_rng_state()
	old_numpy_rng_state = np.random.get_state()
	old_python_rng_state = random.getstate()

	# Set new seed
	if new_seed is not None:
		torch.manual_seed(new_seed)
		torch.cuda.manual_seed(new_seed)
		np.random.seed(new_seed)
		random.seed(new_seed)

	yield

	# Restore RNG state
	torch.set_rng_state(old_torch_rng_state)
	torch.cuda.set_rng_state(old_torch_cuda_rng_state)
	np.random.set_state(old_numpy_rng_state)
	random.setstate(old_python_rng_state)
