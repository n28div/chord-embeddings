from typing import List
import numpy as np
from harte.harte import Harte

def pitchclass_to_onehot(pitchclass: List[int]) -> np.array:
  """
  Convert a pitchclass representation to a one-hot encoding
  of a pitchclass representation: a 12-dimensional vector
  in which each dimension represents the respective pitchclass.

  Args:
      pitchclass (List[int]): Pitchclass representation

  Returns:
      np.array: One hot encoding.
  """
  return np.eye(12)[pitchclass, :].sum(axis=0).clip(0, 1)