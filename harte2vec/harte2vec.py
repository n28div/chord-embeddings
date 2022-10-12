from typing import List, Tuple
import itertools
import numpy as np
import torch.nn as nn
import torch
from harte2vec.harte import HarteToIntervals

def onehot_intervals(intervals: Tuple[int]) -> int:
  """
  Generates a one hot encoding from the intervals composing a chord
  as an integer.

  Args:
      intervals (Tuple[int]): Intervals composing the chord in the form
      (root, target note)

  Returns:
      int: Integer representation of the one hot encoding.
  """
  if len(intervals) == 0:
    chord_encoding = np.array([0])
  else:
    intervals_encoding = np.stack([np.eye(12)[i, :].sum(axis=0).clip(0, 1) for i in intervals])
    whole_chord_encoding = intervals_encoding.sum(axis=0).clip(0, 1)
    chord_encoding = np.vstack((intervals_encoding, whole_chord_encoding))
    # convert chord encoding to indexes
    chord_encoding = chord_encoding.dot(2**np.arange(12)[::-1]).astype(int)
  return chord_encoding

class Harte2Vec(object):

  def __init__(self):
    """
    Initialize an empty model, choosing gpu when available, else using cpu.
    """
    self.device = "cpu"
    self.h2i = HarteToIntervals()
    self.vocab = None
    self.embedding = None

  @classmethod
  def from_pretrained(self, path: str):
    """
    Instantiate an harte2vec model from a pretrained model.

    Args:
        path (str): Path to the saved harte2vec model.

    Returns:
        Harte2Vec: harte2vec instance.
    """
    h2v = Harte2Vec()
    loaded = torch.load(path, map_location=h2v.device)
    h2v.vocab = loaded["vocab"]
    h2v.embedding = nn.EmbeddingBag.from_pretrained(loaded["embedding"], mode="sum")
    return h2v

  def __getitem__(self, chord: str) -> np.ndarray:
    """
    Get the embedding of the provided chord. If the chord is out
    of vocabulary then its embedding is computed using its composing
    intervals.

    Args:
        chord (str): Chord to be embedded

    Returns:
        np.ndarray: Chord embedding
    """
    with torch.no_grad():
      # item not available: compute using intervals
      intervals = self.h2i.convert(chord)
      encoded = torch.tensor(onehot_intervals(intervals)).reshape(1, -1)
      # flatten out batch dimension
      embedded = self.embedding(encoded).flatten() 
    return embedded.cpu().numpy()
