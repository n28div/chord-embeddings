from typing import List, Tuple
from itertools import chain, combinations

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pitchclass2vec.encoding.rootinterval import RootIntervalDataset
from pitchclass2vec.harte import chord_to_pitchclass, pitchclass_to_onehot

class AllIntervalDataset(RootIntervalDataset):
  @staticmethod
  def encode_chord(chord: str) -> List[int]:
    """
    Encode a chord as the one-hot encoding of intervals between
    all the notes in the chord.
    Empty chords are represented as all 0s.

    Args:
        chord (str): 

    Returns:
        List[np.array]: List of encoded components of the chord.
    """
    try:
      pc = chord_to_pitchclass(chord)
    except:
      pc = chord_to_pitchclass("N")

    if len(pc) != 0:
      intervals = list(combinations(pc, 2))
      onehot = list(map(pitchclass_to_onehot, intervals))
    else:
      onehot = [pitchclass_to_onehot(pc)]

    encoding = list(map(lambda x: x.dot(2**np.arange(12)[::-1]).astype(int), onehot))
    return encoding
