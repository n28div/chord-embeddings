from typing import List, Tuple
from itertools import chain
from functools import cache

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pitchclass2vec.data import ChocoChordDataset
from pitchclass2vec.encoding.utils import pitchclass_to_onehot, chord_pitchclass
from harte.harte import Harte

class Chord2vecDataset(ChocoChordDataset):
  @staticmethod
  @cache
  def encode_chord(chord: str) -> List[int]:
    """
    Encode a chord as proposed by chord2vec[1].
    Chords are encoded as a one-hot encoding of the notes they are composed of,
    i.e. their pitchclass.
    Empty chords are represented as all 0s.

    [1] Madjiheurem et al, Chord2Vec: Learning Musical Chord Embeddings
    Args:
        chord (str): 

    Returns:
        List[np.array]: List of encoded components of the chord.
    """
    pc = chord_pitchclass(chord)
    onehot = pitchclass_to_onehot(pc)
    encoding = onehot.dot(2**np.arange(12)[::-1]).astype(int)
    return encoding

  @staticmethod
  def collate_fn(sample: Tuple[List[np.array], List[np.array], np.array]) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Collate the provided samples in a single batch.

    Args:
        sample (List[List[np.array], List[np.array], np.array]): Input sample

    Returns:
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array], np.array]: Output batch.
    """
    source, target, y = zip(*sample)
    
    source = torch.tensor((list(chain(*source))))
    target = torch.tensor((list(chain(*target))))
    y = torch.tensor(list(chain(*y)))

    return source, target, y
