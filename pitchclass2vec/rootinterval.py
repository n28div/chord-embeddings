from typing import List, Tuple, Dict
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from pitchclass2vec.data import ChocoChordDataset
from pitchclass2vec.harte import chord_to_pitchclass, pitchclass_to_onehot

class RootIntervalModel(pl.LightningModule):
  def __init__(self, embedding_dim: int = 10, aggr: str = "sum"):
    """
      Args:
        embedding_dim (int, optional): Embedding dimension. Defaults to 10.
        aggr (str, optional): Chord elements aggregation dimension. Defaults to "sum".
    """
    super().__init__()
    self.save_hyperparameters()
    self.embedding_dim = embedding_dim
    
    # vocabulary size is 2^12 (twelve notes, either in the chord or not)
    # the first element (corresponding to empty chord) can be interpreted
    # as silence
    self.full_vocab_size = 2**12
    self.embedding = nn.EmbeddingBag(self.full_vocab_size, self.embedding_dim,
                                     mode=aggr,
                                     padding_idx=0)

  def _predict(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Perform a prediction step by encoding both the source and the target
    using the embedding method and compute the dot product between the
    vectors.

    Args:
        source (torch.Tensor): Source chords
        target (torch.Tensor): Target chords

    Returns:
        torch.Tensor: Tensor of similarities between chord computes as dot products
    """
    source = self.embedding(source)
    target = self.embedding(target)
    y = torch.einsum("ij,ik->i", source, target)
    return y

  def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.nn.BCEWithLogitsLoss:
    """
    Perform a training step on the provided batch.

    Args:
        batch (torch.Tensor): The provided batch.
        batch_idx (int): The index of the provided batch.

    Returns:
        torch.nn.BCEWithLogitsLoss: Loss on the provided batch.
    """
    source, target, y = batch
    pred = self._predict(source, target)
    loss = nn.functional.binary_cross_entropy_with_logits(pred, y.float())
    self.log("train/loss", loss)
    return loss

  def configure_optimizers(self) -> Tuple[List[torch.optim.Adam], Dict]:
    """
    Lightning optimizers configuration. Uses Adam with 0.1 initial
    learning rate and a scheduler that reduces the learning rate
    after 10 optimizations without any improvement in the training
    loss.

    Returns:
        Tuple[List[torch.optim.Adam], Dict]: The optimizer and the relative scheduler.
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
    scheduler = {
      "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10),
      "monitor": "train/loss",
      "interval": "step" }
    return [optimizer], scheduler

  def __getitem__(self, chord: str) -> np.array:
    encoded = RootIntervalDataset.encode_chord(chord)
    with torch.no_grad():
      embedded = self.embedding(torch.tensor(encoded).unsqueeze(0)).squeeze(0).numpy()

    return embedded

class RootIntervalDataset(ChocoChordDataset):
  @staticmethod
  def encode_chord(chord: str) -> List[int]:
    """
    Encode a chord as the one-hot encoding of intervals between
    the root note and all the other notes in the chord.
    Empty chords are represented as all 0s.

    Args:
        chord (str): 

    Returns:
        List[np.array]: List of encoded components of the chord.
    """
    pc = chord_to_pitchclass(chord)

    if len(pc) != 0:
      intervals = [(pc[0], elem) for elem in pc]
      onehot = list(map(pitchclass_to_onehot, intervals))
    else:
      onehot = [pitchclass_to_onehot(pc)]

    encoding = list(map(lambda x: x.dot(2**np.arange(12)[::-1]).astype(int), onehot))
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
    
    source = torch.nn.utils.rnn.pad_sequence(map(torch.tensor, chain(*source)), 
                                             batch_first=True, 
                                             padding_value=0).int()
    
    target = torch.nn.utils.rnn.pad_sequence(map(torch.tensor, chain(*target)), 
                                             batch_first=True, 
                                             padding_value=0).int()
    
    y = torch.tensor(list(chain(*y)))

    return source, target, y
