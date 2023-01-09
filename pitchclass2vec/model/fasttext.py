from typing import Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.utils.class_weight import compute_sample_weight

from pitchclass2vec.model.base import BaseModel

class FasttextModel(BaseModel):
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
    weight = compute_sample_weight("balanced", y)
    loss = nn.functional.binary_cross_entropy_with_logits(pred, y.float(), torch.tensor(weight).to(pred.device))
    self.log("train/loss", loss)
    return loss
