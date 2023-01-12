from typing import Tuple, Dict

from collections import defaultdict

from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np

from mir_eval.util import boundaries_to_intervals
from mir_eval.segment import pairwise, nce

class LSTMBaselineModel(pl.LightningModule):
  def __init__(self, 
               embedding_dim: int = 10,
               hidden_size: int = 100, 
               dropout: float = 0.0,
               num_layers: int = 1, 
               num_labels: int = 11):
    """
    Model for the functional segmentation of a music piece.

    Args:
        embedding_dim (int, optional): Dimension of the chord embeddings. Defaults to 10.
        hidden_size (int, optional): LSTM hidden size. Defaults to 100.
        dropout (float, optional): LSTM dropout. Defaults to 0.0.
        num_layers (int, optional): Number of stacked layers in the LSTM. Defaults to 1.
        num_labels (int, optional): Number of sections to be predicted. Defaults to 11.
    """
    super().__init__()
    self.save_hyperparameters()
    self.lstm = nn.LSTM(embedding_dim,
                        hidden_size,
                        dropout=dropout,
                        num_layers=num_layers,
                        batch_first=True)
    self.classification = nn.Linear(hidden_size, num_labels)
    self.softmax = nn.Softmax(dim=2)
      
  def _predict(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor]) -> Tuple[torch.tensor, torch.tensor]:
    """
    Perform the prediction step in the specified batch. When computing the loss
    masked elements are ignored.

    Args:
        batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): 
          Input batch in the form (data, labels, padding mask)

    Returns:
        Tuple[torch.tensor, torch.tensor]: The prediction and the loss item
    """
    x, y, mask = batch
    x, _ = self.lstm(x)
    x = self.classification(x)
    x = self.softmax(x)
            
    loss = nn.functional.binary_cross_entropy(x[mask != 0].float(), y[mask != 0].float())
    return x, loss

  def _test(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor]) -> Tuple[torch.tensor, Dict[str, float]]:
    """
    Perform the prediction step in the specified batch. When computing the loss
    masked elements are ignored.

    Args:
        batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): 
          Input batch in the form (data, labels, padding mask)

    Returns:
        Tuple[torch.tensor, Dict[str, float]]: The loss item and the dictionary of metrics
    """
    metrics = defaultdict(list)
    mask = batch[-1]
    y = batch[-2]
    
    with torch.no_grad():
        pred, loss = self._predict(batch)
        
        for pi, yi, mi in zip(pred, y, mask):               
            pi = pi[mi != 0].argmax(axis=-1).cpu().numpy()
            _, pi = np.unique(pi, return_inverse=True)
        
            yi = yi[mi != 0].argmax(axis=-1).cpu().numpy()
            _, yi = np.unique(yi, return_inverse=True)
        
            intervals = boundaries_to_intervals(np.arange(len(yi) + 1))
            precision, recall, f1 = pairwise(intervals, yi, intervals, pi)
            metrics["p_precision"].append(precision)
            metrics["p_recall"].append(recall)
            metrics["p_f1"].append(f1)
            over, under, under_over_f1 = nce(intervals, yi, intervals, pi)
            metrics["under"] = under
            metrics["over"] = over
            metrics["under_over_f1"] = under_over_f1
    
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    return loss, metrics

  def training_step(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor], batch_idx: int) -> torch.tensor:
    """
    Perform a training step.

    Args:
        batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): Input batch composed of (data, label, padding mask).
        batch_idx (int): Batch index.

    Returns:
        torch.tensor: The torch item loss.
    """
    _, loss = self._predict(batch)
    self.log("train_loss", loss)
    return loss
  
  def validation_step(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor], batch_idx: int) -> torch.tensor:
    """
    Perform a validation step.

    Args:
        batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): Input batch composed of (data, label, padding mask).
        batch_idx (int): Batch index.

    Returns:
        torch.tensor: The torch item loss.
    """
    loss, metrics = self._test(batch)
    self.log("val_loss", loss)
    for k, m in metrics.items(): self.log(f"val_{k}", m)
    return loss
  
  def test_step(self, batch: Tuple[torch.tensor, torch.tensor, torch.tensor], batch_idx: int) -> torch.tensor:
    """
    Perform a test step.

    Args:
        batch (Tuple[torch.tensor, torch.tensor, torch.tensor]): Input batch composed of (data, label, padding mask).
        batch_idx (int): Batch index.

    Returns:
        torch.tensor: The torch item loss.
    """
    loss, metrics = self._test(batch)        
    self.log("test_loss", loss)
    for k, m in metrics.items(): self.log(f"test_{k}", m)
    return loss

  def configure_optimizers(self) -> Dict:
    """
    Configure the optimizers such that after 5 epochs without improvement the 
    learning rate is decreased.

    Returns:
        Dict: Optimizer configuration
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "train_loss"
        }
    }