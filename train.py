import argparse

import torch
import numpy as np
import pytorch_lightning as pl

from choco.corpus import ChoCoHarteAnnotationsCorpus
from pitchclass2vec.data import ChocoDataModule
from pitchclass2vec.rootinterval import RootIntervalModel, RootIntervalDataset


ENCODING_MAP = {
  "root-interval": (RootIntervalModel, RootIntervalDataset)
}

parser = argparse.ArgumentParser(description="Train pitchclass2vec embedding.")
parser.add_argument("--choco", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--encoding", choices=["root-interval",], required=True, default="root-interval")
parser.add_argument("--seed", type=int, required=False, default=42)
parser.add_argument("--max-epochs", type=int, required=False, default=5)
parser.add_argument("--batch-size", type=int, required=False, default=512)
parser.add_argument("--embedding-dim", type=int, required=False, default=10)
parser.add_argument("--embedding-aggr", choices=["sum", "mean"], required=False, default="sum")
parser.add_argument("--context", type=int, required=False, default=5)
parser.add_argument("--negative-sampling-k", type=int, required=False, default=20)
parser.add_argument("--early-stop-patience", type=int, required=False, default=2)
parser.add_argument("--wandb", type=str, required=False, default=None)

args = parser.parse_args()

if __name__ == "__main__":
  pl.seed_everything(seed=args.seed, workers=True)

  model_cls, dataset_cls = ENCODING_MAP[args.encoding]
  
  data = ChocoDataModule(
    args.choco, 
    dataset_cls, 
    batch_size=args.batch_size, 
    context_size=args.context,
    negative_sampling_k=args.negative_sampling_k)
  
  model = model_cls(embedding_dim=args.embedding_dim, aggr=args.embedding_aggr)
    
  trainer = pl.Trainer(max_epochs=args.max_epochs, 
                       accelerator="auto",
                       #logger=pl.loggers.WandbLogger(project="pitch2vec", name=args.wandb),
                       devices=1,
                       callbacks=[
                         pl.callbacks.LearningRateMonitor(logging_interval="step"),
                         pl.callbacks.EarlyStopping(monitor="train/loss", 
                                                    min_delta=0.00, 
                                                    patience=args.early_stop_patience),
                         pl.callbacks.ModelCheckpoint(save_top_k=1, 
                                                      monitor="train/loss", 
                                                      mode="min", 
                                                      dirpath=args.out, 
                                                      filename="{epoch}",
                                                      every_n_epochs=1)
                        ])
  
  trainer.fit(model, datamodule=data)