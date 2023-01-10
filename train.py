import argparse
import os
import json

import torch
import numpy as np
import pytorch_lightning as pl
import wandb

from choco.corpus import ChoCoHarteAnnotationsCorpus
from pitchclass2vec.data import ChocoDataModule
import pitchclass2vec.encoding as encoding
import pitchclass2vec.model as model


ENCODING_MAP = {
    "root-interval": encoding.RootIntervalDataset,
    "all-interval": encoding.AllIntervalDataset,
    "chord2vec": encoding.Chord2vecDataset
}

MODEL_MAP = {
    "word2vec": model.Word2vecModel,
    "fasttext": model.FasttextModel
}


def train(choco: str = "", encoding: str = "", model: str = "", out: str = "", seed: int = 42, batch_size: int = 32, context: int = 5,
          negative_sampling_k: int = 20, embedding_dim: int = 10, embedding_aggr: str = "sum",
          early_stop_patience: int = -1, disable_wandb: bool = False, max_epochs: int = 10):
    pl.seed_everything(seed=seed, workers=True)
    
    # save training metadata
    if not os.path.exists(out):
        os.makedirs(out)

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({
            "encoding": encoding,
            "model": model,
            "seed": seed,
            "batch_size": batch_size,
            "context": context,
            "negative_sampling_k": negative_sampling_k,
            "embedding_dim": embedding_dim,
            "embedding_aggr": embedding_aggr,
            "early_stop_patience": early_stop_patience,
            "max_epochs": max_epochs
        }, f)

    dataset_cls = ENCODING_MAP[encoding]
    model_cls = MODEL_MAP[model]

    data = ChocoDataModule(
        choco,
        dataset_cls,
        batch_size=batch_size,
        context_size=context,
        negative_sampling_k=negative_sampling_k)

    model = model_cls(embedding_dim=embedding_dim,
                      aggr=embedding_aggr)

    callbacks = [
        pl.callbacks.ModelCheckpoint(save_top_k=1,
                                     monitor="train/loss",
                                     mode="min",
                                     dirpath=out,
                                     filename="model",
                                     every_n_epochs=1)
    ]

    if early_stop_patience != -1:
        callbacks.append(pl.callbacks.EarlyStopping(
            monitor="train/loss",
            min_delta=0.00,
            patience=early_stop_patience))


    if not disable_wandb:
        logger = pl.loggers.WandbLogger(project="pitchclass2vec",
                                        group=f"{encoding}_{model}",
                                        #log_model=True,
                                        tags=[
                                            f"{embedding_dim} embedding",
                                            f"{embedding_aggr} aggregation",
                                            f"{context} context",
                                            f"{negative_sampling_k} negative sampling k"
                                        ])
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
    else:
        logger = None

    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator="auto",
                         logger=None if disable_wandb else logger,
                         devices=1,
                         callbacks=callbacks)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pitchclass2vec embedding.")
    parser.add_argument("--choco", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--encoding", 
                        choices=list(ENCODING_MAP.keys()), 
                        required=True, 
                        default="root-interval")
    parser.add_argument("--model", 
                        choices=list(MODEL_MAP.keys()), 
                        required=True, 
                        default="fasttext")
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--max-epochs", type=int, required=False, default=5)
    parser.add_argument("--batch-size", type=int, required=False, default=512)
    parser.add_argument("--embedding-dim", type=int, required=False, default=10)
    parser.add_argument("--embedding-aggr",
                        choices=["sum", "mean"], required=False, default="sum")
    parser.add_argument("--context", type=int, required=False, default=5)
    parser.add_argument("--negative-sampling-k", type=int,
                        required=False, default=20)
    parser.add_argument("--early-stop-patience", type=int,
                        required=False, default=2)
    parser.add_argument("--disable-wandb", action='store_const',
                        const=True, default=False)

    args = parser.parse_args()
    train(args.choco, args.encoding, args.model, args.out, seed=args.seed, 
          batch_size=args.batch_size, context=args.context, 
          negative_sampling_k=args.negative_sampling_k, embedding_dim=args.embedding_dim, 
          embedding_aggr=args.embedding_aggr, early_stop_patience=args.early_stop_patience, 
          disable_wandb=args.disable_wandb, max_epochs=args.max_epochs)
