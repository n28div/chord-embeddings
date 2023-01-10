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
    "timed-root-interval": encoding.TimedRootIntervalDataset,
    "all-interval": encoding.AllIntervalDataset,
    "chord2vec": encoding.Chord2vecDataset,
}

MODEL_MAP = {
    "word2vec": model.Word2vecModel,
    "fasttext": model.FasttextModel,
    "scaled-loss-fasttext": model.ScaledLossFasttextModel,
    "emb-weighted-fasttext": model.EmbeddingWeightedFasttextModel,
    "rnn-weighted-fasttext": model.RNNWeightedFasttextModel,
}


def train(choco: str = "", encoding: str = "", model: str = "", out: str = "", **kwargs):
    pl.seed_everything(seed=args.get("seed", 42), workers=True)
    
    # save training metadata
    if not os.path.exists(out):
        os.makedirs(out)

    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump({
            "encoding": encoding,
            "model": model,
            **kwargs
        }, f)

    dataset_cls = ENCODING_MAP[encoding]
    model_cls = MODEL_MAP[model]

    data = ChocoDataModule(
        choco,
        dataset_cls,
        batch_size=kwargs.get(batch_size, 1024),
        context_size=kwargs.get(context, 5),
        negative_sampling_k=kwargs.get(negative_sampling_k, 20))

    model = model_cls(**kwargs)

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


    if not args.get("disable_wandb", False):
        logger = pl.loggers.WandbLogger(project="pitchclass2vec",
                                        group=f"{encoding}_{model}")
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
    else:
        logger = None

    trainer = pl.Trainer(max_epochs=args.get("max_epochs", 5),
                         accelerator="auto",
                         logger=None if args.get("disable_wandb", False) else logger,
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
                        required=False, default=-1)
    parser.add_argument("--disable-wandb", action='store_const',
                        const=True, default=False)

    args = parser.parse_args()
    train(args.choco, args.encoding, args.model, args.out, seed=args.seed, 
          batch_size=args.batch_size, context=args.context, 
          negative_sampling_k=args.negative_sampling_k, embedding_dim=args.embedding_dim, 
          embedding_aggr=args.embedding_aggr, early_stop_patience=args.early_stop_patience, 
          disable_wandb=args.disable_wandb, max_epochs=args.max_epochs)
