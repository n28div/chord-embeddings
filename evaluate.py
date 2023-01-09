import argparse

import torch
import numpy as np
import pytorch_lightning as pl
import json

from choco.corpus import ChoCoHarteAnnotationsCorpus
from pitchclass2vec.data import ChocoDataModule
import pitchclass2vec.encoding as encoding
import pitchclass2vec.model as model
from pitchclass2vec.pitchclass2vec import Pitchclass2VecModel
from gensim_evaluations.methods import odd_one_out


ENCODING_MAP = {
    "root-interval": encoding.RootIntervalDataset,
    "all-interval": encoding.AllIntervalDataset,
    "chord2vec": encoding.Chord2vecDataset
}

MODEL_MAP = {
    "word2vec": model.Word2vecModel,
    "fasttext": model.FasttextModel
}

def evaluate(encoding: str, model: str, path: str, config: str):
    model = Pitchclass2VecModel(ENCODING_MAP[encoding], 
                                MODEL_MAP[model],
                                path)

    with open(config) as f:
        config = json.load(f)

    metrics = {}
    metrics["odd_one_out"] = odd_one_out(
        { k: v for k, v in config.items() if k != "vocab" },
        model, 
        vocab=config["vocab"],
        k_in=4
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pitchclass2vec embedding.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--encoding", 
                        choices=list(ENCODING_MAP.keys()), 
                        required=True, 
                        default="root-interval")
    parser.add_argument("--model", 
                        choices=list(MODEL_MAP.keys()), 
                        required=True, 
                        default="fasttext")
    
    args = parser.parse_args()
    
    evaluation = evaluate(args.encoding, args.model, args.path, args.config)
    for metric, metric_eval in evaluation.items():
        print(f"{metric}:")
        accuracy, accuracy_per_cat, _, _, _ = metric_eval
        print(f"Accuracy: {accuracy}")
        for cat, acc in accuracy_per_cat.items():
            print(f"\ton {cat}: {acc}")