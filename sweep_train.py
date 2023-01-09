import argparse

import wandb
import os
import json
import pandas as pd
from train import train, ENCODING_MAP, MODEL_MAP
from evaluate import evaluate

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--out", type=str, required=True)
arg_parser.add_argument("--config", type=str, required=True)
arg_parser.add_argument("--evaluation", type=str, required=True)
arg_parser.add_argument("--choco", type=str, required=True)

experiments = []
BASE_EXPERIMENTS = None

def run_training(config=None):
    with wandb.init(config=config):
        config = wandb.config
        config["out"] = os.path.join(BASE_EXPERIMENTS, f"experiment_{len(experiments)}/")
        experiments.append(dict(config).values())
        
        train(choco=config["choco"],
              encoding=config["encoding"],
              model=config["model"],
              out=config["out"],
              batch_size=config["batch_size"],
              context=config["context"],
              negative_sampling_k=config["negative_sampling_k"],
              embedding_dim=config["embedding_dim"],
              embedding_aggr=config["embedding_aggr"],
              max_epochs=config["max_epochs"])
        metrics = evaluate(encoding=config["encoding"],
                           model=config["model"],
                           path=os.path.join(config["out"], "model.ckpt"),
                           config=config["evaluation"])
        wandb.log({"odd_one_out_acc": metrics["odd_one_out"][0]})


if __name__ == "__main__":
    args = arg_parser.parse_args()
    BASE_EXPERIMENTS = args.out

    with open(args.config, "r") as f:
        config = json.load(f)
    config["parameters"]["choco"]["value"] = args.choco
    config["parameters"]["config"]["value"] = args.config
    config["parameters"]["evaluation"]["value"] = args.evaluation

    sweep_id = wandb.sweep(config, project="test")
    wandb.agent(sweep_id, function=run_training)
    
    df = pd.DataFrame(experiments)
    df.to_csv(os.path.join(BASE_EXPERIMENTS, "info.csv"))
    


