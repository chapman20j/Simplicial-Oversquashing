# classification.py
"""
Tests graph lifting and rewiring on graph classifiation benchmarks.
"""

import argparse

import torch
import yaml
from ray.tune import TuneConfig, Tuner, grid_search, with_resources

from utils.experiment import experiment
from utils.plot_results import orig_df_to_latex

if __name__ == "__main__":

    # * Load experiment configuration
    with open("experiment_config.yaml", "r") as f:
        args = yaml.safe_load(f)

    NUM_SAMPLES = 10
    MAX_CONCURRENT = 60

    # * Get arguments
    parser = argparse.ArgumentParser(description="Graph Classification")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["MUTAG"],
        help="List of datasets to use for classification",
        choices=["MUTAG", "ENZYMES", "PROTEINS", "NCI1", "IMDB-BINARY"],
    )
    parser.add_argument(
        "--lifts",
        type=str,
        nargs="+",
        default=["none"],
        help="List of graph lifts to use",
        choices=["none", "clique"],
    )
    parser.add_argument(
        "--rewiring",
        type=str,
        nargs="+",
        default=["none"],
        help="List of rewiring strategies to use",
        choices=["none", "fosr", "sdrf", "afr4"],
    )
    command_args = parser.parse_args()

    # * Set up grid search
    args["model_name"] = grid_search(
        [
            "gcn",
            "gin",
            "rgcn",
            "rgin",
            "sgc",
            "sin",
            "cin",
            "cin++",
        ]
    )
    args["dataset"] = grid_search(command_args.datasets)
    args["complex"] = grid_search(command_args.lifts)
    args["rewiring"] = grid_search(command_args.rewiring)
    args["dataset_kwargs"] = {}

    resources_per_trial = (
        {"cpu": 1, "gpu": 0.2} if torch.cuda.is_available() else {"cpu": 1, "gpu": 0}
    )
    tune_config = TuneConfig(
        num_samples=NUM_SAMPLES, max_concurrent_trials=MAX_CONCURRENT
    )
    trainable = with_resources(experiment, resources_per_trial)

    # * Run Experiment
    tuner = Tuner(
        trainable=trainable,
        param_space=args,
        tune_config=tune_config,
    )

    results_grid = tuner.fit()

    print(f"Results saved at {results_grid.experiment_path}")

    # * Compute summary statistics and store in latex tables
    df = results_grid.get_dataframe()

    stat_list = [
        "train_acc",
        "validation_acc",
        "test_acc",
        "best_train_acc",
        "best_validation_acc",
        "best_test_acc",
    ]

    for stat in stat_list:
        orig_df_to_latex(df, stat, results_grid.experiment_path)
