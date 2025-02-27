# ablation.py
"""
Tests graph lifting and rewiring on graph classifiation benchmarks.
"""

import argparse

import yaml
from ray.tune import TuneConfig, Tuner, grid_search, with_resources

from utils.experiment import experiment
from utils.plot_results import orig_df_to_latex

if __name__ == "__main__":

    # * Load experiment configuration
    with open("experiment_config.yaml", "r") as f:
        args = yaml.safe_load(f)

    # * Get arguments
    parser = argparse.ArgumentParser(description="Ablation")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["MUTAG"],
        help="List of datasets to use for experiment",
        choices=[
            "MUTAG",
            "ENZYMES",
            "PROTEINS",
            "NCI1",
            "IMDB-BINARY",
            "ZINC",
        ],
    )
    parser.add_argument(
        "--lifts",
        type=str,
        nargs="+",
        default=["none"],
        help="List of graph lifts to use",
        choices=["none", "clique", "ring"],
    )
    parser.add_argument(
        "--rewiring",
        type=str,
        nargs="+",
        default=["none"],
        help="List of rewiring strategies to use",
        choices=["none", "fosr", "sdrf", "afr4", "prune", "prune1d"],
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["hidden", "layers"],
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of trials to use for each configuration",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=70,
        help="Number of trials to run concurrently",
    )
    command_args = parser.parse_args()
    if command_args.samples <= 0:
        raise ValueError("Number of samples must be non-negative")
    if command_args.max_concurrent <= 0:
        raise ValueError("Number of concurrent trials must be non-negative")

    # * Set up grid search
    args["model_name"] = grid_search(["cin++"])
    args["dataset"] = grid_search(command_args.datasets)
    args["complex"] = grid_search(command_args.lifts)
    args["rewiring"] = grid_search(command_args.rewiring)
    args["rewire_iterations"] = grid_search([0, 10, 20, 40])

    # * Do sweep over network sizes and hidden dimensions
    if command_args.ablation == "layers":
        args["model"]["num_layers"] = grid_search([2, 4, 6])
    elif command_args.ablation == "hidden":
        args["model"]["hidden_dim"] = grid_search([16, 32, 64])
    else:
        raise ValueError("Invalid ablation")

    args["dataset_kwargs"] = {}

    resources_per_trial = {"cpu": 1, "gpu": 0}
    tune_config = TuneConfig(
        num_samples=command_args.samples,
        max_concurrent_trials=command_args.max_concurrent,
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
        "train_metric",
        "validation_metric",
        "test_metric",
        "best_train_metric",
        "best_validation_metric",
        "best_test_metric",
    ]

    for stat in stat_list:
        orig_df_to_latex(df, stat, results_grid.experiment_path)
