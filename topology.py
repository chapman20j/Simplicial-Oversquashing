# topology.py
"""
Contains experiment for the tree task. 
"""

import argparse

import yaml
from ray.tune import TuneConfig, Tuner, grid_search, with_resources

from transfer import process_results
from utils.experiment import experiment


def transfer_size_experiment(
    model_name: str,
    dataset: str,
    num_samples: int,
    num_datapoints: int,
    complex_method: list[str],
    max_concurrent: int,
):
    """Transfer experiment varying the size of the graphs.

    Args:
        model_name: name of the model
        dataset: name of the dataset
        num_samples: number of trials to run
        num_datapoints: number of graphs in the dataset
        complex_method: graph lifting method
        max_concurrent: number of trials to run concurrently

    Raises:
        ValueError: invalid dataset
    """

    with open("experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    param_space = config | {
        "rewiring": "none",
        "rewire_iterations": 1,
        "dataset": dataset,
        "model_name": model_name,
        "complex": grid_search(complex_method),
    }
    # Make model depth depend on problem size
    param_space["dataset_kwargs"] = {
        "max_depth": grid_search([2, 3, 4, 5]),
        "num_datapoints": num_datapoints,
        "cycles": grid_search([True, False]),
    }
    size_config = "config/dataset_kwargs/max_depth"

    group_keys = [
        "config/complex",
        size_config,
    ]

    resources_per_trial = {"cpu": 1, "gpu": 0}
    tune_config = TuneConfig(
        num_samples=num_samples, max_concurrent_trials=max_concurrent
    )
    trainable = with_resources(experiment, resources_per_trial)

    tuner = Tuner(
        trainable=trainable,
        param_space=param_space,
        tune_config=tune_config,
    )

    results_grid = tuner.fit()

    # Compute summary statistics
    process_results(results_grid, dataset, group_keys, size_config, "size")


if __name__ == "__main__":
    NUM_DATAPOINTS = 1000

    parser = argparse.ArgumentParser(description="Transfer experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to run",
        default="tree",
        choices=["tree"],
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to run",
        default="rgcn",
        choices=["rgcn", "gin", "cin++", "rgin", "sin"],
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
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("Number of samples must be non-negative")
    if args.max_concurrent <= 0:
        raise ValueError("Number of concurrent trials must be non-negative")

    model = "transfer" + args.model

    transfer_size_experiment(
        model,
        args.dataset,
        args.samples,
        NUM_DATAPOINTS,
        args.lifts,
        args.max_concurrent,
    )
