# transfer.py
"""
Graph transfer experiments.
    Size
    Rewire
    Hidden Dimension
    Depth
"""

import argparse
import os

import matplotlib.pyplot as plt
import yaml
from ray.tune import ResultGrid, TuneConfig, Tuner, grid_search, with_resources

from utils.experiment import experiment
from utils.plot_results import get_result_stats

DEFAULT_NMATCH_CLIQUES = 3
DEFAULT_NMATCH_CLIQUE_SIZE = 5
DEFAULT_RING_NODES = 10

DEFAULT_RINGTRANSFER_DEPTH = 4
DEFAULT_NMATCH_DEPTH = 4


def process_results(
    results_grid: ResultGrid,
    dataset: str,
    group_keys: list[str],
    size_config: str,
    name: str,
):
    """Process graph transfer results and saves summary statistics.

    Args:
        results_grid: results grid from ray tune
        dataset: dataset name
        group_keys: keys relevant to combining results
        size_config: config variable storing the size parameter
        name: name of the experiment
    """
    print(f"Results saved at {results_grid.experiment_path}")
    df = results_grid.get_dataframe()

    for stat in ["train_metric", "test_metric"]:
        group_df = get_result_stats(df, group_keys, stat)

        mean_df = group_df[stat, "mean"].unstack(level=size_config).T * 100
        std_df = group_df[stat, "std"].unstack(level=size_config).T * 100
        sem_df = group_df[stat, "sem"].unstack(level=size_config).T * 100

        ax = mean_df.plot(kind="bar", yerr=sem_df, capsize=4, rot=0)
        ax.set_xlabel("Size")
        ax.set_ylabel("Accuracy (%)")
        plt.legend(title="Complex")

        # # * Save the results
        path = os.path.join(
            results_grid.experiment_path, f"{name}_{dataset}_{stat}.png"
        )
        plt.savefig(path)
        plt.clf()

        # * Save the data
        path = os.path.join(
            results_grid.experiment_path, f"{name}_{dataset}_{stat}_mean.csv"
        )
        mean_df.to_csv(path)
        path = os.path.join(
            results_grid.experiment_path, f"{name}_{dataset}_{stat}_std.csv"
        )
        std_df.to_csv(path)
        path = os.path.join(
            results_grid.experiment_path, f"{name}_{dataset}_{stat}_sem.csv"
        )
        sem_df.to_csv(path)


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
    param_space["model"]["num_layers"] = None
    if dataset == "ringtransfer":
        param_space["dataset_kwargs"] = {
            "nodes": grid_search(list(range(6, 15, 2))),
            "num_datapoints": num_datapoints,
        }
        size_config = "config/dataset_kwargs/nodes"
    elif dataset == "nmatch":
        param_space["dataset_kwargs"] = {
            "num_cliques": grid_search(list(range(2, 5))),
            "clique_size": DEFAULT_NMATCH_CLIQUE_SIZE,
            "num_datapoints": num_datapoints,
        }
        size_config = "config/dataset_kwargs/num_cliques"
    else:
        raise ValueError(f"Dataset {dataset} not found.")

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


def transfer_rewire_experiment(
    model_name: str,
    dataset: str,
    num_samples: int,
    num_datapoints: int,
    complex_method: list[str],
    rewiring: list[str],
    max_concurrent: int,
):
    """Transfer experiment varying the rewiring iterations.

    Args:
        model_name: name of the model
        dataset: name of the dataset
        num_samples: number of trials to run
        num_datapoints: number of graphs in the dataset
        complex_method: graph lifting method
        rewiring: rewiring method to use
        max_concurrent: number of trials to run concurrently

    Raises:
        ValueError: invalid dataset
    """

    with open("experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    param_space = config | {
        "rewiring": grid_search(rewiring),
        "dataset": dataset,
        "model_name": model_name,
        "complex": grid_search(complex_method),
    }
    if dataset == "ringtransfer":
        param_space["model"]["num_layers"] = DEFAULT_RINGTRANSFER_DEPTH
        param_space["rewire_iterations"] = grid_search(list(range(11)))
        param_space["dataset_kwargs"] = {
            "nodes": DEFAULT_RING_NODES,
            "num_datapoints": num_datapoints,
        }
    elif dataset == "nmatch":
        param_space["model"]["num_layers"] = DEFAULT_NMATCH_DEPTH
        if complex_method[0] == "none":
            param_space["rewire_iterations"] = grid_search(
                list(range(0, 51, 10)) + [100, 150, 200, 250]
            )
        else:
            param_space["rewire_iterations"] = grid_search(list(range(0, 251, 50)))
        param_space["dataset_kwargs"] = {
            "num_cliques": DEFAULT_NMATCH_CLIQUES,
            "clique_size": DEFAULT_NMATCH_CLIQUE_SIZE,
            "num_datapoints": num_datapoints,
        }
    else:
        raise ValueError(f"Dataset {dataset} not found.")

    size_config = "config/rewire_iterations"
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
    process_results(results_grid, dataset, group_keys, size_config, "rewire")


def transfer_hidden_experiment(
    model_name: str,
    dataset: str,
    num_samples: int,
    num_datapoints: int,
    complex_method: list[str],
    max_concurrent: int,
):
    """Transfer experiment varying the hidden dimension of models.

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
    if dataset == "ringtransfer":
        param_space["model"]["num_layers"] = None
        param_space["dataset_kwargs"] = {
            "nodes": DEFAULT_RING_NODES,
            "num_datapoints": num_datapoints,
        }
    elif dataset == "nmatch":
        param_space["model"]["num_layers"] = None
        param_space["dataset_kwargs"] = {
            "num_cliques": DEFAULT_NMATCH_CLIQUES,
            "clique_size": DEFAULT_NMATCH_CLIQUE_SIZE,
            "num_datapoints": num_datapoints,
        }
    else:
        raise ValueError(f"Dataset {dataset} not found.")

    param_space["model"]["hidden_dim"] = grid_search([1, 2, 4, 8, 16, 32, 64, 128])

    size_config = "config/model/hidden_dim"
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
    process_results(results_grid, dataset, group_keys, size_config, "hidden_dim")


if __name__ == "__main__":
    NUM_SAMPLES = 10
    NUM_DATAPOINTS = 1000
    rewire_methods = ["fosr"]

    parser = argparse.ArgumentParser(description="Transfer experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to run",
        choices=["ringtransfer", "nmatch"],
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to run",
        default="rgcn",
        choices=["rgcn", "gin", "cin++", "rgin", "sin"],
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="experiment to run",
        default="size",
        choices=["size", "rewire", "hidden_dim"],
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
    model = "transfer" + args.model
    if args.samples <= 0:
        raise ValueError("Number of samples must be non-negative")
    if args.max_concurrent <= 0:
        raise ValueError("Number of concurrent trials must be non-negative")

    if args.exp == "size":
        transfer_size_experiment(
            model,
            args.dataset,
            args.samples,
            NUM_DATAPOINTS,
            args.lifts,
            args.max_concurrent,
        )
    elif args.exp == "rewire":
        transfer_rewire_experiment(
            model,
            args.dataset,
            args.samples,
            NUM_DATAPOINTS,
            args.lifts,
            rewire_methods,
            args.max_concurrent,
        )
    elif args.exp == "hidden_dim":
        transfer_hidden_experiment(
            model,
            args.dataset,
            args.samples,
            NUM_DATAPOINTS,
            args.lifts,
            args.max_concurrent,
        )
    else:
        raise ValueError("Experiment not found.")
