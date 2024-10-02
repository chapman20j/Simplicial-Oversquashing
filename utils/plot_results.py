# plot_results.py
"""
Code to analyze experiment results and plot them.
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from ray import tune

model_names = {
    "gcn": "GCN",
    "gin": "GIN",
    "rgcn": "RGCN",
    "rgin": "RGIN",
    "sin": "SIN",
    "sgc": "SGC",
    "cin": "CIN",
    "cin++": "CIN++",
}
rewiring_names = {
    "fosr": "FoSR",
    "sdrf": "SDRF",
    "afr4": "AFR4",
    "none": "None",
    "config/dataset": " ",
}


# * Utility
def _row_reduce(x):
    out = []
    for s in x:
        if s in rewiring_names:
            out.append(rewiring_names[s])
        else:
            out.append(s.capitalize())
    return tuple(out)


def _col_reduce(x):
    return x.split("/")[-1].split("_")[0].capitalize() if "config" in x else x


def load_results(path: str) -> DataFrame:
    analysis = tune.ExperimentAnalysis(path)
    result_grid = tune.ResultGrid(analysis)
    df = result_grid.get_dataframe()
    return df


# * General
# Group and compute statistics
def get_result_stats(df: DataFrame, group_keys: list[str], stat: str) -> DataFrame:
    """Computes the mean results for the same keys and different keys.

    Args:
        df: Dataframe of results
        group_keys: Keys that are relevant for combining results

    Returns:
        Dataframe of mean, std, and sem
    """

    group_df = df.groupby(group_keys).aggregate({stat: ["mean", "std", "sem"]})

    return group_df


def df_model_stat(
    df: DataFrame, model: str, stat: str, col_header: str = "config/dataset"
) -> tuple[DataFrame, DataFrame, DataFrame]:
    cin_df = df.xs(model, level="config/model_name")
    cin_df = cin_df[stat].unstack(level=col_header)
    return cin_df["mean"], cin_df["std"], cin_df["sem"]


# * Latex tables
def pd_to_mean_pm_std(df_mean, df_spread) -> DataFrame:
    """Creates a latex table to summarize results.

    Args:
        df_mean: Mean of the data
        df_spread: Spread of the data. Typically std or sem values.

    Returns:
        Formatted dataframe for latex
    """

    formatted_df = DataFrame(
        index=pd.MultiIndex.from_tuples(map(_row_reduce, df_mean.index)),
        columns=list(map(_col_reduce, df_mean.columns)),
        dtype=str,
    )
    formatted_df.index.names = list(map(_col_reduce, df_mean.index.names))

    for col in df_mean.columns:
        for r in df_mean.index:
            row = _row_reduce(r)
            mean = df_mean.at[r, col]
            std = df_spread.at[r, col]
            formatted_df.at[row, col] = f"${100*mean:.1f} \\pm {100*std:.1f}$"

    return formatted_df


def multiple_latex_tables(df_dict: dict[DataFrame]) -> dict[str, str]:
    """Creates multiple latex tables from formatted dataframes."""
    out = dict()

    for key, df in df_dict.items():

        caption = f"\\textbf{{{model_names.get(key, key)}}}"
        out[key] = df.style.to_latex(
            caption=caption,
            label=f"tab:{key}",
            environment="subtable",
            position="h",
            hrules=True,
        )

    return out


def orig_df_to_latex(
    df: DataFrame, stat: str, save_path=None, standard_error=True
) -> str:
    """Converts original dataframe to latex tables."""
    group_df = get_result_stats(
        df,
        [
            "config/complex",
            "config/rewiring",
            "config/dataset",
            "config/model_name",
        ],
        stat,
    )

    model_list = df["config/model_name"].unique()

    out_df = {}

    for model in model_list:
        mean_df, std_df, sem_df = df_model_stat(group_df, model, stat)

        if standard_error:
            formatted_df = pd_to_mean_pm_std(mean_df, sem_df)
        else:
            formatted_df = pd_to_mean_pm_std(mean_df, std_df)

        out_df[model] = formatted_df

    subtables = multiple_latex_tables(out_df)
    if save_path is None:
        for key, table in subtables.items():
            print(table)
    else:
        p = f"{save_path}/00{stat}"
        if not os.path.exists(p):
            os.makedirs(p)
        for key, table in subtables.items():
            tab = table.replace("[h]", "[h]{.5\\textwidth}")
            with open(f"{p}/{key}.tex", "w") as f:
                f.write(tab)

        with open(f"{p}/all.tex", "w") as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\scriptsize")
            for key, table in subtables.items():
                f.write("\\input{tables/00" + stat + "/" + key + "}\n")
            f.write("\\end{table}\n")

    return subtables


if __name__ == "__main__":

    # * Transfer
    # Update path to experiments
    transfer_path = None
    if transfer_path is None:
        raise ValueError("Please update path to results")

    # The following code was used for plotting the results in the paper
    # The folders will change depending on the experimental setup
    rgcn_folders = [
        "1 SIZE/rgcn both/",
        "1 WIDTH/rgcn both/",
        "1 REWIRE/rgcn both/",
    ]
    gin_folders = [
        "1 SIZE/gin none/",
        "1 WIDTH/gin none/",
        "1 REWIRE/gin none/",
    ]
    exp = ["size", "hidden_dim", "rewire"]
    xaxis_map = {
        "size": "Nodes",
        "hidden_dim": "Hidden Dimension",
        "rewire": "Iterations",
    }
    dataset = "ringtransfer"
    stat = "test_acc"

    plt.rcParams.update({"font.size": 18})

    for fgin, frgcn, e in zip(gin_folders, rgcn_folders, exp):
        print(e)

        if e == "size":
            if dataset == "ringtransfer":
                size_var = "config/dataset_kwargs/nodes"
            elif dataset == "nmatch":
                size_var = "config/dataset_kwargs/num_cliques"
        elif e == "hidden_dim":
            size_var = "config/model/hidden_dim"
        elif e == "rewire":
            size_var = "config/rewire_iterations"
        elif e == "depth":
            size_var = "config/model/num_layers"

        save_path = f"./results/transfer/ring_{e}.png"

        # * Plot colors
        color1 = "mediumaquamarine"
        color2 = "cornflowerblue"
        color3 = "darkorange"

        # * GIN
        mean_df = pd.read_csv(transfer_path + fgin + f"{e}_{dataset}_{stat}_mean.csv")
        sem_df = pd.read_csv(transfer_path + fgin + f"{e}_{dataset}_{stat}_sem.csv")

        col = "none"

        plt.plot(
            mean_df[size_var],
            mean_df[col],
            label=f"GIN/{col.capitalize()}",
            color=color1,
        )
        plt.fill_between(
            mean_df[size_var],
            mean_df[col] - sem_df[col],
            mean_df[col] + sem_df[col],
            alpha=0.2,
            color=color1,
        )

        # * RGCN
        mean_df = pd.read_csv(transfer_path + frgcn + f"{e}_{dataset}_{stat}_mean.csv")
        sem_df = pd.read_csv(transfer_path + frgcn + f"{e}_{dataset}_{stat}_sem.csv")

        col = "none"
        plt.plot(
            mean_df[size_var],
            mean_df[col],
            label=f"RGCN/{col.capitalize()}",
            color=color2,
        )

        # Include color fill for standard error
        plt.fill_between(
            mean_df[size_var],
            mean_df[col] - sem_df[col],
            mean_df[col] + sem_df[col],
            alpha=0.2,
            color=color2,
        )

        col = "clique"
        plt.plot(
            mean_df[size_var],
            mean_df[col],
            label=f"RGCN/{col.capitalize()}",
            color=color3,
        )

        # Include color fill for standard error
        plt.fill_between(
            mean_df[size_var],
            mean_df[col] - sem_df[col],
            mean_df[col] + sem_df[col],
            alpha=0.2,
            color=color3,
        )

        if size_var == "config/model/hidden_dim":
            plt.xscale("log")
        plt.xlabel(xaxis_map[e], fontdict={"fontsize": 18})
        plt.ylabel("Accuracy (%)", fontdict={"fontsize": 18})
        plt.tight_layout()
        plt.savefig(f"{save_path}")
        print(f"{save_path}")
        plt.clf()

    # * Neighbors Match
    folders = [
        "2 NMATCH RGCN/rewire none/",
        "2 NMATCH RGCN/rewire clique/",
    ]
    lift_list = ["none", "clique"]
    e = "rewire"
    dataset = "nmatch"
    size_var = "config/rewire_iterations"
    for frgcn, lift in zip(folders, lift_list):

        # RGCN
        mean_df = pd.read_csv(transfer_path + frgcn + f"{e}_{dataset}_{stat}_mean.csv")
        sem_df = pd.read_csv(transfer_path + frgcn + f"{e}_{dataset}_{stat}_sem.csv")

        col = lift
        plt.plot(
            mean_df[size_var],
            mean_df[col],
            label=f"{col.capitalize()}",
        )

        # Include color fill for standard error
        plt.fill_between(
            mean_df[size_var],
            mean_df[col] - sem_df[col],
            mean_df[col] + sem_df[col],
            alpha=0.2,
        )

    plt.rcParams.update({"font.size": 12})

    save_path = f"./results/transfer/nmatch_{e}.png"
    plt.xlabel(xaxis_map[e], fontdict={"fontsize": 12})
    plt.ylabel("Accuracy (%)", fontdict={"fontsize": 12})
    plt.tight_layout()
    plt.legend(title="Lift")
    plt.savefig(f"{save_path}")
    print(f"{save_path}")
    plt.clf()

    # * Neighbors Match with density
    for frgcn, lift in zip(folders, lift_list):

        # RGCN
        mean_df = pd.read_csv(transfer_path + frgcn + f"{e}_{dataset}_{stat}_mean.csv")
        sem_df = pd.read_csv(transfer_path + frgcn + f"{e}_{dataset}_{stat}_sem.csv")

        if lift == "none":
            max_edges = 15 * 7
            base_edges = 32
        else:
            max_edges = 77 * 38
            base_edges = 764

        col = lift
        plt.plot(
            (base_edges + mean_df[size_var]) / max_edges,
            mean_df[col],
            label=f"{col.capitalize()}",
        )

        # Include color fill for standard error
        plt.fill_between(
            (base_edges + mean_df[size_var]) / max_edges,
            mean_df[col] - sem_df[col],
            mean_df[col] + sem_df[col],
            alpha=0.2,
        )

    plt.rcParams.update({"font.size": 12})

    save_path = f"./results/transfer/nmatch_edgedensity_{e}.png"
    plt.xlabel("Density", fontdict={"fontsize": 12})
    plt.xlim(0, 1)
    plt.ylabel("Accuracy (%)", fontdict={"fontsize": 12})
    plt.tight_layout()
    plt.legend(title="Lift")
    plt.savefig(f"{save_path}")
    print(f"{save_path}")
    plt.clf()

    # * RingTransfer Legend
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.legend(
        handles=[
            mpatches.Patch(color=color1, label="GIN/None"),
            mpatches.Patch(color=color2, label="RGCN/None"),
            mpatches.Patch(color=color3, label="RGCN/Clique"),
        ],
        loc="lower left",
        ncol=3,
        title="Model/Lift",
        borderaxespad=0.0,
    )
    # Change fig size
    # fig.set_size_inches(3, 2)
    fig.tight_layout()
    fig.savefig("./results/transfer/ring_legend.png")
    plt.clf()
