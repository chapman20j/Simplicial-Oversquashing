# node_experiment.py
"""
Implements the main training loop for node classification tasks.
"""
import ray
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import get_dataset
from models import build_model
from rewire import rewire


def evaluate(pred: torch.Tensor, label: torch.Tensor) -> float:
    return 100 * (pred.argmax(axis=-1) == label).sum().item() / len(label)


def experiment(args):
    """Main experiment function for node classification."""

    # * Set some parameters
    model_kwargs = args["model"]
    optim_kwargs = args["optim"]

    task = "classification"
    loss_fn = torch.nn.CrossEntropyLoss()

    if task == "regression":
        early_stop_fn = lambda x, y: x < y
        best_fn = lambda x, y: min(x, y)
    elif task == "classification":
        early_stop_fn = lambda x, y: x > y
        best_fn = lambda x, y: max(x, y)
    else:
        raise ValueError("Invalid task")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # * Get dataset
    dataset = get_dataset(args["dataset"], args["complex"], **args["dataset_kwargs"])

    # Set number of classes
    if args["dataset"] in ["TEXAS", "WISCONSIN", "CORNELL"]:
        model_kwargs["num_classes"] = 5
    elif args["dataset"] == "CORA":
        model_kwargs["num_classes"] = 7
    elif args["dataset"] == "CITESEER":
        model_kwargs["num_classes"] = 6
    else:
        raise ValueError("Invalid dataset")

    # * Train, Validation, Test split
    # only predict for the y values in the original graph
    dataset_size = dataset[0].y.size(0)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.1)
    inds = torch.randperm(dataset_size)
    train_mask = inds[:train_size]
    val_mask = inds[train_size : train_size + validation_size]
    test_mask = inds[train_size + validation_size :]

    # * Rewire dataset
    rewire(dataset, args["rewiring"], args["rewire_iterations"])

    # * Build Model
    model_kwargs["num_features"] = dataset[0].x.shape[1]

    # Compute num_relations
    max_edge_type = max([g.edge_type.max().item() for g in dataset])
    model_kwargs["num_relations"] = max_edge_type + 1

    model = build_model(args["model_name"], **model_kwargs).to(device)

    # * Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_kwargs["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer)

    if task == "classification":
        best_validation_metric = 0.0
        best_train_metric = 0.0
        best_test_metric = 0.0
        train_goal = 0.0
        validation_goal = 0.0
        goal_scalar = optim_kwargs["stopping_threshold"]
    else:
        best_validation_metric = 1e5
        best_train_metric = 1e5
        best_test_metric = 1e5
        train_goal = 1e5
        validation_goal = 1e5
        goal_scalar = 1 / optim_kwargs["stopping_threshold"]
    epochs_no_improve = 0

    # * Training Loop
    for epoch in range(1, 1 + optim_kwargs["max_epochs"]):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        graph = dataset[0].to(device)
        y = graph.y.to(device)
        out = model(graph)
        loss = loss_fn(input=out[train_mask], target=y[train_mask])
        total_loss += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step(total_loss)
        if epoch % args["eval_every"] == 0:
            with torch.no_grad():
                train_metric = evaluate(out[train_mask], y[train_mask])
                validation_metric = evaluate(out[val_mask], y[val_mask])
                test_metric = evaluate(out[test_mask], y[test_mask])

            # * Check early stopping criteria
            if optim_kwargs["stopping_criterion"] == "train":
                if early_stop_fn(train_metric, train_goal):
                    epochs_no_improve = 0
                    train_goal = train_metric * goal_scalar
                elif early_stop_fn(train_metric, best_train_metric):
                    epochs_no_improve += 1
                else:
                    epochs_no_improve += 1
            elif optim_kwargs["stopping_criterion"] == "validation":
                if early_stop_fn(validation_metric, validation_goal):
                    epochs_no_improve = 0
                    validation_goal = validation_metric * goal_scalar
                elif early_stop_fn(validation_metric, best_validation_metric):
                    epochs_no_improve += 1
                else:
                    epochs_no_improve += 1
            best_train_metric = best_fn(train_metric, best_train_metric)
            best_validation_metric = best_fn(validation_metric, best_validation_metric)
            best_test_metric = best_fn(test_metric, best_test_metric)

            # * Report metrics
            metrics = {
                "train_metric": train_metric,
                "validation_metric": validation_metric,
                "test_metric": test_metric,
                "best_train_metric": best_train_metric,
                "best_validation_metric": best_validation_metric,
                "best_test_metric": best_test_metric,
            }
            ray.train.report(metrics=metrics)

            if epochs_no_improve > optim_kwargs["patience"]:
                break

    if args.get("save_model_path") is not None:
        torch.save(model.state_dict(), args["save_model_path"])

    return
