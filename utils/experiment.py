# experiment.py
"""
Implements the main training loop for graph classification, regression, and transfer tasks.
"""

import ray
import torch
from ogb.graphproppred import Evaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from data import get_dataset, num_classes_dict
from models import build_model
from rewire import rewire


def evaluate(model, loader, device, loss_fn=None):
    """Evaluates model performance."""
    model.eval()
    sample_size = len(loader.dataset)
    if isinstance(loss_fn, Evaluator):
        trues = []
        probs = []

    with torch.no_grad():
        total = 0
        for graph in loader:
            graph = graph.to(device)
            y = graph.y.to(device)
            out = model(graph)
            if loss_fn is None:
                _, pred = out.max(dim=1)
                total += pred.eq(y).sum().item()
            elif isinstance(loss_fn, Evaluator):
                trues.append(torch.argmax(y, dim=-1, keepdim=True))
                probs.append(
                    torch.nn.functional.softmax(out, dim=-1)[:, 1].unsqueeze(-1)
                )
            else:
                loss = loss_fn(input=out.flatten(), target=y.flatten())
                total += loss.item()

    if isinstance(loss_fn, Evaluator):
        probs = torch.concatenate(probs, dim=0)
        trues = torch.concatenate(trues, dim=0)
        loss = loss_fn.eval({"y_true": trues, "y_pred": probs})["rocauc"]
        return loss

    out = total / sample_size
    return out * 100 if loss_fn is None else out


def experiment(args):
    """Main experiment function."""

    # * Set some parameters
    model_kwargs = args["model"]
    optim_kwargs = args["optim"]

    # * Classification or Regression
    # loss function
    # evaluation metric (None for accuracy)
    # early stopping function (used for time without improvement)
    if args["dataset"] in ["ZINC", "ZINC-FULL"]:
        task = "regression"
        loss_fn = torch.nn.L1Loss()
        eval_loss_fn = torch.nn.L1Loss(reduction="sum")
    else:
        task = "classification"
        loss_fn = torch.nn.CrossEntropyLoss()
        eval_loss_fn = None

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
    # For graph transfer, update model size if None
    if args["dataset"] in num_classes_dict:
        model_kwargs["num_classes"] = num_classes_dict[args["dataset"]]
    elif args["dataset"] == "nmatch":
        model_kwargs["num_classes"] = max([g.y for g in dataset]) + 1
        if model_kwargs["num_layers"] is None:
            model_kwargs["num_layers"] = 2 * args["dataset_kwargs"]["num_cliques"] + 1
    elif "ZINC" in args["dataset"]:
        # This is regression
        model_kwargs["num_classes"] = 1
    else:
        model_kwargs["num_classes"] = max([g.y for g in dataset]) + 1
        if model_kwargs["num_layers"] is None:
            model_kwargs["num_layers"] = args["dataset_kwargs"]["nodes"]

    # * Train, Validation, Test split
    if args["dataset"] in ["ZINC", "ZINC-FULL"]:
        train_dataset, validation_dataset, test_dataset, *_ = dataset
    else:
        dataset_size = len(dataset)
        train_size = int(args["train_fraction"] * dataset_size)
        validation_size = int(args["validation_fraction"] * dataset_size)
        test_size = dataset_size - train_size - validation_size
        train_dataset, validation_dataset, test_dataset = random_split(
            dataset, [train_size, validation_size, test_size]
        )

    # * Rewire dataset
    rewire(train_dataset, args["rewiring"], args["rewire_iterations"])
    rewire(validation_dataset, args["rewiring"], args["rewire_iterations"])
    rewire(test_dataset, args["rewiring"], args["rewire_iterations"])

    train_loader = DataLoader(
        train_dataset, batch_size=optim_kwargs["batch_size"], shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=optim_kwargs["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=optim_kwargs["batch_size"], shuffle=True
    )

    # * Build Model
    model_kwargs["num_features"] = train_dataset[0].x.shape[1]

    # Compute num_relations
    max_train_et = max([g.edge_type.max().item() for g in train_dataset])
    max_validation_et = max([g.edge_type.max().item() for g in validation_dataset])
    max_test_et = max([g.edge_type.max().item() for g in test_dataset])
    max_edge_type = max(max_train_et, max_validation_et, max_test_et)
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

        for graph in train_loader:
            graph = graph.to(device)
            y = graph.y.to(device)
            out = model(graph)
            if task == "regression":
                out = out.flatten()
                y = y.flatten()
            loss = loss_fn(input=out, target=y)
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step(total_loss)
        if epoch % args["eval_every"] == 0:
            train_metric = evaluate(
                model=model, loader=train_loader, device=device, loss_fn=eval_loss_fn
            )
            validation_metric = evaluate(
                model=model,
                loader=validation_loader,
                device=device,
                loss_fn=eval_loss_fn,
            )
            test_metric = evaluate(
                model=model, loader=test_loader, device=device, loss_fn=eval_loss_fn
            )

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

            if epochs_no_improve > optim_kwargs["patience"]:
                ray.train.report(metrics=metrics)
                break
            else:
                ray.train.report(metrics=metrics)

    if args.get("save_model_path") is not None:
        torch.save(model.state_dict(), args["save_model_path"])

    return
