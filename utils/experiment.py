# experiment.py
"""
Implements the main training loop graph classification and transfer tasks.
"""
import ray
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from data import get_dataset, num_classes_dict
from models import build_model
from rewire import rewire


def evaluate(model, loader, device):
    """Evaluates model performance."""
    model.eval()
    sample_size = len(loader.dataset)
    with torch.no_grad():
        total_correct = 0
        for graph in loader:
            graph = graph.to(device)
            y = graph.y.to(device)
            out = model(graph)
            _, pred = out.max(dim=1)
            total_correct += pred.eq(y).sum().item()

    return total_correct / sample_size


def experiment(args):
    """Main experiment function."""

    # * Set some parameters
    model_kwargs = args["model"]
    optim_kwargs = args["optim"]
    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    else:
        model_kwargs["num_classes"] = max([g.y for g in dataset]) + 1
        if model_kwargs["num_layers"] is None:
            model_kwargs["num_layers"] = args["dataset_kwargs"]["nodes"]

    # * Rewire dataset
    rewire(dataset, args["rewiring"], args["rewire_iterations"])

    # * Train, Validation, Test split
    dataset_size = len(dataset)
    train_size = int(args["train_fraction"] * dataset_size)
    validation_size = int(args["validation_fraction"] * dataset_size)
    test_size = dataset_size - train_size - validation_size
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size]
    )
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
    model_kwargs["num_features"] = dataset[0].x.shape[1]

    # Compute num_relations
    max_edge_type = max([g.edge_type.max().item() for g in dataset])
    model_kwargs["num_relations"] = max_edge_type + 1

    model = build_model(args["model_name"], **model_kwargs).to(device)

    # * Set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_kwargs["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer)

    best_validation_acc = 0.0
    best_train_acc = 0.0
    best_test_acc = 0.0
    train_goal = 0.0
    validation_goal = 0.0
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
            loss = loss_fn(input=out, target=y)
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step(total_loss)
        if epoch % args["eval_every"] == 0:
            train_acc = evaluate(model=model, loader=train_loader, device=device)
            validation_acc = evaluate(
                model=model, loader=validation_loader, device=device
            )
            test_acc = evaluate(model=model, loader=test_loader, device=device)

            # * Check early stopping criteria
            if optim_kwargs["stopping_criterion"] == "train":
                if train_acc > train_goal:
                    epochs_no_improve = 0
                    train_goal = train_acc * optim_kwargs["stopping_threshold"]
                elif train_acc > best_train_acc:
                    epochs_no_improve += 1
                else:
                    epochs_no_improve += 1
            elif optim_kwargs["stopping_criterion"] == "validation":
                if validation_acc > validation_goal:
                    epochs_no_improve = 0
                    validation_goal = (
                        validation_acc * optim_kwargs["stopping_threshold"]
                    )
                elif validation_acc > best_validation_acc:
                    epochs_no_improve += 1
                else:
                    epochs_no_improve += 1
            best_train_acc = max(train_acc, best_train_acc)
            best_validation_acc = max(validation_acc, best_validation_acc)
            best_test_acc = max(test_acc, best_test_acc)

            # * Report metrics
            metrics = {
                "train_acc": train_acc,
                "validation_acc": validation_acc,
                "test_acc": test_acc,
                "best_train_acc": best_train_acc,
                "best_validation_acc": best_validation_acc,
                "best_test_acc": best_test_acc,
            }
            ray.train.report(metrics=metrics)

            if epochs_no_improve > optim_kwargs["patience"]:
                return

    return
