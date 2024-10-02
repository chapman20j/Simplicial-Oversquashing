# models/__init__.py
"""
Implements function to build a model based on the model name and the keyword arguments.
"""
from models.cellular.cin import CIN
from models.cellular.cinpp import CINpp
from models.cellular.sin import SIN
from models.gnn.gcn import GCN
from models.gnn.gin import GIN
from models.rgnn.rgcn import RGCN
from models.rgnn.rgin import RGIN
from models.sgc import SGC
from models.transfer.transfer_cinpp import CINpp as TransferCINpp
from models.transfer.transfer_gin import GIN as TransferGIN
from models.transfer.transfer_rgcn import RGCN as TransferRGCN
from models.transfer.transfer_rgin import RGIN as TransferRGIN
from models.transfer.transfer_sin import SIN as TransferSIN

model_names = {
    "gcn": "GCN",
    "gin": "GIN",
    "rgcn": "RGCN",
    "rgin": "RGIN",
    "sin": "SIN",
    "cin": "CIN",
    "cin++": "CIN++",
    "sgc": "SGC",
}


def build_model(model_name: str, **kwargs):
    """Function to build a model based on the model name and the keyword arguments."""

    if model_name == "gcn":
        # GNN
        return GCN(
            num_features=kwargs["num_features"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            num_classes=kwargs["num_classes"],
            dropout=kwargs["dropout"],
            pooling=kwargs["pooling"],
        )
    elif model_name == "gin":
        # GNN
        return GIN(
            num_features=kwargs["num_features"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            num_classes=kwargs["num_classes"],
            dropout=kwargs["dropout"],
            pooling=kwargs["pooling"],
        )
    elif model_name == "rgcn":
        # Relational GNN
        return RGCN(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            num_layers=kwargs["num_layers"],
            hidden_dim=kwargs["hidden_dim"],
            dropout=kwargs["dropout"],
            num_relations=kwargs["num_relations"],
            pooling=kwargs["pooling"],
        )
    elif model_name == "rgin":
        # Relational GNN
        return RGIN(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            num_layers=kwargs["num_layers"],
            hidden_dim=kwargs["hidden_dim"],
            dropout=kwargs["dropout"],
            num_relations=kwargs["num_relations"],
            pooling=kwargs["pooling"],
        )
    elif model_name == "transfergin":
        # GNN
        return TransferGIN(
            num_features=kwargs["num_features"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            num_classes=kwargs["num_classes"],
            dropout=kwargs["dropout"],
            pooling=kwargs["pooling"],
        )
    elif model_name == "transferrgcn":
        # Relational GNN
        return TransferRGCN(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            num_layers=kwargs["num_layers"],
            hidden_dim=kwargs["hidden_dim"],
            dropout=kwargs["dropout"],
            num_relations=kwargs["num_relations"],
        )
    elif model_name == "transferrgin":
        # Relational GNN
        return TransferRGIN(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            num_layers=kwargs["num_layers"],
            hidden_dim=kwargs["hidden_dim"],
            dropout=kwargs["dropout"],
            num_relations=kwargs["num_relations"],
            pooling=kwargs["pooling"],
        )
    elif model_name == "transfersin":
        # Simplicial NN
        return TransferSIN(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            max_dim=kwargs["max_dimension"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            num_relations=kwargs["num_relations"],
            pooling=kwargs["pooling"],
            multi_dim=kwargs["multidimensional"],
        )
    elif model_name == "transfercin++":
        # Simplicial NN
        return TransferCINpp(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            max_dim=kwargs["max_dimension"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            pooling=kwargs["pooling"],
            multi_dim=kwargs["multidimensional"],
        )
    elif model_name == "sin":
        # Simplicial NN
        return SIN(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            max_dim=kwargs["max_dimension"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            num_relations=kwargs["num_relations"],
            pooling=kwargs["pooling"],
            multi_dim=kwargs["multidimensional"],
        )
    elif model_name == "cin":
        # Simplicial NN
        return CIN(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            max_dim=kwargs["max_dimension"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            pooling=kwargs["pooling"],
            multi_dim=kwargs["multidimensional"],
        )
    elif model_name == "cin++":
        # Simplicial NN
        return CINpp(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            max_dim=kwargs["max_dimension"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            dropout=kwargs["dropout"],
            pooling=kwargs["pooling"],
            multi_dim=kwargs["multidimensional"],
        )
    elif model_name == "sgc":
        # Train Free
        return SGC(
            num_features=kwargs["num_features"],
            num_classes=kwargs["num_classes"],
            num_layers=kwargs["num_layers"],
            pooling=kwargs["pooling"],
        )
    else:
        raise ValueError(f"Model {model_name} not found.")
