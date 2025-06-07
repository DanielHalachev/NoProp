import argparse
import os
import torch
import torch.nn as nn
import wandb
from src.data.mnist_dataset import MNISTDatasetManager
from src.models.model_type import NoPropModelType
from src.models.model_wrapper import NoPropModelWrapper
from src.models.noprop_dt import NoPropDT
from src.utils.device import DeviceManager
from src.utils.model_config import NoPropModelConfig
from src.utils.test_utils import test
from src.utils.train_config import NoPropTrainConfig
from src.utils.train_utils import get_scheduler, train


def train_mnist(
    wrapper: NoPropModelWrapper,
):
    """
    Trains the NoProp model on the MNIST dataset.

    :param wrapper: Model wrapper containing the model and device information.
    """

    train_config = wrapper.train_config

    train_dataset, val_dataset, test_dataset = MNISTDatasetManager.get_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=wrapper.train_config.workers,
    )
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=wrapper.train_config.batch_size,
        shuffle=False,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=wrapper.train_config.workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=wrapper.train_config.batch_size,
        shuffle=False,
        pin_memory=(False if wrapper.device.type == "mps" else True),
        num_workers=wrapper.train_config.workers,
    )

    # TODO maybe change optimizer, criterion and scheduler types
    optimizer = torch.optim.Adam(
        wrapper.model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    criterion = nn.MSELoss()
    scheduler = get_scheduler(
        optimizer, total_steps=(wrapper.train_config.epochs * len(train_loader)) // 10
    )

    train(
        wrapper,
        optimizer,
        criterion,
        scheduler,
        train_loader,
        validation_loader,
    )

    test_accuracy = test(wrapper, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")


def main():
    """
    Main function to set up the training environment and start training the model.
    """

    # setup MPS fallback for PyTorch if training on Apple Silicon
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "--model_path",
        type=os.PathLike,
        required=True,
        help="Path where the model will be saved.",
        example="mnist_model.pt",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=True,
        help="Name for the Weights & Biases project.",
        example="NoProp MNIST",
    )
    args = parser.parse_args()

    device = DeviceManager.get_device()
    model_config = NoPropModelConfig()
    train_config = NoPropTrainConfig(args.model_path)

    with wandb.init(project=""):
        wandb.config.update(
            {
                "learning_rate": train_config.lr,
                "epochs": train_config.epochs,
                "batch_size": train_config.batch_size,
                "weight_decay": train_config.weight_decay,
                "workers": train_config.workers,
                "model_path": args.model_path,
            }
        )

        match model_config.type:
            case NoPropModelType.NO_PROP_CT:
                model = NoPropDT(device)
            # case NoPropModelType.NO_PROP_DT:
            #     model = NoPropDT(device)
            # case NoPropModelType.NO_PROP_FM:
            #     model = NoPropDT(device)
            case _:
                raise ValueError(f"Unsupported model type: {model_config.type}")

        wrapper = NoPropModelWrapper(model, train_config, device)

        wandb.watch(model)

        train_mnist(wrapper)

        wandb.finish()
