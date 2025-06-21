import argparse
from pathlib import Path

import torch

import wandb
from src.data.dataset_manager import DatasetManager, DatasetType
from src.models.model_type import NoPropModelType
from src.models.model_wrapper import NoPropModelWrapper
from src.models.noprop_ct import NoPropCT
from src.models.noprop_dt import NoPropDT
from src.utils.device import DeviceManager
from src.utils.model_config import NoPropCTConfig, NoPropDTConfig
from src.utils.test_utils import test
from src.utils.train_config import NoPropTrainConfig
from src.utils.train_utils import get_scheduler, train


def train_on_db(wrapper: NoPropModelWrapper):
    """
    Trains the NoProp model on the MNIST dataset.

    :param wrapper: Model wrapper containing the model and device information.
    :param dataset_path: Path to the MNIST dataset.
    """

    train_config = wrapper.train_config

    train_dataset, val_dataset, test_dataset, number_of_classes = (
        DatasetManager.get_datasets(
            wrapper.train_config.dataset_type, wrapper.train_config.dataset_path
        )
    )

    wrapper.model_config.num_classes = number_of_classes

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

    wrapper.initialize_with_prototypes(train_loader)

    optimizer = torch.optim.AdamW(
        wrapper.model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    scheduler = (
        get_scheduler(
            optimizer, total_steps=(wrapper.train_config.epochs * len(train_loader))
        )
        if train_config.use_scheduler
        else None
    )

    train(
        wrapper,
        optimizer,
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

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a NoProp model.")
    parser.add_argument(
        "--save-model-path",
        type=str,
        required=True,
        help="Path where the model will be saved.",
    )
    parser.add_argument(
        "--model-type",
        type=NoPropModelType,
        choices=list(NoPropModelType),
        required=True,
        help="The type of NoProp model to train.",
    )
    parser.add_argument(
        "--dataset-type",
        type=DatasetType,
        choices=list(DatasetType),
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=False,
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    device = DeviceManager.get_device()

    train_config = NoPropTrainConfig(
        args.save_model_path, args.dataset_type, args.dataset_path
    )

    match args.model_type:
        case NoPropModelType.NO_PROP_CT:
            model_config = NoPropCTConfig()
            if args.config_path:
                train_config = train_config.from_file(args.config_path)
                model_config = model_config.from_file(args.config_path)
            model = NoPropCT(model_config, device)
        case NoPropModelType.NO_PROP_DT:
            model_config = NoPropDTConfig()
            if args.config_path:
                train_config = train_config.from_file(args.config_path)
                model_config = model_config.from_file(args.config_path)
            model = NoPropDT(model_config, device)
        case NoPropModelType.NO_PROP_FM:
            # model_config = NoPropFMConfig()
            # if args.config_path:
            #     train_config = train_config.from_file(args.config_path)
            #     model_config = model_config.from_file(args.config_path)
            # model = NoPropFM(model_config, device)
            raise NotImplementedError(f"Unsupported model type {args.model_type}")
        case _:
            raise NotImplementedError(f"Unsupported model type {args.model_type}")

    project_name = (
        "NoProp" + f"_{args.model_type.value}" + f"_{args.dataset_type.value}"
    )
    with wandb.init(project=project_name):
        wandb.config.update({**vars(train_config), "device": str(device)})
        wandb.watch(model)

        wrapper = NoPropModelWrapper(model, model_config, train_config)
        train_on_db(wrapper)

        wandb.finish()


if __name__ == "__main__":
    main()
