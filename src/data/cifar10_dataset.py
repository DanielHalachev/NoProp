from pathlib import Path

import torchvision  # type:ignore
from torch.utils.data import Dataset


class CIFAR10DatasetManager:
    @classmethod
    def get_datasets(cls, data_root: Path) -> tuple[Dataset, Dataset, Dataset, int]:
        """
        Retrieves the CIFAR10 datasets for training, validation, and testing.

        The paper does not use validation loss for training optimization,
        so the validation and test sets may be the same.

        :return: A tuple containing three datasets:
            train_dataset,
            validation_dataset,
            test_dataset,
            and the number of classes.
        """

        # CIFAR10 has 3 channels (RGB), no need to repeat channels
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),  # Mean for CIFAR10
                    (0.2470, 0.2435, 0.2616)   # Std for CIFAR10
                ),
            ]
        )
        validation_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),  # Mean for CIFAR10
                    (0.2470, 0.2435, 0.2616)   # Std for CIFAR10
                ),
            ]
        )

        num_classes = 10  # CIFAR100 has 10 classes

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )
        validation_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=validation_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=False,
            download=True,
            transform=validation_transform,
        )

        return train_dataset, validation_dataset, test_dataset, num_classes