from pathlib import Path

import torchvision  # type:ignore
from torch.utils.data import Dataset


class CIFAR100DatasetManager:
    @classmethod
    def get_datasets(cls, data_root: Path) -> tuple[Dataset, Dataset, Dataset, int]:
        """
        Retrieves the CIFAR100 datasets for training, validation, and testing.

        The paper does not use validation loss for training optimization,
        so the validation and test sets may be the same.

        :return: A tuple containing three datasets:
            train_dataset,
            validation_dataset,
            test_dataset,
            and the number of classes.
        """

        # CIFAR100 has 3 channels (RGB), no need to repeat channels
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),  # Mean for CIFAR100
                    (0.2675, 0.2565, 0.2761)   # Std for CIFAR100
                ),
            ]
        )
        validation_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),  # Mean for CIFAR100
                    (0.2675, 0.2565, 0.2761)   # Std for CIFAR100
                ),
            ]
        )

        num_classes = 100  # CIFAR100 has 100 classes

        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )
        validation_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=validation_transform,
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=validation_transform,
        )

        return train_dataset, validation_dataset, test_dataset, num_classes