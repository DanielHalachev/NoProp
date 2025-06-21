from pathlib import Path
import torchvision
from torch.utils.data import Dataset
import medmnist
from medmnist import INFO


def repeat_channels(x):
    """
    Converts a 1-channel grayscale image to 3 channels.
    """
    return x.repeat(3, 1, 1)



def identity(x):
    return x


class MedMNISTDatasetManager:
    @classmethod
    def get_datasets(cls, data_root: Path, dataset_name: str = "bloodmnist") -> tuple[Dataset, Dataset, Dataset, int]:
        """
        Retrieves the MedMNIST datasets for training, validation, and testing at 128x128 resolution.

        :param data_root: Path to the directory where MedMNIST data is stored.
        :param dataset_name: Name of the MedMNIST dataset (e.g., 'pathmnist', 'breastmnist').
        :return: A tuple containing three datasets (train_dataset, validation_dataset, test_dataset)
                 and the number of classes.
        """
        dataset_info = INFO[dataset_name]
        num_channels = dataset_info["n_channels"]
        num_classes = len(dataset_info["label"])
        channel_transform = repeat_channels if num_channels == 1 else identity

        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(channel_transform),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
            ]
        )
        validation_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(channel_transform),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        size = 128  

        dataset_class = getattr(medmnist, dataset_info["python_class"])
        
        train_dataset = dataset_class(
            split="train",
            root=data_root,
            download=True,
            transform=train_transform,
            size=size
        )
        validation_dataset = dataset_class(
            split="val",
            root=data_root,
            download=True,
            transform=validation_transform,
            size=size
        )
        test_dataset = dataset_class(
            split="test",
            root=data_root,
            download=True,
            transform=validation_transform,
            size=size
        )

        return train_dataset, validation_dataset, test_dataset, num_classes