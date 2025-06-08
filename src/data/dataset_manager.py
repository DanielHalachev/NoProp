from enum import Enum
from pathlib import Path

from torch.utils.data import Dataset

from src.data.mnist_dataset import MNISTDatasetManager


# TODO add a big dataset
class DatasetType(Enum):
    MNIST = "MNIST"
    CUSTOM = "CUSTOM"


class DatasetManager:
    """
    Convenient class for fetching datasets based on type and path.
    """

    @classmethod
    def get_datasets(
        cls, dataset_type: DatasetType, data_root: Path
    ) -> tuple[Dataset, Dataset, Dataset, int]:
        """
        Retrieves the datasets for training, validation, and testing.

        :param dataset_type: Type of the dataset to retrieve (e.g., MNIST, CIFAR10, CUSTOM).
        :param data_root: Root directory where the dataset is stored.
        :return: A tuple containing train_dataset, validation_dataset, test_dataset, and number_of_classes.
        """
        match dataset_type:
            case DatasetType.MNIST:
                return MNISTDatasetManager.get_datasets(data_root)
            case DatasetType.CUSTOM:
                # return ImageNetDatasetManager._get_datasets(data_root)
                raise NotImplementedError("Dataset type not supported.")
            case _:
                raise NotImplementedError("Dataset type not supported.")
