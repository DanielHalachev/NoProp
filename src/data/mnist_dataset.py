import torch.utils.data as data


class MNISTDataset(data.Dataset):
    def __init__(self) -> None:
        super().__init__()


class MNISTDatasetManager:
    @classmethod
    def get_datasets(cls) -> tuple[data.Dataset, data.Dataset, data.Dataset]:
        # TODO properly split data into train, validation and test in ratio 80-10-10
        # unless MNIST has the data already split
        train_dataset = MNISTDataset()
        validation_dataset = MNISTDataset()
        test_dataset = MNISTDataset()
        return train_dataset, validation_dataset, test_dataset
