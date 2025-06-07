import torch
from tqdm import tqdm
from models.model_wrapper import NoPropModelWrapper  # type:ignore
from torch.utils.data import DataLoader
import wandb


def test(
    wrapper: NoPropModelWrapper,
    dataloader: DataLoader,
) -> float:
    """
    Evaluates the model on the test dataset and logs the accuracy.

    :param wrapper: Model wrapper containing the model and device information.
    :param dataloader: DataLoader for the test dataset.
    :return: The accuracy of the model on the test dataset.
    """

    correct = 0
    total = 0
    wrapper.model.eval()

    with torch.no_grad():
        for src, labels, trg in tqdm(dataloader):
            total += len(labels)  # Batch size
            src = src.to(wrapper.device)
            trg = trg.to(wrapper.device)

            predictions = wrapper.predict(src)  # List of predicted classes
            for pred, label in zip(predictions, labels):
                if pred == label:
                    correct += 1

    test_accuracy = 100 if total == 0 else (correct * 100) / total

    wandb.log(
        {
            "test_accuracy": test_accuracy,
        }
    )

    return test_accuracy
