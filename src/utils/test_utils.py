import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from models.model_wrapper import NoPropModelWrapper  # type:ignore


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

    wrapper.model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for src, trg in tqdm(dataloader):
            total += len(trg)  # batch size
            src = src.to(wrapper.device)
            trg = trg.to(wrapper.device)

            correct, total, predictions = wrapper.model.test_step(src, trg)

    test_accuracy = 100 if total == 0 else (correct * 100) / total

    wandb.log(
        {
            "test_accuracy": test_accuracy,
        }
    )

    return test_accuracy
