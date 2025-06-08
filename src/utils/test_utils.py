import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    correct = 0
    total = 0
    wrapper.model.eval()

    with torch.no_grad():
        for src, trg in tqdm(dataloader):
            total += len(trg)  # batch size
            src = src.to(wrapper.device)
            trg = trg.to(wrapper.device)

            correct, total, predictions = wrapper.model.test_step(
                src, trg, wrapper.train_config.eta
            )

    test_accuracy = 100 if total == 0 else (correct * 100) / total

    wandb.log(
        {
            "test_accuracy": test_accuracy,
        }
    )

    return test_accuracy
