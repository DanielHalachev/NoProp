import random

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from components.dt_block import DTDenoiseBlock
from src.components.dt_noise_scheduler import DTNoiseScheduler
from src.models.base_model import BaseNoPropModel
from src.utils.model_config import NoPropDTConfig


class NoPropDT(BaseNoPropModel):
    """
    NoProp Discrete Time Model.
    """

    def __init__(self, config: NoPropDTConfig, device: torch.device) -> None:
        """
        Initializes the NoPropDT model with the specified configuration and device.

        :param config: Configuration object containing model parameters.
        :param device: The device (CPU or GPU) on which the model will run.
        """

        super().__init__(config, device)
        self.config = config
        self.denoise_blocks = nn.ModuleList(
            DTDenoiseBlock(config) for _ in range(config.number_of_timesteps)
        )
        self.noise_scheduler = DTNoiseScheduler(config.number_of_timesteps)

        alpha_bar = self.noise_scheduler.alpha_bar()
        self.register_buffer("alpha_bar_buf", alpha_bar)

        snr = alpha_bar / (1 - alpha_bar + 1e-8)
        previous_snr = torch.cat(
            [torch.tensor([0.0], dtype=snr.dtype), snr[:-1]], dim=0
        )
        snr_diff = snr - previous_snr
        self.register_buffer("snr_diff_buf", snr_diff)
        self.to(device)

    def alpha_bar(self) -> torch.Tensor:
        """
        Returns the alpha_bar values from the noise scheduler.
        :return: Tensor representing the alpha_bar values.
        """
        return torch.as_tensor(self.alpha_bar_buf)

    def snr_diff(self) -> torch.Tensor:
        """
        Returns the signal to noise ratio (SNR) difference precomputed tensor.

        :return: Tensor representing the SNR difference values.
        """
        return torch.as_tensor(self.snr_diff_buf)

    def forward_denoise(
        self, x: torch.Tensor, z_t: torch.Tensor, t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward denoising step for the NoProp discrete time model.

        :param x: Input tensor.
        :param z_t: Noise tensor.
        :param t: Current timestep in the denoising process.
        :return: Denoised output tensor.
        """
        logits = (self.denoise_blocks[t])(x, z_t)
        probabilities = torch_f.softmax(logits, dim=-1)
        z_next = probabilities @ self.W_Embed
        return logits, z_next

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference step for the NoProp discrete time model.

        :param x: Input tensor.
        :return: Prediction output tensor.
        """
        batch_size = x.size(0)
        z = torch.randn(batch_size, self.config.embedding_dimension, device=self.device)
        for t in range(self.config.number_of_timesteps):
            logits, z = self.forward_denoise(x, z, t)

        return self.classifier(z)

    def train_epoch(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        dataloader: DataLoader,
        eta: float,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """
        Trains the model for one epoch.

        :param optimizer: Optimizer for the model parameters.
        :param scheduler: Learning rate scheduler.
        :param dataloader: DataLoader for the training dataset.
        :param eta: Learning rate.
        :param epoch: Current epoch number.
        :param total_epochs: Total number of epochs for training.
        :return: Average training loss for the epoch.
        """

        self.train()

        training_loss = 0.0

        for timestep in tqdm(
            range(self.config.number_of_timesteps),
            desc="Timestep Loop",
            leave=False,
        ):
            for src, trg in tqdm(
                dataloader, desc=f"Training Epoch {epoch}/{total_epochs}", leave=False
            ):
                src = src.to(self.device)
                trg = trg.to(self.device)

                # TODO should we multiply by batch size?
                training_loss_tensor = self.train_step(src, trg, eta, timestep)
                training_loss += training_loss_tensor.item() * src.size(0)

                # backpropagation and optimization
                optimizer.zero_grad()
                training_loss_tensor.backward()
                optimizer.step()

        avg_loss = training_loss / len(dataloader.dataset)  # type: ignore
        if scheduler is not None:
            scheduler.step()
        return avg_loss

    def validate_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int,
        logs_per_epoch: int,
    ) -> tuple[float, list[tuple[int, object, int, int]]]:
        """
        Validates the model for one epoch.
        :param dataloader: DataLoader for the validation dataset.
        :param epoch: Current epoch number.
        :param total_epochs: Total number of epochs for validation.
        :param logs_per_epoch: Number of samples to log per epoch.
        :return: Tuple containing the validation accuracy and a list of logged samples.
        """

        correct = 0
        total = 0
        logged_samples = 0
        total_samples_to_log = logs_per_epoch
        records: list[tuple[int, object, int, int]] = []

        self.eval()
        with torch.no_grad():
            for timestep in tqdm(
                range(self.config.number_of_timesteps),
                desc="Timestep Loop",
                leave=False,
            ):
                for src, trg in tqdm(
                    dataloader,
                    desc=f"Validation Epoch {epoch}/{total_epochs}",
                    leave=False,
                ):
                    src = src.to(self.device)
                    trg = trg.to(self.device)

                    correct_temp, total_temp, predictions = self.test_step(src, trg)
                    correct += correct_temp
                    total += total_temp

                    remaining_samples_to_log = total_samples_to_log - logged_samples
                    if remaining_samples_to_log > 0:
                        probability = remaining_samples_to_log / (
                            total - logged_samples
                        )
                        for i in range(src.size(0)):
                            if (
                                logged_samples < total_samples_to_log
                                and random.random() < probability
                            ):
                                records.append(
                                    (
                                        epoch,
                                        ToPILImage()(src[i].cpu()),
                                        int(trg[i].item()),
                                        int(predictions[i].item()),
                                    )
                                )
                                logged_samples += 1
            accuracy = correct / total if total > 0 else 0.0
            return accuracy, records

    def train_step(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        eta: float,
        current_timestep: int,
    ) -> torch.Tensor:
        """
        Training and validation step for the NoProp discrete time model.

        :param src: Source tensor (input data).
        :param trg: Target tensor (ground truth labels).
        :param optimizer: Optimizer for updating model parameters.
        :param eta: NoProp hyperparameter.
        :return: Loss value for the training step.
        """
        # u_y
        label_embeddings = self.W_Embed[trg]  # [batch_size, embedding_dim]

        alpha_bar_t = self.alpha_bar()[current_timestep]

        epsilon = torch.randn_like(label_embeddings)  # noise
        z_t = alpha_bar_t.sqrt() * label_embeddings + (1 - alpha_bar_t).sqrt() * epsilon

        # forward pass through the model
        logits, prediction_embeddings = self.forward_denoise(src, z_t, current_timestep)

        # compute losses - MSE, SDM and Kullback-Leibler divergence
        snr_gradient = self.snr_diff()[current_timestep]
        mse_loss = torch_f.mse_loss(
            prediction_embeddings, label_embeddings
        )  #    .sum(dim=1, keepdim=True)

        sdm_loss = 0.5 * eta * (snr_gradient * mse_loss).mean()
        kl_loss = 0.5 * (label_embeddings.pow(2).sum(dim=1)).mean()

        # TODO what to pass to the CE loss function?
        # compute the cross-entropy loss at step t=T
        T = self.config.number_of_timesteps - 1
        alpha_bar_T = self.alpha_bar()[T]
        z_T = alpha_bar_T.sqrt() * label_embeddings + (
            1 - alpha_bar_T
        ).sqrt() * torch.randn_like(label_embeddings)
        logits_T, prediction_embeddings_T = self.forward_denoise(src, z_T, T)
        ce_loss = torch_f.cross_entropy(self.classifier(prediction_embeddings_T), trg)
        # total loss is a combination of cross-entropy, KL divergence, and SDM loss
        total_loss = ce_loss + kl_loss + sdm_loss

        return total_loss
