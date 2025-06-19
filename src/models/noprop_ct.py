import random

import torch
import torch.nn.functional as torch_f
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import wandb
from components.ct_block import CTDenoiseBlock
from components.time_scheduler import TimeScheduler
from src.components.ct_noise_scheduler import CTNoiseScheduler
from src.embeddings.time_encoder import TimeEncoder
from src.models.base_model import BaseNoPropModel
from src.utils.model_config import NoPropCTConfig


class NoPropCT(BaseNoPropModel):
    """
    NoProp Continuous Time Model.
    """

    def __init__(self, config: NoPropCTConfig, device: torch.device) -> None:
        """
        Initializes the NoPropCT model with the specified configuration and device.
        :param config: Configuration object containing model parameters.
        :param device: The device (CPU or GPU) on which the model will run.
        """

        super().__init__(config, device)
        self.config = config
        self.denoise_block = CTDenoiseBlock(self.config)
        self.noise_scheduler = CTNoiseScheduler(config.noise_scheduler_hidden_dimension)
        self.time_encoder = TimeEncoder(
            self.config.time_encoder_hidden_dimension, self.config.embedding_dimension
        )
        self.to(device)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the alpha_bar value for a given time tensor t.
        The alpha_bar value is computed using the parameterized gamma function,
        which adjusts the noise schedule based on the time tensor.

        :param t: A tensor representing time, typically in the range [0, 1].
        :return: A tensor of the same shape as t, with values representing the alpha_bar
        values, which are used in diffusion models to control the noise schedule.
        """
        return self.noise_scheduler.alpha_bar(t)

    def forward_denoise(
        self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the denoise block, which processes the input image features,
        the label-embedding vector, and the time embedding.
        :param x: Input tensor representing the image features.
        :param z_t: Input tensor representing the label-embedding vector.
        :param t: Input tensor representing the time embedding.
        :return: A tuple containing:
            - logits: Output tensor of shape [batch_size, num_classes] representing the logits for each class.
            - z_next: Output tensor of shape [batch_size, embedding_dimension] representing the next label-embedding vector.
        """

        logits = self.denoise_block(x, z_t, t)
        probabilities = torch_f.softmax(logits, dim=1)
        z_next = probabilities @ self.W_Embed  # [batch_size, embedding_dimension]
        return logits, z_next

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs inference on the input tensor x using the NoPropCT model.
        This method iteratively denoises the embeddings using the learned model parameters
        and the noise schedule, and returns the predicted class labels.
        :param x: Input tensor of shape (batch_size, input_dimension).
        :return: Predicted class labels for the input tensor.
        """

        number_of_steps = self.config.inference_number_of_timesteps
        batch_size = x.size(0)
        z = torch.randn(batch_size, self.config.embedding_dimension, device=self.device)
        for t in range(number_of_steps):
            time = torch.full((batch_size, 1), t / number_of_steps, device=self.device)
            logits, z = self.forward_denoise(x, z, time)

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
        :param logs_per_epoch: Number of visualizations to log per epoch.
        :return: Average training loss for the epoch.
        """

        self.train()

        training_loss = 0.0
        for src, trg in tqdm(
            dataloader, desc=f"Training Epoch {epoch}/{total_epochs}", leave=False
        ):
            src = src.to(self.device)
            trg = trg.to(self.device)

            # TODO should we multiply by batch size?
            training_loss_tensor = self.train_step(src, trg, eta)
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
        Trains the model for one epoch.

        :param dataloader: DataLoader for the training dataset.
        :param epoch: Current epoch number.
        :param total_epochs: Total number of epochs for training.
        :param logs_per_epoch: Number of visualizations to log per epoch.
        :return: Average training loss for the epoch.
        """

        correct = 0
        total = 0
        logged_samples = 0
        total_samples_to_log = logs_per_epoch
        records: list[tuple[int, object, int, int]] = []

        self.eval()
        with torch.no_grad():
            for src, trg in tqdm(
                dataloader, desc=f"Validation Epoch {epoch}/{total_epochs}", leave=False
            ):
                src = src.to(self.device)
                trg = trg.to(self.device)

                correct_temp, total_temp, predictions = self.test_step(src, trg)
                correct += correct_temp
                total += total_temp

                remaining_samples_to_log = total_samples_to_log - logged_samples
                if remaining_samples_to_log > 0:
                    probability = remaining_samples_to_log / (total - logged_samples)
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
    ) -> torch.Tensor:
        """
        Performs a single training step for the NoPropCT model.
        This method computes the loss based on the input source tensor and target labels,
        and performs a forward pass through the model to obtain the logits and predictions.

        The training can be split by time steps, but here everything is in one place for simplicity.
        Since training will be done on one GPU anyway.

        :param src: Input tensor of shape (batch_size, input_dimension).
        :param trg: Target tensor of shape (batch_size,) containing class labels.
        :param eta: Hyperparameter for the training step.
        :return: A tensor representing the total loss for the training step.
        """
        batch_size = src.size(0)
        label_embeddings = self.W_Embed[trg]  # [batch_size, embedding_dim]

        # generate random time steps
        time_steps = TimeScheduler.get_uniform_time_schedule(batch_size, self.device)

        alpha_bar = self.alpha_bar(time_steps)
        sig_noise_ratio = alpha_bar / (1 - alpha_bar)
        snr_gradient = torch.autograd.grad(
            sig_noise_ratio.sum(), time_steps, create_graph=True
        )[0]

        epsilon = torch.randn_like(label_embeddings)  # noise
        z_t = alpha_bar.sqrt() * label_embeddings + (1 - alpha_bar).sqrt() * epsilon

        # forward pass through the model
        logits, prediction_embeddings = self.forward_denoise(src, z_t, time_steps)

        # compute losses - MSE, SDM and Kullback-Leibler divergence
        mse_loss = torch_f.mse_loss(
            prediction_embeddings, label_embeddings, reduction="none"
        ).sum(dim=1, keepdim=True)
        sdm_loss = 0.5 * eta * (snr_gradient * mse_loss).mean()
        kl_loss = (label_embeddings.pow(2).sum(dim=1)).mean()

        # compute the cross-entropy loss at step t=1
        time_steps_1 = torch.ones_like(time_steps)
        alpha_bar_1 = self.alpha_bar(time_steps_1)
        z_t_1 = alpha_bar_1.sqrt() * label_embeddings + (
            1 - alpha_bar_1
        ).sqrt() * torch.randn_like(label_embeddings)
        logits_1, prediction_embeddings_1 = self.forward_denoise(
            src, z_t_1, time_steps_1
        )
        ce_loss = torch_f.cross_entropy(torch_f.softmax(self.classifier(prediction_embeddings_1), dim=1), trg)

        # total loss is a combination of cross-entropy, KL divergence, and SDM loss
        wandb.log(
            {
                "train/ce_loss": ce_loss,
                "train/kl_loss": kl_loss,
                "train/sdm_loss": sdm_loss,
            }
        )
        total_loss = ce_loss + kl_loss + sdm_loss

        return total_loss

    def first_order_infer(self, source: torch.Tensor, num_steps: int):
        """
        First-order Euler inference method for NoPropCT.
        This method performs inference by iteratively denoising the embeddings
        using the learned model parameters and the noise schedule.

        This is is a fast solver, but less accurate than the second-order methods.

        :param source: Input tensor of shape (batch_size, input_dimension).
        :param num_steps: Number of steps for the inference process.
        :return: Predicted class labels for the input tensor.
        """

        with torch.no_grad():
            self.eval()
            batch_size = source.size(0)
            embedding_dimension = self.config.embedding_dimension
            time_step_size = 1.0 / num_steps
            z = torch.randn((batch_size, embedding_dimension), device=self.device)
            t = torch.full((batch_size, 1), 0.0, device=self.device)
            for step in range(num_steps):
                t = torch.full(
                    (batch_size, 1), step * time_step_size, device=self.device
                )
                alpha_bar = self.alpha_bar(t)
                logits_t, prediction_embeddings_t = self.forward_denoise(source, z, t)
                z = z + time_step_size * (prediction_embeddings_t - z) / (1 - alpha_bar)
            logits_final, prediction_embeddings_final = self.forward_denoise(
                source, z, torch.ones_like(t)
            )
            return logits_final.argmax(dim=1)

    def second_order_infer(self, source: torch.Tensor, num_steps: int):
        """
        Second-order inference method for NoPropCT.
        This method performs inference by computing two predictions
        at the current and next time steps and returns their average.
        It incorporates a second-order correction step to improve the predictions.

        This is a more accurate solver, but slower than the first-order method.

        :param source: Input tensor of shape (batch_size, input_dimension).
        :param num_steps: Number of steps for the inference process.
        :return: Predicted class labels for the input tensor.
        """

        with torch.no_grad():
            self.eval()
            batch_size = source.size(0)
            embedding_dimension = self.config.embedding_dimension
            time_step_size = 1.0 / num_steps
            z = torch.randn((batch_size, embedding_dimension), device=self.device)
            t = torch.full((batch_size, 1), 0.0, device=self.device)
            for step in range(num_steps):
                t = torch.full(
                    (batch_size, 1), step * time_step_size, device=self.device
                )
                t_1 = torch.full(
                    (batch_size, 1), (step + 1) * time_step_size, device=self.device
                )
                alpha_bar = self.alpha_bar(t)
                logits_t, prediction_embeddings_t = self.forward_denoise(source, z, t)
                f_n = (prediction_embeddings_t - z) / (1 - alpha_bar)
                z_mid = z + time_step_size * f_n
                alpha_bar_1 = self.alpha_bar(t_1)
                logits_t_1, prediction_embeddings_t_1 = self.forward_denoise(
                    source, z_mid, t_1
                )
                f_n_1 = (prediction_embeddings_t_1 - z_mid) / (1 - alpha_bar_1)
                z = z + 0.5 * time_step_size * (f_n + f_n_1)
            logits_final, prediction_embeddings_final = self.forward_denoise(
                source, z, torch.ones_like(t)
            )
            return logits_final.argmax(dim=1)

    def ddim_infer(self, source: torch.Tensor, num_steps: int):
        """
        Denoising Diffusion Implicit Models (DDIM) inference method for NoPropCT.
        This method performs inference by iteratively denoising the embeddings
        using the learned model parameters and the noise schedule.

        This is a fast solver, which is more accurate than the Euler solver,
        but less accurate than the second-order Heun solver.
        Therefore, it is a good trade-off between speed and accuracy.

        :param source: Input tensor of shape (batch_size, input_dimension).
        :param num_steps: Number of steps for the inference process.
        :return: Predicted class labels for the input tensor.
        """

        with torch.no_grad():
            self.eval()
            batch_size = source.size(0)
            embedding_dimension = self.config.embedding_dimension
            time_steps = torch.linspace(1.0, 0.0, num_steps, device=self.device)
            z = torch.randn((batch_size, embedding_dimension), device=self.device)
            t = torch.full((batch_size, 1), 0.0, device=self.device)
            for step in range(num_steps - 1, -1, -1):
                t = time_steps[step]
                alpha_bar_t = self.alpha_bar(t)
                alpha_bar_t_prev = (
                    self.alpha_bar(time_steps[step - 1]) if step > 0 else 1.0
                )
                logits, prediction_embeddings = self.forward_denoise(source, z, t)
                z = (z - (1 - alpha_bar_t) * prediction_embeddings) / (
                    alpha_bar_t_prev**0.5
                )
            logits_final, prediction_embeddings_final = self.forward_denoise(
                source, z, torch.zeros_like(t)
            )
            return logits_final.argmax(dim=1)
