import torch
import torch.nn.functional as torch_f
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.components.ct_concatenator import CTConcatenator
from src.components.ct_noise_scheduler import CTNoiseScheduler
from src.components.time_scheduler import TimeScheduler
from src.embeddings.ct_label_encoder import CTLabelEncoder
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

        self.config = config

        embedding_dimension = config.embedding_dimension
        hidden_dimension = config.label_encoder_hidden_dimension
        label_encoder = CTLabelEncoder(embedding_dimension, hidden_dimension)
        concatenator = CTConcatenator(embedding_dimension, config.num_classes)
        noise_scheduler = CTNoiseScheduler(config.noise_scheduler_hidden_dimension)

        time_embedding_dimension = config.time_embedding_dimension

        super().__init__(config, label_encoder, noise_scheduler, concatenator, device)

        self.time_encoder = TimeEncoder(time_embedding_dimension, embedding_dimension)

    def forward_denoise(
        self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        if t is None:
            raise ValueError("Parameter t cannot be None in NoPropCT")
        fx = self.backbone(x)
        fz = self.label_encoder(z_t)
        ft = self.time_encoder(t)
        return self.concatenator(fx, fz, ft)

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
        :return: Average training loss for the epoch.
        """

        self.train()

        training_loss = 0.0
        for src, trg in tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False
        ):
            src = src.to(self.device)
            trg = trg.to(self.device)

            training_loss += self.train_step(src, trg, optimizer, eta) * src.size(0)
        avg_loss = training_loss / len(dataloader.dataset)  # type: ignore
        if scheduler is not None:
            scheduler.step()
        return avg_loss

    def train_step(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        eta: float,
    ) -> float:
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
        z_t = alpha_bar * label_embeddings + (1 - alpha_bar).sqrt() * epsilon

        # forward pass through the model
        logits = self.forward_denoise(src, z_t, time_steps)
        class_probabilities = torch_f.softmax(logits, dim=1)
        prediction_embeddings = class_probabilities @ self.W_Embed

        # compute losses - MSE, SDM and Kullback-Leibler divergence
        mse_loss = torch_f.mse_loss(
            prediction_embeddings, label_embeddings, reduction="none"
        ).sum(dim=1, keepdim=True)
        sdm_loss = 0.5 * eta * (snr_gradient * mse_loss).mean()
        kl_loss = 0.5 * (label_embeddings.pow(2).sum(dim=1)).mean()

        # compute the cross-entropy loss at step t=1
        time_steps_1 = torch.ones_like(time_steps)
        alpha_bar_1 = self.alpha_bar(time_steps_1)
        z_1 = alpha_bar_1 * label_embeddings + (
            1 - alpha_bar_1
        ).sqrt() * torch.randn_like(label_embeddings)
        ce_loss = torch_f.cross_entropy(
            self.forward_denoise(src, z_1, time_steps_1), trg
        )

        # total loss is a combination of cross-entropy, KL divergence, and SDM loss
        total_loss = ce_loss + kl_loss + sdm_loss

        # backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item()

    def validate_step(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        eta: float,
        inference_number_of_steps,
    ) -> tuple[float, int, int, torch.Tensor]:
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
        z_t = alpha_bar * label_embeddings + (1 - alpha_bar).sqrt() * epsilon

        # forward pass through the model
        logits = self.forward_denoise(src, z_t, time_steps)
        class_probabilities = torch_f.softmax(logits, dim=1)
        prediction_embeddings = class_probabilities @ self.W_Embed

        # compute losses - MSE, SDM and Kullback-Leibler divergence
        mse_loss = torch_f.mse_loss(
            prediction_embeddings, label_embeddings, reduction="none"
        ).sum(dim=1, keepdim=True)
        sdm_loss = 0.5 * eta * (snr_gradient * mse_loss).mean()
        kl_loss = 0.5 * (label_embeddings.pow(2).sum(dim=1)).mean()

        # compute the cross-entropy loss at step t=1
        time_steps_1 = torch.ones_like(time_steps)
        alpha_bar_1 = self.alpha_bar(time_steps_1)
        z_1 = alpha_bar_1 * label_embeddings + (
            1 - alpha_bar_1
        ).sqrt() * torch.randn_like(label_embeddings)
        ce_loss = torch_f.cross_entropy(
            self.forward_denoise(src, z_1, time_steps_1), trg
        )

        # total loss is a combination of cross-entropy, KL divergence, and SDM loss
        total_loss = ce_loss + kl_loss + sdm_loss

        total, correct, predictions = self.test_step(
            src, trg, inference_number_of_steps
        )

        return total_loss.item(), correct, total, predictions

    def test_step(
        self, src: torch.Tensor, trg: torch.Tensor, inference_number_of_steps: int
    ) -> tuple[int, int, torch.Tensor]:
        correct = 0
        predictions = self.first_order_infer(src, inference_number_of_steps)
        correct += int((predictions == trg).sum().item())
        return correct, trg.size(0), predictions

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
                logits_t = self.forward_denoise(source, z, t)
                class_probabilities_t = torch_f.softmax(logits_t, dim=1)
                prediction_embeddings_t = class_probabilities_t @ self.W_Embed
                z = z + time_step_size * (prediction_embeddings_t - z) / (1 - alpha_bar)
            logits = self.forward_denoise(source, z, torch.ones_like(t))
            return logits.argmax(dim=1)

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
                logits_t = self.forward_denoise(source, z, t)
                class_probabilities_t = torch_f.softmax(logits_t, dim=1)
                prediction_embeddings_t = class_probabilities_t @ self.W_Embed
                f_n = (prediction_embeddings_t - z) / (1 - alpha_bar)
                z_mid = z + time_step_size * f_n
                alpha_bar_1 = self.alpha_bar(t_1)
                class_probabilities_t_1 = torch_f.softmax(
                    self.forward_denoise(source, z_mid, t_1), dim=1
                )
                prediction_embeddings_t_1 = class_probabilities_t_1 @ self.W_Embed
                f_n_1 = (prediction_embeddings_t_1 - z_mid) / (1 - alpha_bar_1)
                z = z + 0.5 * time_step_size * (f_n + f_n_1)
            logits = self.forward_denoise(source, z, torch.ones_like(t))
            return logits.argmax(dim=1)

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
                prediction = self.forward_denoise(source, z, t)
                z = (z - (1 - alpha_bar_t) * prediction) / (alpha_bar_t_prev**0.5)
            return self.forward_denoise(source, z, torch.zeros_like(t)).argmax(dim=1)
