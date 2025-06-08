import torch
import torch.nn.functional as torch_f

from src.embeddings.time_encoder import TimeEncoder
from src.models.base_model import BaseNoPropModel
from src.utils.model_config import NoPropCTConfig


class NoPropCT(BaseNoPropModel):
    """
    NoProp Continuous Time Model.
    """

    def __init__(self, config: NoPropCTConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self.time_encoder = TimeEncoder(
            config.time_embedding_dimension, config.embedding_dimension
        )

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise_scheduler.alpha_bar(t)

    def forward_denoise(
        self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        if t is None:
            raise ValueError("Parameter t cannot be None in NoPropCT")
        fx = self.backbone(x)
        fz = self.label_encoder(z_t)
        ft = self.time_encoder(t)
        return self.concatenator(fx, fz, ft)

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
        time_steps = torch.rand(batch_size, 1, device=self.device, requires_grad=True)

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
        time_steps = torch.rand(batch_size, 1, device=self.device, requires_grad=True)

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
        predictions = self.infer(src, inference_number_of_steps)
        correct += int((predictions == trg).sum().item())
        return correct, trg.size(0), predictions

    def infer(self, source: torch.Tensor, num_steps: int):
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
