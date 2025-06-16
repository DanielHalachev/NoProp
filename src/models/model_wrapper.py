import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore

from components.backbone import Backbone
from src.models.base_model import BaseNoPropModel
from src.utils.train_config import NoPropTrainConfig
from utils.model_config import NoPropBaseModelConfig  # type: ignore


class NoPropModelWrapper:
    """
    Wrapper around any NoProp model to store device and training configuration in order to avoid passing too many parameters.
    """

    def __init__(
        self,
        model: BaseNoPropModel,
        model_config: NoPropBaseModelConfig,
        train_config: NoPropTrainConfig,
    ) -> None:
        """
        Initializes the NoPropModelWrapper with an instance of a model, training configuration, and device.

        :param model: An instance of NoPropModel.
        :param train_config: An instance of NoPropTrainConfig containing training parameters.
        :param device: The device (CPU or GPU) on which the model will run.
        """
        self.model = model
        self.model_config = model_config
        self.train_config = train_config
        self.device = model.device

    def save_model(self, path: os.PathLike | None = None):
        """
        Saves the model to the specified path or to the default path in the training configuration.

        :param path: Optional path to save the model. If None, uses the path from train_config.
        """
        save_path = self.train_config.model_path if path is None else path
        self.model.save_model(save_path)

    def initialize_with_prototypes(
        self,
        train_dataloader: DataLoader,
    ) -> tuple[torch.Tensor, list[int]]:
        """
        Initializes class prototypes by selecting representative embeddings for each class from random samples.

        :param train_dataloader: DataLoader providing training data (images and labels).
        :return: A tuple containing:
                - A tensor of class prototypes (shape: [num_classes, embedding_dim]).
                - A list of indexes corresponding to the selected prototype samples.
        """
        self.model.eval()

        # this is a pretrained backbone that will be used to extract features from images
        # it is not the same as the backbone used in the models, which is not pretrained
        self.pretrained_backbone = Backbone(
            self.model_config.backbone_resnet_type,
            self.model_config.embedding_dimension,
            use_resnet=True,
            pretrained=True,
        ).to(self.device)

        num_classes = self.model_config.num_classes
        max_samples_per_class = self.train_config.samples_per_class

        # collect all features and labels from the dataloader
        features_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in tqdm(train_dataloader):
                features = self.pretrained_backbone(images.to(self.device))
                features_list.append(features.cpu())
                labels_list.append(labels)
        all_features = torch.cat(features_list, dim=0)  # [total_samples, embedding_dim]
        all_labels = torch.cat(labels_list, dim=0)  # [total_samples, 1]

        embedding_dimension = all_features.size(1)
        class_prototypes = torch.zeros(
            num_classes, embedding_dimension, device=self.device
        )
        prototype_sample_indexes: list[int] = []

        # compute prototypes for each class
        for class_id in range(num_classes):
            # get indexes of samples belonging to the current class
            class_sample_indexes = (
                (all_labels == class_id).nonzero(as_tuple=True)[0].tolist()
            )

            # randomly select up to max_samples_per_class samples for the current class
            selected_indexes = random.sample(
                class_sample_indexes,
                min(max_samples_per_class, len(class_sample_indexes)),
            )

            # extract embeddings for the selected samples
            selected_embeddings = all_features[selected_indexes].to(self.device)

            # compute pairwise distances [k, k] and find the most central sample
            pairwise_distances = torch.cdist(selected_embeddings, selected_embeddings)
            median_distances = pairwise_distances.median(dim=1).values  # [k]
            central_sample_index = int(torch.argmin(median_distances).item())

            # assign the most central embedding as the prototype for the current class
            class_prototypes[class_id] = selected_embeddings[central_sample_index]

            # store the index of the selected prototype sample
            prototype_sample_indexes.append(selected_indexes[central_sample_index])

        with torch.no_grad():
            self.model.W_Embed.copy_(class_prototypes)
            self.model.W_Embed.data = self.model.W_Embed.data.to(self.device)
        return class_prototypes, prototype_sample_indexes
