import sys

import mlflow
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.entity.config_entity import ModelTrainingConfig, PipelineConfig
from src.exception import MNISTException
from src.logger import logging
from src.ml.model.arch import Net


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config: ModelTrainingConfig = ModelTrainingConfig()

        self.pipeline_config: PipelineConfig = PipelineConfig()

    def train(
        self,
        model: Net,
        train_loader: DataLoader,
        optimizer: Optimizer,
        epoch: int,
    ) -> float:
        """
        The function takes in a model, a train_loader, an optimizer, and an epoch. It then trains the model
        using the train_loader, optimizer, and epoch.

        Args:
          model (nn.Module): nn.Module
          train_loader (DataLoader): DataLoader
          optimizer (Optimizer): The optimizer to use for training.
          epoch (int): int
        """
        try:
            model.train()

            for _, (data, target) in enumerate(train_loader):
                data, target = data.to(self.pipeline_config.device), target.to(
                    self.pipeline_config.device
                )

                optimizer.zero_grad()

                output = model(data)

                loss = F.nll_loss(output, target)

                loss.backward()

                optimizer.step()

            train_loss = loss.item()

            mlflow.log_metric(key="train_loss", value=train_loss)

            print(f"Train Epoch : {epoch},Loss : {train_loss}")

            logging.info(f"Train Epoch : {epoch},Loss : {train_loss}")

            return train_loss

        except Exception as e:
            raise MNISTException(e, sys)
