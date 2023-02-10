import sys

from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.entity.config_entity import ModelTrainingConfig, PipelineConfig
from src.exception import MNISTException


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config: ModelTrainingConfig = ModelTrainingConfig()

        self.pipeline_config: PipelineConfig = PipelineConfig()

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        epoch: int,
    ) -> None:
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

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.pipeline_config.device), target.to(
                    self.pipeline_config.device
                )

                optimizer.zero_grad()

                output = model(data)

                loss = F.nll_loss(output, target)

                loss.backward()

                optimizer.step()

                if batch_idx % self.model_trainer_config.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

        except Exception as e:
            raise MNISTException(e, sys)
