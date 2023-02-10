import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.entity.config_entity import PipelineConfig
from src.exception import MNISTException


class ModelEvaluation:
    def __init__(self):
        self.pipeline_config = PipelineConfig()

    def test(self, model: nn.Module, test_loader: DataLoader) -> None:
        """
        It takes a model and a test_loader as input, and then it evaluates the model on the test_loader.

        Args:
          model: The model to be tested
          test_loader: the test data loader
        """
        try:
            model.eval()

            test_loss: float = 0

            correct: int = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.pipeline_config.device), target.to(
                        self.pipeline_config.device
                    )

                    output = model(data)

                    test_loss += F.nll_loss(output, target, reduction="sum").item()

                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print(
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                )
            )

        except Exception as e:
            raise MNISTException(e, sys)
