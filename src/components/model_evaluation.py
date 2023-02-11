import sys

import mlflow
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.entity.artifact_entity import ModelEvaluationArtifact
from src.entity.config_entity import PipelineConfig
from src.exception import MNISTException
from src.logger import logging
from src.ml.model.arch import Net


class ModelEvaluation:
    def __init__(self):
        self.pipeline_config = PipelineConfig()

    def test(self, model: Net, test_loader: DataLoader) -> ModelEvaluationArtifact:
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

            test_accuracy = correct / len(test_loader.dataset)

            mlflow.log_metric(key="test_loss", value=test_loss)

            mlflow.log_metric(key="test_accuracy", value=test_accuracy)

            model_evaluation_artifact: ModelEvaluationArtifact = (
                ModelEvaluationArtifact(
                    test_loss=test_loss, test_accuracy=test_accuracy
                )
            )

            print(f"Test set : Average loss : {test_loss}, Accuracy : {test_accuracy}")

            logging.info(
                f"Test set : Average loss : {test_loss}, Accuracy : {test_accuracy}"
            )

            logging.info(f"Model Evaluation artifact is {model_evaluation_artifact}")

            return model_evaluation_artifact

        except Exception as e:
            raise MNISTException(e, sys)
