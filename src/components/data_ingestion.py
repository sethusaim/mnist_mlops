import sys

from torchvision import transforms
from torchvision.datasets import MNIST

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import ModelTrainingConfig, PipelineConfig
from src.exception import MNISTException


class DataIngestion:
    def __init__(self):
        self.model_trainer_config: ModelTrainingConfig = ModelTrainingConfig()

        self.pipeline_config: PipelineConfig = PipelineConfig()

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        It takes the MNIST dataset, transforms it into a tensor, and normalizes it

        Returns:
          A tuple of two datasets.
        """
        try:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            train_dataset = MNIST("../data", train=True, transform=transform)

            test_dataset = MNIST("../data", train=False, transform=transform)

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_dataset=train_dataset, test_dataset=test_dataset
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MNISTException(e, sys)
