import sys

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig, PipelineConfig
from src.exception import MNISTException


class DataIngestion:
    def __init__(self):
        self.pipeline_config: PipelineConfig = PipelineConfig()

        self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()

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

            train_dataset: Dataset = MNIST(
                root=self.data_ingestion_config.train_data_path,
                train=True,
                transform=transform,
                download=True,
            )

            test_dataset: Dataset = MNIST(
                root=self.data_ingestion_config.test_data_path,
                train=False,
                transform=transform,
                download=True,
            )

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_dataset=train_dataset, test_dataset=test_dataset
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MNISTException(e, sys)
