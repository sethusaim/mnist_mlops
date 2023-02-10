import sys

from torch.utils.data import DataLoader, Dataset

from src.entity.artifact_entity import DataLoaderArtifact
from src.entity.config_entity import ModelTrainingConfig, PipelineConfig
from src.exception import MNISTException


class MNISTDataLoader:
    def __init__(self):
        self.pipeline_config: PipelineConfig = PipelineConfig()

        self.model_trainer_config: ModelTrainingConfig = ModelTrainingConfig()

    def get_dataloaders(
        self, train_dataset: Dataset, test_dataset: Dataset
    ) -> DataLoaderArtifact:
        """
        It takes in a train and test dataset and returns a train and test dataloader.

        Args:
          train_dataset (Dataset): Dataset = train_dataset
          test_dataset (Dataset): Dataset = MNIST(

        Returns:
          A tuple of two DataLoaders.
        """
        try:
            train_kwargs = {"batch_size": self.model_trainer_config.batch_size}

            test_kwargs = {"batch_size": self.model_trainer_config.test_batch_size}

            if self.pipeline_config.device == "cuda":
                cuda_kwargs = self.pipeline_config.cuda_args

                train_kwargs.update(cuda_kwargs)

                test_kwargs.update(cuda_kwargs)

            train_dataloader: DataLoader = DataLoader(
                dataset=train_dataset, **train_kwargs
            )

            test_dataloader: DataLoader = DataLoader(
                dataset=test_dataset, **test_kwargs
            )

            data_loader_artifact: DataLoaderArtifact = DataLoaderArtifact(
                train_dataloader=train_dataloader, test_dataloader=test_dataloader
            )

            return data_loader_artifact

        except Exception as e:
            raise MNISTException(e, sys)
