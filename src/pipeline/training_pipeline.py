import sys

import torch
from torch import nn
from torch.optim import Adadelta, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler

from src.components.data_ingestion import DataIngestion
from src.components.data_loader import MNISTDataLoader
from src.components.model_evaluation import ModelEvaluation
from src.components.model_training import ModelTrainer
from src.entity.artifact_entity import DataIngestionArtifact, DataLoaderArtifact
from src.entity.config_entity import ModelTrainingConfig, PipelineConfig
from src.exception import MNISTException
from src.ml.model.arch import Net
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        self.model_trainer_config: ModelTrainingConfig = ModelTrainingConfig()

        self.pipeline_config: PipelineConfig = PipelineConfig()

        self.model_trainer: ModelTrainer = ModelTrainer()

        self.data_loader: MNISTDataLoader = MNISTDataLoader()

        self.model_evaluation: ModelEvaluation = ModelEvaluation()

        self.data_ingestion: DataIngestion = DataIngestion()

    def run_pipeline(self):
        """
        We are trying to train a model using the MNIST dataset
        """
        logging.info("Entered run_pipeline method of TrainPipeline class")

        try:
            data_ingestion_artifact: DataIngestionArtifact = (
                self.data_ingestion.initiate_data_ingestion()
            )

            logging.info(f"Data Ingestion artifact is : {data_ingestion_artifact}")

            data_loader_artifact: DataLoaderArtifact = self.data_loader.get_dataloaders(
                train_dataset=data_ingestion_artifact.train_dataset,
                test_dataset=data_ingestion_artifact.test_dataset,
            )

            logging.info(f"Data Loader Artifact is {data_loader_artifact}")

            torch.manual_seed(self.pipeline_config.seed)

            model: nn.Module = Net().to(self.pipeline_config.device)

            logging.info(
                f"Initialized the model architecture at {self.pipeline_config.device}"
            )

            optimizer: Optimizer = Adadelta(
                model.parameters(), **self.model_trainer_config.optimizer_params
            )

            logging.info(
                f"Configured optimizer with params as {self.model_trainer_config.optimizer_params}"
            )

            scheduler: _LRScheduler = StepLR(
                optimizer, **self.model_trainer_config.scheduler_params
            )

            logging.info(
                f"Configured scheduler with params as {self.model_trainer_config.scheduler_params}"
            )

            logging.info("Started model training")

            for epoch in range(1, self.model_trainer_config.epochs + 1):
                self.model_trainer.train(
                    model=model,
                    train_loader=data_loader_artifact.train_dataloader,
                    optimizer=optimizer,
                    epoch=epoch,
                )

                self.model_evaluation.test(
                    model=model, test_loader=data_loader_artifact.test_dataloader
                )

                scheduler.step()

            logging.info("Completed model training")

            torch.save(model, self.model_trainer_config.saved_model_path)

            logging.info(
                f"Saved the trained model to {self.model_trainer_config.saved_model_path}"
            )

        except Exception as e:
            raise MNISTException(e, sys)
