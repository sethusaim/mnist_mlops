from src.constant import training_pipeline
import torch
import os


artifacts_dir: str = os.path.join(
    training_pipeline.ARTIFACTS_DIR, training_pipeline.TIMESTAMP
)


class PipelineConfig:
    def __init__(self):
        self.cuda_args: dict = training_pipeline.CUDA_ARGS

        self.seed: int = training_pipeline.SEED

        self.device: torch.device = training_pipeline.DEVICE


class DataIngestionConfig:
    def __init__(self):
        self.data_path: str = os.path.join(artifacts_dir, "data_ingestion")

        self.train_data_path: str = os.path.join(self.data_path, "train")

        self.test_data_path: str = os.path.join(self.data_path, "test")


class ModelTrainingConfig:
    def __init__(self):
        self.epochs: int = training_pipeline.EPOCHS

        self.log_interval: int = training_pipeline.LOG_INTERVAL

        self.test_batch_size: int = training_pipeline.TEST_BATCH_SIZE

        self.batch_size: int = training_pipeline.BATCH_SIZE

        self.scheduler_params: dict = training_pipeline.SCHEDULER_PARAMS

        self.optimizer_params: dict = training_pipeline.OPTIMIZER_PARAMS

        self.model_dir: str = os.path.join(artifacts_dir, "model_training")

        self.saved_model_path: str = os.path.join(
            self.model_dir, training_pipeline.SAVED_MODEL_NAME
        )


class ModelPusherConfig:
    def __init__(self):
        self.bento_model_service: str = (
            training_pipeline.MODEL_PUSHER_BENTOML_SERVICE_NAME
        )

        self.bento_model_image: str = training_pipeline.MODEL_PUSHER_BENTOML_IMAGE_NAME
