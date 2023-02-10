from src.constant import training_pipeline
import torch


class PipelineConfig:
    def __init__(self):
        self.cuda_args: dict = training_pipeline.CUDA_ARGS

        self.seed: int = training_pipeline.SEED

        self.saved_model_name: str = training_pipeline.SAVED_MODEL_NAME

        self.device: torch.device = training_pipeline.DEVICE


class ModelTrainingConfig:
    def __init__(self):
        self.epochs: int = training_pipeline.EPOCHS

        self.log_interval: int = training_pipeline.LOG_INTERVAL

        self.test_batch_size: int = training_pipeline.TEST_BATCH_SIZE

        self.batch_size: int = training_pipeline.BATCH_SIZE

        self.scheduler_params: dict = training_pipeline.SCHEDULER_PARAMS

        self.optimizer_params: dict = training_pipeline.OPTIMIZER_PARAMS
