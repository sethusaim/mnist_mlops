from datetime import datetime
import torch

TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

MLFLOW_TRACKING_URI_KEY: str = "MLFLOW_TRACKING_URI"

EPOCHS: int = 1

SEED: int = 1

LOG_INTERVAL: int = 10

TEST_BATCH_SIZE: int = 1000

BATCH_SIZE: int = 64

CUDA_ARGS = {"num_workers": 1, "pin_memory": True, "shuffle": True}

OPTIMIZER_PARAMS: dict = {"lr": 1.0}

SCHEDULER_PARAMS = {"step_size": 1, "gamma": 0.7}

SAVED_MODEL_NAME: str = "mnist_model.pt"

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS_DIR: str = "artifacts"

MODEL_PUSHER_BENTOML_MODEL_NAME: str = "mnist_pytorch_model"

MODEL_PUSHER_BENTOML_SERVICE_NAME: str = "mnist_model_service"
