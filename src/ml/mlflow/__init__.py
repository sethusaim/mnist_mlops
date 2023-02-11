import sys

import mlflow

from src.configuration.mlflow_connection import MLFlowClient
from src.exception import MNISTException


class MLFlowOperation:
    def __init__(self):
        self.mlflow_client = MLFlowClient().client

        mlflow.set_tracking_uri(uri=self.mlflow_client.tracking_uri)

        mlflow.set_experiment(experiment_name="test")

    def log_model_params():
        try:
            pass

        except Exception as e:
            raise MNISTException(e, sys)
