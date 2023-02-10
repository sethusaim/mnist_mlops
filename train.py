import sys

from src.exception import MNISTException
from src.pipeline.training_pipeline import TrainPipeline


def start_training():
    """
    It creates an instance of the TrainPipeline class, and then calls the run_pipeline method on that
    instance
    """
    try:
        tp = TrainPipeline()

        tp.run_pipeline()

    except Exception as e:
        raise MNISTException(e, sys)


if __name__ == "__main__":
    start_training()
