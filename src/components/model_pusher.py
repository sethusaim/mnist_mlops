import os
import sys

from src.entity.config_entity import ModelPusherConfig
from src.exception import MNISTException


class ModelPusher:
    def __init__(self):
        self.model_pusher_config: ModelPusherConfig = ModelPusherConfig()

    def build_bento_image(self):
        try:
            os.system("bentoml build")

            os.system(
                f"bentoml containerize {self.model_pusher_config.bento_model_service}:latest -t {self.model_pusher_config.bento_model_image}:latest"
            )

            os.system(
                f"cat '$(bentoml get {self.model_pusher_config.bento_model_service} -o path)/env/docker/Dockerfile' >> Dockerfile"
            )

        except Exception as e:
            raise MNISTException(e, sys)
