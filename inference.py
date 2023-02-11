
import os
import sys

from src.exception import MNISTException


def run_inference():
    try:        
        os.system("curl -F media=@0.png http://localhost:3000/predict_image")
        
    except Exception as e:
        raise MNISTException(e,sys)


if __name__ == "__main__":
    run_inference()

