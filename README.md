# MNIST MLOps

## Points

1. The training can be run using train.py file. 
2. Inference script can be run using inference.py. 
3. Inference Dockerfile is Dockerfile
4. BentoML is used for model serving with prometheus integrated (exposed at /metrics route)
5. Deployment manifests can be found in manifests folder
6. MLFlow is integrated for logging of models and metrics

## How to run

1. Clone the repo 
```bash
git clone https://github.com/sethusaim/mnist_mlops.git
```

2. Setup MLFlow server 

Run this the seprate terminal
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://<your-bucket-name>/ --host 0.0.0.0 -p 8000
```

Export MLFLOW_TRACKING_URI 

```bash
export MLFLOW_TRACKING_URI=http://localhost:8000/
```

3. To run the training script
```bash
python train.py
```

The training script generates bentoml dockerfile and builds and image


4. Run the inference docker image
```bash
docker run -d -p 3000:3000 mnist-mlops
```

On http://localhost:3000/ url, we will get swagger ui, with predict route and prometheus metrics route 

## PostMan Request 

![postman_screenshot](https://user-images.githubusercontent.com/71321529/218254170-f6c6825b-3ff1-43f5-a105-e0b790ab4dc0.png)
