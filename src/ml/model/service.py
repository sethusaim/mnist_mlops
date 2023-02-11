import io
import json
from typing import Any

import bentoml
import numpy as np
import torch
from bentoml import Service
from bentoml._internal.runner import Runner
from bentoml.io import JSON, Image
from numpy.typing import NDArray
from PIL import Image as PILImage
from torchvision.transforms import transforms

from src.constant import training_pipeline

bento_model = bentoml.pytorch.get(training_pipeline.MODEL_PUSHER_BENTOML_MODEL_NAME)

runner: Runner = bento_model.to_runner()

svc: Service = Service(
    name=training_pipeline.MODEL_PUSHER_BENTOML_SERVICE_NAME, runners=[runner]
)


@svc.api(input=Image(), output=JSON())
async def predict_image(f: PILImage) -> NDArray[Any]:
    b = io.BytesIO()

    f.save(b, "jpeg")

    my_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    image = PILImage.open(b)

    image = torch.from_numpy(np.array(my_transform(image).unsqueeze(0)))

    batch_ret = await runner.async_run(image)

    pred = torch.argmax(batch_ret, dim=1).detach().cpu().tolist()

    return json.dumps({"prediction": pred[0]})
