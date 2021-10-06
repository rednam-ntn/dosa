import base64
from io import BytesIO

import numpy as np
from docparser import stage1_entity_detector
from PIL import Image

from app.core.config import settings
from app.schemas.doc_parser import ImageInput  # , ImageOutput
from app.util import NumpyEncoder

entity_detector = stage1_entity_detector.EntityDetector()
entity_detector.init_model(model_log_dir=settings.LOG_DIR)


async def detect_image(img_in: ImageInput):
    # print("Calling `detect_image`")
    image = np.asarray(Image.open(BytesIO(base64.b64decode(str.encode(img_in.img_base64)))))

    # Detect entities
    entity_predictions = entity_detector.predict(image)

    converted = NumpyEncoder.convert(entity_predictions)
    # print("Returning `detect_image`")
    return converted
