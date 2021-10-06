import base64
from io import BytesIO

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from app.schemas.doc_parser import ImageInput  # , ImageOutput

ocr_engine = PaddleOCR(show_log=True, enforce_cpu=True, use_gpu=False)


async def detect_text(img_in: ImageInput):
    # print("Calling `detect_image`")
    image = np.asarray(Image.open(BytesIO(base64.b64decode(str.encode(img_in.img_base64)))))

    result = ocr_engine.ocr(image, rec=False)

    # if result:
    #     print(len(result))
    #     for line in result:
    #         cv2.drawContours(image, [np.int0(line)], 0, (0, 0, 255), 3)

    # cv2.imwrite("visualized.png", image)

    # print("Returning `detect_image`")
    return result
