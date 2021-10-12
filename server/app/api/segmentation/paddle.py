import base64
from io import BytesIO

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from app.schemas.doc_parser import ImageInput  # , ImageOutput
from app.util import NumpyEncoder

ocr_engine = PaddleOCR(
    show_log=False,
    enforce_cpu=True,
    use_gpu=False,
    det_model_dir="/home/shine/.paddleocr/2.2.1/ocr/det/ch/ch_PP-OCRv2_det_infer",
)


async def detect_text(img_in: ImageInput):
    # print("Calling `detect_image`")
    image = np.asarray(Image.open(BytesIO(base64.b64decode(str.encode(img_in.img_base64)))))

    ocr_engine.text_detector.args.det_db_only_bitmap = False
    result = ocr_engine.ocr(image, rec=False)

    # if result:
    #     print(len(result))
    #     for line in result:
    #         cv2.drawContours(image, [np.int0(line)], 0, (0, 0, 255), 3)

    # cv2.imwrite("visualized.png", image)

    # print("Returning `detect_image`")
    return result


async def db_detect_bitmap(img_in: ImageInput):
    # print("Calling `detect_image`")
    image = np.asarray(Image.open(BytesIO(base64.b64decode(str.encode(img_in.img_base64)))))

    ocr_engine.text_detector.args.det_db_only_bitmap = True
    results = ocr_engine.ocr(image, rec=False)
    return NumpyEncoder.convert(results)
