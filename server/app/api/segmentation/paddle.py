import base64
import json
from datetime import datetime as dt
from io import BytesIO

import cv2
import numpy as np
from matplotlib import pyplot as plt, rcParams
from paddleocr import PaddleOCR
from PIL import Image

from app.schemas.doc_parser import ImageInput  # , ImageOutput
from app.util import NumpyEncoder

rcParams["figure.dpi"] = 300


ocr_engine = PaddleOCR(
    show_log=False,
    enforce_cpu=True,
    use_gpu=False,
    det_limit_side_len=1920,
    det_limit_type="max",
    det_db_thresh=0.3,
    det_db_box_thresh=0.5,
    det_db_unclip_ratio=1.5,
    use_dilation=True,
    det_db_score_mode="slow",
    # det_model_dir="/home/shine/.paddleocr/2.2.1/ocr/det/ch/ch_PP-OCRv2_det_infer",  # Defaults better
    # det_model_dir="/home/shine/.paddleocr/2.2.1/ocr/det/ch/ch_ppocr_server_v2.0_det_infer",
)


async def detect_text(img_in: ImageInput):
    # print("Calling `detect_image`")
    image = np.asarray(Image.open(BytesIO(base64.b64decode(str.encode(img_in.img_base64)))))

    ocr_engine.text_detector.args.det_db_only_bitmap = False
    result = ocr_engine.ocr(image, rec=False)

    # if result:
    #     for line in result:
    #         cv2.drawContours(image, [np.int0(line)], 0, (0, 0, 255), 3)

    # cv2.imwrite(f"{dt.now().timestamp()}.png", image)

    # print("Returning `detect_image`")
    return result


async def db_detect_bitmap(img_in: ImageInput):
    # print("Calling `detect_image`")
    image = np.asarray(Image.open(BytesIO(base64.b64decode(str.encode(img_in.img_base64)))))

    ocr_engine.text_detector.args.det_db_only_bitmap = True
    results = ocr_engine.ocr(image, rec=False)

    # pred = results["maps"]
    # pred = pred[:, 0, :, :]
    # plt.imsave("bitmap.png", pred[0])
    return NumpyEncoder.convert(results)
