import json
from datetime import datetime as dt
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm

# font_path = "/home/shine/rednam/sandbox/Paddle/PaddleOCR/doc/fonts/simfang.ttf"
output_path = Path("./output/") / f"{dt.now().timestamp()}"
output_path.mkdir(exist_ok=True, parents=True)

sample_path = Path("/home/shine/rednam/source/mrcnn-doc-parser/data/sample/resized")

ocr_engine = PaddleOCR(show_log=True, enforce_cpu=True, use_gpu=False)

for img_path in tqdm(list(sample_path.glob("*.png"))):
    print(img_path.name)
    result_path = output_path / img_path.stem
    result_path.mkdir(exist_ok=True, parents=True)

    img = cv2.imread(str(img_path.absolute()))

    result = ocr_engine.ocr(img, rec=False)
    print(len(result))
    for line in result:
        cv2.drawContours(img, [np.int0(line)], 0, (0, 0, 255), 3)

    with open(str(result_path / "contours.json"), "w") as f:
        json.dump(result, f)

    copy2(img_path, result_path / "input.png")

    cv2.imwrite(str(result_path / "visualized.png"), img)
