from typing import Tuple

from pydantic import BaseModel


class ImageInput(BaseModel):
    img_base64: str


class ImageOutput(BaseModel):
    class_name: str
    pred_score: float
    bbox_orig_coords: Tuple[int, int, int, int]
