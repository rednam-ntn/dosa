from typing import List, Tuple

from pydantic import BaseModel


class ImageInput(BaseModel):
    img_base64: str
    file_name: str


class ImageOutput(BaseModel):
    class_name: str
    pred_score: float
    bbox_orig_coords: Tuple[int, int, int, int]
