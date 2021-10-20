from fastapi import APIRouter

from app.api.segmentation import doc_parser, paddle
from app.core.config import settings
from app.schemas.doc_parser import ImageInput, ImageOutput

router = APIRouter(prefix="/segment", tags=["segment"])


@router.post("/detect-image")
async def doc_parser_api(*, img_in: ImageInput):
    print("requesting `doc_parser_api`")
    return await doc_parser.detect_image(img_in)


@router.post("/detect-text")
async def paddle_api(*, img_in: ImageInput):
    print("requesting `paddle_text_api`")
    return await paddle.detect_text(img_in)


@router.post("/detect-bitmap")
async def paddle_bitmap_api(*, img_in: ImageInput):
    print("requesting `paddle_bitmap_api`")
    return await paddle.db_detect_bitmap(img_in)


@router.post("/detect-text-fast")
async def paddle_math_api(*, img_in: ImageInput):
    print("requesting `detect_text_fast`")
    return await paddle.detect_text_fast(img_in)
