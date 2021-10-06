import time
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request  # Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api import router as api_router
from app.core.config import settings, silence_packages_logger

BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))

silence_packages_logger()
root_router = APIRouter()
app = FastAPI(title="Doc Segment API", openapi_url="/openapi.json")
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")


@root_router.get("/", status_code=200)
def root(
    request: Request,
):
    """
    Root GET
    """
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


app.include_router(api_router)
app.include_router(root_router)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
