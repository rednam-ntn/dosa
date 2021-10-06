import logging
from typing import List, Union

from pydantic import AnyHttpUrl, BaseSettings, DirectoryPath, validator


class Settings(BaseSettings):
    STATIC_DIR: DirectoryPath
    DATA_DIR: DirectoryPath
    LOG_DIR: DirectoryPath

    SERVER_URL: AnyHttpUrl

    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200"]'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


def silence_packages_logger() -> None:
    for package_name in [
        "PIL",
        "tensorflow",
    ]:
        _logger = logging.getLogger(package_name)
        _logger.setLevel(0)
        _logger.propagate = False


settings = Settings()
