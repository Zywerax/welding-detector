from pydantic_settings import BaseSettings #type: ignore


class Settings(BaseSettings):
    # USB Camera
    CAMERA_INDEX: int = 0
    CAMERA_USB_FPS: int = 60
    CAMERA_USB_WIDTH: int = 1280
    CAMERA_USB_HEIGHT: int = 720
    CAMERA_JPEG_QUALITY: int = 95

    # Application
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_TITLE: str = "Welding Detector API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
