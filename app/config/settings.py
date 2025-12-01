from pydantic_settings import BaseSettings # type: ignore
from typing import Optional


class Settings(BaseSettings):
    """
    Klasa konfiguracyjna ładująca zmienne środowiskowe z pliku .env
    """
    # Konfiguracja kamery USB
    CAMERA_INDEX: int = 0
    CAMERA_USB_FPS: int = 60
    CAMERA_USB_WIDTH: int = 1080
    CAMERA_USB_HEIGHT: int = 720

    # Konfiguracja kamery - docker
    CAMERA_SERVER_URL: str = "http://host.docker.internal:8001"
    
    # Konfiguracja aplikacji
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_TITLE: str = "Welding Detector API"
    APP_VERSION: str = "1.0.0"
    
    # Dodatkowe ustawienia (opcjonalne)
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton - jedna instancja dla całej aplikacji
settings = Settings()
