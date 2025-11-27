"""
Welding Detector API - Główny punkt wejścia aplikacji.

Aplikacja do detekcji krawędzi stołu za pomocą kamery MJPEG.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.api.routes import camera_router, edge_router, recording_router
from app.services.remote_camera_service import get_camera_service
from app.api.models import AppHealthResponse, HealthStatus

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager dla aplikacji FastAPI.
    Inicjalizuje i czyści zasoby przy starcie/zamknięciu.
    """
    # Startup
    logger.info("🚀 Starting Welding Detector API...")
    logger.info(f"📡 Camera server URL: {settings.CAMERA_SERVER_URL}")
    
    # Inicjalizacja camera service
    camera_service = get_camera_service()
    health = await camera_service.health_check()
    
    if health.get("status") == "healthy":
        logger.info("✅ Camera connection verified")
    else:
        logger.warning(f"⚠️ Camera not available: {health.get('error', 'unknown error')}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Welding Detector API...")


# Tworzenie aplikacji FastAPI
app = FastAPI(
    title=settings.APP_TITLE,
    description="""
## Welding Detector API

API do detekcji krawędzi stołu za pomocą kamery MJPEG.

### Funkcje:
- 📹 **Stream MJPEG** - podgląd na żywo z kamery (z timestampem)
- 📸 **Capture** - pojedyncze zdjęcia z timestampem
- 🔴 **Recording** - wskaźnik nagrywania (migająca czerwona kropka)
- 🔍 **Edge Detection** - wykrywanie krawędzi stołu (wkrótce)

### Architektura:
```
[Kamera USB] --> [Camera Server :8001] --> [Welding Detector API :8000]
```
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware (dla frontendów)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji ogranicz do konkretnych domen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== MAIN ROUTES ==============

@app.get(
    "/",
    summary="Root endpoint",
    description="Informacje o API"
)
async def root():
    """Główny endpoint z informacjami o API."""
    return {
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "camera_stream": "/camera/stream",
            "camera_capture": "/camera/capture",
            "camera_health": "/camera/health",
            "recording_start": "/recording/start",
            "recording_stop": "/recording/stop",
            "recording_status": "/recording/status",
            "edge_detect": "/edge/detect (coming soon)",
            "health": "/health"
        }
    }


@app.get(
    "/health",
    response_model=AppHealthResponse,
    summary="Health check",
    description="Sprawdza status aplikacji i połączenia z kamerą"
)
async def health_check():
    """
    Główny health check aplikacji.
    Sprawdza status API i połączenie z kamerą.
    """
    camera_service = get_camera_service()
    camera_health = await camera_service.health_check()
    
    # Określ ogólny status
    if camera_health.get("status") == "healthy":
        overall_status = HealthStatus.HEALTHY
    elif camera_health.get("has_cached_frame"):
        overall_status = HealthStatus.DEGRADED
    else:
        overall_status = HealthStatus.UNHEALTHY
    
    return AppHealthResponse(
        status=overall_status,
        app_name=settings.APP_TITLE,
        version=settings.APP_VERSION,
        camera_status={
            "status": camera_health.get("status", "error"),
            "camera_url": camera_health.get("camera_url", ""),
            "response_code": camera_health.get("response_code"),
            "error": camera_health.get("error"),
            "has_cached_frame": camera_health.get("has_cached_frame", False)
        }
    )


# ============== INCLUDE ROUTERS ==============

app.include_router(camera_router)
app.include_router(edge_router)
app.include_router(recording_router)


# ============== MAIN ENTRY POINT ==============

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )
