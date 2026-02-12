"""Welding Detector API - Main entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.api.routes import camera_router, recording_router, labeling_router, ml_router, defect_router
from app.services.camera_service import get_camera_service

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Welding Detector API...")
    
    camera = get_camera_service()
    health = await camera.health_check()
    
    if health.get("status") == "healthy":
        logger.info(f"Camera ready: {health.get('resolution')} @ {health.get('fps')}fps")
    else:
        logger.warning("Camera not available")
    
    yield
    
    logger.info("Shutting down...")
    camera.release()


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="API for welding process monitoring with USB camera.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "stream": "/camera/stream",
            "capture": "/camera/capture",
            "health": "/camera/health",
            "recording": "/recording/start | /recording/stop"
        }
    }


@app.get("/health")
async def health():
    camera = get_camera_service()
    camera_health = await camera.health_check()
    return {
        "status": camera_health.get("status"),
        "app": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "camera": camera_health
    }


app.include_router(camera_router)
app.include_router(recording_router)
app.include_router(labeling_router)
app.include_router(ml_router)
app.include_router(defect_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=settings.DEBUG)
