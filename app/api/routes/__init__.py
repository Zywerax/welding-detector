"""API routes module - exports all routers."""

from app.api.routes.camera import router as camera_router
from app.api.routes.recording import router as recording_router
from app.api.routes.labeling import router as labeling_router
from app.api.routes.ml import router as ml_router
from app.api.routes.defects import router as defect_router

__all__ = [
    "camera_router",
    "recording_router",
    "labeling_router",
    "ml_router",
    "defect_router"
]
