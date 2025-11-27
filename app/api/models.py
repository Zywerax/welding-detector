"""
Pydantic models dla API.
"""

from pydantic import BaseModel
from typing import Optional
from enum import Enum


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


class CameraHealthResponse(BaseModel):
    status: HealthStatus
    camera_url: str
    response_code: Optional[int] = None
    error: Optional[str] = None
    has_cached_frame: bool = False


class AppHealthResponse(BaseModel):
    status: HealthStatus
    app_name: str
    version: str
    camera_status: CameraHealthResponse


class RecordingStatusResponse(BaseModel):
    is_recording: bool
    duration_seconds: Optional[float] = None


class RecordingStartResponse(BaseModel):
    status: str
    message: str


class RecordingStopResponse(BaseModel):
    status: str
    duration_seconds: Optional[float] = None
    message: str
