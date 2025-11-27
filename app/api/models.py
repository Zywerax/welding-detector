"""
Pydantic models dla API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class HealthStatus(str, Enum):
    """Status zdrowia serwisu."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


class CameraHealthResponse(BaseModel):
    """Response dla health check kamery."""
    status: HealthStatus
    camera_url: str
    response_code: Optional[int] = None
    error: Optional[str] = None
    has_cached_frame: bool = False


class CameraStatsResponse(BaseModel):
    """Statystyki kamery."""
    camera_index: Optional[int] = None
    is_opened: bool = False
    consecutive_failures: int = 0
    has_cached_frame: bool = False
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    backend: Optional[float] = None


class AppHealthResponse(BaseModel):
    """Response dla głównego health check aplikacji."""
    status: HealthStatus
    app_name: str
    version: str
    camera_status: CameraHealthResponse


class EdgeCoordinate(BaseModel):
    """Koordynaty wykrytej krawędzi."""
    x1: int = Field(..., description="Początek X linii krawędzi")
    y1: int = Field(..., description="Początek Y linii krawędzi")
    x2: int = Field(..., description="Koniec X linii krawędzi")
    y2: int = Field(..., description="Koniec Y linii krawędzi")
    confidence: float = Field(..., ge=0, le=1, description="Pewność detekcji (0-1)")


class EdgeDetectionResponse(BaseModel):
    """Response z wynikiem detekcji krawędzi."""
    detected: bool = Field(..., description="Czy wykryto krawędź")
    edges: List[EdgeCoordinate] = Field(default_factory=list, description="Lista wykrytych krawędzi")
    primary_edge: Optional[EdgeCoordinate] = Field(None, description="Główna wykryta krawędź (najdłuższa)")
    processing_time_ms: float = Field(..., description="Czas przetwarzania w milisekundach")
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None


class EdgeDetectionConfig(BaseModel):
    """Konfiguracja algorytmu detekcji krawędzi."""
    canny_threshold1: int = Field(50, ge=0, le=255, description="Dolny próg Canny")
    canny_threshold2: int = Field(150, ge=0, le=255, description="Górny próg Canny")
    hough_threshold: int = Field(100, ge=1, description="Próg Hough Transform")
    min_line_length: int = Field(100, ge=1, description="Minimalna długość linii")
    max_line_gap: int = Field(10, ge=0, description="Maksymalna przerwa między segmentami linii")
    roi_top: float = Field(0.0, ge=0, le=1, description="ROI góra (0-1)")
    roi_bottom: float = Field(1.0, ge=0, le=1, description="ROI dół (0-1)")
    roi_left: float = Field(0.0, ge=0, le=1, description="ROI lewa (0-1)")
    roi_right: float = Field(1.0, ge=0, le=1, description="ROI prawa (0-1)")


class ErrorResponse(BaseModel):
    """Standardowy response dla błędów."""
    error: str
    detail: Optional[str] = None
    status_code: int


# ============== RECORDING MODELS ==============

class RecordingStatusResponse(BaseModel):
    """Response ze statusem nagrywania."""
    is_recording: bool = Field(..., description="Czy nagrywanie jest aktywne")
    duration_seconds: Optional[float] = Field(None, description="Czas nagrywania w sekundach")


class RecordingStartResponse(BaseModel):
    """Response po rozpoczęciu nagrywania."""
    status: str = Field(..., description="Status operacji")
    message: str = Field(..., description="Komunikat")


class RecordingStopResponse(BaseModel):
    """Response po zatrzymaniu nagrywania."""
    status: str = Field(..., description="Status operacji")
    duration_seconds: Optional[float] = Field(None, description="Całkowity czas nagrywania")
    message: str = Field(..., description="Komunikat")
