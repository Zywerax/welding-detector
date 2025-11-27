"""
API Routes - Endpointy do obsługi kamery i detekcji krawędzi.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, Response
import logging

from app.services.remote_camera_service import RemoteCameraService, get_camera_service
from app.api.models import (
    CameraHealthResponse,
    CameraStatsResponse,
    ErrorResponse,
    HealthStatus
)

logger = logging.getLogger(__name__)

# Router dla endpointów kamery
camera_router = APIRouter(prefix="/camera", tags=["Camera"])

# Router dla endpointów detekcji krawędzi (na przyszłość)
edge_router = APIRouter(prefix="/edge", tags=["Edge Detection"])


# ============== CAMERA ENDPOINTS ==============

@camera_router.get(
    "/stream",
    summary="Stream MJPEG z kamery",
    description="Zwraca ciągły stream klatek w formacie MJPEG. Używaj do podglądu na żywo.",
    responses={
        200: {"content": {"multipart/x-mixed-replace": {}}},
        503: {"model": ErrorResponse, "description": "Kamera niedostępna"}
    }
)
async def stream_camera(
    camera_service: RemoteCameraService = Depends(get_camera_service)
):
    """
    Endpoint streamujący wideo z kamery w formacie MJPEG.
    
    Format streamu: multipart/x-mixed-replace z boundary=frame
    Każda klatka to osobny JPEG.
    """
    async def generate_mjpeg():
        async for frame in camera_service.stream_frames():
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
    
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@camera_router.get(
    "/capture",
    summary="Pobierz pojedynczą klatkę",
    description="Zwraca pojedynczy snapshot z kamery jako obraz JPEG.",
    responses={
        200: {"content": {"image/jpeg": {}}},
        503: {"model": ErrorResponse, "description": "Nie udało się pobrać klatki"}
    }
)
async def capture_frame(
    camera_service: RemoteCameraService = Depends(get_camera_service)
):
    """
    Endpoint zwracający pojedynczą klatkę z kamery.
    
    Przydatny do:
    - Robienia zdjęć
    - Testowania połączenia
    - Analizy pojedynczych klatek
    """
    frame = await camera_service.get_single_frame()
    
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail="Nie udało się pobrać klatki z kamery. Sprawdź połączenie z serwerem kamery."
        )
    
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@camera_router.get(
    "/health",
    response_model=CameraHealthResponse,
    summary="Status połączenia z kamerą",
    description="Sprawdza czy kamera jest dostępna i zwraca status połączenia."
)
async def camera_health(
    camera_service: RemoteCameraService = Depends(get_camera_service)
):
    """
    Health check dla połączenia z kamerą.
    
    Returns:
        CameraHealthResponse: Status połączenia z kamerą
    """
    health_data = await camera_service.health_check()
    
    return CameraHealthResponse(
        status=HealthStatus(health_data.get("status", "error")),
        camera_url=health_data.get("camera_url", ""),
        response_code=health_data.get("response_code"),
        error=health_data.get("error"),
        has_cached_frame=health_data.get("has_cached_frame", False)
    )


@camera_router.get(
    "/stats",
    summary="Statystyki kamery",
    description="Zwraca szczegółowe statystyki kamery (FPS, rozdzielczość, itp.)"
)
async def camera_stats(
    camera_service: RemoteCameraService = Depends(get_camera_service)
):
    """
    Pobiera statystyki z serwera kamery.
    
    Returns:
        dict: Statystyki kamery lub komunikat o błędzie
    """
    stats = await camera_service.get_camera_stats()
    
    if stats is None:
        raise HTTPException(
            status_code=503,
            detail="Nie udało się pobrać statystyk kamery"
        )
    
    return stats


# ============== EDGE DETECTION ENDPOINTS (placeholder) ==============

@edge_router.get(
    "/detect",
    summary="Wykryj krawędź stołu",
    description="Pobiera klatkę z kamery i wykrywa krawędź stołu. (W budowie)"
)
async def detect_edge():
    """
    Placeholder dla detekcji krawędzi.
    Zostanie zaimplementowane w następnym kroku.
    """
    return {
        "message": "Edge detection endpoint - coming soon!",
        "status": "not_implemented"
    }


@edge_router.get(
    "/stream",
    summary="Stream z zaznaczoną krawędzią",
    description="Stream MJPEG z nałożoną wizualizacją wykrytej krawędzi. (W budowie)"
)
async def stream_with_edge():
    """
    Placeholder dla streamu z detekcją krawędzi.
    """
    return {
        "message": "Edge detection stream - coming soon!",
        "status": "not_implemented"
    }
