"""
API Routes - Endpointy do obsługi kamery i detekcji krawędzi.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
import logging

from app.services.remote_camera_service import RemoteCameraService, get_camera_service
from app.services.frame_overlay_service import FrameOverlayService, get_overlay_service
from app.api.models import (
    CameraHealthResponse,
    CameraStatsResponse,
    ErrorResponse,
    HealthStatus,
    RecordingStatusResponse,
    RecordingStartResponse,
    RecordingStopResponse
)

logger = logging.getLogger(__name__)

# Router dla endpointów kamery
camera_router = APIRouter(prefix="/camera", tags=["Camera"])

# Router dla endpointów detekcji krawędzi (na przyszłość)
edge_router = APIRouter(prefix="/edge", tags=["Edge Detection"])

# Router dla nagrywania
recording_router = APIRouter(prefix="/recording", tags=["Recording"])


# ============== CAMERA ENDPOINTS ==============

@camera_router.get(
    "/stream",
    summary="Stream MJPEG z kamery",
    description="Zwraca ciągły stream klatek w formacie MJPEG z timestampem i wskaźnikiem nagrywania.",
    responses={
        200: {"content": {"multipart/x-mixed-replace": {}}},
        503: {"model": ErrorResponse, "description": "Kamera niedostępna"}
    }
)
async def stream_camera(
    overlay: bool = Query(True, description="Czy nakładać overlay (timestamp + REC indicator)"),
    camera_service: RemoteCameraService = Depends(get_camera_service),
    overlay_service: FrameOverlayService = Depends(get_overlay_service)
):
    """
    Endpoint streamujący wideo z kamery w formacie MJPEG.
    
    Format streamu: multipart/x-mixed-replace z boundary=frame
    Każda klatka zawiera:
    - Timestamp (lewy górny róg)
    - Wskaźnik REC gdy nagrywanie aktywne (prawy górny róg, migająca czerwona kropka)
    """
    async def generate_mjpeg():
        async for frame in camera_service.stream_frames():
            if overlay:
                frame = overlay_service.apply_overlay_to_jpeg(frame)
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
    description="Zwraca pojedynczy snapshot z kamery jako obraz JPEG z timestampem.",
    responses={
        200: {"content": {"image/jpeg": {}}},
        503: {"model": ErrorResponse, "description": "Nie udało się pobrać klatki"}
    }
)
async def capture_frame(
    overlay: bool = Query(True, description="Czy nakładać overlay (timestamp + REC indicator)"),
    camera_service: RemoteCameraService = Depends(get_camera_service),
    overlay_service: FrameOverlayService = Depends(get_overlay_service)
):
    """
    Endpoint zwracający pojedynczą klatkę z kamery.
    
    Przydatny do:
    - Robienia zdjęć z timestampem
    - Testowania połączenia
    - Analizy pojedynczych klatek
    """
    frame = await camera_service.get_single_frame()
    
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail="Nie udało się pobrać klatki z kamery. Sprawdź połączenie z serwerem kamery."
        )
    
    if overlay:
        frame = overlay_service.apply_overlay_to_jpeg(frame)
    
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


# ============== RECORDING ENDPOINTS ==============

@recording_router.get(
    "/status",
    response_model=RecordingStatusResponse,
    summary="Status nagrywania",
    description="Sprawdza czy nagrywanie jest aktywne i zwraca czas trwania."
)
async def recording_status(
    overlay_service: FrameOverlayService = Depends(get_overlay_service)
):
    """
    Zwraca aktualny status nagrywania.
    """
    return RecordingStatusResponse(
        is_recording=overlay_service.is_recording,
        duration_seconds=overlay_service.get_recording_duration()
    )


@recording_router.post(
    "/start",
    response_model=RecordingStartResponse,
    summary="Rozpocznij nagrywanie",
    description="Rozpoczyna nagrywanie - aktywuje czerwoną kropkę na streamie."
)
async def start_recording(
    overlay_service: FrameOverlayService = Depends(get_overlay_service)
):
    """
    Rozpoczyna nagrywanie.
    Aktywuje wskaźnik REC (migająca czerwona kropka) na streamie.
    """
    if overlay_service.is_recording:
        raise HTTPException(
            status_code=400,
            detail="Nagrywanie już jest aktywne"
        )
    
    overlay_service.start_recording()
    
    return RecordingStartResponse(
        status="started",
        message="Nagrywanie rozpoczęte. Czerwona kropka będzie widoczna na streamie."
    )


@recording_router.post(
    "/stop",
    response_model=RecordingStopResponse,
    summary="Zatrzymaj nagrywanie",
    description="Zatrzymuje nagrywanie i zwraca czas trwania."
)
async def stop_recording(
    overlay_service: FrameOverlayService = Depends(get_overlay_service)
):
    """
    Zatrzymuje nagrywanie.
    Wyłącza wskaźnik REC na streamie.
    """
    if not overlay_service.is_recording:
        raise HTTPException(
            status_code=400,
            detail="Nagrywanie nie jest aktywne"
        )
    
    duration = overlay_service.stop_recording()
    
    return RecordingStopResponse(
        status="stopped",
        duration_seconds=duration,
        message=f"Nagrywanie zatrzymane. Czas trwania: {duration:.2f}s"
    )
