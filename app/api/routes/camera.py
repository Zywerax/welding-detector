"""Camera API routes."""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response

from app.services.camera_service import CameraService, get_camera_service
from app.services.frame_overlay_service import FrameOverlayService, get_overlay_service
from app.api.models import CameraHealthResponse, HealthStatus, CameraSettingsRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/camera", tags=["Camera"])


@router.get("/stream")
async def stream_camera(camera: CameraService = Depends(get_camera_service)):
    return StreamingResponse(
        camera.stream_raw(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.get("/stream/overlay")
async def stream_camera_overlay(
    camera: CameraService = Depends(get_camera_service),
    overlay: FrameOverlayService = Depends(get_overlay_service)
):
    async def gen():
        async for frame in camera.stream_frames():
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + overlay.apply_overlay_to_jpeg(frame) + b'\r\n'
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.get("/capture")
async def capture_frame(
    overlay: bool = True,
    camera: CameraService = Depends(get_camera_service),
    overlay_svc: FrameOverlayService = Depends(get_overlay_service)
):
    frame = await camera.get_single_frame()
    if not frame:
        raise HTTPException(503, "Camera unavailable")
    return Response(overlay_svc.apply_overlay_to_jpeg(frame) if overlay else frame, media_type="image/jpeg")


@router.get("/health", response_model=CameraHealthResponse)
async def camera_health(camera: CameraService = Depends(get_camera_service)):
    data = await camera.health_check()
    return CameraHealthResponse(
        status=HealthStatus(data.get("status", "disconnected")),
        **{k: data.get(k) for k in ["camera_index", "fps", "resolution", "is_recording"]}
    )


@router.get("/monochrome")
async def get_monochrome(camera: CameraService = Depends(get_camera_service)):
    return {"monochrome": camera.monochrome}


@router.post("/monochrome")
async def set_monochrome(enabled: bool = Query(...), camera: CameraService = Depends(get_camera_service)):
    camera.monochrome = enabled
    return {"monochrome": camera.monochrome}


@router.get("/settings")
async def get_settings(camera: CameraService = Depends(get_camera_service)):
    return camera.get_settings()


@router.put("/settings")
async def update_settings(req: CameraSettingsRequest, camera: CameraService = Depends(get_camera_service)):
    return camera.apply_settings(**req.model_dump(exclude_none=True))


@router.post("/start")
async def start_camera(camera: CameraService = Depends(get_camera_service)):
    """Turn on the camera."""
    if camera._running:
        return {"status": "already_running", "running": True}
    
    camera._start_capture()
    return {"status": "started", "running": True}


@router.post("/stop")
async def stop_camera(camera: CameraService = Depends(get_camera_service)):
    """Turn off the camera."""
    if not camera._running:
        return {"status": "already_stopped", "running": False}
    
    camera._running = False
    time.sleep(0.2)
    
    if camera.cap:
        camera.cap.release()
        camera.cap = None
    
    return {"status": "stopped", "running": False}


@router.get("/running")
async def get_camera_status(camera: CameraService = Depends(get_camera_service)):
    """Check if the camera is running."""
    return {"running": camera._running}
