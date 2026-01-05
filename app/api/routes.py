"""API Routes - Camera, Recording, Edge Detection."""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response, FileResponse
from datetime import datetime
from typing import Optional

from app.services.camera_service import CameraService, get_camera_service
from app.services.frame_overlay_service import FrameOverlayService, get_overlay_service
from app.services.video_overlay_service import VideoOverlayService, get_video_overlay_service
from app.api.models import (
    CameraHealthResponse, HealthStatus,
    RecordingStatusResponse, RecordingStartResponse, RecordingStopResponse, RecordingListResponse,
    CameraSettingsRequest
)

camera_router = APIRouter(prefix="/camera", tags=["Camera"])
edge_router = APIRouter(prefix="/edge", tags=["Edge Detection"])
recording_router = APIRouter(prefix="/recording", tags=["Recording"])


# ============== CAMERA ==============

@camera_router.get("/stream")
async def stream_camera(camera: CameraService = Depends(get_camera_service)):
    return StreamingResponse(camera.stream_raw(), media_type="multipart/x-mixed-replace; boundary=frame",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@camera_router.get("/stream/overlay")
async def stream_camera_overlay(camera: CameraService = Depends(get_camera_service), overlay: FrameOverlayService = Depends(get_overlay_service)):
    async def gen():
        async for frame in camera.stream_frames():
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + overlay.apply_overlay_to_jpeg(frame) + b'\r\n'
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@camera_router.get("/capture")
async def capture_frame(overlay: bool = True, camera: CameraService = Depends(get_camera_service), overlay_svc: FrameOverlayService = Depends(get_overlay_service)):
    frame = await camera.get_single_frame()
    if not frame:
        raise HTTPException(503, "Camera unavailable")
    return Response(overlay_svc.apply_overlay_to_jpeg(frame) if overlay else frame, media_type="image/jpeg")


@camera_router.get("/health", response_model=CameraHealthResponse)
async def camera_health(camera: CameraService = Depends(get_camera_service)):
    data = await camera.health_check()
    return CameraHealthResponse(status=HealthStatus(data.get("status", "disconnected")), **{k: data.get(k) for k in ["camera_index", "fps", "resolution", "is_recording"]})


@camera_router.get("/monochrome")
async def get_monochrome(camera: CameraService = Depends(get_camera_service)):
    return {"monochrome": camera.monochrome}


@camera_router.post("/monochrome")
async def set_monochrome(enabled: bool = Query(...), camera: CameraService = Depends(get_camera_service)):
    camera.monochrome = enabled
    return {"monochrome": camera.monochrome}


@camera_router.get("/settings")
async def get_settings(camera: CameraService = Depends(get_camera_service)):
    return camera.get_settings()


@camera_router.put("/settings")
async def update_settings(req: CameraSettingsRequest, camera: CameraService = Depends(get_camera_service)):
    return camera.apply_settings(**req.model_dump(exclude_none=True))


# ============== EDGE ==============

@edge_router.get("/detect")
async def detect_edge():
    return {"status": "not_implemented"}


# ============== RECORDING ==============

@recording_router.get("/status", response_model=RecordingStatusResponse)
async def recording_status(camera: CameraService = Depends(get_camera_service)):
    return RecordingStatusResponse(is_recording=camera.is_recording, duration_seconds=camera.get_recording_duration(), frames=camera._frame_count)


@recording_router.post("/start", response_model=RecordingStartResponse)
async def start_recording(camera: CameraService = Depends(get_camera_service), overlay: FrameOverlayService = Depends(get_overlay_service)):
    if camera.is_recording:
        raise HTTPException(400, "Already recording")
    overlay.start_recording()
    return RecordingStartResponse(status="started", filename=camera.start_recording())


@recording_router.post("/stop", response_model=RecordingStopResponse)
async def stop_recording(camera: CameraService = Depends(get_camera_service), overlay: FrameOverlayService = Depends(get_overlay_service)):
    if not camera.is_recording:
        raise HTTPException(400, "Not recording")
    overlay.stop_recording()
    return RecordingStopResponse(status="stopped", **camera.stop_recording())


@recording_router.get("/list", response_model=RecordingListResponse)
async def list_recordings(camera: CameraService = Depends(get_camera_service)):
    return RecordingListResponse(recordings=camera.list_recordings())


@recording_router.get("/download/{filename}")
async def download_recording(filename: str, camera: CameraService = Depends(get_camera_service)):
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename, media_type="video/mp4")


@recording_router.delete("/{filename}")
async def delete_recording(filename: str, camera: CameraService = Depends(get_camera_service)):
    if not camera.delete_recording(filename):
        raise HTTPException(404, "File not found")
    return {"status": "deleted", "filename": filename}


@recording_router.put("/{filename}/note")
async def set_recording_note(filename: str, note: str = Query("", max_length=500), camera: CameraService = Depends(get_camera_service)):
    if not camera.set_note(filename, note):
        raise HTTPException(404, "File not found")
    return {"status": "saved", "filename": filename, "note": note}

@recording_router.post("/{filename}/apply-overlay")
async def apply_overlay(filename: str, start_time: Optional[str] = None, camera: CameraService = Depends(get_camera_service), overlay: VideoOverlayService = Depends(get_video_overlay_service)):
    if not camera.get_recording_path(filename):
        raise HTTPException(404, "File not found")
    parsed_time = datetime.fromisoformat(start_time) if start_time else None
    if not overlay.process_video(filename, parsed_time):
        return {"status": "already_processing"}
    return {"status": "processing_started", "filename": filename}


@recording_router.get("/{filename}/overlay-status")
async def overlay_status(filename: str, overlay: VideoOverlayService = Depends(get_video_overlay_service)):
    return overlay.get_status(filename) or {"status": "not_found"}
