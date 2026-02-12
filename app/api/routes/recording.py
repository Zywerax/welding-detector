"""Recording API routes - video recording, frame extraction, motion detection."""

import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.services.camera_service import CameraService, get_camera_service
from app.services.frame_overlay_service import FrameOverlayService, get_overlay_service
from app.services.video_overlay_service import VideoOverlayService, get_video_overlay_service
from app.services.frame_extractor_service import FrameExtractorService, get_frame_extractor_service
from app.services.motion_detection_service import MotionDetectionService, get_motion_detection_service
from app.services.image_enhancement_service import (
    ImageEnhancementService,
    get_enhancement_service,
    EnhancementPreset,
    EnhancementParams
)
from app.api.models import (
    RecordingStatusResponse,
    RecordingStartResponse,
    RecordingStopResponse,
    RecordingListResponse,
    VideoInfoResponse,
    ExtractFramesRequest,
    ExtractFramesResponse,
    MotionAnalysisResponse,
    MotionSegmentResponse,
    TrimToMotionRequest,
    TrimToMotionResponse,
    EnhancementPresetEnum,
    EnhancementPresetsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recording", tags=["Recording"])


@router.get("/status", response_model=RecordingStatusResponse)
async def recording_status(camera: CameraService = Depends(get_camera_service)):
    return RecordingStatusResponse(
        is_recording=camera.is_recording,
        duration_seconds=camera.get_recording_duration(),
        frames=camera._frame_count
    )


@router.post("/start", response_model=RecordingStartResponse)
async def start_recording(
    camera: CameraService = Depends(get_camera_service),
    overlay: FrameOverlayService = Depends(get_overlay_service)
):
    if camera.is_recording:
        raise HTTPException(400, "Already recording")
    overlay.start_recording()
    return RecordingStartResponse(status="started", filename=camera.start_recording())


@router.post("/stop", response_model=RecordingStopResponse)
async def stop_recording(
    camera: CameraService = Depends(get_camera_service),
    overlay: FrameOverlayService = Depends(get_overlay_service)
):
    if not camera.is_recording:
        raise HTTPException(400, "Not recording")
    overlay.stop_recording()
    return RecordingStopResponse(status="stopped", **camera.stop_recording())


@router.get("/list", response_model=RecordingListResponse)
async def list_recordings(camera: CameraService = Depends(get_camera_service)):
    return RecordingListResponse(recordings=camera.list_recordings())


@router.get("/download/{filename}")
async def download_recording(filename: str, camera: CameraService = Depends(get_camera_service)):
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename, media_type="video/mp4")


@router.delete("/{filename}")
async def delete_recording(filename: str, camera: CameraService = Depends(get_camera_service)):
    if not camera.delete_recording(filename):
        raise HTTPException(404, "File not found")
    return {"status": "deleted", "filename": filename}


@router.put("/{filename}/note")
async def set_recording_note(
    filename: str,
    note: str = Query("", max_length=500),
    camera: CameraService = Depends(get_camera_service)
):
    if not camera.set_note(filename, note):
        raise HTTPException(404, "File not found")
    return {"status": "saved", "filename": filename, "note": note}


@router.post("/{filename}/apply-overlay")
async def apply_overlay(
    filename: str,
    start_time: Optional[str] = None,
    camera: CameraService = Depends(get_camera_service),
    overlay: VideoOverlayService = Depends(get_video_overlay_service)
):
    if not camera.get_recording_path(filename):
        raise HTTPException(404, "File not found")
    parsed_time = datetime.fromisoformat(start_time) if start_time else None
    if not overlay.process_video(filename, parsed_time):
        return {"status": "already_processing"}
    return {"status": "processing_started", "filename": filename}


@router.get("/{filename}/overlay-status")
async def overlay_status(filename: str, overlay: VideoOverlayService = Depends(get_video_overlay_service)):
    return overlay.get_status(filename) or {"status": "not_found"}


# Frame extraction endpoints

@router.get("/{filename}/info", response_model=VideoInfoResponse)
async def get_video_info(
    filename: str,
    camera: CameraService = Depends(get_camera_service),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service)
):
    """Get video file information (frame count, fps, resolution)."""
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    try:
        info = extractor.get_video_info(path)
        return VideoInfoResponse(filename=filename, **info)
    except Exception as e:
        raise HTTPException(500, f"Cannot read video info: {e}")


@router.post("/{filename}/extract-frames", response_model=ExtractFramesResponse)
async def extract_frames(
    filename: str,
    req: ExtractFramesRequest,
    camera: CameraService = Depends(get_camera_service),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service)
):
    """Extract frames from video and save as JPEG."""
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    base_name = Path(filename).stem
    output_folder = req.output_folder or f"frames/{base_name}"
    
    try:
        frames = extractor.extract_frames(path, step=req.step, max_frames=req.max_frames)
        saved = extractor.save_frames_to_folder(
            frames,
            output_folder,
            prefix=req.prefix,
            jpeg_quality=req.jpeg_quality
        )
        
        return ExtractFramesResponse(
            status="completed",
            filename=filename,
            frames_extracted=len(frames),
            output_folder=output_folder,
            files=[f.name for f in saved]
        )
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {e}")


@router.get("/{filename}/frame/{frame_index}")
async def get_single_frame(
    filename: str,
    frame_index: int,
    preset: Optional[EnhancementPresetEnum] = Query(None, description="Preset: original, weld_enhance, high_contrast, edge_overlay, heatmap, denoise"),
    clahe: Optional[float] = Query(None, description="CLAHE clip_limit (1.0-4.0)"),
    sharpen: Optional[float] = Query(None, description="Sharpen amount (0.5-3.0)"),
    gamma: Optional[float] = Query(None, description="Gamma (<1 darker, >1 brighter)"),
    contrast: Optional[float] = Query(None, description="Contrast (1.0-3.0)"),
    brightness: Optional[int] = Query(None, description="Brightness (-100 to 100)"),
    denoise: Optional[int] = Query(None, description="Denoise strength (5-15)"),
    edges: bool = Query(False, description="Overlay edge detection"),
    heatmap: Optional[str] = Query(None, description="Heatmap colormap: jet, hot, turbo, viridis"),
    camera: CameraService = Depends(get_camera_service),
    enhancer: ImageEnhancementService = Depends(get_enhancement_service)
):
    """Get single frame from video as JPEG with optional enhancement."""
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise HTTPException(500, "Cannot open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index < 0 or frame_index >= total_frames:
            cap.release()
            raise HTTPException(400, f"Frame index out of range (0-{total_frames-1})")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(500, "Cannot read frame")
        
        has_custom_params = any([clahe, sharpen, gamma, contrast, brightness, denoise, edges, heatmap])
        
        if preset or has_custom_params:
            if preset:
                params = enhancer.get_preset_params(EnhancementPreset(preset.value))
            else:
                params = EnhancementParams()
            
            if clahe is not None:
                params.clahe_enabled = True
                params.clahe_clip_limit = clahe
            if sharpen is not None:
                params.sharpen_enabled = True
                params.sharpen_amount = sharpen
            if gamma is not None:
                params.gamma_enabled = True
                params.gamma_value = gamma
            if contrast is not None:
                params.contrast_enabled = True
                params.contrast_alpha = contrast
            if brightness is not None:
                params.contrast_enabled = True
                params.contrast_beta = brightness
            if denoise is not None:
                params.denoise_enabled = True
                params.denoise_strength = denoise
            if edges:
                params.edge_overlay_enabled = True
            if heatmap:
                params.heatmap_enabled = True
                colormaps = enhancer.list_colormaps()
                params.heatmap_colormap = colormaps.get(heatmap, cv2.COLORMAP_JET)
            
            frame = enhancer.enhance(frame, params)
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(buffer.tobytes(), media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get frame: {e}")


@router.get("/enhancement/presets", response_model=EnhancementPresetsResponse)
async def list_enhancement_presets(enhancer: ImageEnhancementService = Depends(get_enhancement_service)):
    """List available image enhancement presets and options."""
    return EnhancementPresetsResponse(
        presets=enhancer.list_presets(),
        colormaps=list(enhancer.list_colormaps().keys()),
        edge_colors=["green", "red", "blue", "yellow", "cyan", "magenta"]
    )


# Motion detection endpoints

@router.get("/{filename}/detect-motion", response_model=MotionAnalysisResponse)
async def detect_motion(
    filename: str,
    threshold: int = 25,
    min_area_percent: float = 0.5,
    camera: CameraService = Depends(get_camera_service),
    motion: MotionDetectionService = Depends(get_motion_detection_service)
):
    """Analyze recording and detect motion segments."""
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    try:
        result = motion.detect_motion(path, threshold=threshold, min_area_percent=min_area_percent)
        return MotionAnalysisResponse(
            filename=result.filename,
            total_frames=result.total_frames,
            fps=result.fps,
            duration_seconds=result.duration_seconds,
            segments=[MotionSegmentResponse(
                start_frame=s.start_frame,
                end_frame=s.end_frame,
                start_time_ms=s.start_time_ms,
                end_time_ms=s.end_time_ms,
                duration_ms=s.duration_ms
            ) for s in result.segments],
            motion_percentage=result.motion_percentage
        )
    except Exception as e:
        raise HTTPException(500, f"Motion detection failed: {e}")


@router.post("/{filename}/trim-to-motion", response_model=TrimToMotionResponse)
async def trim_to_motion(
    filename: str,
    req: TrimToMotionRequest,
    camera: CameraService = Depends(get_camera_service),
    motion: MotionDetectionService = Depends(get_motion_detection_service)
):
    """Trim recording to motion segments only."""
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    output_path = None
    if req.output_filename:
        output_path = path.parent / req.output_filename
    
    try:
        result = motion.trim_to_motion(
            path,
            output_path=output_path,
            threshold=req.threshold,
            min_area_percent=req.min_area_percent,
            include_all_segments=req.include_all_segments
        )
        return TrimToMotionResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Trim failed: {e}")


@router.post("/{filename}/trim-to-postprocessing")
async def trim_to_postprocessing(
    filename: str,
    output_filename: Optional[str] = Query(None),
    brightness_threshold: int = Query(150, ge=100, le=255),
    min_bright_percent: float = Query(2.0, ge=0.5, le=20.0),
    camera: CameraService = Depends(get_camera_service),
    motion: MotionDetectionService = Depends(get_motion_detection_service)
):
    """Remove welding process (bright laser), keep only post-processing segment."""
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    output_path = None
    if output_filename:
        output_path = path.parent / output_filename
    
    try:
        result = motion.trim_to_post_processing(
            path,
            output_path=output_path,
            brightness_threshold=brightness_threshold,
            min_bright_percent=min_bright_percent
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Trim to post-processing failed: {e}")
