"""API Routes - Camera, Recording, Edge Detection."""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, Response, FileResponse
from datetime import datetime
from typing import Optional
from pathlib import Path

from app.services.camera_service import CameraService, get_camera_service
from app.services.frame_overlay_service import FrameOverlayService, get_overlay_service
from app.services.video_overlay_service import VideoOverlayService, get_video_overlay_service
from app.services.frame_extractor_service import FrameExtractorService, get_frame_extractor_service
from app.services.motion_detection_service import MotionDetectionService, get_motion_detection_service
from app.services.image_enhancement_service import (
    ImageEnhancementService, get_enhancement_service,
    EnhancementPreset, EnhancementParams
)
from app.services.labeling_service import LabelingService, get_labeling_service
from app.services.ml_classification_service import MLClassificationService, get_ml_service
from app.services.defect_classifier_service import DefectClassifierService, get_defect_classifier_service
from app.services.video_analysis_service import VideoAnalysisService, get_video_analysis_service
from app.api.models import (
    CameraHealthResponse, HealthStatus,
    RecordingStatusResponse, RecordingStartResponse, RecordingStopResponse, RecordingListResponse,
    CameraSettingsRequest,
    VideoInfoResponse, ExtractFramesRequest, ExtractFramesResponse, FrameResponse,
    MotionAnalysisResponse, MotionSegmentResponse, TrimToMotionRequest, TrimToMotionResponse,
    EnhancementPresetEnum, ImageEnhancementParams, EnhancementPresetsResponse,
    LabelType, AddLabelRequest, FrameLabelResponse, LabelingStatsResponse, TrainingDataResponse
)

logger = logging.getLogger(__name__)

camera_router = APIRouter(prefix="/camera", tags=["Camera"])
edge_router = APIRouter(prefix="/edge", tags=["Edge Detection"])
recording_router = APIRouter(prefix="/recording", tags=["Recording"])
labeling_router = APIRouter(prefix="/labeling", tags=["Labeling"])
ml_router = APIRouter(prefix="/ml", tags=["Machine Learning"])
defect_router = APIRouter(prefix="/defects", tags=["Defect Classification"])


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


@camera_router.post("/start")
async def start_camera(camera: CameraService = Depends(get_camera_service)):
    """Włącz kamerę"""
    if camera._running:
        return {"status": "already_running", "running": True}
    
    camera._start_capture()
    return {"status": "started", "running": True}


@camera_router.post("/stop")
async def stop_camera(camera: CameraService = Depends(get_camera_service)):
    """Wyłącz kamerę"""
    if not camera._running:
        return {"status": "already_stopped", "running": False}
    
    camera._running = False
    time.sleep(0.2)  # Wait for capture loop to stop
    
    if camera.cap:
        camera.cap.release()
        camera.cap = None
    
    return {"status": "stopped", "running": False}


@camera_router.get("/running")
async def get_camera_status(camera: CameraService = Depends(get_camera_service)):
    """Sprawdź czy kamera jest włączona"""
    return {"running": camera._running}


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


# ============== FRAME EXTRACTION ==============

@recording_router.get("/{filename}/info", response_model=VideoInfoResponse)
async def get_video_info(
    filename: str,
    camera: CameraService = Depends(get_camera_service),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service)
):
    """Pobiera informacje o pliku wideo (liczba klatek, fps, rozdzielczość)."""
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    try:
        info = extractor.get_video_info(path)
        return VideoInfoResponse(filename=filename, **info)
    except Exception as e:
        raise HTTPException(500, f"Cannot read video info: {e}")


@recording_router.post("/{filename}/extract-frames", response_model=ExtractFramesResponse)
async def extract_frames(
    filename: str,
    req: ExtractFramesRequest,
    camera: CameraService = Depends(get_camera_service),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service)
):
    """
    Ekstrahuje klatki z nagrania i zapisuje jako JPEG.
    
    - step: co która klatka (1 = każda, 2 = co druga, itd.)
    - max_frames: limit klatek do wyekstrahowania
    - output_folder: folder docelowy (domyślnie: frames/{filename}/)
    """
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    # Ustal folder docelowy
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


@recording_router.get("/{filename}/frame/{frame_index}")
async def get_single_frame(
    filename: str,
    frame_index: int,
    # Preset (szybki wybór)
    preset: Optional[EnhancementPresetEnum] = Query(None, description="Preset: original, weld_enhance, high_contrast, edge_overlay, heatmap, denoise"),
    # Ręczne parametry (nadpisują preset)
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
    """
    Pobiera pojedynczą klatkę z nagrania jako JPEG z opcjonalnym przetwarzaniem.
    
    **Presety:**
    - `original` - bez zmian
    - `weld_enhance` - najlepszy dla spawów (CLAHE + sharpen + denoise)
    - `high_contrast` - mocny kontrast
    - `edge_overlay` - kolorowe krawędzie spawu
    - `heatmap` - pseudokolory
    - `denoise` - redukcja szumu
    
    **Przykłady:**
    - `/frame/100?preset=weld_enhance` - użyj presetu
    - `/frame/100?clahe=2.5&sharpen=1.5` - ręczne parametry
    - `/frame/100?preset=weld_enhance&gamma=1.3` - preset + dostrojenie
    """
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
        
        # Zastosuj przetwarzanie obrazu
        has_custom_params = any([clahe, sharpen, gamma, contrast, brightness, denoise, edges, heatmap])
        
        if preset or has_custom_params:
            # Zacznij od presetu lub pustych parametrów
            if preset:
                params = enhancer.get_preset_params(EnhancementPreset(preset.value))
            else:
                params = EnhancementParams()
            
            # Nadpisz ręcznymi parametrami
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
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return Response(buffer.tobytes(), media_type="image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get frame: {e}")


@recording_router.get("/enhancement/presets", response_model=EnhancementPresetsResponse)
async def list_enhancement_presets(enhancer: ImageEnhancementService = Depends(get_enhancement_service)):
    """Zwraca listę dostępnych presetów i opcji przetwarzania obrazu."""
    return EnhancementPresetsResponse(
        presets=enhancer.list_presets(),
        colormaps=list(enhancer.list_colormaps().keys()),
        edge_colors=["green", "red", "blue", "yellow", "cyan", "magenta"]
    )


# ============== MOTION DETECTION ==============

@recording_router.get("/{filename}/detect-motion", response_model=MotionAnalysisResponse)
async def detect_motion(
    filename: str,
    threshold: int = 25,
    min_area_percent: float = 0.5,
    camera: CameraService = Depends(get_camera_service),
    motion: MotionDetectionService = Depends(get_motion_detection_service)
):
    """
    Analizuje nagranie i wykrywa segmenty z ruchem kamery/obiektu.
    
    - threshold: próg różnicy pikseli (0-255), wyższy = mniej czuły
    - min_area_percent: minimalny % powierzchni ze zmianą
    """
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


@recording_router.post("/{filename}/trim-to-motion", response_model=TrimToMotionResponse)
async def trim_to_motion(
    filename: str,
    req: TrimToMotionRequest,
    camera: CameraService = Depends(get_camera_service),
    motion: MotionDetectionService = Depends(get_motion_detection_service)
):
    """
    Przycina nagranie do segmentów z wykrytym ruchem.
    
    Tworzy nowy plik zawierający tylko fragmenty z ruchem kamery.
    """
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    # Ustal ścieżkę wyjściową
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


@recording_router.post("/{filename}/trim-to-postprocessing")
async def trim_to_postprocessing(
    filename: str,
    output_filename: Optional[str] = Query(None, description="Nazwa pliku wyjściowego"),
    brightness_threshold: int = Query(150, ge=100, le=255, description="Próg jasności dla detekcji spawania"),
    min_bright_percent: float = Query(2.0, ge=0.5, le=20.0, description="Minimalny % jasnych pikseli"),
    camera: CameraService = Depends(get_camera_service),
    motion: MotionDetectionService = Depends(get_motion_detection_service)
):
    """
    Przycina wideo usuwając proces spawania (jasne światło lasera).
    Zostawia tylko post-processing - fragment po spawaniu.
    
    Wykrywa moment spawania na podstawie jasności obrazu (laser = jasne światło).
    """
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    # Ustal ścieżkę wyjściową
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


# ============== LABELING ==============

@labeling_router.post("/{filename}/frame/{frame_index}", response_model=FrameLabelResponse)
async def add_label(
    filename: str,
    frame_index: int,
    req: AddLabelRequest,
    camera: CameraService = Depends(get_camera_service),
    labeling: LabelingService = Depends(get_labeling_service)
):
    """
    Dodaje etykietę OK/NOK/SKIP do klatki.
    
    Automatycznie zapisuje klatkę do folderu treningowego (labels/training_data/ok lub nok).
    Dla NOK można podać defect_type określający typ wady.
    """
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, "File not found")
    
    try:
        label = labeling.add_label(
            video_filename=filename,
            frame_index=frame_index,
            label=req.label.value,
            defect_type=req.defect_type.value if req.defect_type else None,
            notes=req.notes,
            filters_used=req.filters_used
        )
        return FrameLabelResponse(
            video_filename=label.video_filename,
            frame_index=label.frame_index,
            label=label.label,
            defect_type=label.defect_type,
            timestamp=label.timestamp,
            notes=label.notes
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to add label: {e}")


@labeling_router.get("/{filename}/frame/{frame_index}", response_model=Optional[FrameLabelResponse])
async def get_label(
    filename: str,
    frame_index: int,
    labeling: LabelingService = Depends(get_labeling_service)
):
    """Pobiera etykietę dla klatki (lub null jeśli nie ma)."""
    label = labeling.get_label(filename, frame_index)
    if not label:
        return None
    return FrameLabelResponse(
        video_filename=label.video_filename,
        frame_index=label.frame_index,
        label=label.label,
        defect_type=label.defect_type,
        timestamp=label.timestamp,
        notes=label.notes
    )


@labeling_router.delete("/{filename}/frame/{frame_index}")
async def remove_label(
    filename: str,
    frame_index: int,
    labeling: LabelingService = Depends(get_labeling_service)
):
    """Usuwa etykietę z klatki."""
    if labeling.remove_label(filename, frame_index):
        return {"status": "deleted"}
    raise HTTPException(404, "Label not found")


@labeling_router.get("/{filename}/labels")
async def get_video_labels(
    filename: str,
    labeling: LabelingService = Depends(get_labeling_service)
):
    """Pobiera wszystkie etykiety dla danego wideo."""
    labels = labeling.get_labels_for_video(filename)
    return {
        "filename": filename,
        "labels": [
            {
                "frame_index": l.frame_index,
                "label": l.label,
                "defect_type": l.defect_type,
                "timestamp": l.timestamp,
                "notes": l.notes
            }
            for l in labels
        ],
        "count": len(labels)
    }


@labeling_router.get("/stats", response_model=LabelingStatsResponse)
async def get_labeling_stats(labeling: LabelingService = Depends(get_labeling_service)):
    """Zwraca statystyki etykietowania."""
    stats = labeling.get_stats()
    return LabelingStatsResponse(
        total_labeled=stats.total_labeled,
        ok_count=stats.ok_count,
        nok_count=stats.nok_count,
        skip_count=stats.skip_count,
        videos_labeled=stats.videos_labeled,
        defect_counts=stats.defect_counts
    )


@labeling_router.get("/training-data", response_model=TrainingDataResponse)
async def get_training_data_info(labeling: LabelingService = Depends(get_labeling_service)):
    """Zwraca informacje o danych treningowych."""
    return TrainingDataResponse(**labeling.export_for_training())


# ============== ML CLASSIFICATION ==============

# Zmienna do śledzenia statusu treningu
_training_status = {
    "in_progress": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "history": None,
    "error": None
}


@ml_router.get("/info")
async def get_ml_info(ml: MLClassificationService = Depends(get_ml_service)):
    """Informacje o modelu ML i statusie."""
    info = ml.get_model_info()
    info["training_status"] = _training_status
    info["training_data_stats"] = ml.get_training_data_stats()
    return info


@ml_router.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    epochs: int = Query(20, ge=5, le=100),
    batch_size: int = Query(16, ge=4, le=64),
    learning_rate: float = Query(0.001, ge=0.00001, le=0.1),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """
    Rozpocznij trening modelu w tle.
    Wymaga minimum 20 próbek OK i 20 próbek NOK.
    """
    global _training_status
    
    if _training_status["in_progress"]:
        raise HTTPException(400, "Training already in progress")
    
    # Sprawdź dane treningowe
    stats = ml.get_training_data_stats()
    if not stats["ready_for_training"]:
        raise HTTPException(
            400, 
            f"Insufficient training data. Need 20+ samples per class. "
            f"Current: OK={stats['ok_samples']}, NOK={stats['nok_samples']}"
        )
    
    # Reset status
    _training_status = {
        "in_progress": True,
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": epochs,
        "history": None,
        "error": None
    }
    
    # Trenuj w tle
    def train_in_background():
        global _training_status
        try:
            history = ml.train(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            _training_status["history"] = history
            _training_status["progress"] = 100
        except Exception as e:
            _training_status["error"] = str(e)
        finally:
            _training_status["in_progress"] = False
    
    background_tasks.add_task(train_in_background)
    
    return {
        "status": "training_started",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_data": stats
    }


@ml_router.get("/training-status")
async def get_training_status():
    """Status treningu modelu."""
    return _training_status


@ml_router.post("/predict/{filename}/frame/{frame_index}")
async def predict_frame(
    filename: str,
    frame_index: int,
    with_gradcam: bool = Query(True, description="Include Grad-CAM heatmap"),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """
    Klasyfikuj klatkę wideo jako OK/NOK.
    Zwraca predykcję, pewność i opcjonalnie Grad-CAM.
    """
    try:
        # Pobierz klatkę
        frame = extractor.get_frame(filename, frame_index)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        
        # Predykcja
        result = ml.predict(frame, with_gradcam=with_gradcam)
        
        return {
            "filename": filename,
            "frame_index": frame_index,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["class_probabilities"],
            "has_gradcam": result["gradcam_heatmap"] is not None
        }
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@ml_router.get("/predict/{filename}/frame/{frame_index}/gradcam")
async def get_gradcam_overlay(
    filename: str,
    frame_index: int,
    alpha: float = Query(0.4, ge=0.0, le=1.0),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """
    Pobierz obraz z nałożoną heatmapą Grad-CAM.
    Pokazuje obszary, na które model zwraca uwagę.
    """
    import cv2
    
    try:
        # Pobierz klatkę
        frame = extractor.get_frame(filename, frame_index)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        
        # Predykcja z Grad-CAM
        result = ml.predict(frame, with_gradcam=True)
        
        if result["gradcam_heatmap"] is None:
            raise HTTPException(400, "Grad-CAM not available")
        
        # Stwórz overlay
        overlay = ml.create_gradcam_overlay(frame, result["gradcam_heatmap"], alpha=alpha)
        
        # Dodaj tekst z predykcją
        label = result["prediction"].upper()
        confidence = result["confidence"]
        color = (0, 255, 0) if label == "OK" else (0, 0, 255)
        cv2.putText(overlay, f"{label}: {confidence}%", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Enkoduj do JPEG
        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"X-Prediction": label, "X-Confidence": str(confidence)}
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Grad-CAM failed: {e}")


@ml_router.post("/load")
async def load_model(
    model_name: str = Query("best_model.pth", description="Model filename"),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """Wczytaj zapisany model."""
    if ml.load_model(model_name):
        return {"status": "loaded", "model": model_name}
    raise HTTPException(404, f"Model not found: {model_name}")


@ml_router.post("/export-onnx")
async def export_onnx_model(
    filename: str = Query("model.onnx"),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """Eksportuj model do formatu ONNX dla szybszego inference."""
    raise HTTPException(501, "ONNX export not implemented yet")


# ============== DEFECT CLASSIFICATION ==============

_defect_training_status = {
    "in_progress": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "history": None,
    "error": None
}


@defect_router.get("/info")
async def get_defect_info(defect: DefectClassifierService = Depends(get_defect_classifier_service)):
    """Informacje o modelu klasyfikacji defektów."""
    info = defect.get_model_info()
    info["training_status"] = _defect_training_status
    info["training_data_stats"] = defect.get_training_data_stats()
    return info


@defect_router.post("/train")
async def train_defect_classifier(
    background_tasks: BackgroundTasks,
    epochs: int = Query(30, ge=10, le=100),
    batch_size: int = Query(16, ge=4, le=64),
    learning_rate: float = Query(0.001, ge=0.00001, le=0.1),
    defect: DefectClassifierService = Depends(get_defect_classifier_service)
):
    """
    Trenuj model klasyfikacji typów defektów w tle.
    Wymaga minimum 10 próbek każdego typu i 50 próbek łącznie.
    """
    global _defect_training_status
    
    if _defect_training_status["in_progress"]:
        raise HTTPException(400, "Defect classifier training already in progress")
    
    stats = defect.get_training_data_stats()
    if not stats["ready_for_training"]:
        raise HTTPException(
            400,
            f"Insufficient training data. Need 10+ samples per class and 50+ total. "
            f"Current: {stats['class_counts']}"
        )
    
    _defect_training_status = {
        "in_progress": True,
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": epochs,
        "history": None,
        "error": None
    }
    
    def train_in_background():
        global _defect_training_status
        try:
            history = defect.train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
            _defect_training_status["history"] = history
            _defect_training_status["progress"] = 100
        except Exception as e:
            _defect_training_status["error"] = str(e)
        finally:
            _defect_training_status["in_progress"] = False
    
    background_tasks.add_task(train_in_background)
    
    return {
        "status": "started",
        "epochs": epochs,
        "training_samples": stats["total_samples"],
        "class_counts": stats["class_counts"]
    }


@defect_router.post("/predict")
async def predict_defect_type(
    filename: str = Query(..., description="Video filename"),
    frame_index: int = Query(..., ge=0),
    camera: CameraService = Depends(get_camera_service),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service),
    defect: DefectClassifierService = Depends(get_defect_classifier_service)
):
    """Klasyfikuj typ defektu dla konkretnej klatki."""
    try:
        frame = extractor.get_frame(filename, frame_index)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        
        result = defect.predict(frame, with_gradcam=False)
        return result
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@defect_router.get("/predict/{filename}/frame/{frame_index}/gradcam")
async def get_defect_gradcam(
    filename: str,
    frame_index: int,
    camera: CameraService = Depends(get_camera_service),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service),
    defect: DefectClassifierService = Depends(get_defect_classifier_service)
):
    """Grad-CAM dla klasyfikacji typu defektu."""
    try:
        frame = extractor.get_frame(filename, frame_index)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        
        result = defect.predict(frame, with_gradcam=True)
        
        if result["gradcam_heatmap"] is None:
            raise HTTPException(400, "Grad-CAM not available")
        
        import cv2
        overlay = defect.create_gradcam_overlay(frame, result["gradcam_heatmap"])
        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Prediction": result["prediction"],
                "X-Confidence": str(result["confidence"])
            }
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Grad-CAM failed: {e}")


# ============== VIDEO ANALYSIS ==============

_video_analysis_status = {}  # filename -> status dict


@ml_router.post("/analyze-video/{filename}")
async def analyze_video(
    filename: str,
    background_tasks: BackgroundTasks,
    skip_frames: int = Query(1, ge=1, le=10, description="Analizuj co N-tą klatkę"),
    analyze_defects: bool = Query(True, description="Czy klasyfikować typy defektów"),
    analysis: VideoAnalysisService = Depends(get_video_analysis_service)
):
    """
    Analizuje całe wideo klatka po klatce: OK/NOK + typ defektu.
    Wykonuje się w tle, zwraca natychmiastowy status.
    """
    global _video_analysis_status
    
    if filename in _video_analysis_status and _video_analysis_status[filename]["in_progress"]:
        raise HTTPException(400, f"Analysis already in progress for {filename}")
    
    _video_analysis_status[filename] = {
        "in_progress": True,
        "progress": 0,
        "current_frame": 0,
        "total_frames": 0,
        "results": None,
        "error": None
    }
    
    def analyze_in_background():
        global _video_analysis_status
        try:
            def progress_callback(current, total, frame_result):
                _video_analysis_status[filename]["progress"] = int((current / total) * 100)
                _video_analysis_status[filename]["current_frame"] = current
                _video_analysis_status[filename]["total_frames"] = total
            
            results = analysis.analyze_video(
                filename=filename,
                analyze_defects=analyze_defects,
                skip_frames=skip_frames,
                progress_callback=progress_callback
            )
            _video_analysis_status[filename]["results"] = results
            _video_analysis_status[filename]["progress"] = 100
        except Exception as e:
            logger.error(f"Video analysis failed for {filename}: {e}")
            _video_analysis_status[filename]["error"] = str(e)
        finally:
            _video_analysis_status[filename]["in_progress"] = False
    
    background_tasks.add_task(analyze_in_background)
    
    return {
        "status": "started",
        "filename": filename,
        "skip_frames": skip_frames,
        "analyze_defects": analyze_defects
    }


@ml_router.get("/analyze-video/{filename}/status")
async def get_analysis_status(filename: str):
    """Pobierz status analizy wideo"""
    if filename not in _video_analysis_status:
        return {"status": "not_started"}
    
    state = _video_analysis_status[filename]
    
    # Format response for frontend
    if state["error"]:
        return {
            "status": "error",
            "error": state["error"],
            "progress": state.get("progress", 0)
        }
    elif state["in_progress"]:
        return {
            "status": "in_progress",
            "progress": state.get("progress", 0),
            "current_frame": state.get("current_frame", 0),
            "total_frames": state.get("total_frames", 0)
        }
    else:
        # Completed
        return {
            "status": "completed",
            "progress": 100
        }


@ml_router.get("/analyze-video/{filename}/results")
async def get_analysis_results(
    filename: str,
    analysis: VideoAnalysisService = Depends(get_video_analysis_service)
):
    """Pobierz wyniki analizy wideo"""
    results = analysis.get_analysis_results(filename)
    if not results:
        raise HTTPException(404, "Analysis results not found")
    
    return results


@ml_router.get("/analyze-video/{filename}/defect-frames")
async def get_defect_frames(
    filename: str,
    defect_type: Optional[str] = Query(None, description="Filtruj po typie defektu"),
    analysis: VideoAnalysisService = Depends(get_video_analysis_service)
):
    """Pobierz listę klatek z defektami"""
    frames = analysis.get_defect_frames(filename, defect_type)
    return {
        "filename": filename,
        "defect_type": defect_type,
        "frames": frames,
        "count": len(frames)
    }
