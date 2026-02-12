"""Machine Learning API routes - model training, prediction, and video analysis."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import Response

from app.services.frame_extractor_service import FrameExtractorService, get_frame_extractor_service
from app.services.ml_classification_service import MLClassificationService, get_ml_service
from app.services.video_analysis_service import VideoAnalysisService, get_video_analysis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Training status tracking (global state for background training)
_training_status = {
    "in_progress": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "history": None,
    "error": None
}

# Video analysis status tracking (per-video state)
_video_analysis_status = {}


@router.get("/info")
async def get_ml_info(ml: MLClassificationService = Depends(get_ml_service)):
    """Get ML model information and status."""
    info = ml.get_model_info()
    info["training_status"] = _training_status
    info["training_data_stats"] = ml.get_training_data_stats()
    return info


@router.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    epochs: int = Query(20, ge=5, le=100),
    batch_size: int = Query(16, ge=4, le=64),
    learning_rate: float = Query(0.001, ge=0.00001, le=0.1),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """Start model training in background (requires 20+ samples per class)."""
    global _training_status
    
    if _training_status["in_progress"]:
        raise HTTPException(400, "Training already in progress")
    
    stats = ml.get_training_data_stats()
    if not stats["ready_for_training"]:
        raise HTTPException(
            400,
            f"Insufficient training data. Need 20+ samples per class. "
            f"Current: OK={stats['ok_samples']}, NOK={stats['nok_samples']}"
        )
    
    _training_status = {
        "in_progress": True,
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": epochs,
        "history": None,
        "error": None
    }
    
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
            logger.error(f"Training failed: {e}", exc_info=True)
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


@router.get("/training-status")
async def get_training_status():
    """Get model training status."""
    return _training_status


@router.post("/predict/{filename}/frame/{frame_index}")
async def predict_frame(
    filename: str,
    frame_index: int,
    with_gradcam: bool = Query(True, description="Include Grad-CAM heatmap"),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """Classify frame as OK/NOK with confidence and optional Grad-CAM."""
    try:
        frame = extractor.get_frame(filename, frame_index)
        if frame is None:
            raise HTTPException(404, f"Frame {frame_index} not found in '{filename}'")
        
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
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(500, f"Prediction failed: {e}")


@router.get("/predict/{filename}/frame/{frame_index}/gradcam")
async def get_gradcam_overlay(
    filename: str,
    frame_index: int,
    alpha: float = Query(0.4, ge=0.0, le=1.0),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """Get image with Grad-CAM heatmap showing model attention areas."""
    import cv2
    
    try:
        frame = extractor.get_frame(filename, frame_index)
        if frame is None:
            raise HTTPException(404, f"Frame {frame_index} not found in '{filename}'")
        
        result = ml.predict(frame, with_gradcam=True)
        
        if result["gradcam_heatmap"] is None:
            raise HTTPException(400, "Grad-CAM not available")
        
        overlay = ml.create_gradcam_overlay(frame, result["gradcam_heatmap"], alpha=alpha)
        
        label = result["prediction"].upper()
        confidence = result["confidence"]
        color = (0, 255, 0) if label == "OK" else (0, 0, 255)
        cv2.putText(overlay, f"{label}: {confidence}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={"X-Prediction": label, "X-Confidence": str(confidence)}
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}", exc_info=True)
        raise HTTPException(500, f"Grad-CAM failed: {e}")


@router.post("/load")
async def load_model(
    model_name: str = Query("best_model.pth", description="Model filename"),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """Load saved model from file."""
    if ml.load_model(model_name):
        return {"status": "loaded", "model": model_name}
    raise HTTPException(404, f"Model not found: {model_name}")


@router.post("/export-onnx")
async def export_onnx_model(
    filename: str = Query("model.onnx"),
    ml: MLClassificationService = Depends(get_ml_service)
):
    """Export model to ONNX format for faster inference."""
    raise HTTPException(501, "ONNX export not implemented yet")


# Video analysis endpoints

@router.post("/analyze-video/{filename}")
async def analyze_video(
    filename: str,
    background_tasks: BackgroundTasks,
    skip_frames: int = Query(1, ge=1, le=10, description="Analyze every N-th frame"),
    analyze_defects: bool = Query(True, description="Classify defect types"),
    analysis: VideoAnalysisService = Depends(get_video_analysis_service)
):
    """Analyze entire video frame-by-frame: OK/NOK + defect classification."""
    global _video_analysis_status
    
    if filename in _video_analysis_status and _video_analysis_status[filename]["in_progress"]:
        raise HTTPException(400, f"Analysis already in progress for '{filename}'")
    
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
            logger.error(f"Video analysis failed for {filename}: {e}", exc_info=True)
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


@router.get("/analyze-video/{filename}/status")
async def get_analysis_status(filename: str):
    """Get video analysis status."""
    if filename not in _video_analysis_status:
        return {"status": "not_started"}
    
    state = _video_analysis_status[filename]
    
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
        return {
            "status": "completed",
            "progress": 100
        }


@router.get("/analyze-video/{filename}/results")
async def get_analysis_results(
    filename: str,
    analysis: VideoAnalysisService = Depends(get_video_analysis_service)
):
    """Get completed video analysis results."""
    results = analysis.get_analysis_results(filename)
    if not results:
        raise HTTPException(404, f"Analysis results not found for '{filename}'")
    
    return results


@router.get("/analyze-video/{filename}/defect-frames")
async def get_defect_frames(
    filename: str,
    defect_type: Optional[str] = Query(None, description="Filter by defect type"),
    analysis: VideoAnalysisService = Depends(get_video_analysis_service)
):
    """Get list of frames with detected defects."""
    frames = analysis.get_defect_frames(filename, defect_type)
    return {
        "filename": filename,
        "defect_type": defect_type,
        "frames": frames,
        "count": len(frames)
    }
