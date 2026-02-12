"""Defect Classification API routes - train and classify weld defect types."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import Response

from app.services.camera_service import CameraService, get_camera_service
from app.services.frame_extractor_service import FrameExtractorService, get_frame_extractor_service
from app.services.defect_classifier_service import DefectClassifierService, get_defect_classifier_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/defects", tags=["Defect Classification"])

# Training status tracking (global state for background training)
_defect_training_status = {
    "in_progress": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "history": None,
    "error": None
}


@router.get("/info")
async def get_defect_info(defect: DefectClassifierService = Depends(get_defect_classifier_service)):
    """Get defect classification model information."""
    info = defect.get_model_info()
    info["training_status"] = _defect_training_status
    info["training_data_stats"] = defect.get_training_data_stats()
    return info


@router.post("/train")
async def train_defect_classifier(
    background_tasks: BackgroundTasks,
    epochs: int = Query(30, ge=10, le=100),
    batch_size: int = Query(16, ge=4, le=64),
    learning_rate: float = Query(0.001, ge=0.00001, le=0.1),
    defect: DefectClassifierService = Depends(get_defect_classifier_service)
):
    """Train defect classifier in background (requires 10+ samples per class, 50+ total)."""
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
            logger.error(f"Defect training failed: {e}", exc_info=True)
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


@router.post("/predict")
async def predict_defect_type(
    filename: str = Query(..., description="Video filename"),
    frame_index: int = Query(..., ge=0),
    camera: CameraService = Depends(get_camera_service),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service),
    defect: DefectClassifierService = Depends(get_defect_classifier_service)
):
    """Classify defect type for a specific frame."""
    try:
        frame = extractor.get_frame(filename, frame_index)
        if frame is None:
            raise HTTPException(404, f"Frame {frame_index} not found in '{filename}'")
        
        result = defect.predict(frame, with_gradcam=False)
        return result
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Defect prediction failed: {e}", exc_info=True)
        raise HTTPException(500, f"Prediction failed: {e}")


@router.get("/predict/{filename}/frame/{frame_index}/gradcam")
async def get_defect_gradcam(
    filename: str,
    frame_index: int,
    camera: CameraService = Depends(get_camera_service),
    extractor: FrameExtractorService = Depends(get_frame_extractor_service),
    defect: DefectClassifierService = Depends(get_defect_classifier_service)
):
    """Get Grad-CAM visualization for defect type classification."""
    try:
        frame = extractor.get_frame(filename, frame_index)
        if frame is None:
            raise HTTPException(404, f"Frame {frame_index} not found in '{filename}'")
        
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
        logger.error(f"Defect Grad-CAM failed: {e}", exc_info=True)
        raise HTTPException(500, f"Grad-CAM failed: {e}")
