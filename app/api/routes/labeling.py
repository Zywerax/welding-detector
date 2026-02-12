"""Labeling API routes - frame labeling for training data."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from app.services.camera_service import CameraService, get_camera_service
from app.services.labeling_service import LabelingService, get_labeling_service
from app.api.models import (
    AddLabelRequest,
    FrameLabelResponse,
    LabelingStatsResponse,
    TrainingDataResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/labeling", tags=["Labeling"])


@router.post("/{filename}/frame/{frame_index}", response_model=FrameLabelResponse)
async def add_label(
    filename: str,
    frame_index: int,
    req: AddLabelRequest,
    camera: CameraService = Depends(get_camera_service),
    labeling: LabelingService = Depends(get_labeling_service)
):
    """Add OK/NOK/SKIP label to a frame and save to training folder."""
    path = camera.get_recording_path(filename)
    if not path:
        raise HTTPException(404, f"Recording '{filename}' not found")
    
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
        logger.error(f"Failed to add label: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to add label: {e}")


@router.get("/{filename}/frame/{frame_index}", response_model=Optional[FrameLabelResponse])
async def get_label(
    filename: str,
    frame_index: int,
    labeling: LabelingService = Depends(get_labeling_service)
):
    """Get label for a frame (returns null if none exists)."""
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


@router.delete("/{filename}/frame/{frame_index}")
async def remove_label(
    filename: str,
    frame_index: int,
    labeling: LabelingService = Depends(get_labeling_service)
):
    """Remove label from a frame."""
    if labeling.remove_label(filename, frame_index):
        return {"status": "deleted"}
    raise HTTPException(404, "Label not found")


@router.get("/{filename}/labels")
async def get_video_labels(
    filename: str,
    labeling: LabelingService = Depends(get_labeling_service)
):
    """Get all labels for a video."""
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


@router.get("/stats", response_model=LabelingStatsResponse)
async def get_labeling_stats(labeling: LabelingService = Depends(get_labeling_service)):
    """Get labeling statistics."""
    stats = labeling.get_stats()
    return LabelingStatsResponse(
        total_labeled=stats.total_labeled,
        ok_count=stats.ok_count,
        nok_count=stats.nok_count,
        skip_count=stats.skip_count,
        videos_labeled=stats.videos_labeled,
        defect_counts=stats.defect_counts
    )


@router.get("/training-data", response_model=TrainingDataResponse)
async def get_training_data_info(labeling: LabelingService = Depends(get_labeling_service)):
    """Get training data information."""
    return TrainingDataResponse(**labeling.export_for_training())
