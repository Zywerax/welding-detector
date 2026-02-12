"""
Labeling Service - management of OK/NOK labels for video frames.
Saves labels as JSON + copies labeled frames to training folders.
"""

import cv2  # type: ignore
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


LabelType = Literal["ok", "nok", "skip"]
DefectType = Literal[
    "porosity",        
    "crack",          
    "lack_of_fusion", 
    "undercut",         
    "burn_through",   
    "spatter",         
    "irregular_bead",  
    "contamination",   
    "other"            
]

DEFECT_TYPES = [
    "porosity", "crack", "lack_of_fusion", "undercut", 
    "burn_through", "spatter", "irregular_bead", "contamination", "other"
]


@dataclass
class FrameLabel:
    """Label for a single frame."""
    video_filename: str
    frame_index: int
    label: LabelType
    timestamp: str
    defect_type: Optional[DefectType] = None  # Defect type (when label=nok)
    notes: str = ""
    filters_used: Optional[dict] = None  # Jakie filtry były użyte przy etykietowaniu


@dataclass 
class LabelingStats:
    """Labeling statistics."""
    total_labeled: int
    ok_count: int
    nok_count: int
    skip_count: int
    videos_labeled: int
    defect_counts: dict = field(default_factory=dict)  # Defect counts by type (e.g., {"porosity": 10, "crack": 5})


class LabelingService:
    """
    Service for managing weld labels.
    
    Folder structure:
        labels/
            labels.json          # All labels in JSON format
            training_data/
                ok/              # Frames labeled as OK
                    video1_frame100.jpg
                    video2_frame50.jpg
                nok/             # Frames labeled as NOK
                    video1_frame200.jpg
    """
    
    def __init__(
        self,
        labels_dir: Path = Path("labels"),
        recordings_dir: Path = Path("recordings")
    ):
        self.labels_dir = labels_dir
        self.recordings_dir = recordings_dir
        self.labels_file = labels_dir / "labels.json"
        self.training_dir = labels_dir / "training_data"
        
        # Create folder structure
        self.labels_dir.mkdir(exist_ok=True)
        (self.training_dir / "ok").mkdir(parents=True, exist_ok=True)
        (self.training_dir / "nok").mkdir(parents=True, exist_ok=True)
        
        # Folders for defect types (only for NOK frames)
        for defect in DEFECT_TYPES:
            (self.training_dir / "defects" / defect).mkdir(parents=True, exist_ok=True)
        
        # Load existing labels
        self._labels: dict[str, FrameLabel] = {}
        self._load_labels()
        
        logger.info(f"LabelingService initialized ({len(self._labels)} labels loaded)")
    
    def _get_label_key(self, video_filename: str, frame_index: int) -> str:
        """Generates a unique key for a label."""
        return f"{video_filename}:{frame_index}"
    
    def _load_labels(self):
        """Loads labels from the JSON file."""
        if self.labels_file.exists():
            try:
                with open(self.labels_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, label_data in data.items():
                        self._labels[key] = FrameLabel(**label_data)
            except Exception as e:
                logger.error(f"Failed to load labels: {e}")
    
    def _save_labels(self):
        """Saves labels to the JSON file."""
        try:
            data = {key: asdict(label) for key, label in self._labels.items()}
            with open(self.labels_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save labels: {e}")
            raise
    
    def add_label(
        self, 
        video_filename: str, # file name of the video (e.g., "weld1.mp4")
        frame_index: int, # index of the frame in the video (e.g., 150)
        label: LabelType, # "ok", "nok" or "skip"
        defect_type: Optional[DefectType] = None, # required if label="nok"
        notes: str = "", # optional notes
        filters_used: Optional[dict] = None, # which filters were used during labeling (e.g., {"edge_detection": "canny", "threshold": 100})
        save_frame: bool = True # whether to save the frame to the training folder
    ) -> FrameLabel: #returns the created label

        key = self._get_label_key(video_filename, frame_index)
        
        # delete old training image if label is being updated
        if key in self._labels:
            old_label = self._labels[key]
            if old_label.label != label or old_label.defect_type != defect_type:
                self._remove_training_image(video_filename, frame_index, old_label.label, old_label.defect_type)
        
        frame_label = FrameLabel(
            video_filename=video_filename,
            frame_index=frame_index,
            label=label,
            defect_type=defect_type if label == "nok" else None,
            timestamp=datetime.now().isoformat(),
            notes=notes,
            filters_used=filters_used
        )
        
        self._labels[key] = frame_label
        self._save_labels()
        
        #save_frame is True by default, but we can set it to False 
        # if we want to add labels without saving images (e.g., for skipped frames)
        if save_frame and label in ("ok", "nok"):
            self._save_training_image(video_filename, frame_index, label, defect_type)
        
        defect_info = f"({defect_type})" if defect_type else ""
        logger.info(f"Label added: {video_filename} frame {frame_index} = {label.upper()}{defect_info}")
        return frame_label
    
    def get_label(self, video_filename: str, frame_index: int) -> Optional[FrameLabel]:
        '''get label for a specific frame'''
        key = self._get_label_key(video_filename, frame_index)
        return self._labels.get(key)
    
    def get_labels_for_video(self, video_filename: str) -> list[FrameLabel]:
        """get all labels for a specific video"""
        return [
            label for label in self._labels.values()
            if label.video_filename == video_filename
        ]
    
    def remove_label(self, video_filename: str, frame_index: int) -> bool:
        """remove a label"""
        key = self._get_label_key(video_filename, frame_index)
        if key in self._labels:
            label = self._labels[key]
            self._remove_training_image(video_filename, frame_index, label.label, label.defect_type)
            del self._labels[key]
            self._save_labels()
            logger.info(f"Label removed: {video_filename} frame {frame_index}")
            return True
        return False
    
    def get_stats(self) -> LabelingStats:
        """get labeling statistics"""
        ok_count = sum(1 for l in self._labels.values() if l.label == "ok")
        nok_count = sum(1 for l in self._labels.values() if l.label == "nok")
        skip_count = sum(1 for l in self._labels.values() if l.label == "skip")
        videos = set(l.video_filename for l in self._labels.values())
        
        # number of labels for each defect type
        defect_counts = {defect: 0 for defect in DEFECT_TYPES}
        for l in self._labels.values():
            if l.defect_type and l.defect_type in defect_counts:
                defect_counts[l.defect_type] += 1
        
        # remove zero counts
        defect_counts = {k: v for k, v in defect_counts.items() if v > 0}
        
        return LabelingStats(
            total_labeled=len(self._labels),
            ok_count=ok_count,
            nok_count=nok_count,
            skip_count=skip_count,
            videos_labeled=len(videos),
            defect_counts=defect_counts
        )
    
    def get_all_labels(self) -> list[FrameLabel]:
        """get all labels"""
        return list(self._labels.values())
    
    def _save_training_image(
        self, 
        video_filename: str, 
        frame_index: int, 
        label: LabelType,
        defect_type: Optional[DefectType] = None
    ):
        """save frame to training folder"""
        if label == "skip":
            return
            
        video_path = self.recordings_dir / video_filename
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            return
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                stem = Path(video_filename).stem
                
                # Save to main ok/nok folder
                output_path = self.training_dir / label / f"{stem}_frame{frame_index:05d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.debug(f"Saved training image: {output_path}")
                
                # Additionally save to defect type folder
                if label == "nok" and defect_type:
                    defect_path = self.training_dir / "defects" / defect_type / f"{stem}_frame{frame_index:05d}.jpg"
                    cv2.imwrite(str(defect_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    logger.debug(f"Saved defect image: {defect_path}")
                    
        except Exception as e:
            logger.error(f"Failed to save training image: {e}")
    
    def _remove_training_image(
        self, 
        video_filename: str, 
        frame_index: int, 
        label: LabelType,
        defect_type: Optional[DefectType] = None
    ):
        """remove frame from training folder"""
        if label == "skip":
            return
            
        stem = Path(video_filename).stem
        
        # Remove from main folder
        image_path = self.training_dir / label / f"{stem}_frame{frame_index:05d}.jpg"
        if image_path.exists():
            image_path.unlink()
            logger.debug(f"Removed training image: {image_path}")
        
        # Remove from defect folder
        if defect_type:
            defect_path = self.training_dir / "defects" / defect_type / f"{stem}_frame{frame_index:05d}.jpg"
            if defect_path.exists():
                defect_path.unlink()
                logger.debug(f"Removed defect image: {defect_path}")
    
    def get_training_data_path(self) -> Path:
        """get path to training data"""
        return self.training_dir
    
    def export_for_training(self) -> dict:
        """export data for ML training"""
        ok_images = list((self.training_dir / "ok").glob("*.jpg"))
        nok_images = list((self.training_dir / "nok").glob("*.jpg"))
        
        # number of labels for each defect type
        defect_counts = {}
        defects_dir = self.training_dir / "defects"
        if defects_dir.exists():
            for defect in DEFECT_TYPES:
                defect_dir = defects_dir / defect
                if defect_dir.exists():
                    count = len(list(defect_dir.glob("*.jpg")))
                    if count > 0:
                        defect_counts[defect] = count
        
        return {
            "training_data_path": str(self.training_dir),
            "ok_count": len(ok_images),
            "nok_count": len(nok_images),
            "total": len(ok_images) + len(nok_images),
            "ready_for_training": len(ok_images) >= 20 and len(nok_images) >= 20,
            "defect_counts": defect_counts
        }


# Singleton
_labeling_service: Optional[LabelingService] = None


def get_labeling_service() -> LabelingService:
    """FastAPI dependency."""
    global _labeling_service
    if _labeling_service is None:
        _labeling_service = LabelingService()
    return _labeling_service
