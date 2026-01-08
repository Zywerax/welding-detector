"""
Labeling Service - zarządzanie etykietami OK/NOK dla klatek wideo.

Zapisuje etykiety jako JSON + kopiuje oznaczone klatki do folderów treningowych.
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
    "porosity",        # Porowatość - pęcherzyki gazu
    "crack",           # Pęknięcia
    "lack_of_fusion",  # Brak przetopu
    "undercut",        # Podtopienia przy krawędzi
    "burn_through",    # Przepalenie
    "spatter",         # Rozpryski
    "irregular_bead",  # Nierówna spoina
    "contamination",   # Zanieczyszczenia
    "other"            # Inna wada
]

DEFECT_TYPES = [
    "porosity", "crack", "lack_of_fusion", "undercut", 
    "burn_through", "spatter", "irregular_bead", "contamination", "other"
]


@dataclass
class FrameLabel:
    """Etykieta pojedynczej klatki."""
    video_filename: str
    frame_index: int
    label: LabelType
    timestamp: str
    defect_type: Optional[DefectType] = None  # Typ wady (gdy label=nok)
    notes: str = ""
    filters_used: Optional[dict] = None  # Jakie filtry były użyte przy etykietowaniu


@dataclass 
class LabelingStats:
    """Statystyki etykietowania."""
    total_labeled: int
    ok_count: int
    nok_count: int
    skip_count: int
    videos_labeled: int
    defect_counts: dict = field(default_factory=dict)  # Liczniki wad


class LabelingService:
    """
    Serwis do zarządzania etykietami spawów.
    
    Struktura folderów:
        labels/
            labels.json          # Wszystkie etykiety
            training_data/
                ok/              # Klatki oznaczone jako OK
                    video1_frame100.jpg
                    video2_frame50.jpg
                nok/             # Klatki oznaczone jako NOK
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
        
        # Utwórz strukturę folderów
        self.labels_dir.mkdir(exist_ok=True)
        (self.training_dir / "ok").mkdir(parents=True, exist_ok=True)
        (self.training_dir / "nok").mkdir(parents=True, exist_ok=True)
        
        # Foldery dla typów wad
        for defect in DEFECT_TYPES:
            (self.training_dir / "defects" / defect).mkdir(parents=True, exist_ok=True)
        
        # Załaduj istniejące etykiety
        self._labels: dict[str, FrameLabel] = {}
        self._load_labels()
        
        logger.info(f"🏷️ LabelingService initialized ({len(self._labels)} labels loaded)")
    
    def _get_label_key(self, video_filename: str, frame_index: int) -> str:
        """Generuje unikalny klucz dla etykiety."""
        return f"{video_filename}:{frame_index}"
    
    def _load_labels(self):
        """Wczytuje etykiety z pliku JSON."""
        if self.labels_file.exists():
            try:
                with open(self.labels_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, label_data in data.items():
                        self._labels[key] = FrameLabel(**label_data)
            except Exception as e:
                logger.error(f"Failed to load labels: {e}")
    
    def _save_labels(self):
        """Zapisuje etykiety do pliku JSON."""
        try:
            data = {key: asdict(label) for key, label in self._labels.items()}
            with open(self.labels_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save labels: {e}")
            raise
    
    def add_label(
        self,
        video_filename: str,
        frame_index: int,
        label: LabelType,
        defect_type: Optional[DefectType] = None,
        notes: str = "",
        filters_used: Optional[dict] = None,
        save_frame: bool = True
    ) -> FrameLabel:
        """
        Dodaje lub aktualizuje etykietę dla klatki.
        
        Args:
            video_filename: Nazwa pliku wideo
            frame_index: Numer klatki
            label: "ok", "nok" lub "skip"
            defect_type: Typ wady (wymagane gdy label="nok")
            notes: Opcjonalne notatki
            filters_used: Filtry użyte przy etykietowaniu
            save_frame: Czy zapisać klatkę do folderu treningowego
            
        Returns:
            Utworzona etykieta
        """
        key = self._get_label_key(video_filename, frame_index)
        
        # Usuń starą klatkę jeśli istniała z inną etykietą
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
        
        # Zapisz klatkę do odpowiedniego folderu
        if save_frame and label in ("ok", "nok"):
            self._save_training_image(video_filename, frame_index, label, defect_type)
        
        defect_info = f" ({defect_type})" if defect_type else ""
        logger.info(f"🏷️ Label added: {video_filename} frame {frame_index} = {label.upper()}{defect_info}")
        return frame_label
    
    def get_label(self, video_filename: str, frame_index: int) -> Optional[FrameLabel]:
        """Pobiera etykietę dla klatki."""
        key = self._get_label_key(video_filename, frame_index)
        return self._labels.get(key)
    
    def get_labels_for_video(self, video_filename: str) -> list[FrameLabel]:
        """Pobiera wszystkie etykiety dla danego wideo."""
        return [
            label for label in self._labels.values()
            if label.video_filename == video_filename
        ]
    
    def remove_label(self, video_filename: str, frame_index: int) -> bool:
        """Usuwa etykietę."""
        key = self._get_label_key(video_filename, frame_index)
        if key in self._labels:
            label = self._labels[key]
            self._remove_training_image(video_filename, frame_index, label.label, label.defect_type)
            del self._labels[key]
            self._save_labels()
            logger.info(f"🏷️ Label removed: {video_filename} frame {frame_index}")
            return True
        return False
    
    def get_stats(self) -> LabelingStats:
        """Zwraca statystyki etykietowania."""
        ok_count = sum(1 for l in self._labels.values() if l.label == "ok")
        nok_count = sum(1 for l in self._labels.values() if l.label == "nok")
        skip_count = sum(1 for l in self._labels.values() if l.label == "skip")
        videos = set(l.video_filename for l in self._labels.values())
        
        # Liczniki typów wad
        defect_counts = {defect: 0 for defect in DEFECT_TYPES}
        for l in self._labels.values():
            if l.defect_type and l.defect_type in defect_counts:
                defect_counts[l.defect_type] += 1
        
        # Usuń zerowe liczniki
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
        """Zwraca wszystkie etykiety."""
        return list(self._labels.values())
    
    def _save_training_image(
        self, 
        video_filename: str, 
        frame_index: int, 
        label: LabelType,
        defect_type: Optional[DefectType] = None
    ):
        """Zapisuje klatkę do folderu treningowego."""
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
                
                # Zapisz do głównego folderu ok/nok
                output_path = self.training_dir / label / f"{stem}_frame{frame_index:05d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                logger.debug(f"Saved training image: {output_path}")
                
                # Dodatkowo zapisz do folderu typu wady
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
        """Usuwa klatkę z folderu treningowego."""
        if label == "skip":
            return
            
        stem = Path(video_filename).stem
        
        # Usuń z głównego folderu
        image_path = self.training_dir / label / f"{stem}_frame{frame_index:05d}.jpg"
        if image_path.exists():
            image_path.unlink()
            logger.debug(f"Removed training image: {image_path}")
        
        # Usuń z folderu wady
        if defect_type:
            defect_path = self.training_dir / "defects" / defect_type / f"{stem}_frame{frame_index:05d}.jpg"
            if defect_path.exists():
                defect_path.unlink()
                logger.debug(f"Removed defect image: {defect_path}")
    
    def get_training_data_path(self) -> Path:
        """Zwraca ścieżkę do danych treningowych."""
        return self.training_dir
    
    def export_for_training(self) -> dict:
        """Eksportuje dane do treningu ML."""
        ok_images = list((self.training_dir / "ok").glob("*.jpg"))
        nok_images = list((self.training_dir / "nok").glob("*.jpg"))
        
        # Liczniki dla każdego typu wady
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
