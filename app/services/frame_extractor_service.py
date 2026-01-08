"""
Frame Extractor Service - ekstrakcja klatek z nagrań wideo.

Serwis do wczytywania klatek z plików MP4 do pamięci (lista numpy arrays)
oraz opcjonalnego zapisu do folderu jako JPEG.
"""

import cv2  # type: ignore
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Dane klatki z metadanymi."""
    index: int           # Numer klatki (0-based)
    frame: np.ndarray    # Obraz BGR jako numpy array
    timestamp_ms: float  # Pozycja w milisekundach


class FrameExtractorService:
    """
    Serwis do ekstrakcji klatek z plików wideo.
    
    Użycie:
        extractor = FrameExtractorService()
        frames = extractor.extract_frames("recordings/rec_20260105_120000.mp4")
        extractor.save_frames_to_folder(frames, "output/frames")
    """
    
    def __init__(self, recordings_dir: Path = Path("recordings")):
        self.recordings_dir = recordings_dir
        logger.info("🎞️ FrameExtractorService initialized")
    
    def extract_frames(
        self, 
        video_path: str | Path, 
        step: int = 1,
        max_frames: Optional[int] = None
    ) -> list[FrameData]:
        """
        Ekstrahuje klatki z pliku wideo do pamięci.
        
        Args:
            video_path: Ścieżka do pliku wideo (absolutna lub względna do recordings_dir)
            step: Co która klatka ma być pobrana (1 = każda, 2 = co druga, itd.)
            max_frames: Maksymalna liczba klatek do pobrania (None = wszystkie)
            
        Returns:
            Lista FrameData z klatkami i metadanymi
            
        Raises:
            FileNotFoundError: Gdy plik nie istnieje
            ValueError: Gdy nie można otworzyć pliku jako wideo
        """
        path = self._resolve_path(video_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        
        try:
            frames: list[FrameData] = []
            frame_index = 0
            extracted_count = 0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📹 Extracting frames from {path.name} ({total_frames} frames, {fps:.1f} fps)")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Pobierz tylko co N-tą klatkę
                if frame_index % step == 0:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    frames.append(FrameData(
                        index=frame_index,
                        frame=frame.copy(),
                        timestamp_ms=timestamp_ms
                    ))
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_index += 1
            
            logger.info(f"✅ Extracted {len(frames)} frames (step={step})")
            return frames
            
        finally:
            cap.release()
    
    def get_frame(self, filename: str, frame_index: int) -> Optional[np.ndarray]:
        """
        Pobierz pojedynczą klatkę z wideo.
        
        Args:
            filename: Nazwa pliku wideo
            frame_index: Indeks klatki
            
        Returns:
            Klatka jako numpy array (BGR) lub None jeśli nie znaleziono
        """
        video_path = self.recordings_dir / filename
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if not ret:
                logger.error(f"Cannot read frame {frame_index} from {filename}")
                return None
            
            return frame
        finally:
            cap.release()
    
    def extract_frames_generator(
        self, 
        video_path: str | Path,
        step: int = 1
    ) -> Generator[FrameData, None, None]:
        """
        Generator klatek - dla dużych plików (oszczędza RAM).
        
        Yields:
            FrameData z kolejnymi klatkami
        """
        path = self._resolve_path(video_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        
        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_index % step == 0:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    yield FrameData(
                        index=frame_index,
                        frame=frame,
                        timestamp_ms=timestamp_ms
                    )
                
                frame_index += 1
        finally:
            cap.release()
    
    def save_frames_to_folder(
        self,
        frames: list[FrameData],
        output_dir: str | Path,
        prefix: str = "frame",
        jpeg_quality: int = 95
    ) -> list[Path]:
        """
        Zapisuje klatki jako pliki JPEG.
        
        Args:
            frames: Lista FrameData do zapisania
            output_dir: Folder docelowy (zostanie utworzony jeśli nie istnieje)
            prefix: Prefix nazwy pliku (np. "frame" -> "frame_00001.jpg")
            jpeg_quality: Jakość JPEG (1-100)
            
        Returns:
            Lista ścieżek do zapisanych plików
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files: list[Path] = []
        
        for frame_data in frames:
            filename = f"{prefix}_{frame_data.index:05d}.jpg"
            filepath = output_path / filename
            
            cv2.imwrite(
                str(filepath),
                frame_data.frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            saved_files.append(filepath)
        
        logger.info(f"💾 Saved {len(saved_files)} frames to {output_path}")
        return saved_files
    
    def get_video_info(self, video_path: str | Path) -> dict:
        """
        Pobiera informacje o pliku wideo.
        
        Returns:
            Dict z: frame_count, fps, width, height, duration_seconds
        """
        path = self._resolve_path(video_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            return {
                "frame_count": frame_count,
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "duration_seconds": round(duration, 2)
            }
        finally:
            cap.release()
    
    def _resolve_path(self, video_path: str | Path) -> Path:
        """Rozwiązuje ścieżkę - absolutna lub względna do recordings_dir."""
        path = Path(video_path)
        if path.is_absolute():
            return path
        
        # Najpierw sprawdź czy istnieje względna do CWD
        if path.exists():
            return path
        
        # Potem względna do recordings_dir
        return self.recordings_dir / path


# Singleton
_frame_extractor_service: Optional[FrameExtractorService] = None


def get_frame_extractor_service() -> FrameExtractorService:
    """Singleton getter dla FrameExtractorService."""
    global _frame_extractor_service
    if _frame_extractor_service is None:
        _frame_extractor_service = FrameExtractorService()
    return _frame_extractor_service
