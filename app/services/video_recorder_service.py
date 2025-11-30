"""Serwis nagrywania wideo do MP4."""

import cv2  # type: ignore
import numpy as np
import threading
import time
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Folder na nagrania - Docker: /app/recordings, Lokalnie: ./recordings
def _get_recordings_dir() -> Path:
    """Zwraca folder nagrań (kompatybilny z Docker i Windows)."""
    docker_path = Path("/app/recordings")
    local_path = Path(__file__).parent.parent.parent / "recordings"
    
    # Użyj Docker path jeśli istnieje lub jesteśmy w Linux
    if docker_path.exists() or os.name != 'nt':
        docker_path.mkdir(exist_ok=True)
        return docker_path
    
    # Lokalnie (Windows)
    local_path.mkdir(exist_ok=True)
    return local_path

RECORDINGS_DIR = _get_recordings_dir()


class VideoRecorderService:
    """Singleton - zapisuje klatki JPEG do pliku MP4."""
    
    _instance: Optional["VideoRecorderService"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.is_recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_file: Optional[Path] = None
        self.start_time: Optional[float] = None
        self.frame_count = 0
        self.lock = threading.Lock()
        
        logger.info(f"📼 VideoRecorderService: {RECORDINGS_DIR}")
    
    def start(self) -> str:
        """Rozpocznij nagrywanie. Zwraca nazwę pliku."""
        with self.lock:
            if self.is_recording:
                return self.current_file.name
            
            filename = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            self.current_file = RECORDINGS_DIR / filename
            self.is_recording = True
            self.start_time = time.time()
            self.frame_count = 0
            self.writer = None  # Lazy init przy pierwszej klatce
            
            logger.info(f"🔴 Recording started: {filename}")
            return filename
    
    def add_frame(self, jpeg_bytes: bytes) -> None:
        """Dodaj klatkę do nagrania (thread-safe)."""
        if not self.is_recording:
            return
        
        with self.lock:
            try:
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    return
                
                # Lazy init writera przy pierwszej klatce
                if self.writer is None:
                    h, w = frame.shape[:2]
                    self.writer = cv2.VideoWriter(
                        str(self.current_file),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        15.0, (w, h)
                    )
                
                self.writer.write(frame)
                self.frame_count += 1
            except Exception as e:
                logger.error(f"Frame error: {e}")
    
    def stop(self) -> dict:
        """Zatrzymaj nagrywanie. Zwraca info o pliku."""
        with self.lock:
            if not self.is_recording:
                return {}
            
            self.is_recording = False
            duration = time.time() - self.start_time if self.start_time else 0
            
            if self.writer:
                self.writer.release()
                self.writer = None
            
            size_mb = self.current_file.stat().st_size / 1024 / 1024 if self.current_file and self.current_file.exists() else 0
            
            result = {
                "filename": self.current_file.name,
                "duration_seconds": round(duration, 1),
                "frames": self.frame_count,
                "size_mb": round(size_mb, 2)
            }
            
            logger.info(f"⏹️ Recording stopped: {result}")
            self.current_file = None
            return result
    
    def list_files(self) -> list[dict]:
        """Lista nagrań."""
        return [
            {
                "filename": f.name,
                "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
                "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat()
            }
            for f in sorted(RECORDINGS_DIR.glob("*.mp4"), key=lambda x: x.stat().st_ctime, reverse=True)
        ]
    
    def get_path(self, filename: str) -> Optional[Path]:
        """Zwraca ścieżkę do pliku (bezpieczne)."""
        if ".." in filename or "/" in filename:
            return None
        path = RECORDINGS_DIR / filename
        return path if path.exists() else None


# Singleton getter
_recorder: Optional[VideoRecorderService] = None

def get_recorder_service() -> VideoRecorderService:
    global _recorder
    if _recorder is None:
        _recorder = VideoRecorderService()
    return _recorder
