"""CameraService - DirectShow capture + MJPEG streaming + MP4 recording."""

import cv2 #type: ignore
import asyncio
import threading
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, AsyncGenerator
from app.config import settings

logger = logging.getLogger(__name__)


class CameraService:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        
        self._last_frame: Optional[bytes] = None
        self._last_raw_frame = None
        self._running = False
        
        # Recording
        self._recording = False
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[Path] = None
        self._temp_recording_path: Optional[Path] = None
        self._recording_start: Optional[float] = None
        self._frame_count = 0
        self._record_width = 0
        self._record_height = 0
        
        # Settings (requested)
        self.requested_fps = settings.CAMERA_USB_FPS
        self.width = settings.CAMERA_USB_WIDTH
        self.height = settings.CAMERA_USB_HEIGHT
        self.jpeg_quality = settings.CAMERA_JPEG_QUALITY
        self.monochrome = False
        
        # Actual FPS measurement
        self.actual_fps = 30.0  # Default, will be measured
        self._fps_samples = []
        self._last_frame_time = 0
        
        self.recordings_dir = Path("recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        
        self._init_camera()
        self._start_capture()
    
    def _init_camera(self) -> bool:
        """Initialize camera with best available backend."""
        if self.cap:
            self.cap.release()
        
        # Try different backends in order of preference
        backends = [
            (cv2.CAP_MSMF, "MSMF"),      # Media Foundation - usually fastest on Windows
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_ANY, "Auto"),
        ]
        
        for backend, name in backends:
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if self.cap.isOpened():
                logger.info(f"🎥 Using backend: {name}")
                break
        else:
            logger.error(f"Failed to open camera {self.camera_index}")
            return False
        
        # Optimize for speed
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        # Try MJPEG format - faster than raw
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # type: ignore
        
        # Measure actual FPS (includes warmup)
        self._measure_actual_fps()
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"📹 Camera opened: {w}x{h} @ {self.actual_fps:.1f}fps (requested: {self.requested_fps})")
        return True
    
    def _measure_actual_fps(self):
        """Measure real FPS by timing actual frame capture rate."""
        if not self.cap:
            return
        
        # Warmup - discard first frames
        for _ in range(5):
            self.cap.read()
        
        # Measure time for N frames
        start = time.perf_counter()
        frames_captured = 0
        for _ in range(60):  # Try to capture 60 frames
            ret, _ = self.cap.read()
            if ret:
                frames_captured += 1
        elapsed = time.perf_counter() - start
        
        if frames_captured > 10 and elapsed > 0:
            self.actual_fps = frames_captured / elapsed
            # Clamp to reasonable range
            self.actual_fps = max(5.0, min(self.actual_fps, 120.0))
        
        logger.info(f"📊 Measured FPS: {self.actual_fps:.1f} ({frames_captured} frames in {elapsed:.2f}s)")
    
    def _start_capture(self):
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
    
    def _capture_loop(self):
        """Capture frames as fast as camera allows, measure real FPS."""
        frame_times = []
        
        while self._running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(1)
                self._init_camera()
                continue
            
            t0 = time.perf_counter()
            ret, frame = self.cap.read()
            
            if not ret:
                continue
            
            # Measure actual FPS continuously
            now = time.perf_counter()
            if self._last_frame_time > 0:
                frame_times.append(now - self._last_frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                if len(frame_times) >= 10:
                    avg_interval = sum(frame_times) / len(frame_times)
                    self.actual_fps = 1.0 / avg_interval if avg_interval > 0 else 30.0
            self._last_frame_time = now
            
            if self.monochrome:
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            
            with self.lock:
                self._last_frame = buf.tobytes()
                self._last_raw_frame = frame
                if self._recording and self._video_writer:
                    try:
                        # Ensure frame matches VideoWriter size
                        fh, fw = frame.shape[:2]
                        if fw != self._record_width or fh != self._record_height:
                            frame = cv2.resize(frame, (self._record_width, self._record_height))
                        self._video_writer.write(frame)
                        self._frame_count += 1
                    except Exception as e:
                        logger.error(f"Write frame error: {e}")
            
            # No artificial sleep - capture as fast as camera provides
    
    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            return self._last_frame
    
    async def get_single_frame(self) -> Optional[bytes]:
        return self.get_frame()
    
    async def stream_raw(self) -> AsyncGenerator[bytes, None]:
        """Stream at actual camera FPS."""
        while True:
            frame = self.get_frame()
            if frame:
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n'
            await asyncio.sleep(1.0 / self.actual_fps)
    
    async def stream_frames(self) -> AsyncGenerator[bytes, None]:
        """Stream raw JPEG frames at actual camera FPS."""
        while True:
            frame = self.get_frame()
            if frame:
                yield frame
            await asyncio.sleep(1.0 / self.actual_fps)
    
    # Recording
    def start_recording(self) -> str:
        if self._recording:
            return self._recording_path.name if self._recording_path else ""
        
        filename = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self._recording_path = self.recordings_dir / filename
        self._temp_recording_path = self.recordings_dir / f"temp_{filename}"
        
        # Get frame size from last captured frame or use settings
        with self.lock:
            if self._last_raw_frame is not None:
                h, w = self._last_raw_frame.shape[:2]
            else:
                w, h = self.width, self.height
        
        self._record_width = w
        self._record_height = h
        
        # Record at placeholder FPS - will be fixed in stop_recording
        self._video_writer = cv2.VideoWriter(
            str(self._temp_recording_path), 
            cv2.VideoWriter_fourcc(*'mp4v'), #type: ignore
            30.0, 
            (w, h)
        )
        
        if not self._video_writer.isOpened():
            logger.error(f"❌ Failed to create VideoWriter: {self._temp_recording_path}")
            return ""
        
        self._recording = True
        self._recording_start = time.perf_counter()
        self._frame_count = 0
        logger.info(f"🔴 Recording: {filename} ({w}x{h})")
        return filename
    
    def stop_recording(self) -> dict:
        if not self._recording:
            return {}
        self._recording = False
        
        duration = time.perf_counter() - self._recording_start if self._recording_start else 0
        frames = self._frame_count
        
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        
        # Calculate real FPS based on actual recording
        real_fps = frames / duration if duration > 0 else 30.0
        real_fps = max(10.0, min(real_fps, 60.0))
        logger.info(f"📊 Recording stats: {frames} frames in {duration:.1f}s = {real_fps:.1f} fps")
        
        # Re-encode with correct FPS
        if self._temp_recording_path and self._temp_recording_path.exists():
            self._reencode_video(real_fps)
        
        size_mb = self._recording_path.stat().st_size / (1024 * 1024) if self._recording_path and self._recording_path.exists() else 0
        
        result = {"filename": self._recording_path.name if self._recording_path else "", "duration_seconds": round(duration, 1), "frames": frames, "fps": round(real_fps, 1), "size_mb": round(size_mb, 2)}
        logger.info(f"⏹️ Stopped: {result}")
        self._recording_path = self._recording_start = self._temp_recording_path = None
        self._frame_count = 0
        return result
    
    def _reencode_video(self, target_fps: float):
        """Re-encode temp video with correct FPS for proper playback speed."""
        try:
            cap = cv2.VideoCapture(str(self._temp_recording_path))
            writer = cv2.VideoWriter(str(self._recording_path), cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (self._record_width, self._record_height)) # type: ignore
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            
            cap.release()
            writer.release()
            if self._temp_recording_path:
                self._temp_recording_path.unlink()  # Delete temp file
            logger.info(f"✅ Re-encoded with {target_fps:.1f} fps")
        except Exception as e:
            logger.error(f"Re-encode failed: {e}")
            # Fallback - just rename temp to final
            if self._temp_recording_path and self._temp_recording_path.exists() and self._recording_path:
                self._temp_recording_path.rename(self._recording_path)
    
    @property
    def is_recording(self) -> bool:
        return self._recording
    
    def get_recording_duration(self) -> float:
        return time.perf_counter() - self._recording_start if self._recording and self._recording_start else 0
    
    def list_recordings(self) -> list:
        notes = self._load_notes()
        return sorted([{
            "filename": f.name, 
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2), 
            "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat(),
            "note": notes.get(f.name, "")
        } for f in self.recordings_dir.glob("*.mp4")], key=lambda x: x["created"], reverse=True)
    
    def _load_notes(self) -> dict:
        notes_file = self.recordings_dir / "notes.json"
        if notes_file.exists():
            try:
                import json
                return json.loads(notes_file.read_text(encoding="utf-8"))
            except:
                return {}
        return {}
    
    def _save_notes(self, notes: dict):
        import json
        notes_file = self.recordings_dir / "notes.json"
        notes_file.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def set_note(self, filename: str, note: str) -> bool:
        path = self.recordings_dir / filename
        if not path.exists():
            return False
        notes = self._load_notes()
        if note.strip():
            notes[filename] = note.strip()
        elif filename in notes:
            del notes[filename]
        self._save_notes(notes)
        return True
    
    def get_note(self, filename: str) -> str:
        return self._load_notes().get(filename, "")
    
    def get_recording_path(self, filename: str) -> Optional[Path]:
        path = self.recordings_dir / filename
        return path if path.exists() else None
    
    def delete_recording(self, filename: str) -> bool:
        path = self.recordings_dir / filename
        if path.exists():
            path.unlink()
            return True
        return False
    
    async def health_check(self) -> dict:
        return {
            "status": "healthy" if self.cap and self.cap.isOpened() else "disconnected",
            "camera_index": self.camera_index,
            "fps": round(self.actual_fps, 1),
            "requested_fps": self.requested_fps,
            "resolution": f"{self.width}x{self.height}",
            "is_recording": self._recording
        }
    
    def apply_settings(self, contrast: Optional[int] = None, fps: Optional[int] = None, jpeg_quality: Optional[int] = None, resolution: Optional[str] = None) -> dict:
        results = {}
        if contrast is not None and self.cap:
            self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            results["contrast"] = self.cap.get(cv2.CAP_PROP_CONTRAST)
        if fps is not None:
            self.requested_fps = fps
            if self.cap:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
            results["fps"] = fps
        if jpeg_quality is not None:
            self.jpeg_quality = jpeg_quality
            results["jpeg_quality"] = jpeg_quality
        if resolution:
            res = {"HD": (1280, 720), "FHD": (1920, 1080)}.get(resolution.upper())
            if res and self.cap:
                self.width, self.height = res
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                results["resolution"] = resolution
        return results
    
    def get_settings(self) -> dict:
        return {
            "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST) if self.cap else 0,
            "fps": round(self.actual_fps, 1),
            "requested_fps": self.requested_fps,
            "jpeg_quality": self.jpeg_quality,
            "resolution": "FHD" if self.width >= 1920 else "HD"
        }
    
    def release(self):
        self._running = False
        if self._recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()


_camera_service: Optional[CameraService] = None

def get_camera_service() -> CameraService:
    global _camera_service
    if _camera_service is None:
        _camera_service = CameraService(settings.CAMERA_INDEX)
    return _camera_service
