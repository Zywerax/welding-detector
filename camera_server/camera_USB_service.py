# to start camera server use:
# uvicorn camera_server.stream:app --host 0.0.0.0 --port 8001


import cv2 # type: ignore
import threading
import time
import logging
from typing import Optional
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Camera_USB_Service:
    """
    Thread-safe camera service with background capture thread for minimal latency.
    
    Features:
    - Background thread continuously captures frames (no blocking on request)
    - Thread-safe frame access with lock
    - Automatic reconnection on camera failure
    - Optimized JPEG encoding for high resolution
    - Minimal latency design
    """
    
    def __init__(self, camera_index=None):
        """
        Initialize camera service.
        
        Args:
            camera_index: Index kamery USB (0, 1, 2...) lub None (remote mode - no hardware)
        """
        # Ustaw camera_index - jeśli None podany, użyj z settings (USB mode)
        if camera_index is None and not hasattr(self, '_remote_mode'):
            camera_index = settings.CAMERA_INDEX
        
        self.camera_index = camera_index
        self.lock = threading.Lock()  # Thread-safety for frame access
        self.last_frame: Optional[bytes] = None  # Latest captured frame
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Background capture thread
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        
        # Performance settings - optimize for resolution
        self.target_fps = settings.CAMERA_USB_FPS
        self.frame_interval = 1.0 / self.target_fps
        
        # JPEG quality - wysoka jakość (90 dla wszystkich rozdzielczości)
        self.jpeg_quality = 95
        
        # Initialize camera hardware tylko jeśli camera_index podany
        if camera_index is not None:
            self.cap: Optional[cv2.VideoCapture] = None
            self._initialize_camera()
            self._start_capture_thread()
        else:
            # Remote mode - no hardware initialization
            self.cap = None
            logger.info("📡 Camera service initialized in remote mode (no hardware)")
    
    def _initialize_camera(self) -> bool:
        """
        Initialize USB camera hardware with optimal settings.
        
        Returns:
            bool: True if camera initialized successfully
        """
        try:
            logger.info(f"📹 Initializing USB camera {self.camera_index}")
            
            # Release old capture if exists
            if self.cap is not None:
                self.cap.release()
            
            # Create new capture with DirectShow backend (more stable than MSMF)
            # CAP_DSHOW = 700 (DirectShow on Windows)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties from settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            self.cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_USB_FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_USB_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_USB_HEIGHT)
            
            logger.info(f"📐 Camera settings: {settings.CAMERA_USB_WIDTH}x{settings.CAMERA_USB_HEIGHT} @ {settings.CAMERA_USB_FPS}fps")
            
            # Warm up camera - discard first few frames
            for _ in range(5):
                self.cap.read()
            
            logger.info(f"✅ Camera {self.camera_index} initialized successfully")
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def _start_capture_thread(self):
        """Start background thread for continuous frame capture."""
        if self._running:
            return
        
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("🔄 Background capture thread started")
    
    def _stop_capture_thread(self):
        """Stop background capture thread."""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        logger.info("⏹️ Background capture thread stopped")
    
    def _capture_loop(self):
        """
        Background loop that continuously captures frames.
        This runs in a separate thread to minimize latency.
        """
        last_capture_time = 0.0
        
        while self._running:
            try:
                # Kontrola tempa - nie szybciej niż target FPS
                current_time = time.time()
                elapsed = current_time - last_capture_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                
                if self.cap is None or not self.cap.isOpened():
                    logger.warning("Camera not opened in capture loop, reconnecting...")
                    if not self._reconnect_camera():
                        time.sleep(1.0)
                        continue
                
                # Grab + retrieve (najszybsza metoda)
                grabbed = self.cap.grab()
                if not grabbed:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self._reconnect_camera()
                    continue
                
                success, frame = self.cap.retrieve()
                if not success or frame is None:
                    self.consecutive_failures += 1
                    continue
                
                # Encode to JPEG (optimized quality for resolution)
                ret, buffer = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1  # Optimize Huffman tables
                ])
                
                if ret:
                    with self.lock:
                        self.last_frame = buffer.tobytes()
                    self.consecutive_failures = 0
                    last_capture_time = time.time()
                    
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                self.consecutive_failures += 1
                time.sleep(0.1)
    
    def _reconnect_camera(self) -> bool:
        """
        Attempt to reconnect to camera after failure.
        
        Returns:
            bool: True if reconnection successful
        """
        logger.warning(f"Attempting to reconnect camera {self.camera_index}...")
        time.sleep(0.5)  # Short wait before reconnecting
        return self._initialize_camera()
    
    def get_frame(self) -> Optional[bytes]:
        """
        Get latest captured frame (non-blocking).
        Frame is captured by background thread, this just returns the latest one.
        
        Returns:
            Optional[bytes]: JPEG encoded frame or None if no frame available
        """
        with self.lock:
            return self.last_frame
    
    def get_stats(self) -> dict:
        """
        Get camera statistics for monitoring.
        
        Returns:
            dict: Camera statistics
        """
        stats = {
            "camera_index": self.camera_index,
            "is_opened": self.cap.isOpened() if self.cap else False,
            "consecutive_failures": self.consecutive_failures,
            "has_cached_frame": self.last_frame is not None,
            "capture_thread_running": self._running,
            "target_fps": self.target_fps,
            "jpeg_quality": self.jpeg_quality
        }
        
        if self.cap and self.cap.isOpened():
            stats.update({
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "backend": self.cap.get(cv2.CAP_PROP_BACKEND)
            })
        
        return stats
    
    def release(self):
        """Release camera resources."""
        self._stop_capture_thread()
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.last_frame = None
            logger.info(f"Camera {self.camera_index} released")