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
    Thread-safe camera service with automatic reconnection and error handling.
    
    Features:
    - Thread-safe frame access with lock
    - Automatic reconnection on camera failure
    - Retry logic for transient errors
    - Frame caching for better performance
    - Detailed error logging
    """
    
    def __init__(self, camera_index=None):
        """
        Initialize camera service.
        
        Args:
            camera_index: Index kamery USB (0, 1, 2...) lub None (remote mode - no hardware)
        """
        # Ustaw camera_index - jeśli None podany, użyj z settings (USB mode)
        # Jeśli jawnie None (z child class), pozostaw None (remote mode)
        if camera_index is None and not hasattr(self, '_remote_mode'):
            camera_index = settings.CAMERA_INDEX
        
        self.camera_index = camera_index
        self.lock = threading.Lock()  # Thread-safety
        self.last_frame: Optional[bytes] = None  # Cached last good frame
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Camera settings for stability
        self.retry_delay = 0.1  # seconds
        self.max_retries = 3
        
        # Initialize camera hardware tylko jeśli camera_index podany
        if camera_index is not None:
            self.cap: Optional[cv2.VideoCapture] = None
            self._initialize_camera()
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
            
            # Set camera properties for stability
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set consistent FPS
            
            # Warm up camera - discard first few frames
            for _ in range(5):
                self.cap.read()
            
            logger.info(f"✅ Camera {self.camera_index} initialized successfully")
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def _reconnect_camera(self) -> bool:
        """
        Attempt to reconnect to camera after failure.
        
        Returns:
            bool: True if reconnection successful
        """
        logger.warning(f"Attempting to reconnect camera {self.camera_index}...")
        time.sleep(1.0)  # Wait before reconnecting
        return self._initialize_camera()
    
    def get_frame(self) -> Optional[bytes]:
        """
        Get current frame from camera with retry logic and error handling.
        
        Returns:
            Optional[bytes]: JPEG encoded frame or None if failed
        """
        with self.lock:  # Thread-safe access
            for attempt in range(self.max_retries):
                try:
                    if self.cap is None or not self.cap.isOpened():
                        logger.warning("Camera not opened, attempting reconnection")
                        if not self._reconnect_camera():
                            return self.last_frame  # Return cached frame
                    
                    # Grab frame (this pulls from buffer)
                    grabbed = self.cap.grab()
                    if not grabbed:
                        logger.warning(f"Failed to grab frame (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue
                    
                    # Retrieve frame (this decodes the grabbed frame)
                    success, frame = self.cap.retrieve()
                    
                    if not success or frame is None:
                        logger.warning(f"Failed to retrieve frame (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay)
                        continue
                    
                    # Encode to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if not ret:
                        logger.error("Failed to encode frame to JPEG")
                        time.sleep(self.retry_delay)
                        continue
                    
                    # Success - cache frame and reset failure counter
                    self.last_frame = buffer.tobytes()
                    self.consecutive_failures = 0
                    return self.last_frame
                    
                except Exception as e:
                    logger.error(f"Error reading frame (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                    continue
            
            # All retries failed
            self.consecutive_failures += 1
            logger.error(f"Failed to get frame after {self.max_retries} retries (consecutive failures: {self.consecutive_failures})")
            
            # If too many consecutive failures, try reconnecting
            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.critical(f"Too many consecutive failures ({self.consecutive_failures}), reconnecting camera...")
                self._reconnect_camera()
            
            # Return last good frame if available
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
            "has_cached_frame": self.last_frame is not None
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
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.last_frame = None
            logger.info(f"Camera {self.camera_index} released")