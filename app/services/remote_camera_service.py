"""
Remote Camera Service - Serwis do komunikacji z serwerem kamery MJPEG.

Ten serwis łączy się z zewnętrznym endpointem kamery (np. camera_server)
i pobiera klatki ze streamu MJPEG.
"""

import httpx
import logging
import asyncio
from typing import Optional, AsyncGenerator
import numpy as np
import cv2 # type: ignore

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RemoteCameraService:
    """
    Serwis do pobierania klatek z zewnętrznego serwera kamery MJPEG.
    
    Features:
    - Async HTTP client dla wydajności
    - Automatyczny retry przy błędach
    - Streaming MJPEG
    - Konwersja JPEG -> numpy array dla przetwarzania CV
    """
    
    def __init__(self, camera_url: Optional[str] = None):
        """
        Inicjalizacja serwisu.
        
        Args:
            camera_url: Base URL serwera kamery (np. http://localhost:8001)
        """
        self.camera_url = camera_url or settings.CAMERA_SERVER_URL
        self.stream_endpoint = f"{self.camera_url}/stream"
        self.capture_endpoint = f"{self.camera_url}/capture"
        self.stats_endpoint = f"{self.camera_url}/stats"
        
        # HTTP client settings
        self.timeout = httpx.Timeout(10.0, read=30.0)
        self.max_retries = 3
        self.retry_delay = 0.5
        
        # Cached last frame
        self._last_frame: Optional[bytes] = None
        self._last_frame_np: Optional[np.ndarray] = None
        
        logger.info(f"📡 RemoteCameraService initialized with URL: {self.camera_url}")
    
    async def get_single_frame(self) -> Optional[bytes]:
        """
        Pobiera pojedynczą klatkę z kamery.
        
        Returns:
            Optional[bytes]: Klatka JPEG jako bytes lub None przy błędzie
        """
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(self.capture_endpoint)
                    
                    if response.status_code == 200:
                        self._last_frame = response.content
                        logger.debug("✅ Frame captured successfully")
                        return self._last_frame
                    else:
                        logger.warning(f"Camera returned status {response.status_code}")
                        
            except httpx.TimeoutException:
                logger.warning(f"Timeout getting frame (attempt {attempt + 1}/{self.max_retries})")
            except httpx.ConnectError:
                logger.error(f"Cannot connect to camera server at {self.capture_endpoint}")
            except Exception as e:
                logger.error(f"Error getting frame: {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
        
        # Return cached frame if available
        if self._last_frame:
            logger.info("Returning cached frame")
            return self._last_frame
        
        return None
    
    async def get_frame_as_numpy(self) -> Optional[np.ndarray]:
        """
        Pobiera klatkę i konwertuje do numpy array (BGR format).
        Przydatne do przetwarzania OpenCV.
        
        Returns:
            Optional[np.ndarray]: Obraz w formacie BGR lub None
        """
        frame_bytes = await self.get_single_frame()
        
        if frame_bytes is None:
            return self._last_frame_np
        
        try:
            # Decode JPEG to numpy array
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self._last_frame_np = frame
                return frame
            else:
                logger.error("Failed to decode JPEG frame")
                return self._last_frame_np
                
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return self._last_frame_np
    
    async def stream_frames(self) -> AsyncGenerator[bytes, None]:
        """
        Generator asynchroniczny do streamowania klatek MJPEG.
        
        Yields:
            bytes: Kolejne klatki JPEG ze streamu
        """
        while True:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("GET", self.stream_endpoint) as response:
                        if response.status_code != 200:
                            logger.error(f"Stream returned status {response.status_code}")
                            await asyncio.sleep(1)
                            continue
                        
                        logger.info("📹 Connected to MJPEG stream")
                        
                        # Buffer do parsowania MJPEG
                        buffer = b""
                        
                        async for chunk in response.aiter_bytes():
                            buffer += chunk
                            
                            # Szukamy granic klatek JPEG
                            while True:
                                # Znajdź początek JPEG (FFD8)
                                start = buffer.find(b'\xff\xd8')
                                if start == -1:
                                    break
                                
                                # Znajdź koniec JPEG (FFD9)
                                end = buffer.find(b'\xff\xd9', start)
                                if end == -1:
                                    break
                                
                                # Wyodrębnij klatkę
                                frame = buffer[start:end + 2]
                                buffer = buffer[end + 2:]
                                
                                self._last_frame = frame
                                yield frame
                                
            except httpx.TimeoutException:
                logger.warning("Stream timeout, reconnecting...")
            except httpx.ConnectError:
                logger.error(f"Cannot connect to stream at {self.stream_endpoint}")
            except Exception as e:
                logger.error(f"Stream error: {e}")
            
            # Czekaj przed reconnect
            await asyncio.sleep(1)
            logger.info("🔄 Attempting to reconnect to stream...")
    
    async def get_camera_stats(self) -> Optional[dict]:
        """
        Pobiera statystyki z serwera kamery.
        
        Returns:
            Optional[dict]: Statystyki kamery lub None
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.stats_endpoint)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Stats endpoint returned {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting camera stats: {e}")
            return None
    
    async def health_check(self) -> dict:
        """
        Sprawdza połączenie z serwerem kamery.
        
        Returns:
            dict: Status połączenia
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(self.capture_endpoint)
                
                return {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "camera_url": self.camera_url,
                    "response_code": response.status_code,
                    "has_cached_frame": self._last_frame is not None
                }
                
        except httpx.ConnectError:
            return {
                "status": "disconnected",
                "camera_url": self.camera_url,
                "error": "Cannot connect to camera server",
                "has_cached_frame": self._last_frame is not None
            }
        except Exception as e:
            return {
                "status": "error",
                "camera_url": self.camera_url,
                "error": str(e),
                "has_cached_frame": self._last_frame is not None
            }


# Singleton instance
_camera_service: Optional[RemoteCameraService] = None


def get_camera_service() -> RemoteCameraService:
    """
    Zwraca singleton instance RemoteCameraService.
    Używaj tej funkcji do dependency injection w FastAPI.
    """
    global _camera_service
    if _camera_service is None:
        _camera_service = RemoteCameraService()
    return _camera_service
