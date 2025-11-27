"""
Remote Camera Service - Pobieranie klatek z serwera kamery MJPEG.
"""

import httpx
import asyncio
import logging
from typing import Optional, AsyncGenerator

from app.config import settings

logger = logging.getLogger(__name__)


class RemoteCameraService:
    """Serwis do pobierania klatek z serwera kamery MJPEG."""
    
    def __init__(self):
        self.camera_url = settings.CAMERA_SERVER_URL
        self.timeout = httpx.Timeout(10.0, read=30.0)
        self._last_frame: Optional[bytes] = None
        logger.info(f"📡 RemoteCameraService: {self.camera_url}")
    
    async def get_single_frame(self) -> Optional[bytes]:
        """Pobiera pojedynczą klatkę z kamery."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.camera_url}/capture")
                if response.status_code == 200:
                    self._last_frame = response.content
                    return self._last_frame
        except Exception as e:
            logger.warning(f"Frame error: {e}")
        
        return self._last_frame  # Fallback to cached
    
    async def stream_frames(self) -> AsyncGenerator[bytes, None]:
        """Generator do streamowania klatek MJPEG."""
        while True:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("GET", f"{self.camera_url}/stream") as response:
                        if response.status_code != 200:
                            await asyncio.sleep(1)
                            continue
                        
                        logger.info("📹 Connected to MJPEG stream")
                        buffer = b""
                        
                        async for chunk in response.aiter_bytes():
                            buffer += chunk
                            
                            while True:
                                start = buffer.find(b'\xff\xd8')
                                if start == -1:
                                    break
                                end = buffer.find(b'\xff\xd9', start)
                                if end == -1:
                                    break
                                
                                frame = buffer[start:end + 2]
                                buffer = buffer[end + 2:]
                                self._last_frame = frame
                                yield frame
                                
            except Exception as e:
                logger.warning(f"Stream error: {e}")
            
            await asyncio.sleep(1)
    
    async def health_check(self) -> dict:
        """Sprawdza połączenie z kamerą."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.camera_url}/capture")
                return {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "camera_url": self.camera_url,
                    "response_code": response.status_code,
                    "has_cached_frame": self._last_frame is not None
                }
        except Exception as e:
            return {
                "status": "disconnected",
                "camera_url": self.camera_url,
                "error": str(e),
                "has_cached_frame": self._last_frame is not None
            }


# Singleton
_camera_service: Optional[RemoteCameraService] = None

def get_camera_service() -> RemoteCameraService:
    global _camera_service
    if _camera_service is None:
        _camera_service = RemoteCameraService()
    return _camera_service
