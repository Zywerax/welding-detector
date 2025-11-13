"""
Remote Camera Service - Proxy do camera-server
Streamuje dane BEZPOŚREDNIO z camera-server bez dekodowania
"""
import httpx
import re
from app.core.config import settings


class RemoteCameraService:
    """Serwis pobierający stream ze zdalnego camera-server przez HTTP"""
    
    def __init__(self):
        self.camera_server_url = settings.CAMERA_SERVER_URL
        self.stream_endpoint = f"{self.camera_server_url}/stream"
        self.health_endpoint = f"{self.camera_server_url}/health"
    
    async def get_stream(self):
        """
        Pobiera stream z camera-server i przekazuje dalej jako PROXY.
        NIE dekoduje ani nie przetwarza obrazu - tylko przekazuje bajty.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream('GET', self.stream_endpoint) as response:
                    if response.status_code != 200:
                        print(f"❌ Camera server błąd: {response.status_code}")
                        yield b''
                        return
                    
                    print(f"✅ Połączono z camera-server: {self.stream_endpoint}")
                    
                    # Streamuj bajty bezpośrednio bez przetwarzania
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        yield chunk
                        
        except httpx.ConnectError as e:
            print(f"❌ Nie można połączyć się z camera-server: {e}")
            print(f"   Sprawdź czy camera-server działa na: {self.camera_server_url}")
            yield b''
        except Exception as e:
            print(f"❌ Błąd streamowania: {e}")
            yield b''
    
    async def health_check(self) -> dict:
        """Sprawdza status camera-server"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.health_endpoint)
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "camera_server": response.json()
                    }
                return {
                    "status": "unhealthy",
                    "code": response.status_code
                }
        except httpx.ConnectError:
            return {
                "status": "error",
                "message": f"Cannot connect to camera-server at {self.camera_server_url}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def capture_frame_from_stream(self) -> bytes:
        """
        Wyciąga pojedynczą klatkę JPEG ze strumienia MJPEG.
        
        Działa w Dockerze! Nie wymaga opencv-python.
        Parsuje strumień MJPEG i wyciąga pierwszą kompletną klatkę JPEG.
        
        Returns:
            bytes: Surowe dane JPEG obrazu lub b'' w przypadku błędu
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                async with client.stream('GET', self.stream_endpoint) as response:
                    if response.status_code != 200:
                        print(f"❌ Camera server błąd: {response.status_code}")
                        return b''
                    
                    print(f"✅ Pobieranie klatki ze strumienia: {self.stream_endpoint}")
                    
                    # Bufor na dane
                    buffer = b''
                    jpeg_data = b''
                    in_jpeg = False
                    
                    # Czytaj strumień chunk po chunk
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        buffer += chunk
                        
                        # Szukaj początku JPEG (FF D8)
                        if not in_jpeg:
                            jpeg_start = buffer.find(b'\xff\xd8')
                            if jpeg_start != -1:
                                in_jpeg = True
                                buffer = buffer[jpeg_start:]  # Odetnij wszystko przed JPEG
                                jpeg_data = buffer
                        else:
                            jpeg_data += chunk
                        
                        # Szukaj końca JPEG (FF D9)
                        if in_jpeg:
                            jpeg_end = jpeg_data.find(b'\xff\xd9')
                            if jpeg_end != -1:
                                # Znaleziono kompletny JPEG!
                                jpeg_frame = jpeg_data[:jpeg_end + 2]  # +2 dla FF D9
                                print(f"✅ Pobrano klatkę: {len(jpeg_frame)} bytes")
                                return jpeg_frame
                        
                        # Bezpieczeństwo: jeśli bufor przekroczy 5MB, coś jest nie tak
                        if len(jpeg_data) > 5 * 1024 * 1024:
                            print("❌ Przekroczono maksymalny rozmiar bufora")
                            return b''
                    
                    # Jeśli dotarliśmy tu, stream się zakończył bez kompletnej klatki
                    print("❌ Stream zakończył się przed znalezieniem klatki")
                    return b''
                    
        except httpx.ConnectError as e:
            print(f"❌ Nie można połączyć się z camera-server: {e}")
            return b''
        except Exception as e:
            print(f"❌ Błąd pobierania klatki: {e}")
            return b''

