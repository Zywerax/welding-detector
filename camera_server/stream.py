## uvicorn camera_server.stream:app --host 0.0.0.0 --port 8001


import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from camera_server.camera_USB_service import Camera_USB_Service
from app.config import settings

app = FastAPI(
    title="Camera Server",
    description="Lokalny serwer dostępu do kamery USB dla welding detector",
    version="1.0.0"
)
camera = Camera_USB_Service()

# Oblicz interwał między klatkami na podstawie FPS
FRAME_INTERVAL = 1.0 / settings.CAMERA_USB_FPS

async def generate_frames():
    """Async generator klatek MJPEG - minimalne opóźnienie."""
    while True:
        frame = camera.get_frame()
        
        if frame:
            # Wyślij klatkę natychmiast
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame)).encode() + b'\r\n'
                   b'\r\n' + frame + b'\r\n')
        
        # Kontrola tempa - async sleep nie blokuje event loop
        await asyncio.sleep(FRAME_INTERVAL)

@app.get("/stream", tags=["Camera"])
async def stream():
    """
    Zwraca ciągłe klatki z kamery w formacie MJPEG.
    Endpoint do strumieniowania wideo z minimalnym opóźnieniem.
    """
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers
    )

@app.get("/capture", tags=["Camera"])
def capture():
    """
    Zwraca pojedynczą klatkę z kamery jako obraz JPEG.
    Endpoint do robienia zdjęć lub pobierania snapshot'ów.
    """
    frame = camera.get_frame()
    if frame:
        return Response(content=frame, media_type="image/jpeg")
    else:
        return Response(content=b"", status_code=503, media_type="text/plain")

@app.get("/stats", tags=["Camera"])
def get_camera_stats():
    """
    Endpoint zwracający szczegółowe statystyki kamery.
    Użyteczny do monitorowania i debugowania.
    """
    return camera.get_stats()

 