## uvicorn camera_server.stream:app --host 0.0.0.0 --port 8001



from fastapi import APIRouter
from fastapi.responses import StreamingResponse, Response
from app.services.camera_service import CameraService
import time

app = APIRouter()
camera = CameraService()

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Jeśli nie ma klatki, czekaj chwilę
            time.sleep(0.1)

@app.get("/stream")
def stream():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/capture")
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

@app.get("/health")
def health():
    """
    Health check endpoint - sprawdza czy kamera jest dostępna.
    """
    frame = camera.get_frame()
    if frame:
        return {
            "status": "healthy",
            "camera": "connected",
            "frame_size": len(frame)
        }
    else:
        return {
            "status": "unhealthy",
            "camera": "disconnected"
        }
