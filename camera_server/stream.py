## uvicorn camera_server.stream:app --host 0.0.0.0 --port 8001



from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response
from camera_server.camera_USB_service import Camera_USB_Service
import time

app = FastAPI(
    title="Camera Server",
    description="Lokalny serwer dostępu do kamery USB dla welding detector",
    version="1.0.0"
)
camera = Camera_USB_Service()

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Jeśli nie ma klatki, czekaj chwilę
            time.sleep(0.1)

@app.get("/stream", tags=["Camera"])
def stream():
    """
    Zwraca ciągłe klatki z kamery w formacie MJPEG.
    Endpoint do strumieniowania wideo.
    """
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

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

 