from fastapi import FastAPI
from starlette.responses import StreamingResponse
import cv2
import threading
import time

app = FastAPI()

# --- Globalny obiekt kamery (otwieramy raz) ---
CAMERA_INDEX = 1  # Zmieniono z 1 na 0 (domyślna kamera)
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    # Jeśli kamera nie otwarta — logujemy (ale nie crashujemy aplikacji od razu)
    print(f"❌ Nie można otworzyć kamery index={CAMERA_INDEX}. Sprawdź, czy inna aplikacja nie używa kamery.")
    print("⚠️  Spróbuj zmienić CAMERA_INDEX na inną wartość (0, 1, 2...)")

# --- Opcjonalny worker odczytujący klatki w tle (bardziej stabilne) ---
# Możesz pominąć ten mechanizm i czytać klatki bezpośrednio w generatorze,
# ale worker pozwala buforować ostatnią klatkę i zmniejsza blocking w głównym wątku.
latest_frame = None
frame_lock = threading.Lock()
stop_worker = False
def camera_worker():
    global latest_frame, stop_worker
    while not stop_worker:
        if not cap.isOpened():
            time.sleep(0.1)
            continue
        ret, frame = cap.read()
        if not ret:
            # Nie udało się odczytać klatki, spróbuj ponownie
            time.sleep(0.05)
            continue
        with frame_lock:
            latest_frame = frame
    # Koniec pętli -> release
    try:
        cap.release()
    except Exception:
        pass

# Uruchom worker w tle
worker_thread = threading.Thread(target=camera_worker, daemon=True)
worker_thread.start()

def generate_frames():
    """Generator zwracający klatki JPEG w formacie MJPEG."""
    global latest_frame
    boundary = b'--frame\r\n'
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            # Jeśli nie mamy klatki, poczekaj i spróbuj dalej
            time.sleep(0.05)
            continue

        # Kodowanie klatki do JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            # nieudane kodowanie - pomiń klatkę
            continue

        frame_bytes = buffer.tobytes()

        # Zwracamy fragment zgodny z MJPEG/multipart
        yield (boundary +
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/stream")
def stream():
    """
    Endpoint zwracający MJPEG stream.
    Użyj w przeglądarce: http://127.0.0.1:8000/stream
    """
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# Opcjonalnie endpoint zwracający pojedynczą klatkę (JPEG)
@app.get("/frame.jpg")
def single_frame():
    global latest_frame
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        # Zwróć prosty obraz lub HTTP 503
        from starlette.responses import Response
        return Response(content="No frame", status_code=503)

    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        from starlette.responses import Response
        return Response(content="Failed to encode", status_code=500)

    return StreamingResponse(iter([buffer.tobytes()]), media_type="image/jpeg")

# Cleanup (przy shutdown można ustawić stop_worker = True — opcjonalne)
