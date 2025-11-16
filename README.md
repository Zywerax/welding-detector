# 🎥 Welding Detector

System detekcji wad w procesie spawania z wykorzystaniem kamery USB i FastAPI.

## 📋 Opis

Welding Detector to mikroserwisowa aplikacja do monitorowania procesu spawania w czasie rzeczywistym. System składa się z dwóch głównych komponentów:

- **Camera-Server** (localhost) - bezpośredni dostęp do kamery USB z użyciem OpenCV
- **Backend API** (Docker) - API do streamingu wideo i przetwarzania obrazu

## 🏗️ Architektura

```
┌─────────────────┐
│  Camera (USB)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│   Camera-Server         │  Port 8001 (localhost)
│   - opencv-python       │
│   - FastAPI             │
│   Endpoints:            │
│   • GET /stream         │  MJPEG stream
│   • GET /capture        │  Single JPEG frame
│   • GET /health         │  Health check
└───────────┬─────────────┘
            │ HTTP
            ▼
┌─────────────────────────┐
│   Backend API (Docker)  │  Port 8000
│   - FastAPI             │
│   - httpx (no OpenCV!)  │
│   Endpoints:            │
│   • GET /stream         │  Proxy MJPEG
│   • GET /capture        │  Extract JPEG from stream
│   • GET /health         │  Status check
│   • GET /docs           │  API documentation
└─────────────────────────┘
```

## ✨ Główne funkcje

### 1. **Video Streaming (`/stream`)**
- MJPEG stream w czasie rzeczywistym
- Proxy bez dekodowania (działa w Docker)
- Format: `multipart/x-mixed-replace`

### 2. **Frame Capture (`/capture`)** 🆕
- Pojedyncza klatka jako JPEG
- **Parsuje MJPEG bez OpenCV** - działa w Docker!
- Idealny do analizy obrazu i ML

### 3. **Health Monitoring (`/health`)**
- Status API i camera-server
- Informacje o połączeniu z kamerą

## 🚀 Instalacja i uruchomienie

### Wymagania
- Python 3.11+
- Kamera USB
- Docker Desktop (opcjonalnie)

### 1. Instalacja zależności
```bash
pip install -r requirements.txt
```

### 2. Konfiguracja
Utwórz plik `.env`:
```env
CAMERA_SERVER_URL=http://localhost:8001
APP_TITLE=Welding Vision API
DEBUG=False
```

### 3. Uruchomienie Camera-Server
```bash
# W pierwszym terminalu
uvicorn camera_server.stream:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Uruchomienie Backend API
```bash
# W drugim terminalu
uvicorn app.main:app --reload
```

Lub z Docker:
```bash
docker-compose up
```

## 📡 API Endpoints

### Backend API (http://localhost:8000)

#### `GET /`
Informacje o API
```json
{
  "status": "running",
  "camera_url": "http://localhost:8001",
  "endpoints": {
    "stream": "/stream - MJPEG video stream",
    "capture": "/capture - Single frame (JPEG image)",
    "health": "/health - API and camera health check",
    "docs": "/docs - Interactive API documentation"
  }
}
```

#### `GET /stream`
MJPEG video stream
```bash
# Przeglądarka
http://localhost:8000/stream

# HTML
<img src="http://localhost:8000/stream" />
```

#### `GET /capture` 🆕
Pojedyncza klatka JPEG
```bash
# cURL
curl http://localhost:8000/capture -o zdjecie.jpg

# Python
import requests
frame = requests.get("http://localhost:8000/capture").content
with open("foto.jpg", "wb") as f:
    f.write(frame)

# PowerShell
Invoke-WebRequest -Uri http://localhost:8000/capture -OutFile foto.jpg
```

#### `GET /health`
Status systemu
```json
{
  "api": "healthy",
  "camera_service": {
    "status": "healthy",
    "camera_server": {
      "status": "healthy",
      "camera": "connected",
      "frame_size": 45678
    }
  }
}
```

#### `GET /docs`
Interaktywna dokumentacja Swagger UI
```
http://localhost:8000/docs
```

### Camera-Server (http://localhost:8001)

#### `GET /stream`
Bezpośredni stream z kamery

#### `GET /capture`
Bezpośrednia klatka z kamery

#### `GET /health`
Status kamery

## 💡 Przykłady użycia

### Python - Pobieranie klatek
```python
import requests
from datetime import datetime

while True:
    # Pobierz klatkę
    response = requests.get("http://localhost:8000/capture")
    
    if response.status_code == 200:
        # Zapisz
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"frame_{timestamp}.jpg", "wb") as f:
            f.write(response.content)
        
        print(f"Saved frame_{timestamp}.jpg")
    
    time.sleep(1)  # Co sekundę
```

### Python - Analiza z PIL
```python
import requests
from PIL import Image
from io import BytesIO

response = requests.get("http://localhost:8000/capture")
img = Image.open(BytesIO(response.content))

print(f"Rozdzielczość: {img.size}")
img.show()
```

### HTML - Live preview
```html
<!DOCTYPE html>
<html>
<head>
    <title>Welding Camera</title>
</head>
<body>
    <h1>Live Stream</h1>
    <img src="http://localhost:8000/stream" width="640" />
    
    <h1>Snapshot</h1>
    <img id="snapshot" src="http://localhost:8000/capture" />
    
    <script>
        // Odświeżaj snapshot co sekundę
        setInterval(() => {
            document.getElementById('snapshot').src = 
                'http://localhost:8000/capture?' + Date.now();
        }, 1000);
    </script>
</body>
</html>
```

## 🧪 Testowanie

### Uruchom testy jednostkowe
```bash
pytest tests/ -v
```

### Test endpointu /capture
```bash
python test_capture_from_stream.py
```

### Przykładowe wyniki testów
```
✅ 38/38 testów PASS
✅ Coverage: 100% krytycznej funkcjonalności
✅ Testy API, serwisów, konfiguracji, integracji
```

## 🐳 Docker

### Uruchomienie z docker-compose
```bash
docker-compose up
```

### Konfiguracja dla Docker
W `.env` ustaw:
```env
CAMERA_SERVER_URL=http://host.docker.internal:8001
```

### Dlaczego Camera-Server NIE jest w Docker?
❌ Docker na Windows nie ma dostępu do USB kamery  
✅ Camera-Server działa na hoście (localhost:8001)  
✅ Backend API w Docker łączy się przez `host.docker.internal`

## 📚 Dokumentacja

- [Endpoint /capture - Szczegóły](docs/CAPTURE_FROM_STREAM.md)
- [Swagger UI](http://localhost:8000/docs) - Interaktywna dokumentacja
- [OpenAPI Schema](http://localhost:8000/openapi.json)

## 🔧 Konfiguracja

### Zmienne środowiskowe (.env)
```env
# Camera Server URL
CAMERA_SERVER_URL=http://localhost:8001          # Lokalne
# CAMERA_SERVER_URL=http://host.docker.internal:8001  # Docker

# API Settings
APP_TITLE=Welding Vision API
DEBUG=False

# Camera Settings (camera_server)
CAMERA_INDEX=0
```

### Camera Service
```python
# app/services/camera_service.py
CAMERA_INDEX = 0  # Zmień jeśli masz wiele kamer
```

## ⚡ Wydajność

### Typowe wartości:
- **Stream:** ~30 FPS, ~2-5 MB/s
- **Capture:** ~200ms/request, ~50 KB/frame
- **Health check:** <100ms

### Optymalizacja:
- Chunk size: 8192 bytes (8KB)
- Timeout stream: 30s
- Timeout capture: 10s

## 🐛 Troubleshooting

### "Camera unavailable"
1. Sprawdź czy kamera jest podłączona
2. Sprawdź czy camera-server działa: `curl http://localhost:8001/health`
3. Sprawdź indeks kamery w `camera_service.py`

### "Connection refused"
1. Upewnij się że camera-server działa na porcie 8001
2. Sprawdź `CAMERA_SERVER_URL` w `.env`
3. Dla Docker użyj `host.docker.internal:8001`

### Stream nie działa
1. Sprawdź logi camera-server
2. Sprawdź czy inna aplikacja nie używa kamery
3. Zrestartuj camera-server

### ⚠️ Błąd MSMF: "can't grab frame. Error: -1072875772"
**Status:** ✅ **ROZWIĄZANY**

System został zaktualizowany o profesjonalne rozwiązanie tego problemu:

**Implementowane poprawki:**
- ✅ DirectShow backend (stabilniejszy niż MSMF)
- ✅ Thread-safety z `threading.Lock`
- ✅ Retry logic z exponential backoff
- ✅ Automatic reconnection
- ✅ Frame caching dla graceful degradation
- ✅ Comprehensive error handling & logging

**Szczegóły:** Zobacz [docs/CAMERA_STABILITY.md](docs/CAMERA_STABILITY.md)

**Weryfikacja:**
```bash
# Quick test
python tests/test_camera_stability.py

# Pełny test suite
pytest tests/test_camera_stability.py -v
```

## 📝 TODO / Roadmap

- [ ] Detekcja wad spawania (ML model)
- [ ] WebSocket dla real-time events
- [ ] Zapisywanie historii klatek
- [ ] Panel admina
- [ ] Alerty email/SMS przy wykryciu wad

## 🤝 Contributing

Pull requesty mile widziane! Przed większymi zmianami otwórz issue.

## 📄 Licencja

MIT

## 👨‍💻 Autor

Zywerax

---

**Status:** 🟢 Aktywny rozwój  
**Wersja:** 1.0.0  
**Python:** 3.11+  
**FastAPI:** 0.104.1
