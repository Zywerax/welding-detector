"""
Pytest Fixtures - Welding Detector Tests.
"""

import sys
from pathlib import Path

# Dodaj folder projektu do PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import cv2 # type: ignore
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# ============== JPEG FIXTURES ==============

@pytest.fixture(scope="session")
def valid_jpeg_bytes() -> bytes:
    """Generuje prawidłowy obraz JPEG 640x480 do testów."""
    # Tworzenie kolorowego obrazu testowego
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Gradient tła
    frame[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)  # Blue gradient
    frame[:, :, 1] = 128  # Green
    frame[:, :, 2] = 64   # Red
    
    # Biały prostokąt w centrum (symulacja obiektu)
    cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
    
    # Enkodowanie do JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


@pytest.fixture(scope="session")
def small_jpeg_bytes() -> bytes:
    """Mały obraz JPEG 100x100 do szybkich testów."""
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


@pytest.fixture(scope="session")
def invalid_jpeg_bytes() -> bytes:
    """Nieprawidłowe dane - nie jest to JPEG."""
    return b"This is not a valid JPEG image data"


# ============== SINGLETON RESET ==============

@pytest.fixture(autouse=True)
def reset_singletons():
    """Resetuje singletony przed każdym testem."""
    # Import singletonów
    import app.services.frame_overlay_service as overlay_module
    import app.services.video_recorder_service as recorder_module
    import app.services.remote_camera_service as camera_module
    
    # Reset przed testem
    overlay_module._overlay_service = None
    recorder_module._recorder = None
    camera_module._camera_service = None
    
    # Reset klasy singleton (dla VideoRecorderService)
    if hasattr(recorder_module.VideoRecorderService, '_instance'):
        recorder_module.VideoRecorderService._instance = None
    
    yield
    
    # Cleanup po teście
    overlay_module._overlay_service = None
    recorder_module._recorder = None
    camera_module._camera_service = None
    
    if hasattr(recorder_module.VideoRecorderService, '_instance'):
        recorder_module.VideoRecorderService._instance = None


# ============== TEMP RECORDINGS DIR ==============

@pytest.fixture
def temp_recordings_dir(tmp_path: Path):
    """Tymczasowy folder na nagrania."""
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    return recordings


@pytest.fixture
def patch_recordings_dir(temp_recordings_dir: Path):
    """Patchuje RECORDINGS_DIR w video_recorder_service."""
    with patch('app.services.video_recorder_service.RECORDINGS_DIR', temp_recordings_dir):
        yield temp_recordings_dir


# ============== MOCK SERVICES ==============

@pytest.fixture
def mock_camera_service(valid_jpeg_bytes: bytes):
    """Mock RemoteCameraService z predefiniowaną klatką."""
    mock = MagicMock()
    mock.get_single_frame = AsyncMock(return_value=valid_jpeg_bytes)
    mock._last_frame = valid_jpeg_bytes
    
    async def mock_stream():
        for _ in range(5):
            yield valid_jpeg_bytes
    
    mock.stream_frames = mock_stream
    mock.health_check = AsyncMock(return_value={
        "status": "healthy",
        "camera_url": "http://test:8001",
        "response_code": 200,
        "has_cached_frame": True
    })
    return mock


@pytest.fixture
def mock_overlay_service():
    """Mock FrameOverlayService."""
    mock = MagicMock()
    mock.is_recording = False
    mock.get_recording_duration.return_value = None
    mock.apply_overlay_to_jpeg = lambda x: x  # Passthrough
    return mock


@pytest.fixture
def mock_recorder_service(temp_recordings_dir: Path):
    """Mock VideoRecorderService z temp directory."""
    mock = MagicMock()
    mock.is_recording = False
    mock.frame_count = 0
    mock.start.return_value = "test_recording.mp4"
    mock.stop.return_value = {
        "filename": "test_recording.mp4",
        "duration_seconds": 10.5,
        "frames": 150,
        "size_mb": 2.5
    }
    mock.list_files.return_value = []
    mock.get_path.return_value = None
    return mock


# ============== TEST CLIENT ==============

@pytest.fixture
def test_client(mock_camera_service, mock_overlay_service, mock_recorder_service):
    """FastAPI TestClient z mockami serwisów."""
    from app.main import app
    from app.services.remote_camera_service import get_camera_service
    from app.services.frame_overlay_service import get_overlay_service
    from app.services.video_recorder_service import get_recorder_service
    
    # Override dependencies
    app.dependency_overrides[get_camera_service] = lambda: mock_camera_service
    app.dependency_overrides[get_overlay_service] = lambda: mock_overlay_service
    app.dependency_overrides[get_recorder_service] = lambda: mock_recorder_service
    
    with TestClient(app) as client:
        yield client
    
    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def test_client_real_services(patch_recordings_dir):
    """TestClient z prawdziwymi serwisami (bez kamery)."""
    from app.main import app
    from app.services.remote_camera_service import get_camera_service
    
    # Tylko mock kamery, reszta prawdziwa
    mock_camera = MagicMock()
    mock_camera.get_single_frame = AsyncMock(return_value=None)
    mock_camera.health_check = AsyncMock(return_value={"status": "disconnected"})
    
    app.dependency_overrides[get_camera_service] = lambda: mock_camera
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()
