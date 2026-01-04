"""Pytest Fixtures - Welding Detector Tests."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import cv2
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def valid_jpeg_bytes() -> bytes:
    """Generuje prawidłowy obraz JPEG 640x480."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)
    frame[:, :, 1] = 128
    frame[:, :, 2] = 64
    cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


@pytest.fixture(scope="session")
def small_jpeg_bytes() -> bytes:
    """Mały obraz JPEG 100x100."""
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Resetuje singletony przed każdym testem."""
    import app.services.frame_overlay_service as overlay_module
    import app.services.camera_service as camera_module
    
    overlay_module._overlay_service = None
    camera_module._camera_service = None
    yield
    overlay_module._overlay_service = None
    camera_module._camera_service = None


@pytest.fixture
def temp_recordings_dir(tmp_path: Path):
    """Tymczasowy folder na nagrania."""
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    return recordings


@pytest.fixture
def mock_camera_service(valid_jpeg_bytes: bytes):
    """Mock CameraService."""
    mock = MagicMock()
    mock.get_frame.return_value = valid_jpeg_bytes
    mock.get_single_frame = AsyncMock(return_value=valid_jpeg_bytes)
    mock.monochrome = False
    mock.is_recording = False
    mock._frame_count = 0
    mock.fps = 30
    mock.width = 1920
    mock.height = 1080
    mock.jpeg_quality = 95
    
    async def mock_stream():
        for _ in range(5):
            yield valid_jpeg_bytes
    
    mock.stream_frames = mock_stream
    mock.stream_raw = mock_stream
    mock.health_check = AsyncMock(return_value={"status": "healthy", "camera_index": 0, "fps": 30, "resolution": "1920x1080", "is_recording": False})
    mock.get_settings.return_value = {"contrast": 50, "fps": 30, "jpeg_quality": 95, "resolution": "FHD"}
    mock.apply_settings.return_value = {}
    mock.start_recording.return_value = "test.mp4"
    mock.stop_recording.return_value = {"filename": "test.mp4", "duration_seconds": 10, "frames": 300, "size_mb": 5.0}
    mock.get_recording_duration.return_value = 0
    mock.list_recordings.return_value = []
    mock.get_recording_path.return_value = None
    mock.delete_recording.return_value = False
    return mock


@pytest.fixture
def mock_overlay_service():
    """Mock FrameOverlayService."""
    mock = MagicMock()
    mock.is_recording = False
    mock.get_recording_duration.return_value = None
    mock.apply_overlay_to_jpeg = lambda x: x
    return mock


@pytest.fixture
def test_client(mock_camera_service, mock_overlay_service):
    """FastAPI TestClient z mockami."""
    from app.main import app
    from app.services.camera_service import get_camera_service
    from app.services.frame_overlay_service import get_overlay_service
    
    app.dependency_overrides[get_camera_service] = lambda: mock_camera_service
    app.dependency_overrides[get_overlay_service] = lambda: mock_overlay_service
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()
