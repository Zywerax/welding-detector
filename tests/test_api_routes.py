"""Testy API Routes."""

import pytest
from unittest.mock import AsyncMock


class TestCameraEndpoints:
    @pytest.mark.unit
    def test_camera_health(self, test_client):
        response = test_client.get("/camera/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.unit
    def test_camera_capture(self, test_client, valid_jpeg_bytes):
        response = test_client.get("/camera/capture")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content[:2] == b'\xff\xd8'
    
    @pytest.mark.unit
    def test_camera_capture_no_frame(self, test_client, mock_camera_service):
        mock_camera_service.get_single_frame = AsyncMock(return_value=None)
        response = test_client.get("/camera/capture")
        assert response.status_code == 503
    
    @pytest.mark.unit
    def test_camera_stream(self, test_client):
        response = test_client.get("/camera/stream", timeout=2)
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers["content-type"]
    
    @pytest.mark.unit
    def test_camera_settings_get(self, test_client):
        response = test_client.get("/camera/settings")
        assert response.status_code == 200
        data = response.json()
        assert "fps" in data
        assert "jpeg_quality" in data
    
    @pytest.mark.unit
    def test_camera_monochrome(self, test_client):
        response = test_client.get("/camera/monochrome")
        assert response.status_code == 200
        assert "monochrome" in response.json()


class TestEdgeEndpoints:
    @pytest.mark.unit
    def test_edge_detect_not_implemented(self, test_client):
        response = test_client.get("/edge/detect")
        assert response.status_code == 200
        assert response.json()["status"] == "not_implemented"


class TestRecordingEndpoints:
    @pytest.mark.unit
    def test_recording_status(self, test_client):
        response = test_client.get("/recording/status")
        assert response.status_code == 200
        assert "is_recording" in response.json()
    
    @pytest.mark.unit
    def test_recording_start(self, test_client, mock_camera_service):
        mock_camera_service.is_recording = False
        response = test_client.post("/recording/start")
        assert response.status_code == 200
        assert response.json()["status"] == "started"
    
    @pytest.mark.unit
    def test_recording_start_already_recording(self, test_client, mock_camera_service):
        mock_camera_service.is_recording = True
        response = test_client.post("/recording/start")
        assert response.status_code == 400
    
    @pytest.mark.unit
    def test_recording_stop_not_recording(self, test_client, mock_camera_service):
        mock_camera_service.is_recording = False
        response = test_client.post("/recording/stop")
        assert response.status_code == 400
    
    @pytest.mark.unit
    def test_recording_list(self, test_client):
        response = test_client.get("/recording/list")
        assert response.status_code == 200
        assert "recordings" in response.json()
    
    @pytest.mark.unit
    def test_recording_download_not_found(self, test_client, mock_camera_service):
        mock_camera_service.get_recording_path.return_value = None
        response = test_client.get("/recording/download/nonexistent.mp4")
        assert response.status_code == 404
    
    @pytest.mark.unit
    def test_recording_delete_not_found(self, test_client, mock_camera_service):
        mock_camera_service.delete_recording.return_value = False
        response = test_client.delete("/recording/nonexistent.mp4")
        assert response.status_code == 404
