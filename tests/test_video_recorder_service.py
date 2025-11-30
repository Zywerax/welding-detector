"""
Testy jednostkowe - VideoRecorderService.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import patch


class TestVideoRecorderService:
    """Testy dla VideoRecorderService."""
    
    # ============== START/STOP TESTS ==============
    
    @pytest.mark.unit
    def test_start_creates_filename(self, patch_recordings_dir: Path):
        """Sprawdza czy start() zwraca prawidłową nazwę pliku."""
        from app.services.video_recorder_service import VideoRecorderService
        
        # Reset singleton
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            filename = service.start()
        
        assert filename.startswith("rec_"), f"Nazwa powinna zaczynać się od 'rec_': {filename}"
        assert filename.endswith(".mp4"), f"Nazwa powinna kończyć się na '.mp4': {filename}"
        assert service.is_recording is True
    
    @pytest.mark.unit
    def test_start_sets_recording_state(self, patch_recordings_dir: Path):
        """Sprawdza czy start() ustawia stan nagrywania."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            
            assert service.is_recording is False
            assert service.frame_count == 0
            
            service.start()
            
            assert service.is_recording is True
            assert service.start_time is not None
    
    @pytest.mark.unit
    def test_double_start_returns_same_file(self, patch_recordings_dir: Path):
        """Sprawdza czy podwójny start zwraca tę samą nazwę pliku."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            
            filename1 = service.start()
            filename2 = service.start()
            
            assert filename1 == filename2, "Podwójny start powinien zwrócić tę samą nazwę"
    
    @pytest.mark.unit
    def test_stop_without_start_returns_empty(self, patch_recordings_dir: Path):
        """Sprawdza czy stop() bez start() zwraca pusty dict."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            result = service.stop()
        
        assert result == {}, "Stop bez start powinien zwrócić pusty dict"
    
    @pytest.mark.unit
    def test_stop_returns_stats(self, patch_recordings_dir: Path, valid_jpeg_bytes: bytes):
        """Sprawdza czy stop() zwraca prawidłowe statystyki."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            
            filename = service.start()
            
            # Dodaj kilka klatek
            for _ in range(5):
                service.add_frame(valid_jpeg_bytes)
            
            time.sleep(0.1)  # Krótka przerwa dla duration
            result = service.stop()
        
        assert "filename" in result
        assert result["filename"] == filename
        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0.1
        assert "frames" in result
        assert result["frames"] == 5
        assert "size_mb" in result
    
    # ============== ADD FRAME TESTS ==============
    
    @pytest.mark.unit
    def test_add_frame_increments_count(self, patch_recordings_dir: Path, valid_jpeg_bytes: bytes):
        """Sprawdza czy add_frame() zwiększa licznik klatek."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            service.start()
            
            assert service.frame_count == 0
            
            service.add_frame(valid_jpeg_bytes)
            assert service.frame_count == 1
            
            service.add_frame(valid_jpeg_bytes)
            assert service.frame_count == 2
            
            service.stop()
    
    @pytest.mark.unit
    def test_add_frame_without_recording_ignored(self, patch_recordings_dir: Path, valid_jpeg_bytes: bytes):
        """Sprawdza czy add_frame() jest ignorowane gdy nie nagrywamy."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            
            # Nie startujemy nagrywania
            service.add_frame(valid_jpeg_bytes)
            
            assert service.frame_count == 0, "Klatki nie powinny być dodawane bez nagrywania"
    
    @pytest.mark.unit
    def test_add_frame_handles_invalid_jpeg(self, patch_recordings_dir: Path, invalid_jpeg_bytes: bytes):
        """Sprawdza czy nieprawidłowy JPEG nie crashuje serwisu."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            service.start()
            
            # Nie powinno rzucić wyjątku
            service.add_frame(invalid_jpeg_bytes)
            
            # Klatka nie powinna być dodana (decode failed)
            assert service.frame_count == 0
            
            service.stop()
    
    # ============== LIST FILES TESTS ==============
    
    @pytest.mark.unit
    def test_list_files_empty(self, patch_recordings_dir: Path):
        """Sprawdza czy list_files() zwraca pustą listę gdy brak nagrań."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            files = service.list_files()
        
        assert files == [], "Lista powinna być pusta"
    
    @pytest.mark.unit
    def test_list_files_returns_recordings(self, patch_recordings_dir: Path, valid_jpeg_bytes: bytes):
        """Sprawdza czy list_files() zwraca nagrania."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            
            # Utwórz nagranie
            service.start()
            service.add_frame(valid_jpeg_bytes)
            service.stop()
            
            files = service.list_files()
        
        assert len(files) == 1
        assert "filename" in files[0]
        assert "size_mb" in files[0]
        assert "created" in files[0]
    
    # ============== PATH SECURITY TESTS ==============
    
    @pytest.mark.unit
    def test_get_path_blocks_traversal(self, patch_recordings_dir: Path):
        """Sprawdza czy get_path() blokuje path traversal."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            
            # Próba path traversal
            assert service.get_path("../etc/passwd") is None
            assert service.get_path("..\\windows\\system32") is None
            assert service.get_path("/etc/passwd") is None
    
    @pytest.mark.unit
    def test_get_path_nonexistent_file(self, patch_recordings_dir: Path):
        """Sprawdza czy get_path() zwraca None dla nieistniejącego pliku."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            result = service.get_path("nonexistent.mp4")
        
        assert result is None
    
    @pytest.mark.unit
    def test_get_path_existing_file(self, patch_recordings_dir: Path, valid_jpeg_bytes: bytes):
        """Sprawdza czy get_path() zwraca ścieżkę dla istniejącego pliku."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service = VideoRecorderService()
            
            # Utwórz nagranie
            filename = service.start()
            service.add_frame(valid_jpeg_bytes)
            service.stop()
            
            path = service.get_path(filename)
        
        assert path is not None
        assert path.exists()
        assert path.name == filename
    
    # ============== SINGLETON TESTS ==============
    
    @pytest.mark.unit
    def test_singleton_pattern(self, patch_recordings_dir: Path):
        """Sprawdza czy VideoRecorderService jest singletonem."""
        from app.services.video_recorder_service import VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service1 = VideoRecorderService()
            service2 = VideoRecorderService()
        
        assert service1 is service2, "Singleton powinien zwracać tę samą instancję"
    
    @pytest.mark.unit
    def test_get_recorder_service_singleton(self, patch_recordings_dir: Path):
        """Sprawdza czy get_recorder_service() zwraca singleton."""
        from app.services.video_recorder_service import get_recorder_service, VideoRecorderService
        
        VideoRecorderService._instance = None
        
        with patch('app.services.video_recorder_service.RECORDINGS_DIR', patch_recordings_dir):
            service1 = get_recorder_service()
            service2 = get_recorder_service()
        
        assert service1 is service2
