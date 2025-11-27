"""
Frame Overlay Service - Nakładanie timestampa i wskaźnika nagrywania na klatki.

Funkcjonalności:
- Timestamp w formacie YYYY-MM-DD HH:MM:SS.mmm
- Czerwona migająca kropka podczas nagrywania
- Konfigurowalny wygląd (pozycja, kolor, rozmiar)
"""

import cv2  # type: ignore
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class FrameOverlayService:
    """
    Serwis do nakładania overlayów na klatki wideo.
    
    Features:
    - Timestamp z milisekundami
    - Wskaźnik nagrywania (czerwona kropka)
    - Miganie kropki (blink effect)
    - Tło pod tekstem dla lepszej czytelności
    """
    
    def __init__(self):
        """Inicjalizacja serwisu overlay."""
        # Stan nagrywania
        self._is_recording = False
        self._recording_start_time: Optional[datetime] = None
        
        # Ustawienia timestamp
        self.timestamp_position = "top-left"  # top-left, top-right, bottom-left, bottom-right
        self.timestamp_font = cv2.FONT_HERSHEY_SIMPLEX
        self.timestamp_font_scale = 0.4
        self.timestamp_color = (255, 255, 255)  # Biały
        self.timestamp_thickness = 1
        self.timestamp_bg_color = (0, 0, 0)  # Czarne tło
        self.timestamp_bg_opacity = 0.4
        self.timestamp_padding = 8
        
        # Ustawienia wskaźnika nagrywania
        self.rec_indicator_radius = 10
        self.rec_indicator_color = (0, 0, 255)  # Czerwony (BGR)
        self.rec_indicator_blink_interval = 0.5  # sekundy
        self.rec_text = "REC"
        self.rec_text_color = (0, 0, 255)  # Czerwony
        
        # Marginesy
        self.margin_x = 15
        self.margin_y = 25
        
        logger.info("🎨 FrameOverlayService initialized")
    
    @property
    def is_recording(self) -> bool:
        """Zwraca czy nagrywanie jest aktywne."""
        return self._is_recording
    
    def start_recording(self) -> None:
        """Rozpoczyna nagrywanie - aktywuje wskaźnik."""
        self._is_recording = True
        self._recording_start_time = datetime.now()
        logger.info("🔴 Recording started")
    
    def stop_recording(self) -> Optional[float]:
        """
        Zatrzymuje nagrywanie.
        
        Returns:
            Optional[float]: Czas nagrywania w sekundach lub None
        """
        if not self._is_recording:
            return None
        
        self._is_recording = False
        duration = None
        
        if self._recording_start_time:
            duration = (datetime.now() - self._recording_start_time).total_seconds()
            logger.info(f"⏹️ Recording stopped. Duration: {duration:.2f}s")
        
        self._recording_start_time = None
        return duration
    
    def get_recording_duration(self) -> Optional[float]:
        """
        Zwraca aktualny czas nagrywania.
        
        Returns:
            Optional[float]: Czas nagrywania w sekundach lub None jeśli nie nagrywa
        """
        if not self._is_recording or not self._recording_start_time:
            return None
        return (datetime.now() - self._recording_start_time).total_seconds()
    
    def _get_timestamp_text(self) -> str:
        """Generuje tekst timestamp z milisekundami."""
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"
    
    def _should_show_rec_indicator(self) -> bool:
        """
        Sprawdza czy pokazać wskaźnik nagrywania (efekt migania).
        
        Returns:
            bool: True jeśli kropka powinna być widoczna
        """
        if not self._is_recording:
            return False
        
        # Miganie co rec_indicator_blink_interval sekund
        elapsed = time.time() % (self.rec_indicator_blink_interval * 2)
        return elapsed < self.rec_indicator_blink_interval
    
    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font: int,
        font_scale: float,
        text_color: Tuple[int, int, int],
        thickness: int,
        bg_color: Tuple[int, int, int],
        padding: int = 5
    ) -> np.ndarray:
        """
        Rysuje tekst z półprzezroczystym tłem.
        
        Args:
            frame: Klatka obrazu
            text: Tekst do narysowania
            position: Pozycja (x, y) - lewy dolny róg tekstu
            font: Czcionka OpenCV
            font_scale: Skala czcionki
            text_color: Kolor tekstu (BGR)
            thickness: Grubość tekstu
            bg_color: Kolor tła (BGR)
            padding: Padding wokół tekstu
            
        Returns:
            np.ndarray: Klatka z nałożonym tekstem
        """
        # Oblicz rozmiar tekstu
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        x, y = position
        
        # Współrzędne prostokąta tła
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding
        
        # Rysuj półprzezroczyste tło
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.addWeighted(overlay, self.timestamp_bg_opacity, frame, 1 - self.timestamp_bg_opacity, 0, frame)
        
        # Rysuj tekst
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return frame
    
    def _draw_recording_indicator(self, frame: np.ndarray) -> np.ndarray:
        """
        Rysuje wskaźnik nagrywania (czerwona kropka + "REC").
        
        Args:
            frame: Klatka obrazu
            
        Returns:
            np.ndarray: Klatka z wskaźnikiem nagrywania
        """
        if not self._should_show_rec_indicator():
            return frame
        
        height, width = frame.shape[:2]
        
        # Pozycja - prawy górny róg
        circle_x = width - self.margin_x - self.rec_indicator_radius
        circle_y = self.margin_y
        
        # Rysuj czerwoną kropkę
        cv2.circle(
            frame,
            (circle_x, circle_y),
            self.rec_indicator_radius,
            self.rec_indicator_color,
            -1  # Wypełniona
        )
        
        # Rysuj "REC" obok kropki
        rec_text_x = circle_x - self.rec_indicator_radius - 45
        rec_text_y = circle_y + 5
        
        cv2.putText(
            frame,
            self.rec_text,
            (rec_text_x, rec_text_y),
            self.timestamp_font,
            0.6,
            self.rec_text_color,
            2,
            cv2.LINE_AA
        )
        
        # Opcjonalnie: pokaż czas nagrywania
        duration = self.get_recording_duration()
        if duration is not None:
            duration_text = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
            duration_x = circle_x - self.rec_indicator_radius - 95
            cv2.putText(
                frame,
                duration_text,
                (duration_x, rec_text_y),
                self.timestamp_font,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return frame
    
    def apply_overlay(
        self,
        frame: np.ndarray,
        show_timestamp: bool = True,
        show_recording_indicator: bool = True
    ) -> np.ndarray:
        """
        Nakłada overlay na klatkę (timestamp + wskaźnik nagrywania).
        
        Args:
            frame: Klatka obrazu (numpy array BGR)
            show_timestamp: Czy pokazać timestamp
            show_recording_indicator: Czy pokazać wskaźnik nagrywania
            
        Returns:
            np.ndarray: Klatka z nałożonym overlayem
        """
        if frame is None:
            return frame
        
        # Kopiuj klatkę żeby nie modyfikować oryginału
        result = frame.copy()
        
        # Timestamp
        if show_timestamp:
            timestamp_text = self._get_timestamp_text()
            result = self._draw_text_with_background(
                result,
                timestamp_text,
                (self.margin_x, self.margin_y),
                self.timestamp_font,
                self.timestamp_font_scale,
                self.timestamp_color,
                self.timestamp_thickness,
                self.timestamp_bg_color,
                self.timestamp_padding
            )
        
        # Wskaźnik nagrywania
        if show_recording_indicator and self._is_recording:
            result = self._draw_recording_indicator(result)
        
        return result
    
    def apply_overlay_to_jpeg(
        self,
        jpeg_bytes: bytes,
        show_timestamp: bool = True,
        show_recording_indicator: bool = True
    ) -> Optional[bytes]:
        """
        Nakłada overlay na klatkę JPEG i zwraca jako JPEG.
        
        Args:
            jpeg_bytes: Klatka JPEG jako bytes
            show_timestamp: Czy pokazać timestamp
            show_recording_indicator: Czy pokazać wskaźnik nagrywania
            
        Returns:
            Optional[bytes]: Klatka JPEG z overlayem lub None przy błędzie
        """
        try:
            # Dekoduj JPEG
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode JPEG for overlay")
                return jpeg_bytes  # Zwróć oryginalną klatkę
            
            # Nałóż overlay
            result = self.apply_overlay(frame, show_timestamp, show_recording_indicator)
            
            # Zakoduj z powrotem do JPEG
            ret, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if not ret:
                logger.error("Failed to encode frame with overlay")
                return jpeg_bytes
            
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error applying overlay: {e}")
            return jpeg_bytes


# Singleton instance
_overlay_service: Optional[FrameOverlayService] = None


def get_overlay_service() -> FrameOverlayService:
    """
    Zwraca singleton instance FrameOverlayService.
    """
    global _overlay_service
    if _overlay_service is None:
        _overlay_service = FrameOverlayService()
    return _overlay_service
