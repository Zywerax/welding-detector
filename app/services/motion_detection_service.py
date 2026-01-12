"""
Motion Detection Service - wykrywanie ruchu w nagraniach wideo.

Serwis do detekcji segmentów z ruchem w nagraniach spawalniczych.
Używa cv2.absdiff do porównywania kolejnych klatek.
"""

import cv2  # type: ignore
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MotionSegment:
    """Segment wideo z ruchem."""
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float


@dataclass
class MotionAnalysisResult:
    """Wynik analizy ruchu w wideo."""
    filename: str
    total_frames: int
    fps: float
    duration_seconds: float
    segments: list[MotionSegment]
    motion_percentage: float  # Procent klatek z ruchem


class MotionDetectionService:
    """
    Serwis do detekcji ruchu w nagraniach wideo.
    
    Użycie:
        service = MotionDetectionService()
        result = service.detect_motion("recordings/rec_20260105_120000.mp4")
        
        # Przytnij wideo do segmentów z ruchem
        service.trim_to_motion("input.mp4", "output.mp4")
    """
    
    def __init__(
        self,
        recordings_dir: Path = Path("recordings"),
        threshold: int = 25,           # Próg różnicy pikseli
        min_area_percent: float = 0.5, # Min % powierzchni ze zmianą
        min_segment_frames: int = 5,   # Min klatek na segment
        padding_frames: int = 30       # Padding przed/po segmencie (0.5s @60fps)
    ):
        self.recordings_dir = recordings_dir
        self.threshold = threshold
        self.min_area_percent = min_area_percent
        self.min_segment_frames = min_segment_frames
        self.padding_frames = padding_frames
        logger.info("🔍 MotionDetectionService initialized")
    
    def detect_motion(
        self,
        video_path: str | Path,
        threshold: Optional[int] = None,
        min_area_percent: Optional[float] = None,
        analyze_step: int = 1
    ) -> MotionAnalysisResult:
        """
        Analizuje wideo i wykrywa segmenty z ruchem.
        
        Args:
            video_path: Ścieżka do pliku wideo
            threshold: Próg różnicy pikseli (0-255)
            min_area_percent: Minimalny % powierzchni ze zmianą
            analyze_step: Co która klatka analizować (1 = każda)
            
        Returns:
            MotionAnalysisResult z listą segmentów
        """
        path = self._resolve_path(video_path)
        threshold = threshold or self.threshold
        min_area = min_area_percent or self.min_area_percent
        
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"🎬 Analyzing motion in {path.name} ({total_frames} frames)")
            
            # Wczytaj pierwszą klatkę
            ret, prev_frame = cap.read()
            if not ret:
                raise ValueError("Cannot read first frame")
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
            
            frame_height, frame_width = prev_frame.shape[:2]
            total_pixels = frame_width * frame_height
            min_changed_pixels = int(total_pixels * min_area / 100)
            
            motion_frames: list[int] = []
            frame_idx = 1
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % analyze_step == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    
                    # Różnica między klatkami
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Policz piksele ze zmianą
                    changed_pixels = cv2.countNonZero(thresh)
                    
                    if changed_pixels >= min_changed_pixels:
                        motion_frames.append(frame_idx)
                    
                    prev_gray = gray
                
                frame_idx += 1
            
            # Grupuj klatki w segmenty
            segments = self._group_into_segments(motion_frames, total_frames, fps)
            
            motion_pct = (len(motion_frames) / (total_frames / analyze_step) * 100) if total_frames > 0 else 0
            
            logger.info(f"✅ Found {len(segments)} motion segments ({motion_pct:.1f}% motion)")
            
            return MotionAnalysisResult(
                filename=path.name,
                total_frames=total_frames,
                fps=fps,
                duration_seconds=duration,
                segments=segments,
                motion_percentage=round(motion_pct, 2)
            )
        finally:
            cap.release()
    
    def _group_into_segments(
        self, 
        motion_frames: list[int], 
        total_frames: int,
        fps: float
    ) -> list[MotionSegment]:
        """Grupuje klatki z ruchem w ciągłe segmenty."""
        if not motion_frames:
            return []
        
        segments = []
        start = motion_frames[0]
        end = motion_frames[0]
        
        # Max przerwa między klatkami w jednym segmencie (0.5s)
        max_gap = int(fps * 0.5)
        
        for frame in motion_frames[1:]:
            if frame - end <= max_gap:
                end = frame
            else:
                # Dodaj padding: pełny na początku, minimalny na końcu
                seg_start = max(0, start - self.padding_frames)
                seg_end = min(total_frames - 1, end + 5)  # Tylko 5 klatek na końcu (~0.08s)
                
                if seg_end - seg_start >= self.min_segment_frames:
                    segments.append(MotionSegment(
                        start_frame=seg_start,
                        end_frame=seg_end,
                        start_time_ms=seg_start / fps * 1000,
                        end_time_ms=seg_end / fps * 1000,
                        duration_ms=(seg_end - seg_start) / fps * 1000
                    ))
                
                start = frame
                end = frame
        
        # Ostatni segment
        seg_start = max(0, start - self.padding_frames)
        seg_end = min(total_frames - 1, end + 5)  # Tylko 5 klatek na końcu
        
        if seg_end - seg_start >= self.min_segment_frames:
            segments.append(MotionSegment(
                start_frame=seg_start,
                end_frame=seg_end,
                start_time_ms=seg_start / fps * 1000,
                end_time_ms=seg_end / fps * 1000,
                duration_ms=(seg_end - seg_start) / fps * 1000
            ))
        
        # Scal nakładające się segmenty
        return self._merge_overlapping(segments, fps)
    
    def _merge_overlapping(self, segments: list[MotionSegment], fps: float) -> list[MotionSegment]:
        """Scala nakładające się lub bliskie segmenty."""
        if len(segments) <= 1:
            return segments
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            # Jeśli segmenty nakładają się lub są blisko
            if seg.start_frame <= last.end_frame + self.padding_frames:
                # Rozszerz ostatni segment
                merged[-1] = MotionSegment(
                    start_frame=last.start_frame,
                    end_frame=max(last.end_frame, seg.end_frame),
                    start_time_ms=last.start_time_ms,
                    end_time_ms=max(last.end_frame, seg.end_frame) / fps * 1000,
                    duration_ms=(max(last.end_frame, seg.end_frame) - last.start_frame) / fps * 1000
                )
            else:
                merged.append(seg)
        
        return merged
    
    def trim_to_motion(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        threshold: Optional[int] = None,
        min_area_percent: Optional[float] = None,
        include_all_segments: bool = True
    ) -> dict:
        """
        Przycina wideo do segmentów z ruchem.
        
        Args:
            video_path: Ścieżka do wideo źródłowego
            output_path: Ścieżka wyjściowa (domyślnie: {input}_trimmed.mp4)
            threshold: Próg detekcji ruchu
            min_area_percent: Min % powierzchni ze zmianą
            include_all_segments: True = wszystkie segmenty, False = tylko najdłuższy
            
        Returns:
            Słownik z informacjami o przyciętym wideo
        """
        path = self._resolve_path(video_path)
        
        # Wykryj segmenty
        analysis = self.detect_motion(path, threshold, min_area_percent)
        
        if not analysis.segments:
            logger.warning(f"⚠️ No motion detected in {path.name}")
            return {
                "status": "no_motion",
                "filename": path.name,
                "message": "No motion segments detected"
            }
        
        # Wybierz segmenty
        if include_all_segments:
            segments = analysis.segments
        else:
            # Tylko najdłuższy segment
            segments = [max(analysis.segments, key=lambda s: s.duration_ms)]
        
        # Ustal ścieżkę wyjściową
        if output_path:
            out_path = Path(output_path)
        else:
            out_path = path.parent / f"{path.stem}_trimmed.mp4"
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Otwórz źródło
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            
            if not writer.isOpened():
                raise ValueError("Cannot create output video")
            
            frames_written = 0
            
            for seg in segments:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame)
                
                for _ in range(seg.end_frame - seg.start_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                    frames_written += 1
            
            writer.release()
            
            # Informacje o wyniku
            output_size = out_path.stat().st_size / (1024 * 1024)
            original_size = path.stat().st_size / (1024 * 1024)
            
            logger.info(f"✂️ Trimmed {path.name} -> {out_path.name} ({frames_written} frames)")
            
            return {
                "status": "completed",
                "input_filename": path.name,
                "output_filename": out_path.name,
                "output_path": str(out_path),
                "segments_count": len(segments),
                "frames_written": frames_written,
                "duration_seconds": round(frames_written / fps, 2),
                "original_size_mb": round(original_size, 2),
                "output_size_mb": round(output_size, 2),
                "reduction_percent": round((1 - output_size / original_size) * 100, 1) if original_size > 0 else 0
            }
        finally:
            cap.release()
    
    def detect_welding_process(
        self,
        video_path: str | Path,
        brightness_threshold: int = 150,
        min_bright_percent: float = 2.0
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Wykrywa moment spawania (jasne światło lasera).
        
        Args:
            video_path: Ścieżka do pliku wideo
            brightness_threshold: Próg jasności (0-255) dla detekcji spawania
            min_bright_percent: Minimalny % jasnych pikseli aby uznać za spawanie
            
        Returns:
            (start_frame, end_frame) spawania lub (None, None) jeśli nie wykryto
        """
        path = self._resolve_path(video_path)
        cap = cv2.VideoCapture(str(path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"🔍 Detecting welding process in {path.name}")
            
            welding_frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sprawdź zarówno jasność jak i kolory (biały/żółty/czerwony laser)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Metoda 1: Jasne piksele
                bright_pixels = np.sum(gray > brightness_threshold)
                
                # Metoda 2: Bardzo jasne piksele (białe/żółte centrum) - TYLKO dla aktywnego spawania
                very_bright_pixels = np.sum(gray > 220)  # Podwyższony próg
                
                # Metoda 3: Czerwone/pomarańczowe światło (spawanie może być czerwone)
                # Wysokie wartości R i G, niskie B
                b, g, r = cv2.split(frame)
                red_hot = np.sum((r > 220) & (g > 180) & (b < 120))  # Bardziej restrykcyjne
                
                total_pixels = gray.shape[0] * gray.shape[1]
                bright_percent = (bright_pixels / total_pixels) * 100
                very_bright_percent = (very_bright_pixels / total_pixels) * 100
                red_hot_percent = (red_hot / total_pixels) * 100
                
                # Wykryj spawanie jeśli:
                # - Dużo bardzo jasnych pikseli (aktywny laser)
                # - LUB intensywne czerwone światło
                is_welding = (
                    very_bright_percent >= 1.0 or  # Co najmniej 1% bardzo jasnych (aktywny laser)
                    red_hot_percent >= 3.0  # Lub 3% intensywnie czerwonych
                )
                
                if is_welding:
                    welding_frames.append(frame_idx)
                
                frame_idx += 1
            
            if not welding_frames:
                logger.info("❌ No welding process detected")
                return None, None
            
            # Znajdź ciągły segment spawania - grupuj klatki z tolerancją
            # Jeśli przerwa > 10 klatek (0.3s), to traktuj jako koniec spawania
            gap_tolerance = 10  # Zmniejszone z 30 na 10
            segments = []
            current_start = welding_frames[0]
            prev_frame = welding_frames[0]
            
            for frame in welding_frames[1:]:
                if frame - prev_frame > gap_tolerance:
                    # Koniec obecnego segmentu
                    segments.append((current_start, prev_frame))
                    current_start = frame
                prev_frame = frame
            
            # Dodaj ostatni segment
            segments.append((current_start, prev_frame))
            
            # Wybierz najdłuższy segment (główny proces spawania)
            if segments:
                longest_segment = max(segments, key=lambda s: s[1] - s[0])
                start_frame, end_frame = longest_segment
                
                logger.info(f"✅ Welding detected: frames {start_frame}-{end_frame} ({len(welding_frames)} frames total, {len(segments)} segments)")
                return start_frame, end_frame
            
            return None, None
            
        finally:
            cap.release()
    
    def trim_to_post_processing(
        self,
        video_path: str | Path,
        output_path: Optional[Path] = None,
        brightness_threshold: int = 150,
        min_bright_percent: float = 2.0
    ) -> dict:
        """
        Przycina wideo usuwając TYLKO proces spawania (jasne światło).
        Zachowuje wszystko przed i po spawaniu.
        
        Args:
            video_path: Ścieżka do wideo wejściowego
            output_path: Ścieżka wyjściowa (opcjonalna)
            brightness_threshold: Próg jasności dla detekcji spawania
            min_bright_percent: Minimalny % jasnych pikseli
            
        Returns:
            Dict z informacjami o wyniku
        """
        path = self._resolve_path(video_path)
        
        # Wykryj moment spawania
        weld_start, weld_end = self.detect_welding_process(
            path, 
            brightness_threshold=brightness_threshold,
            min_bright_percent=min_bright_percent
        )
        
        # Ustal ścieżkę wyjściową
        if output_path is None:
            stem = path.stem
            output_path = path.parent / f"{stem}_postprocess.mp4"
        
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Jeśli nie wykryto spawania, zachowaj całe wideo
            if weld_start is None or weld_end is None:
                logger.warning("No welding detected - keeping entire video")
                segments = [(0, total_frames - 1)]
            else:
                # Dodaj buffer tylko na POCZĄTKU spawania (1 klatka)
                # Koniec zostawiamy dokładny - moment zgaśnięcia lasera
                buffer_frames = 1
                weld_start_buffered = max(0, weld_start - buffer_frames)
                weld_end_buffered = weld_end  # Bez bufora na końcu!
                
                # Sprawdź czy spawanie zajmuje większość wideo (>80%)
                weld_duration = weld_end_buffered - weld_start_buffered + 1
                weld_percent = (weld_duration / total_frames) * 100
                
                if weld_percent > 80:
                    # Spawanie wykryte w prawie całym wideo
                    # Prawdopodobnie gotowy spaw też jest wykrywany jako jasny
                    # Zachowaj drugą połowę wideo (post-processing/inspekcja)
                    logger.warning(f"Welding detected in {weld_percent:.1f}% of video - keeping second half")
                    segments = [(total_frames // 2, total_frames - 1)]
                elif weld_end_buffered >= total_frames - 1:
                    # Spawanie do końca wideo - zachowaj tylko początek
                    logger.info("Welding extends to end - keeping only pre-weld footage")
                    segments = [(0, weld_start_buffered - 1)] if weld_start_buffered > 0 else []
                    if not segments:
                        logger.warning("No frames before welding - keeping second half anyway")
                        segments = [(total_frames // 2, total_frames - 1)]
                else:
                    # Normalna sytuacja: zachowaj przed i po spawaniu
                    segments = []
                    
                    # Segment PRZED spawaniem (jeśli istnieje)
                    if weld_start_buffered > 0:
                        segments.append((0, weld_start_buffered - 1))
                        logger.info(f"Keeping pre-weld segment: frames 0-{weld_start_buffered - 1}")
                    
                    # Segment PO spawaniu (jeśli istnieje)
                    if weld_end_buffered < total_frames - 1:
                        segments.append((weld_end_buffered + 1, total_frames - 1))
                        logger.info(f"Keeping post-weld segment: frames {weld_end_buffered + 1}-{total_frames - 1}")
                    
                    if not segments:
                        logger.warning("Entire video is welding - keeping second half")
                        segments = [(total_frames // 2, total_frames - 1)]
                
                logger.info(f"Removing welding frames {weld_start_buffered}-{weld_end_buffered} (detected: {weld_start}-{weld_end}, buffer: {buffer_frames})")
            
            # Twórz writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {output_path}")
            
            # Zapisz wszystkie segmenty
            frames_written = 0
            for start_frame, end_frame in segments:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for _ in range(end_frame - start_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                    frames_written += 1
            
            writer.release()
            
            # Informacje o wyniku
            output_size = output_path.stat().st_size / (1024 * 1024)
            original_size = path.stat().st_size / (1024 * 1024)
            
            frames_removed = total_frames - frames_written
            
            logger.info(f"✅ Welding removed: {frames_written} frames kept, {frames_removed} removed ({frames_written/fps:.1f}s)")
            
            return {
                "status": "completed",
                "input_filename": path.name,
                "output_filename": output_path.name,
                "output_path": str(output_path),
                "welding_start_frame": weld_start if weld_start is not None else "not_detected",
                "welding_end_frame": weld_end if weld_end is not None else "not_detected",
                "frames_removed": frames_removed,
                "kept_frames": frames_written,
                "duration_seconds": round(frames_written / fps, 2),
                "original_size_mb": round(original_size, 2),
                "output_size_mb": round(output_size, 2),
                "reduction_percent": round((1 - output_size / original_size) * 100, 1) if original_size > 0 else 0
            }
        finally:
            cap.release()
    
    def _resolve_path(self, video_path: str | Path) -> Path:
        """Rozwiązuje ścieżkę do pliku wideo."""
        path = Path(video_path)
        # Jeśli ścieżka absolutna lub już istnieje - użyj jej
        if path.is_absolute() or path.exists():
            return path
        # W przeciwnym razie szukaj w recordings_dir
        return self.recordings_dir / path


# Singleton dla FastAPI dependency injection
_motion_detection_service: Optional[MotionDetectionService] = None


def get_motion_detection_service() -> MotionDetectionService:
    """FastAPI dependency - zwraca singleton MotionDetectionService."""
    global _motion_detection_service
    if _motion_detection_service is None:
        _motion_detection_service = MotionDetectionService()
    return _motion_detection_service
