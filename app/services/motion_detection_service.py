"""
Motion Detection Service - detection of motion in video recordings.
Service for detecting segments with motion in welding recordings.
Uses cv2.absdiff to compare consecutive frames.
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
    """Video segment with motion."""
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float


@dataclass
class MotionAnalysisResult:
    """Result of motion analysis in a video."""
    filename: str
    total_frames: int
    fps: float
    duration_seconds: float
    segments: list[MotionSegment]
    motion_percentage: float  # Percentage of frames with motion


class MotionDetectionService:
    """
    Service for detecting motion in video recordings, specifically for welding processes.
    
    Use case:
        service = MotionDetectionService()
        result = service.detect_motion("recordings/rec_20260105_120000.mp4")
        
        # Trim video to segments with motion
        service.trim_to_motion("input.mp4", "output.mp4")
    """
    
    def __init__(
        self,
        recordings_dir: Path = Path("recordings"),
        threshold: int = 25,           # Pixel difference threshold
        min_area_percent: float = 0.5, # Minimum % of area with changes
        min_segment_frames: int = 5,   # Minimum frames per segment
        padding_frames: int = 30       # Padding before/after segment (0.5s @60fps)
    ):
        self.recordings_dir = recordings_dir
        self.threshold = threshold
        self.min_area_percent = min_area_percent
        self.min_segment_frames = min_segment_frames
        self.padding_frames = padding_frames
        logger.info("MotionDetectionService initialized")
    
    def detect_motion(
        self,
        video_path: str | Path,
        threshold: Optional[int] = None,
        min_area_percent: Optional[float] = None,
        analyze_step: int = 1
    ) -> MotionAnalysisResult:
        """
        Analyzes a video and detects segments with motion.
        
        Args:
            video_path: Path to the video file
            threshold: Pixel difference threshold (0-255)
            min_area_percent: Minimum % of area with changes
            analyze_step: Analyze every nth frame (1 = every frame, 2 = every second frame, etc.)
            
        Returns:
            MotionAnalysisResult with a list of motion segments
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
            
            logger.info(f"Analyzing motion in {path.name} ({total_frames} frames)")
            
            # Read the first frame to initialize
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
                    
                    # Difference between frames
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Count changed pixels
                    changed_pixels = cv2.countNonZero(thresh)
                    
                    if changed_pixels >= min_changed_pixels:
                        motion_frames.append(frame_idx)
                    
                    prev_gray = gray
                
                frame_idx += 1
            
            # Group frames into segments
            segments = self._group_into_segments(motion_frames, total_frames, fps)
            
            motion_pct = (len(motion_frames) / (total_frames / analyze_step) * 100) if total_frames > 0 else 0
            
            logger.info(f"Found {len(segments)} motion segments ({motion_pct:.1f}% motion)")
            
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
        """Groups motion frames into continuous segments."""
        if not motion_frames:
            return []
        
        segments = []
        start = motion_frames[0]
        end = motion_frames[0]
        
        # Max gap between frames in a single segment (0.5s)
        max_gap = int(fps * 0.5)
        
        for frame in motion_frames[1:]:
            if frame - end <= max_gap:
                end = frame
            else:
                # Add padding: full at the start, minimal at the end
                seg_start = max(0, start - self.padding_frames)
                seg_end = min(total_frames - 1, end + 5)  # Only 5 frames at the end (~0.08s)
                
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
        
        # Last segment
        seg_start = max(0, start - self.padding_frames)
        seg_end = min(total_frames - 1, end + 5)  # Only 5 frames at the end
        
        if seg_end - seg_start >= self.min_segment_frames:
            segments.append(MotionSegment(
                start_frame=seg_start,
                end_frame=seg_end,
                start_time_ms=seg_start / fps * 1000,
                end_time_ms=seg_end / fps * 1000,
                duration_ms=(seg_end - seg_start) / fps * 1000
            ))
        
        # Merge overlapping segments
        return self._merge_overlapping(segments, fps)
    
    def _merge_overlapping(self, segments: list[MotionSegment], fps: float) -> list[MotionSegment]:
        """Merge overlapping or close segments."""
        if len(segments) <= 1:
            return segments
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            last = merged[-1]
            # If segments overlap or are close (within padding), merge them
            if seg.start_frame <= last.end_frame + self.padding_frames:
                # Extend the last segment
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
        Trims the video to motion segments.
        
        Args:
            video_path: Path to the source video
            output_path: Output path (default: {input}_trimmed.mp4)
            threshold: Motion detection threshold
            min_area_percent: Min % of area with changes
            include_all_segments: True = all segments, False = only the longest segment
            
        Returns:
            Dictionary with information about the trimmed video
        """
        path = self._resolve_path(video_path)
        
        # Detect segments
        analysis = self.detect_motion(path, threshold, min_area_percent)
        
        if not analysis.segments:
            logger.warning(f"No motion detected in {path.name}")
            return {
                "status": "no_motion",
                "filename": path.name,
                "message": "No motion segments detected"
            }
        
        # Select segments
        if include_all_segments:
            segments = analysis.segments
        else:
            # Only the longest segment
            segments = [max(analysis.segments, key=lambda s: s.duration_ms)]
        
        # Determine output path
        if output_path:
            out_path = Path(output_path)
        else:
            out_path = path.parent / f"{path.stem}_trimmed.mp4"
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open source
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
            
            # Information about the result
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
        Detects the welding process (bright laser light).
        
        Args:
            video_path: Path to the video file
            brightness_threshold: Brightness threshold (0-255) for welding detection
            min_bright_percent: Minimum % of bright pixels to consider as welding
            
        Returns:
            (start_frame, end_frame) of the welding process or (None, None) if not detected
        """
        path = self._resolve_path(video_path)
        cap = cv2.VideoCapture(str(path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Detecting welding process in {path.name}")
            
            welding_frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check both brightness and colors (white/yellow/red laser)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Method 1: Bright pixels (laser light) - general detection
                bright_pixels = np.sum(gray > brightness_threshold)
                
                # Method 2: Very bright pixels (white/yellow center) - ONLY for active welding
                very_bright_pixels = np.sum(gray > 220)  # Elevated threshold
                
                # Method 3: Red/orange light (welding may be red)
                # High R and G values, low B
                b, g, r = cv2.split(frame)
                red_hot = np.sum((r > 220) & (g > 180) & (b < 120))  # More restrictive
                
                total_pixels = gray.shape[0] * gray.shape[1]
                bright_percent = (bright_pixels / total_pixels) * 100
                very_bright_percent = (very_bright_pixels / total_pixels) * 100
                red_hot_percent = (red_hot / total_pixels) * 100
                
                # Detect welding if:
                # - Many very bright pixels (active laser)
                # - Or intense red light (welding glow)
                is_welding = (
                    very_bright_percent >= 1.0 or  # At least 1% very bright (active laser)
                    red_hot_percent >= 3.0  # Or 3% intense red (welding glow) - increased from 2% to 3%
                )
                
                if is_welding:
                    welding_frames.append(frame_idx)
                
                frame_idx += 1
            
            if not welding_frames:
                logger.info("No welding process detected")
                return None, None
            
            # Find continuous welding segment - group frames with tolerance
            # If gap > 10 frames (0.3s), consider it as end of welding
            gap_tolerance = 10  # Reduced from 30 to 10
            segments = []
            current_start = welding_frames[0]
            prev_frame = welding_frames[0]
            
            for frame in welding_frames[1:]:
                if frame - prev_frame > gap_tolerance:
                    # End of current segment
                    segments.append((current_start, prev_frame))
                    current_start = frame
                prev_frame = frame
            
            # Add the last segment
            segments.append((current_start, prev_frame))
            
            # Choose the longest segment (main welding process)
            if segments:
                longest_segment = max(segments, key=lambda s: s[1] - s[0])
                start_frame, end_frame = longest_segment
                
                logger.info(f"Welding detected: frames {start_frame}-{end_frame} ({len(welding_frames)} frames total, {len(segments)} segments)")
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
        Trim video by removing ONLY the welding process (bright light).
        Keeps everything before and after welding.
        
        Args:
            video_path: Path to the input video
            output_path: Path to the output video (optional, default: {input}_postprocess.mp4)
            brightness_threshold: Brightness threshold for welding detection
            min_bright_percent: Minimum % of bright pixels to consider as welding
            
        Returns:
            Dict with information about the result
        """
        path = self._resolve_path(video_path)
        
        # Detect welding moment
        weld_start, weld_end = self.detect_welding_process(
            path, 
            brightness_threshold=brightness_threshold,
            min_bright_percent=min_bright_percent
        )
        
        # Set output path
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
            
            # If no welding detected, keep the entire video
            if weld_start is None or weld_end is None:
                logger.warning("No welding detected - keeping entire video")
                segments = [(0, total_frames - 1)]
            else:
                # Add buffer only at the START of welding (1 frame)
                # Keep the end exact - moment of laser shutdown
                buffer_frames = 1
                weld_start_buffered = max(0, weld_start - buffer_frames)
                weld_end_buffered = weld_end  # No buffer at the end
                
                # Check if welding occupies most of the video (>80%)
                weld_duration = weld_end_buffered - weld_start_buffered + 1
                weld_percent = (weld_duration / total_frames) * 100
                
                if weld_percent > 80:
                    # Welding detected in almost the entire video - likely a false positive or very short pre/post footage
                    # Probably the finished weld is also detected as bright
                    # Keep the second half of the video (post-processing/inspection footage)
                    logger.warning(f"Welding detected in {weld_percent:.1f}% of video - keeping second half")
                    segments = [(total_frames // 2, total_frames - 1)]
                elif weld_end_buffered >= total_frames - 1:
                    # Welding extends to the end of the video - keep only the beginning
                    logger.info("Welding extends to end - keeping only pre-weld footage")
                    segments = [(0, weld_start_buffered - 1)] if weld_start_buffered > 0 else []
                    if not segments:
                        logger.warning("No frames before welding - keeping second half anyway")
                        segments = [(total_frames // 2, total_frames - 1)]
                else:
                    # Normal situation: keep pre- and post-weld segments
                    segments = []
                    
                    # Segment BEFORE welding (if exists)
                    if weld_start_buffered > 0:
                        segments.append((0, weld_start_buffered - 1))
                        logger.info(f"Keeping pre-weld segment: frames 0-{weld_start_buffered - 1}")
                    
                    # Segment AFTER welding (if exists)
                    if weld_end_buffered < total_frames - 1:
                        segments.append((weld_end_buffered + 1, total_frames - 1))
                        logger.info(f"Keeping post-weld segment: frames {weld_end_buffered + 1}-{total_frames - 1}")
                    
                    if not segments:
                        logger.warning("Entire video is welding - keeping second half")
                        segments = [(total_frames // 2, total_frames - 1)]
                
                logger.info(f"Removing welding frames {weld_start_buffered}-{weld_end_buffered} (detected: {weld_start}-{weld_end}, buffer: {buffer_frames})")
            
            # Create writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create output video: {output_path}")
            
            # Save all segments
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
            
            # Information about the result
            output_size = output_path.stat().st_size / (1024 * 1024)
            original_size = path.stat().st_size / (1024 * 1024)
            
            frames_removed = total_frames - frames_written
            
            logger.info(f"Welding removed: {frames_written} frames kept, {frames_removed} removed ({frames_written/fps:.1f}s)")
            
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
        """Resolves the path to the video file."""
        path = Path(video_path)
        # If the path is absolute or already exists - use it
        if path.is_absolute() or path.exists():
            return path
        # Otherwise, look in recordings_dir
        return self.recordings_dir / path


# Singleton for FastAPI dependency injection
_motion_detection_service: Optional[MotionDetectionService] = None


def get_motion_detection_service() -> MotionDetectionService:
    """FastAPI dependency - returns the singleton MotionDetectionService."""
    global _motion_detection_service
    if _motion_detection_service is None:
        _motion_detection_service = MotionDetectionService()
    return _motion_detection_service
