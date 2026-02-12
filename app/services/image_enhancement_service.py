"""
Image Enhancement Service - filter for improving weld visibility on frames.
Presets and manual parameters for sharpening, contrast enhancement,
and visualization of cold welds on images.
"""

import cv2  # type: ignore
import numpy as np
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EnhancementPreset(str, Enum):
    """Presets for image enhancement."""
    ORIGINAL = "original"           # No changes
    WELD_ENHANCE = "weld_enhance"   # Best for welds - CLAHE + sharpen
    HIGH_CONTRAST = "high_contrast" # Strong contrast
    EDGE_OVERLAY = "edge_overlay"   # Weld edges in color
    HEATMAP = "heatmap"             # Pseudocolors
    DENOISE = "denoise"             # Noise reduction + slight sharpening


@dataclass
class EnhancementParams:
    """Image enhancement parameters."""
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0      # 1.0-4.0, higher = more contrast
    clahe_grid_size: int = 8           # Grid size (8x8)
    
    # Sharpening
    sharpen_enabled: bool = False
    sharpen_amount: float = 1.0        # 0.5-3.0, sharpening strength
    
    # Unsharp Mask
    unsharp_enabled: bool = False
    unsharp_amount: float = 1.5        # Effect strength
    unsharp_radius: float = 1.0        # Blur radius
    
    # Gamma correction
    gamma_enabled: bool = False
    gamma_value: float = 1.0           # <1 darker, >1 brighter
    
    # Contrast/Brightness
    contrast_enabled: bool = False
    contrast_alpha: float = 1.0        # Contrast (1.0-3.0)
    contrast_beta: int = 0             # Brightness (-100 to 100)
    
    # Denoise (bilateral filter)
    denoise_enabled: bool = False
    denoise_strength: int = 9          # Filter strength (5-15)
    
    # Edge detection overlay
    edge_overlay_enabled: bool = False
    edge_color: tuple = (0, 255, 0)    # Edge color (BGR - green)
    edge_threshold1: int = 50
    edge_threshold2: int = 150
    
    # Heatmap (pseudocolors based on intensity)
    heatmap_enabled: bool = False
    heatmap_colormap: int = cv2.COLORMAP_JET


class ImageEnhancementService:
    """
    Service for processing images to improve weld visibility.
    
    Usage:
        service = ImageEnhancementService()
        
        # Use a preset
        enhanced = service.apply_preset(frame, EnhancementPreset.WELD_ENHANCE)
        
        # Or manual parameters
        params = EnhancementParams(clahe_enabled=True, sharpen_enabled=True)
        enhanced = service.enhance(frame, params)
    """
    
    # Predefined preset settings
    PRESETS = {
        EnhancementPreset.ORIGINAL: EnhancementParams(),
        
        EnhancementPreset.WELD_ENHANCE: EnhancementParams(
            clahe_enabled=True,
            clahe_clip_limit=2.5,
            clahe_grid_size=8,
            sharpen_enabled=True,
            sharpen_amount=1.2,
            denoise_enabled=True,
            denoise_strength=7
        ),
        
        EnhancementPreset.HIGH_CONTRAST: EnhancementParams(
            clahe_enabled=True,
            clahe_clip_limit=4.0,
            contrast_enabled=True,
            contrast_alpha=1.5,
            contrast_beta=10
        ),
        
        EnhancementPreset.EDGE_OVERLAY: EnhancementParams(
            clahe_enabled=True,
            clahe_clip_limit=2.0,
            edge_overlay_enabled=True,
            edge_color=(0, 255, 0),
            edge_threshold1=30,
            edge_threshold2=100
        ),
        
        EnhancementPreset.HEATMAP: EnhancementParams(
            clahe_enabled=True,
            clahe_clip_limit=2.0,
            heatmap_enabled=True,
            heatmap_colormap=cv2.COLORMAP_JET
        ),
        
        EnhancementPreset.DENOISE: EnhancementParams(
            denoise_enabled=True,
            denoise_strength=9,
            unsharp_enabled=True,
            unsharp_amount=1.2,
            unsharp_radius=1.0
        )
    }
    
    def __init__(self):
        logger.info("ImageEnhancementService initialized")
    
    def apply_preset(self, frame: np.ndarray, preset: EnhancementPreset) -> np.ndarray:
        """Applies a preset to the image."""
        params = self.PRESETS.get(preset, EnhancementParams())
        return self.enhance(frame, params)
    
    def enhance(self, frame: np.ndarray, params: EnhancementParams) -> np.ndarray:
        """
        Applies filters to the image according to the parameters.
        
        The order of filters is optimized:
        1. Denoise (first noise reduction)
        2. CLAHE (contrast enhancement before sharpening)
        3. Gamma (brightness correction before contrast)
        4. Contrast/Brightness
        5. Sharpen/Unsharp (sharpening at the end)
        6. Edge overlay / Heatmap (visual effects at the end)
        """
        result = frame.copy()
        
        # 1. Denoise
        if params.denoise_enabled:
            result = self._apply_denoise(result, params.denoise_strength)
        
        # 2. CLAHE
        if params.clahe_enabled:
            result = self._apply_clahe(result, params.clahe_clip_limit, params.clahe_grid_size)
        
        # 3. Gamma
        if params.gamma_enabled and params.gamma_value != 1.0:
            result = self._apply_gamma(result, params.gamma_value)
        
        # 4. Contrast/Brightness
        if params.contrast_enabled:
            result = self._apply_contrast(result, params.contrast_alpha, params.contrast_beta)
        
        # 5. Sharpening
        if params.sharpen_enabled:
            result = self._apply_sharpen(result, params.sharpen_amount)
        elif params.unsharp_enabled:
            result = self._apply_unsharp_mask(result, params.unsharp_amount, params.unsharp_radius)
        
        # 6. Edge overlay (overlays edges on the image)
        if params.edge_overlay_enabled:
            result = self._apply_edge_overlay(
                result, params.edge_color, 
                params.edge_threshold1, params.edge_threshold2
            )
        
        # 7. Heatmap (converts to pseudocolors)
        if params.heatmap_enabled:
            result = self._apply_heatmap(result, params.heatmap_colormap)
        
        return result
    
    def _apply_clahe(self, frame: np.ndarray, clip_limit: float, grid_size: int) -> np.ndarray:
        """CLAHE - Contrast Limited Adaptive Histogram Equalization."""
        # Convert to LAB (better for CLAHE than directly on BGR)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _apply_sharpen(self, frame: np.ndarray, amount: float) -> np.ndarray:
        """Sharpening kernel."""
        # Sharpening kernel - standard unsharp mask
        kernel = np.array([
            [0, -1, 0],
            [-1, 4 + amount, -1],
            [0, -1, 0]
        ]) / amount
        
        # More aggressive kernel for stronger effect
        if amount > 1.5:
            kernel = np.array([
                [-1, -1, -1],
                [-1, 8 + amount, -1],
                [-1, -1, -1]
            ]) / amount
        
        return cv2.filter2D(frame, -1, kernel)
    
    def _apply_unsharp_mask(self, frame: np.ndarray, amount: float, radius: float) -> np.ndarray:
        """Unsharp mask - classic sharpening."""
        blurred = cv2.GaussianBlur(frame, (0, 0), radius)
        return cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
    
    def _apply_gamma(self, frame: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma correction."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)
    
    def _apply_contrast(self, frame: np.ndarray, alpha: float, beta: int) -> np.ndarray:
        """Contrast and brightness adjustment."""
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    def _apply_denoise(self, frame: np.ndarray, strength: int) -> np.ndarray:
        """Bilateral filter - reduces noise while preserving edges."""
        return cv2.bilateralFilter(frame, strength, strength * 10, strength * 10)
    
    def _apply_edge_overlay(
        self, 
        frame: np.ndarray, 
        color: tuple, 
        threshold1: int, 
        threshold2: int
    ) -> np.ndarray:
        """Overlays detected edges in color on the image."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        
        # Create a colored edge mask
        edge_colored = np.zeros_like(frame)
        edge_colored[edges > 0] = color
        
        # Overlay on the original
        return cv2.addWeighted(frame, 1, edge_colored, 0.7, 0)
    
    def _apply_heatmap(self, frame: np.ndarray, colormap: int) -> np.ndarray:
        """Converts to pseudocolors (heatmap)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, colormap)
    
    def get_preset_params(self, preset: EnhancementPreset) -> EnhancementParams:
        """Returns the parameters for a given preset (modifiable)."""
        return self.PRESETS.get(preset, EnhancementParams())
    
    @staticmethod
    def list_presets() -> list[str]:
        """List of available presets."""
        return [p.value for p in EnhancementPreset]
    
    @staticmethod
    def list_colormaps() -> dict[str, int]:
        """List of available colormaps for heatmap."""
        return {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
        }


# Singleton
_enhancement_service: Optional[ImageEnhancementService] = None


def get_enhancement_service() -> ImageEnhancementService:
    """FastAPI dependency."""
    global _enhancement_service
    if _enhancement_service is None:
        _enhancement_service = ImageEnhancementService()
    return _enhancement_service
