"""Pydantic models dla API."""

from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"


class CameraHealthResponse(BaseModel):
    status: HealthStatus
    camera_index: Optional[int] = None
    fps: Optional[float] = None
    resolution: Optional[str] = None
    is_recording: Optional[bool] = None
    error: Optional[str] = None


class RecordingStatusResponse(BaseModel):
    is_recording: bool
    duration_seconds: Optional[float] = None
    frames: int = 0


class RecordingStartResponse(BaseModel):
    status: str
    filename: str


class RecordingStopResponse(BaseModel):
    status: str
    filename: str
    duration_seconds: float
    frames: int
    size_mb: float


class RecordingFile(BaseModel):
    filename: str
    size_mb: float
    created: str
    note: str = ""


class RecordingListResponse(BaseModel):
    recordings: List[RecordingFile]


class CameraSettingsRequest(BaseModel):
    contrast: Optional[int] = None
    fps: Optional[int] = None
    jpeg_quality: Optional[int] = None
    resolution: Optional[str] = None


# ============== FRAME EXTRACTION ==============

class VideoInfoResponse(BaseModel):
    """Informacje o pliku wideo."""
    filename: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float


class ExtractFramesRequest(BaseModel):
    """Request do ekstrakcji klatek."""
    step: int = 1              # Co która klatka (1 = każda)
    max_frames: Optional[int] = None  # Limit klatek
    output_folder: Optional[str] = None  # Folder docelowy (domyślnie: frames/{filename}/)
    prefix: str = "frame"      # Prefix nazwy pliku
    jpeg_quality: int = 95     # Jakość JPEG


class ExtractFramesResponse(BaseModel):
    """Response z wynikiem ekstrakcji."""
    status: str
    filename: str
    frames_extracted: int
    output_folder: str
    files: List[str]


class FrameResponse(BaseModel):
    """Informacje o pojedynczej klatce."""
    index: int
    timestamp_ms: float
    width: int
    height: int


# ============== MOTION DETECTION ==============

class MotionSegmentResponse(BaseModel):
    """Segment wideo z wykrytym ruchem."""
    start_frame: int
    end_frame: int
    start_time_ms: float
    end_time_ms: float
    duration_ms: float


class MotionAnalysisResponse(BaseModel):
    """Wynik analizy ruchu w wideo."""
    filename: str
    total_frames: int
    fps: float
    duration_seconds: float
    segments: List[MotionSegmentResponse]
    motion_percentage: float


class TrimToMotionRequest(BaseModel):
    """Request do przycinania wideo."""
    threshold: Optional[int] = None       # Próg detekcji (0-255)
    min_area_percent: Optional[float] = None  # Min % powierzchni ze zmianą
    include_all_segments: bool = True     # True = wszystkie, False = najdłuższy
    output_filename: Optional[str] = None # Nazwa pliku wyjściowego


class TrimToMotionResponse(BaseModel):
    """Response z wynikiem przycinania."""
    status: str
    input_filename: str
    output_filename: Optional[str] = None
    output_path: Optional[str] = None
    segments_count: Optional[int] = None
    frames_written: Optional[int] = None
    duration_seconds: Optional[float] = None
    original_size_mb: Optional[float] = None
    output_size_mb: Optional[float] = None
    reduction_percent: Optional[float] = None
    message: Optional[str] = None


# ============== IMAGE ENHANCEMENT ==============

class EnhancementPresetEnum(str, Enum):
    """Dostępne presety przetwarzania obrazu."""
    ORIGINAL = "original"           # Bez zmian
    WELD_ENHANCE = "weld_enhance"   # Najlepszy dla spawów
    HIGH_CONTRAST = "high_contrast" # Mocny kontrast
    EDGE_OVERLAY = "edge_overlay"   # Krawędzie spawu kolorowo
    HEATMAP = "heatmap"             # Pseudokolory
    DENOISE = "denoise"             # Redukcja szumu


class ImageEnhancementParams(BaseModel):
    """Parametry przetwarzania obrazu - do ręcznego dostrajania."""
    # CLAHE
    clahe: Optional[float] = None        # clip_limit (1.0-4.0), None = wyłączone
    clahe_grid: int = 8                  # Rozmiar siatki
    
    # Sharpening
    sharpen: Optional[float] = None      # amount (0.5-3.0)
    
    # Unsharp mask
    unsharp: Optional[float] = None      # amount
    unsharp_radius: float = 1.0
    
    # Gamma
    gamma: Optional[float] = None        # <1 ciemniej, >1 jaśniej
    
    # Contrast/Brightness
    contrast: Optional[float] = None     # alpha (1.0-3.0)
    brightness: int = 0                  # beta (-100 do 100)
    
    # Denoise
    denoise: Optional[int] = None        # strength (5-15)
    
    # Edge overlay
    edges: bool = False                  # Włącz nakładkę krawędzi
    edge_color: str = "green"            # green, red, blue, yellow
    
    # Heatmap
    heatmap: Optional[str] = None        # colormap: jet, hot, turbo, etc.


class EnhancementPresetsResponse(BaseModel):
    """Lista dostępnych presetów i opcji."""
    presets: List[str]
    colormaps: List[str]
    edge_colors: List[str]


# ============== LABELING ==============

class LabelType(str, Enum):
    """Typ etykiety."""
    OK = "ok"
    NOK = "nok"
    SKIP = "skip"


class DefectType(str, Enum):
    """Typ wady spawu (gdy etykieta = NOK)."""
    POROSITY = "porosity"           # Porowatość - pęcherzyki gazu
    CRACK = "crack"                 # Pęknięcia
    LACK_OF_FUSION = "lack_of_fusion"  # Brak przetopu
    UNDERCUT = "undercut"           # Podtopienia przy krawędzi
    BURN_THROUGH = "burn_through"   # Przepalenie
    SPATTER = "spatter"             # Rozpryski
    IRREGULAR_BEAD = "irregular_bead"  # Nierówna spoina
    CONTAMINATION = "contamination" # Zanieczyszczenia
    OTHER = "other"                 # Inna wada


class AddLabelRequest(BaseModel):
    """Request do dodania etykiety."""
    label: LabelType
    defect_type: Optional[DefectType] = None  # Wymagane gdy label=nok
    notes: str = ""
    filters_used: Optional[dict] = None


class FrameLabelResponse(BaseModel):
    """Odpowiedź z etykietą klatki."""
    video_filename: str
    frame_index: int
    label: str
    defect_type: Optional[str] = None
    timestamp: str
    notes: str = ""


class LabelingStatsResponse(BaseModel):
    """Statystyki etykietowania."""
    total_labeled: int
    ok_count: int
    nok_count: int
    skip_count: int
    videos_labeled: int
    defect_counts: Optional[dict] = None  # Liczba każdego typu wady


class TrainingDataResponse(BaseModel):
    """Informacje o danych treningowych."""
    training_data_path: str
    ok_count: int
    nok_count: int
    total: int
    ready_for_training: bool
    defect_counts: Optional[dict] = None
