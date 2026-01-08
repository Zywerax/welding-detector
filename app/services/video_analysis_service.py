"""
Video Analysis Service - Batch analysis całych nagrań
Analizuje wszystkie klatki: OK/NOK + typ defektu
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import cv2

from app.services.ml_classification_service import get_ml_service
from app.services.defect_classifier_service import get_defect_classifier_service
from app.services.frame_extractor_service import get_frame_extractor_service

logger = logging.getLogger(__name__)


class VideoAnalysisService:
    """Serwis do batch analizy wideo"""
    
    def __init__(self, results_dir: str = "recordings/analysis"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.ml_service = get_ml_service()
        self.defect_service = get_defect_classifier_service()
        self.frame_extractor = get_frame_extractor_service()
    
    def analyze_video(
        self,
        filename: str,
        analyze_defects: bool = True,
        skip_frames: int = 1,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Analizuje całe wideo klatka po klatce
        
        Args:
            filename: Nazwa pliku wideo
            analyze_defects: Czy klasyfikować typy defektów dla NOK
            skip_frames: Analizuj co N-tą klatkę (1 = wszystkie)
            progress_callback: Funkcja callback(current, total, frame_result)
        """
        logger.info(f"Starting video analysis: {filename}")
        
        # Pobierz liczbę klatek
        video_info = self.frame_extractor.get_video_info(filename)
        if not video_info:
            raise ValueError(f"Video not found: {filename}")
        
        total_frames = video_info['frame_count']
        frames_to_analyze = list(range(0, total_frames, skip_frames))
        
        results = {
            "filename": filename,
            "analyzed_at": datetime.now().isoformat(),
            "total_frames": total_frames,
            "analyzed_frames": len(frames_to_analyze),
            "skip_frames": skip_frames,
            "summary": {
                "ok": 0,
                "nok": 0
            },
            "defect_summary": {},
            "frames": []
        }
        
        # Analizuj klatki
        for idx, frame_index in enumerate(frames_to_analyze):
            try:
                frame = self.frame_extractor.get_frame(filename, frame_index)
                if frame is None:
                    logger.warning(f"Failed to extract frame {frame_index}")
                    continue
                
                # Klasyfikacja OK/NOK
                prediction = self.ml_service.predict(frame, with_gradcam=False)
                
                frame_result = {
                    "frame": frame_index,
                    "prediction": prediction["prediction"],
                    "confidence": prediction["confidence"]
                }
                
                # Update summary
                if prediction["prediction"] == "ok":
                    results["summary"]["ok"] += 1
                else:
                    results["summary"]["nok"] += 1
                    
                    # Klasyfikacja typu defektu jeśli NOK
                    if analyze_defects and self.defect_service.model:
                        try:
                            defect_pred = self.defect_service.predict(frame, with_gradcam=False)
                            frame_result["defect_type"] = defect_pred["prediction"]
                            frame_result["defect_confidence"] = defect_pred["confidence"]
                            
                            # Update defect summary
                            defect_type = defect_pred["prediction"]
                            results["defect_summary"][defect_type] = results["defect_summary"].get(defect_type, 0) + 1
                        except Exception as e:
                            logger.warning(f"Defect classification failed for frame {frame_index}: {e}")
                
                results["frames"].append(frame_result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(idx + 1, len(frames_to_analyze), frame_result)
                
            except Exception as e:
                logger.error(f"Error analyzing frame {frame_index}: {e}")
        
        # Zapisz wyniki
        self._save_results(filename, results)
        
        logger.info(f"Analysis complete: {filename} - OK: {results['summary']['ok']}, NOK: {results['summary']['nok']}")
        return results
    
    def _save_results(self, filename: str, results: Dict[str, Any]):
        """Zapisz wyniki analizy do JSON"""
        result_file = self.results_dir / f"{Path(filename).stem}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {result_file}")
    
    def get_analysis_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """Pobierz zapisane wyniki analizy"""
        result_file = self.results_dir / f"{Path(filename).stem}.json"
        if not result_file.exists():
            return None
        
        with open(result_file, 'r') as f:
            return json.load(f)
    
    def has_analysis(self, filename: str) -> bool:
        """Sprawdź czy wideo ma zapisaną analizę"""
        result_file = self.results_dir / f"{Path(filename).stem}.json"
        return result_file.exists()
    
    def get_defect_frames(self, filename: str, defect_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Pobierz listę klatek z defektami"""
        results = self.get_analysis_results(filename)
        if not results:
            return []
        
        defect_frames = [f for f in results["frames"] if f["prediction"] == "nok"]
        
        if defect_type:
            defect_frames = [f for f in defect_frames if f.get("defect_type") == defect_type]
        
        return defect_frames


# Singleton instance
_analysis_service_instance = None

def get_video_analysis_service() -> VideoAnalysisService:
    """Pobierz singleton VideoAnalysisService"""
    global _analysis_service_instance
    if _analysis_service_instance is None:
        _analysis_service_instance = VideoAnalysisService()
    return _analysis_service_instance
