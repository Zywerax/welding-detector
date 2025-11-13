"""
Image Processing Service - Wykrywanie krawędzi i analiza obrazu
"""
import cv2 # type: ignore
import numpy as np
from typing import Tuple, List, Optional
import base64


class ImageProcessingService:
    """Serwis do przetwarzania obrazu i detekcji krawędzi"""
    
    
    @staticmethod
    def detect_edges_simple(image_bytes: bytes, threshold1: int = 50, threshold2: int = 150) -> bytes:
        """
        Prosta detekcja krawędzi - zwraca obraz z krawędziami.
        
        Args:
            image_bytes: Surowe bajty JPEG
            threshold1: Dolny próg Canny
            threshold2: Górny próg Canny
            
        Returns:
            bytes: Obraz JPEG z wykrytymi krawędziami
        """
        try:
            # Dekoduj obraz
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return b''
            
            # Skala szarości
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Rozmycie
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detekcja krawędzi
            edges = cv2.Canny(blurred, threshold1, threshold2)
            
            # Konwertuj z powrotem do BGR dla lepszej wizualizacji
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Zakoduj do JPEG
            _, encoded = cv2.imencode('.jpg', edges_colored)
            
            return encoded.tobytes()
            
        except Exception as e:
            print(f"❌ Błąd detekcji krawędzi: {e}")
            return b''
