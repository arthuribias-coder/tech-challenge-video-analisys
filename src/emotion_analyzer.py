"""
Tech Challenge - Fase 4: Analisador de Emoções
Módulo responsável pela análise de expressões emocionais em rostos detectados.
Usa DeepFace como método principal.
"""

import cv2
import numpy as np
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class EmotionResult:
    """Resultado da análise emocional de um rosto."""
    face_id: int
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    confidence: float
    emotion_pt: str  # Emoção em português


class EmotionAnalyzer:
    """
    Analisador de expressões emocionais usando DeepFace.
    """
    
    def __init__(self, temporal_window: int = 5):
        """
        Inicializa o analisador de emoções.
        """
        self.temporal_window = temporal_window
        self.emotion_history: Dict[int, deque] = {}
        
        try:
            import os
            from .config import MODELS_DIR
            
            # Configura diretório de cache ANTES de importar DeepFace
            os.environ['DEEPFACE_HOME'] = str(MODELS_DIR)
            
            # Garante que a estrutura de diretórios existe para evitar erro do gdown
            weights_dir = MODELS_DIR / ".deepface" / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            
            from deepface import DeepFace
            
            self.analyzer = DeepFace
            print(f"[INFO] DeepFace inicializado com sucesso (Cache: {weights_dir})")
            
        except ImportError:
            print(f"[ERRO] DeepFace não instalado. Instale com 'pip install deepface'")
            self.analyzer = None # Desabilita análise
        except Exception as e:
            print(f"[ERRO] Falha ao inicializar DeepFace: {e}")
            self.analyzer = None

    def analyze(
        self, 
        frame: np.ndarray, 
        face_bbox: Tuple[int, int, int, int],
        face_id: int
    ) -> Optional[EmotionResult]:
        """
        Analisa a emoção de um rosto no frame.
        """
        if self.analyzer is None:
            return None

        x, y, w, h = face_bbox
        
        # Margem de segurança
        margin = int(min(w, h) * 0.1) 
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
            
        return self._analyze_deepface(face_roi, face_id)
    
    def _analyze_deepface(self, face_roi: np.ndarray, face_id: int) -> Optional[EmotionResult]:
        from .config import EMOTION_LABELS, DEEPFACE_BACKBONE
        
        try:
            # Análise com DeepFace
            # actions=['emotion'] garante apenas análise emocional (rápido)
            # enforce_detection=False permite analisar ROI já recortada
            results = self.analyzer.analyze(
                img_path=face_roi, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='skip', # Já detectamos o rosto
                silent=True
            )
            
            if not results:
                return None
                
            # DeepFace retorna lista
            result = results[0]
            emotions = result['emotion']
            
            # Suavização temporal
            if face_id not in self.emotion_history:
                self.emotion_history[face_id] = deque(maxlen=self.temporal_window)
            
            self.emotion_history[face_id].append(emotions)
            
            # Média das emoções no histórico
            avg_emotions = {}
            for key in emotions.keys():
                vals = [h[key] for h in self.emotion_history[face_id]]
                avg_emotions[key] = sum(vals) / len(vals)
            
            # Normalizar (0-1)
            total = sum(avg_emotions.values())
            normalized_emotions = {k: v/total for k, v in avg_emotions.items()}
            
            dominant = max(normalized_emotions, key=normalized_emotions.get)
            confidence = normalized_emotions[dominant]
            
            return EmotionResult(
                face_id=face_id,
                dominant_emotion=dominant,
                emotion_scores=normalized_emotions,
                confidence=confidence,
                emotion_pt=EMOTION_LABELS.get(dominant, dominant)
            )
            
        except Exception as e:
            # Log apenas em caso de erro real para depuração, sem parar processamento
            print(f"[ERRO] Falha na análise de emoção (face_id={face_id}): {e}")
            return None
