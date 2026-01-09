"""
Tech Challenge - Fase 4: Analisador de Emoções
Módulo responsável pela análise de expressões emocionais em rostos detectados.
Usa DeepFace como método principal.
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

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
            logger.info(f"DeepFace inicializado com sucesso (Cache: {weights_dir})")
            
        except ImportError:
            logger.error("DeepFace não instalado. Instale com 'pip install deepface'")
            self.analyzer = None # Desabilita análise
        except Exception as e:
            logger.error(f"Falha ao inicializar DeepFace: {e}")
            self.analyzer = None

    def analyze(
        self, 
        frame: np.ndarray, 
        face_bbox: Tuple[int, int, int, int],
        face_id: int,
        scene_context: str = "unknown"
    ) -> Optional[EmotionResult]:
        """
        Analisa a emoção de um rosto no frame.
        ARGS:
            frame: Frame original
            face_bbox: Bounding box do rosto (x, y, w, h)
            face_id: ID da pessoa rastreada
            scene_context: Contexto atual da cena (ex: 'office', 'home') para ajuste de pesos
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
            
        return self._analyze_deepface(face_roi, face_id, scene_context)
    
    def _analyze_deepface(self, face_roi: np.ndarray, face_id: int, scene_context: str) -> Optional[EmotionResult]:
        from .config import EMOTION_LABELS, DEEPFACE_BACKBONE, EMOTION_THRESHOLDS, SCENE_EMOTION_WEIGHTS
        
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
            
            # Normalizar (0-1) iniciais
            total = sum(avg_emotions.values())
            normalized_emotions = {k: v/total for k, v in avg_emotions.items()}

            # --- APLICAÇÃO DE PESOS POR CONTEXTO (SCENE AWARENESS) ---
            # Se sabemos que é um escritório, reduz probabilidade de medo/tristeza (falsos positivos de leitura)
            context_weights = SCENE_EMOTION_WEIGHTS.get(scene_context, {})
            if context_weights:
                for emo, weight in context_weights.items():
                    if emo in normalized_emotions:
                        normalized_emotions[emo] *= weight
                
                # Renormaliza após pesos
                new_total = sum(normalized_emotions.values())
                if new_total > 0:
                    normalized_emotions = {k: v/new_total for k, v in normalized_emotions.items()}

            # --- APLICAÇÃO DE LIMIARES CONFIGURÁVEIS ---
            # Filtra emoções que não atingem a confiança mínima configurada
            
            # 1. Identifica candidato dominante original
            dominant_candidate = max(normalized_emotions, key=normalized_emotions.get)
            dominant_score = normalized_emotions[dominant_candidate]
            
            # 2. Verifica se atinge o limiar
            threshold = EMOTION_THRESHOLDS.get(dominant_candidate, 0.0)
            
            final_emotion = dominant_candidate
            final_confidence = dominant_score
            
            # Se não atingir limiar, penaliza ou troca
            if dominant_score < threshold:
                # Regra específica para Medo/Tristeza (falsos positivos comuns):
                # Se não atingiu limiar, forçamos verificação de 'neutral' ou a próxima mais provável
                
                # Tenta encontrar a próxima emoção que satisfaça seu limiar
                sorted_emotions = sorted(normalized_emotions.items(), key=lambda x: x[1], reverse=True)
                found_valid = False
                
                for emo, score in sorted_emotions:
                    if emo == dominant_candidate: continue # Já falhou
                    
                    t = EMOTION_THRESHOLDS.get(emo, 0.0)
                    if score >= t:
                        final_emotion = emo
                        final_confidence = score
                        found_valid = True
                        break
                
                # Se nenhuma passou no teste, fallback para 'neutral' se disponível e razoável
                if not found_valid:
                    if 'neutral' in normalized_emotions:
                        final_emotion = 'neutral'
                        final_confidence = normalized_emotions['neutral']
            
            return EmotionResult(
                face_id=face_id,
                dominant_emotion=final_emotion,
                emotion_scores=normalized_emotions,
                confidence=final_confidence,
                emotion_pt=EMOTION_LABELS.get(final_emotion, final_emotion)
            )
            
        except Exception as e:
            logger.error(f"Falha na análise de emoção (face_id={face_id}): {e}")
            return None
