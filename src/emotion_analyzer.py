"""
Tech Challenge - Fase 4: Analisador de Emoções
Módulo responsável pela análise de expressões emocionais em rostos detectados.
Usa DeepFace como método principal (2-3x mais rápido que FER).
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from .config import EMOTION_LABELS


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
    Analisador de expressões emocionais usando DeepFace (principal) ou FER (fallback).
    DeepFace é 2-3x mais rápido que FER e mais preciso em expressões sutis.
    Suporta rastreamento temporal de emoções para suavização.
    """
    
    def __init__(self, method: str = None, temporal_window: int = 5):
        """
        Inicializa o analisador de emoções.
        
        Args:
            method: Método de análise ('deepface', 'fer', None=auto).
                   Se None, usa EMOTION_ANALYZER_METHOD do config.
            temporal_window: Janela temporal para suavização (em frames)
        """
        from .config import EMOTION_ANALYZER_METHOD
        
        # Se não especificado, usa config
        if method is None:
            method = EMOTION_ANALYZER_METHOD
        
        self.method = method
        self.temporal_window = temporal_window
        self.emotion_history: Dict[int, deque] = {}  # face_id -> histórico
        
        self._init_analyzer()
    
    def _init_analyzer(self):
        """Inicializa o analisador baseado no método escolhido."""
        if self.method == "deepface":
            self._init_deepface()
        elif self.method == "fer":
            self._init_fer()
        else:
            raise ValueError(f"Método desconhecido: {self.method}")
    
    def _init_fer(self):
        """Inicializa FER (Facial Expression Recognition)."""
        try:
            # Nova versão do FER usa fer.fer.FER
            try:
                from fer.fer import FER
            except ImportError:
                from fer import FER
            self.analyzer = FER(mtcnn=False)  # Usamos nosso próprio detector
        except (ImportError, Exception) as e:
            print(f"[AVISO] FER não disponível ({e}), usando análise simplificada")
            self.method = "simple"
            self.analyzer = None
    
    def _init_deepface(self):
        """Inicializa DeepFace para análise de emoções."""
        try:
            from deepface import DeepFace
            import os as os_module
            
            # Configura cache directory para evitar redownload
            from .config import DEEPFACE_CACHE_DIR
            os_module.environ['DEEPFACE_HOME'] = DEEPFACE_CACHE_DIR
            
            # Pré-carrega modelo para validar funcionamento
            # (isso acontece na primeira execução)
            self.analyzer = DeepFace
            print("[INFO] DeepFace inicializado com sucesso")
            
        except ImportError as e:
            print(f"[AVISO] DeepFace não instalado ({e}), tentando FER")
            self.method = "fer"
            self._init_fer()
        except Exception as e:
            print(f"[AVISO] Erro ao inicializar DeepFace ({e}), tentando FER")
            self.method = "fer"
            self._init_fer()
    
    def analyze(
        self, 
        frame: np.ndarray, 
        face_bbox: Tuple[int, int, int, int],
        face_id: int
    ) -> Optional[EmotionResult]:
        """
        Analisa a emoção de um rosto no frame.
        
        Args:
            frame: Imagem BGR
            face_bbox: Bounding box do rosto (x, y, w, h)
            face_id: ID do rosto para rastreamento temporal
            
        Returns:
            Resultado da análise emocional ou None se falhar
        """
        x, y, w, h = face_bbox
        
        # Extrai região do rosto com margem
        margin = int(min(w, h) * 0.1)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
        
        # Log de performance
        t0 = time.time()
        
        if self.method == "deepface":
            result = self._analyze_deepface(face_roi, face_id)
        elif self.method == "fer":
            result = self._analyze_fer(face_roi, face_id)
        else:
            result = self._analyze_simple(face_roi, face_id)
        
        # Log de tempo se tempo > 150ms (aviso de performance)
        elapsed = (time.time() - t0) * 1000
        if elapsed > 150:
            print(f"[AVISO] EmotionAnalyzer.{self.method} lento: {elapsed:.1f}ms para face_id={face_id}")
        
        return result
    
    def _analyze_fer(
        self, 
        face_roi: np.ndarray, 
        face_id: int
    ) -> Optional[EmotionResult]:
        """Análise usando FER."""
        if self.analyzer is None:
            return self._analyze_simple(face_roi, face_id)
        
        try:
            # FER espera imagem BGR
            emotions = self.analyzer.detect_emotions(face_roi)
            
            if not emotions:
                return None
            
            # Pega a primeira detecção (já sabemos que há um rosto)
            emotion_scores = emotions[0]["emotions"]
            
            # Aplica suavização temporal
            smoothed_scores = self._smooth_emotions(face_id, emotion_scores)
            
            # === DETECÇÃO DE CARETAS ===
            # Caretas são expressões exageradas que causam:
            # 1. Alta variância nos scores (múltiplas emoções ativadas)
            # 2. Combinações incomuns (ex: surprise + disgust)
            # 3. Scores extremos em emoções "ativas"
            
            is_grimace, grimace_confidence = self._detect_grimace(smoothed_scores, face_id)
            
            if is_grimace:
                # Adiciona "grimace" como emoção detectada
                smoothed_scores["grimace"] = grimace_confidence
                dominant = "grimace"
                confidence = grimace_confidence
            else:
                # Tunning de Sensibilidade: Reduz o viés de "Neutro"
                # Modelos FER tendem a exagerar na classificação 'neutral' quando a expressão é sutil.
                # Aqui aplicamos uma penalização leve ao score 'neutral' se houver outra emoção competitiva.
                if "neutral" in smoothed_scores:
                    neutral_score = smoothed_scores["neutral"]
                    # Se não é totalmente óbvio que é neutro (>0.7), damos chance para outras emoções
                    # AJUSTE 2: Mas protegemos contra "Sad" falso positivo (comum ao olhar para baixo/ler)
                    # Se o "runner-up" (a segunda maior) for "sad", NÃO penalizamos o neutral.
                    
                    # Encontra a segunda maior emoção
                    sorted_emotions = sorted(smoothed_scores.items(), key=lambda x: x[1], reverse=True)
                    runner_up = sorted_emotions[1][0] if len(sorted_emotions) > 1 else None
                    
                    # Só penaliza neutral se o concorrente for 'ativo' (happy, surprise, fear, angry)
                    # Se o concorrente for 'sad' ou 'disgust', mantemos o neutral forte para evitar falsos positivos de leitura
                    if neutral_score < 0.7 and runner_up in ["happy", "surprise", "fear", "angry"]: 
                        smoothed_scores["neutral"] = neutral_score * 0.85
                        
                    # AJUSTE 3: Se "Sad" for o dominante, mas "Neutral" for alto, força Neutral.
                    # Isso corrige pessoas lendo/olhando para baixo sendo classificadas como tristes.
                    sad_score = smoothed_scores.get("sad", 0)
                    if sad_score > neutral_score and neutral_score > 0.3:
                        # Se Sad ganha por pouco ou Neutral ainda é relevante, penaliza Sad
                        if sad_score < 0.6: # Não é uma tristeza profunda/óbvia
                             smoothed_scores["sad"] = sad_score * 0.8  # Reduz confiança do triste
                
                # Encontra emoção dominante
                dominant = max(smoothed_scores, key=smoothed_scores.get)
                confidence = smoothed_scores[dominant]
            
            return EmotionResult(
                face_id=face_id,
                dominant_emotion=dominant,
                emotion_scores=smoothed_scores,
                confidence=confidence,
                emotion_pt=EMOTION_LABELS.get(dominant, dominant)
            )
            
        except Exception as e:
            print(f"[ERRO] FER: {e}")
            return self._analyze_simple(face_roi, face_id)
    
    def _analyze_deepface(
        self, 
        face_roi: np.ndarray, 
        face_id: int
    ) -> Optional[EmotionResult]:
        """
        Análise usando DeepFace.
        
        DeepFace retorna scores de 0-100 (percentual) que normalizamos para 0-1.
        """
        try:
            result = self.analyzer.analyze(
                face_roi,
                actions=["emotion"],
                enforce_detection=False,
                silent=True,
                anti_spoofing=False  # Desabilita verificação anti-spoofing para performance
            )
            
            if not result:
                return None
            
            # Extrai scores de emoção
            emotion_scores_raw = result[0]["emotion"]
            
            # Normaliza scores: DeepFace retorna 0-100, convertemos para 0-1
            emotion_scores = {
                k: v / 100.0 for k, v in emotion_scores_raw.items()
            }
            
            # Valida scores
            if not emotion_scores or sum(emotion_scores.values()) == 0:
                return None
            
            # Aplica suavização temporal
            smoothed_scores = self._smooth_emotions(face_id, emotion_scores)
            
            # Encontra emoção dominante
            dominant = max(smoothed_scores, key=smoothed_scores.get)
            confidence = smoothed_scores[dominant]
            
            return EmotionResult(
                face_id=face_id,
                dominant_emotion=dominant,
                emotion_scores=smoothed_scores,
                confidence=confidence,
                emotion_pt=EMOTION_LABELS.get(dominant, dominant)
            )
            
        except Exception as e:
            print(f"[ERRO] DeepFace: {e}")
            return None
    
    def _analyze_simple(
        self, 
        face_roi: np.ndarray, 
        face_id: int
    ) -> EmotionResult:
        """
        Análise simplificada baseada em características básicas.
        Usado como fallback quando bibliotecas principais não estão disponíveis.
        """
        # Análise muito básica baseada em brilho/contraste
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        std_contrast = np.std(gray) / 128.0
        
        # Heurística simples (não é precisa, apenas para demonstração)
        if std_contrast > 0.5:
            dominant = "surprise" if mean_brightness > 0.5 else "angry"
        elif mean_brightness > 0.6:
            dominant = "happy"
        elif mean_brightness < 0.4:
            dominant = "sad"
        else:
            dominant = "neutral"
        
        emotion_scores = {
            "angry": 0.1,
            "disgust": 0.05,
            "fear": 0.1,
            "happy": 0.1,
            "sad": 0.1,
            "surprise": 0.1,
            "neutral": 0.45
        }
        emotion_scores[dominant] = 0.6
        
        return EmotionResult(
            face_id=face_id,
            dominant_emotion=dominant,
            emotion_scores=emotion_scores,
            confidence=0.6,
            emotion_pt=EMOTION_LABELS.get(dominant, dominant)
        )
    
    def _smooth_emotions(
        self, 
        face_id: int, 
        current_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Aplica suavização temporal nas emoções para reduzir ruído.
        
        Args:
            face_id: ID do rosto
            current_scores: Scores atuais
            
        Returns:
            Scores suavizados
        """
        # Inicializa histórico se necessário
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=self.temporal_window)
        
        self.emotion_history[face_id].append(current_scores)
        
        # Média ponderada (mais recentes têm mais peso)
        if len(self.emotion_history[face_id]) == 1:
            return current_scores
        
        smoothed = {}
        weights = np.linspace(0.5, 1.0, len(self.emotion_history[face_id]))
        weights /= weights.sum()
        
        for emotion in current_scores.keys():
            values = [h.get(emotion, 0) for h in self.emotion_history[face_id]]
            smoothed[emotion] = float(np.average(values, weights=weights))
        
        return smoothed
    
    def _detect_grimace(
        self,
        emotion_scores: Dict[str, float],
        face_id: int
    ) -> Tuple[bool, float]:
        """
        Detecta se a expressão é uma careta (expressão exagerada/engraçada).
        
        Caretas são identificadas por combinações MUITO incomuns de emoções
        que raramente ocorrem naturalmente.
        
        Args:
            emotion_scores: Scores das emoções detectadas
            face_id: ID do rosto para análise temporal
            
        Returns:
            Tuple (is_grimace, confidence)
        """
        # Extrai scores das emoções principais
        surprise = emotion_scores.get("surprise", 0)
        disgust = emotion_scores.get("disgust", 0)
        fear = emotion_scores.get("fear", 0)
        happy = emotion_scores.get("happy", 0)
        angry = emotion_scores.get("angry", 0)
        neutral = emotion_scores.get("neutral", 0)
        sad = emotion_scores.get("sad", 0)
        
        # Se neutro é dominante, NÃO é careta
        if neutral > 0.4:
            return False, 0.0
        
        # Se uma emoção é muito dominante (>0.6), provavelmente é emoção real, não careta
        max_score = max(emotion_scores.values())
        if max_score > 0.65:
            return False, 0.0
        
        grimace_score = 0.0
        
        # === COMBINAÇÕES MUITO ESPECÍFICAS DE CARETAS ===
        # Apenas combinações que são MUITO raras naturalmente
        
        # 1. Surprise + Disgust AMBOS ALTOS (careta clássica de "eca/bleh")
        if surprise > 0.25 and disgust > 0.25:
            grimace_score += 0.7
        
        # 2. Fear + Disgust AMBOS ALTOS (careta de nojo exagerado)
        if fear > 0.25 and disgust > 0.25:
            grimace_score += 0.6
        
        # 3. Happy + Disgust (sorriso + nojo = careta engraçada)
        if happy > 0.25 and disgust > 0.2:
            grimace_score += 0.6
        
        # 4. Angry + Surprise AMBOS ALTOS (raiva cômica exagerada)
        if angry > 0.25 and surprise > 0.25:
            grimace_score += 0.5
        
        # 5. Instabilidade temporal EXTREMA (muitas mudanças rápidas)
        if face_id in self.emotion_history and len(self.emotion_history[face_id]) >= 4:
            history = list(self.emotion_history[face_id])
            dominants = [max(h, key=h.get) for h in history[-6:]]
            changes = sum(1 for i in range(1, len(dominants)) if dominants[i] != dominants[i-1])
            # Só conta se tiver MUITAS mudanças (3+)
            if changes >= 3:
                grimace_score += 0.4
        
        # Normaliza para 0-1
        grimace_score = min(grimace_score, 1.0)
        
        # Threshold ALTO para considerar careta (muito restritivo)
        is_grimace = grimace_score >= 0.55
        
        return is_grimace, grimace_score
    
    def get_emotion_trend(self, face_id: int) -> Optional[str]:
        """
        Analisa a tendência emocional de um rosto.
        
        Returns:
            'increasing', 'decreasing', 'stable' ou None
        """
        if face_id not in self.emotion_history:
            return None
        
        history = list(self.emotion_history[face_id])
        if len(history) < 3:
            return None
        
        # Analisa tendência da emoção dominante
        dominant_values = []
        for h in history:
            dominant = max(h, key=h.get)
            dominant_values.append(h[dominant])
        
        trend = np.polyfit(range(len(dominant_values)), dominant_values, 1)[0]
        
        if trend > 0.05:
            return "increasing"
        elif trend < -0.05:
            return "decreasing"
        return "stable"
    
    def reset_history(self, face_id: Optional[int] = None):
        """Reseta o histórico de emoções."""
        if face_id is not None:
            self.emotion_history.pop(face_id, None)
        else:
            self.emotion_history.clear()
    
    def draw_emotion(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        result: EmotionResult,
        color: Tuple[int, int, int] = (255, 255, 0)
    ) -> np.ndarray:
        """
        Desenha a emoção detectada no frame.
        
        Args:
            frame: Imagem BGR
            bbox: Bounding box do rosto
            result: Resultado da análise
            color: Cor do texto (BGR)
            
        Returns:
            Frame anotado
        """
        annotated = frame.copy()
        x, y, w, h = bbox
        
        # Label da emoção
        label = f"{result.emotion_pt} ({result.confidence:.0%})"
        
        # Posição abaixo do rosto
        text_y = y + h + 20
        
        # Fundo do texto
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        cv2.rectangle(
            annotated,
            (x, text_y - text_h - 5),
            (x + text_w + 4, text_y + 5),
            (0, 0, 0),
            -1
        )
        
        # Texto
        cv2.putText(
            annotated, label,
            (x + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 1, cv2.LINE_AA
        )
        
        # Barra de emoções (mini gráfico)
        bar_y = text_y + 15
        bar_height = 8
        bar_width = w
        
        for i, (emotion, score) in enumerate(sorted(
            result.emotion_scores.items(), 
            key=lambda x: -x[1]
        )[:3]):  # Top 3 emoções
            bar_x = x
            filled_width = int(bar_width * score)
            
            # Cores diferentes para cada emoção
            emotion_colors = {
                "happy": (0, 255, 0),
                "sad": (255, 0, 0),
                "angry": (0, 0, 255),
                "surprise": (0, 255, 255),
                "fear": (255, 0, 255),
                "disgust": (0, 128, 0),
                "neutral": (128, 128, 128)
            }
            bar_color = emotion_colors.get(emotion, (200, 200, 200))
            
            cv2.rectangle(
                annotated,
                (bar_x, bar_y + i * (bar_height + 2)),
                (bar_x + filled_width, bar_y + i * (bar_height + 2) + bar_height),
                bar_color,
                -1
            )
        
        return annotated
