"""
Tech Challenge - Fase 4: Detector de Anomalias
Módulo responsável pela detecção de comportamentos anômalos no vídeo.
Anomalias incluem: movimentos bruscos, mudanças emocionais súbitas, padrões atípicos.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime


class AnomalyType(Enum):
    """Tipos de anomalias detectáveis."""
    SUDDEN_MOVEMENT = "sudden_movement"
    EMOTION_SPIKE = "emotion_spike"
    UNUSUAL_ACTIVITY = "unusual_activity"
    CROWD_ANOMALY = "crowd_anomaly"
    PROLONGED_INACTIVITY = "prolonged_inactivity"


@dataclass
class AnomalyEvent:
    """Representa um evento anômalo detectado."""
    anomaly_type: AnomalyType
    timestamp: float  # Segundos desde o início do vídeo
    frame_number: int
    person_id: Optional[int]
    severity: float  # 0.0 a 1.0
    description: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    details: Dict = field(default_factory=dict)


@dataclass
class PersonMetrics:
    """Métricas acumuladas de uma pessoa para análise de anomalias."""
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    emotion_history: deque = field(default_factory=lambda: deque(maxlen=30))
    activity_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen_frame: int = 0
    frames_inactive: int = 0


class AnomalyDetector:
    """
    Detector de anomalias comportamentais.
    Analisa padrões de movimento, emoção e atividade para identificar comportamentos atípicos.
    """
    
    def __init__(
        self,
        sudden_movement_threshold: float = 80.0,
        emotion_change_threshold: float = 0.5,
        inactivity_threshold: int = 90,  # frames
        fps: float = 30.0
    ):
        """
        Inicializa o detector de anomalias.
        
        Args:
            sudden_movement_threshold: Limiar de velocidade para movimento brusco (pixels/frame)
            emotion_change_threshold: Limiar de mudança emocional (0-1)
            inactivity_threshold: Frames de inatividade para considerar anomalia
            fps: Frames por segundo do vídeo
        """
        self.sudden_movement_threshold = sudden_movement_threshold
        self.emotion_change_threshold = emotion_change_threshold
        self.inactivity_threshold = inactivity_threshold
        self.fps = fps
        
        # Métricas por pessoa
        self.person_metrics: Dict[int, PersonMetrics] = {}
        
        # Histórico de anomalias
        self.anomaly_history: List[AnomalyEvent] = []
        
        # Estatísticas globais para baseline
        self.global_velocity_mean = 0.0
        self.global_velocity_std = 1.0
        self.velocity_samples: deque = deque(maxlen=1000)
        
        # Contadores
        self.frame_count = 0
        self.total_detections = 0
    
    def update(
        self,
        frame_number: int,
        face_detections: List,
        emotion_results: List,
        activity_detections: List
    ) -> List[AnomalyEvent]:
        """
        Atualiza o detector com novos dados e retorna anomalias detectadas.
        
        Args:
            frame_number: Número do frame atual
            face_detections: Lista de detecções de rostos
            emotion_results: Lista de resultados de análise emocional
            activity_detections: Lista de atividades detectadas
            
        Returns:
            Lista de anomalias detectadas neste frame
        """
        self.frame_count = frame_number
        anomalies = []
        
        # Atualiza métricas de cada pessoa
        seen_persons = set()
        
        # Processa detecções de face/emoção
        for face, emotion in zip(face_detections, emotion_results):
            if emotion is None:
                continue
            
            person_id = face.face_id
            seen_persons.add(person_id)
            
            self._ensure_person_metrics(person_id)
            metrics = self.person_metrics[person_id]
            
            # Atualiza histórico
            center = np.array([
                face.bbox[0] + face.bbox[2]/2,
                face.bbox[1] + face.bbox[3]/2
            ])
            metrics.position_history.append(center)
            metrics.emotion_history.append(emotion.emotion_scores.copy())
            metrics.last_seen_frame = frame_number
            metrics.frames_inactive = 0
            
            # Detecta anomalias de emoção
            emotion_anomaly = self._check_emotion_anomaly(person_id, emotion, frame_number)
            if emotion_anomaly:
                emotion_anomaly.bbox = face.bbox
                anomalies.append(emotion_anomaly)
        
        # Processa detecções de atividade
        for activity in activity_detections:
            person_id = activity.person_id
            seen_persons.add(person_id)
            
            self._ensure_person_metrics(person_id)
            metrics = self.person_metrics[person_id]
            
            # Atualiza histórico
            metrics.activity_history.append(activity.activity.value)
            metrics.velocity_history.append(activity.velocity)
            
            # Atualiza estatísticas globais
            self.velocity_samples.append(activity.velocity)
            if len(self.velocity_samples) > 100:
                self.global_velocity_mean = np.mean(self.velocity_samples)
                self.global_velocity_std = max(np.std(self.velocity_samples), 1.0)
            
            # Detecta anomalias de movimento
            movement_anomaly = self._check_movement_anomaly(person_id, activity, frame_number)
            if movement_anomaly:
                anomalies.append(movement_anomaly)
            
            # Detecta anomalias de atividade
            activity_anomaly = self._check_activity_anomaly(person_id, activity, frame_number)
            if activity_anomaly:
                anomalies.append(activity_anomaly)
        
        # Verifica inatividade prolongada
        for person_id, metrics in self.person_metrics.items():
            if person_id not in seen_persons:
                metrics.frames_inactive += 1
                
                if metrics.frames_inactive == self.inactivity_threshold:
                    anomalies.append(AnomalyEvent(
                        anomaly_type=AnomalyType.PROLONGED_INACTIVITY,
                        timestamp=frame_number / self.fps,
                        frame_number=frame_number,
                        person_id=person_id,
                        severity=0.4,
                        description=f"Pessoa #{person_id} desapareceu por {self.inactivity_threshold} frames"
                    ))
        
        # Registra anomalias no histórico
        self.anomaly_history.extend(anomalies)
        self.total_detections += len(face_detections) + len(activity_detections)
        
        return anomalies
    
    def _ensure_person_metrics(self, person_id: int):
        """Garante que métricas existam para a pessoa."""
        if person_id not in self.person_metrics:
            self.person_metrics[person_id] = PersonMetrics()
    
    def _check_emotion_anomaly(
        self,
        person_id: int,
        emotion_result,
        frame_number: int
    ) -> Optional[AnomalyEvent]:
        """Verifica anomalias de mudança emocional."""
        metrics = self.person_metrics[person_id]
        history = metrics.emotion_history
        
        if len(history) < 3:
            return None
        
        # Calcula mudança emocional
        prev_scores = history[-2]
        curr_scores = emotion_result.emotion_scores
        
        change = 0.0
        for emotion in curr_scores:
            prev_val = prev_scores.get(emotion, 0)
            curr_val = curr_scores.get(emotion, 0)
            change += abs(curr_val - prev_val)
        
        change /= len(curr_scores)
        
        if change > self.emotion_change_threshold:
            # Identifica qual emoção mudou mais
            max_change_emotion = max(
                curr_scores.keys(),
                key=lambda e: abs(curr_scores[e] - prev_scores.get(e, 0))
            )
            
            return AnomalyEvent(
                anomaly_type=AnomalyType.EMOTION_SPIKE,
                timestamp=frame_number / self.fps,
                frame_number=frame_number,
                person_id=person_id,
                severity=min(change / self.emotion_change_threshold, 1.0),
                description=f"Mudança emocional brusca para '{max_change_emotion}'",
                details={
                    "emotion": max_change_emotion,
                    "change_magnitude": change,
                    "previous_dominant": max(prev_scores, key=prev_scores.get),
                    "current_dominant": emotion_result.dominant_emotion
                }
            )
        
        return None
    
    def _check_movement_anomaly(
        self,
        person_id: int,
        activity_detection,
        frame_number: int
    ) -> Optional[AnomalyEvent]:
        """Verifica anomalias de movimento brusco."""
        velocity = activity_detection.velocity
        
        # Usa threshold absoluto e relativo
        abs_threshold = self.sudden_movement_threshold
        rel_threshold = self.global_velocity_mean + 3 * self.global_velocity_std
        
        threshold = max(abs_threshold, rel_threshold)
        
        if velocity > threshold:
            severity = min((velocity - threshold) / threshold + 0.5, 1.0)
            
            return AnomalyEvent(
                anomaly_type=AnomalyType.SUDDEN_MOVEMENT,
                timestamp=frame_number / self.fps,
                frame_number=frame_number,
                person_id=person_id,
                severity=severity,
                description=f"Movimento brusco detectado (velocidade: {velocity:.1f} px/frame)",
                bbox=activity_detection.bbox,
                details={
                    "velocity": velocity,
                    "threshold": threshold,
                    "activity": activity_detection.activity.value
                }
            )
        
        return None
    
    def _check_activity_anomaly(
        self,
        person_id: int,
        activity_detection,
        frame_number: int
    ) -> Optional[AnomalyEvent]:
        """Verifica anomalias de padrão de atividade."""
        metrics = self.person_metrics[person_id]
        history = list(metrics.activity_history)
        
        if len(history) < 10:
            return None
        
        current_activity = activity_detection.activity.value
        
        # Conta frequência de cada atividade no histórico
        activity_counts = {}
        for act in history[:-1]:  # Exclui o atual
            activity_counts[act] = activity_counts.get(act, 0) + 1
        
        total = len(history) - 1
        if total == 0:
            return None
        
        # Calcula probabilidade da atividade atual baseado no histórico
        current_freq = activity_counts.get(current_activity, 0) / total
        
        # Se atividade atual é muito rara no histórico (< 5%), é anômala
        if current_freq < 0.05 and len(set(history)) > 2:
            most_common = max(activity_counts, key=activity_counts.get)
            
            return AnomalyEvent(
                anomaly_type=AnomalyType.UNUSUAL_ACTIVITY,
                timestamp=frame_number / self.fps,
                frame_number=frame_number,
                person_id=person_id,
                severity=0.5,
                description=f"Atividade incomum: '{current_activity}' (usual: '{most_common}')",
                bbox=activity_detection.bbox,
                details={
                    "activity": current_activity,
                    "frequency": current_freq,
                    "usual_activity": most_common
                }
            )
        
        return None
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do detector."""
        anomaly_counts = {}
        for anomaly in self.anomaly_history:
            atype = anomaly.anomaly_type.value
            anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1
        
        severity_avg = 0.0
        if self.anomaly_history:
            severity_avg = np.mean([a.severity for a in self.anomaly_history])
        
        return {
            "total_frames": self.frame_count,
            "total_anomalies": len(self.anomaly_history),
            "anomalies_by_type": anomaly_counts,
            "average_severity": severity_avg,
            "persons_tracked": len(self.person_metrics),
            "global_velocity_mean": self.global_velocity_mean,
            "global_velocity_std": self.global_velocity_std
        }
    
    def get_anomalies_summary(self) -> List[Dict]:
        """Retorna resumo das anomalias para relatório."""
        summary = []
        for anomaly in self.anomaly_history:
            summary.append({
                "tipo": anomaly.anomaly_type.value,
                "timestamp": f"{anomaly.timestamp:.2f}s",
                "frame": anomaly.frame_number,
                "pessoa_id": anomaly.person_id,
                "severidade": f"{anomaly.severity:.0%}",
                "descricao": anomaly.description,
                "detalhes": anomaly.details
            })
        return summary
    
    def reset(self):
        """Reseta o estado do detector."""
        self.person_metrics.clear()
        self.anomaly_history.clear()
        self.velocity_samples.clear()
        self.frame_count = 0
        self.total_detections = 0
        self.global_velocity_mean = 0.0
        self.global_velocity_std = 1.0


def draw_anomaly(
    frame: np.ndarray,
    anomaly: AnomalyEvent,
    color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """
    Desenha indicação de anomalia no frame.
    
    Args:
        frame: Imagem BGR
        anomaly: Evento de anomalia
        color: Cor do indicador (BGR)
        
    Returns:
        Frame anotado
    """
    import cv2
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # Se tem bbox, destaca a região
    if anomaly.bbox:
        x, y, bw, bh = anomaly.bbox
        
        # Borda pulsante (mais grossa para severidade maior)
        thickness = int(2 + anomaly.severity * 4)
        cv2.rectangle(annotated, (x, y), (x+bw, y+bh), color, thickness)
        
        # Ícone de alerta
        alert_x = x + bw - 25
        alert_y = y + 5
        cv2.putText(
            annotated, "!",
            (alert_x, alert_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            color, 2, cv2.LINE_AA
        )
    
    # Banner de anomalia no topo
    banner_text = f"ANOMALIA: {anomaly.description}"
    (text_w, text_h), _ = cv2.getTextSize(
        banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    
    # Fundo semi-transparente
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, text_h + 20), color, -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
    
    # Texto
    cv2.putText(
        annotated, banner_text,
        (10, text_h + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    
    return annotated
