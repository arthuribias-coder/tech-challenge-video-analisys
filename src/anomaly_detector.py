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
from .config import SCENE_CONTEXT_RULES


class AnomalyType(Enum):
    """Tipos de anomalias detectáveis."""
    # Anomalias comportamentais (originais)
    SUDDEN_MOVEMENT = "sudden_movement"
    EMOTION_SPIKE = "emotion_spike"
    UNUSUAL_ACTIVITY = "unusual_activity"
    CROWD_ANOMALY = "crowd_anomaly"
    PROLONGED_INACTIVITY = "prolonged_inactivity"
    
    # Novas anomalias visuais/contextuais (Fase 4)
    VISUAL_OVERLAY = "visual_overlay"           # Texto/logo sobreposto detectado
    SCENE_INCONSISTENCY = "scene_inconsistency" # Objeto fora de contexto
    SUDDEN_OBJECT_APPEAR = "sudden_object_appear"  # Objeto surge do nada
    SILHOUETTE_ANOMALY = "silhouette_anomaly"   # Silhueta não-humana detectada como pessoa


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
    Detector de anomalias comportamentais e visuais.
    Analisa padrões de movimento, emoção, atividade, objetos e overlays.
    """
    
    def __init__(
        self,
        sudden_movement_threshold: float = 80.0,
        emotion_change_threshold: float = 0.5,
        inactivity_threshold: int = 90,  # frames
        fps: float = 30.0,
        # Novos parâmetros para anomalias visuais
        require_persistence: int = 3,  # Frames para confirmar anomalia
        enable_object_anomalies: bool = True,
        enable_overlay_anomalies: bool = True
    ):
        """
        Inicializa o detector de anomalias.
        
        Args:
            sudden_movement_threshold: Limiar de velocidade para movimento brusco (pixels/frame)
            emotion_change_threshold: Limiar de mudança emocional (0-1)
            inactivity_threshold: Frames de inatividade para considerar anomalia
            fps: Frames por segundo do vídeo
            require_persistence: Frames consecutivos para confirmar uma anomalia (evita falsos positivos)
            enable_object_anomalies: Habilita detecção de anomalias baseadas em objetos
            enable_overlay_anomalies: Habilita detecção de overlays/texto
        """
        self.sudden_movement_threshold = sudden_movement_threshold
        self.emotion_change_threshold = emotion_change_threshold
        self.inactivity_threshold = inactivity_threshold
        self.fps = fps
        self.require_persistence = require_persistence
        self.enable_object_anomalies = enable_object_anomalies
        self.enable_overlay_anomalies = enable_overlay_anomalies
        
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
        
        # Cache de anomalias pendentes (para persistência temporal)
        self._pending_anomalies: Dict[str, Dict] = {}  # key -> {count, data}
    
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
        self._pending_anomalies.clear()
        self.frame_count = 0
        self.total_detections = 0
        self.global_velocity_mean = 0.0
        self.global_velocity_std = 1.0
    
    # ========== NOVOS MÉTODOS PARA ANOMALIAS VISUAIS ==========
    
    def process_object_detections(
        self,
        frame_number: int,
        object_detections: List
    ) -> List[AnomalyEvent]:
        """
        Processa detecções de objetos e gera anomalias contextuais.
        
        Args:
            frame_number: Número do frame atual
            object_detections: Lista de ObjectDetection do ObjectDetector
            
        Returns:
            Lista de anomalias relacionadas a objetos
        """
        if not self.enable_object_anomalies:
            return []
        
        anomalies = []
        
        for obj_det in object_detections:
            if obj_det.is_anomalous:
                # Usa persistência temporal para evitar falsos positivos
                anomaly_key = f"obj_{obj_det.class_name}_{obj_det.bbox[0]//50}_{obj_det.bbox[1]//50}"
                
                if self._confirm_anomaly(anomaly_key, {
                    "type": AnomalyType.SCENE_INCONSISTENCY,
                    "reason": obj_det.anomaly_reason,
                    "bbox": obj_det.bbox,
                    "class_name": obj_det.class_name
                }):
                    anomalies.append(AnomalyEvent(
                        anomaly_type=AnomalyType.SCENE_INCONSISTENCY,
                        timestamp=frame_number / self.fps,
                        frame_number=frame_number,
                        person_id=None,
                        severity=0.6,
                        description=obj_det.anomaly_reason or f"Objeto '{obj_det.class_name}' fora de contexto",
                        bbox=obj_det.bbox,
                        details={
                            "object_class": obj_det.class_name,
                            "category": obj_det.category.value,
                            "confidence": obj_det.confidence
                        }
                    ))
        
        return anomalies
    
    def process_overlay_detections(
        self,
        frame_number: int,
        overlay_detections: List
    ) -> List[AnomalyEvent]:
        """
        Processa detecções de overlays/texto e gera anomalias.
        
        Args:
            frame_number: Número do frame atual
            overlay_detections: Lista de OverlayDetection do OverlayDetector
            
        Returns:
            Lista de anomalias relacionadas a overlays
        """
        if not self.enable_overlay_anomalies:
            return []
        
        anomalies = []
        
        for overlay in overlay_detections:
            if overlay.is_anomalous:
                # Usa persistência temporal para confirmar
                anomaly_key = f"overlay_{overlay.overlay_type.value}_{overlay.position_zone}"
                
                if self._confirm_anomaly(anomaly_key, {
                    "type": AnomalyType.VISUAL_OVERLAY,
                    "reason": overlay.anomaly_reason,
                    "text": overlay.text,
                    "bbox": overlay.bbox
                }):
                    anomalies.append(AnomalyEvent(
                        anomaly_type=AnomalyType.VISUAL_OVERLAY,
                        timestamp=frame_number / self.fps,
                        frame_number=frame_number,
                        person_id=None,
                        severity=0.5,
                        description=overlay.anomaly_reason or f"Overlay detectado: '{overlay.text[:30]}...'",
                        bbox=overlay.bbox,
                        details={
                            "overlay_type": overlay.overlay_type.value,
                            "text": overlay.text[:100],
                            "position": overlay.position_zone,
                            "confidence": overlay.confidence
                        }
                    ))
        
        return anomalies
    
    def process_segment_validation(
        self,
        frame_number: int,
        segment_results: List
    ) -> List[AnomalyEvent]:
        """
        Processa validação de segmentação para detectar silhuetas anômalas.
        
        Args:
            frame_number: Número do frame atual
            segment_results: Lista de resultados de validação de segmentação
            
        Returns:
            Lista de anomalias relacionadas a silhuetas
        """
        anomalies = []
        
        for result in segment_results:
            if result.get("is_anomalous", False):
                anomaly_key = f"segment_{result.get('person_id', 0)}"
                
                if self._confirm_anomaly(anomaly_key, result):
                    anomalies.append(AnomalyEvent(
                        anomaly_type=AnomalyType.SILHOUETTE_ANOMALY,
                        timestamp=frame_number / self.fps,
                        frame_number=frame_number,
                        person_id=result.get("person_id"),
                        severity=result.get("severity", 0.5),
                        description=result.get("reason", "Silhueta anômala detectada"),
                        bbox=result.get("bbox"),
                        details=result
                    ))
        
        return anomalies
    
    def _confirm_anomaly(self, key: str, data: Dict) -> bool:
        """
        Confirma uma anomalia após persistência temporal.
        Evita falsos positivos exigindo detecção em múltiplos frames.
        
        Args:
            key: Chave única para a anomalia
            data: Dados da anomalia
            
        Returns:
            True se a anomalia foi confirmada
        """
        if key not in self._pending_anomalies:
            self._pending_anomalies[key] = {"count": 1, "data": data, "first_frame": self.frame_count}
            return False
        
        self._pending_anomalies[key]["count"] += 1
        
        # Limpa pendências antigas (mais de 30 frames sem ver)
        if self.frame_count - self._pending_anomalies[key].get("last_frame", self.frame_count) > 30:
            self._pending_anomalies[key] = {"count": 1, "data": data, "first_frame": self.frame_count}
            return False
        
        self._pending_anomalies[key]["last_frame"] = self.frame_count
        
        # Confirma se atingiu o threshold de persistência
        if self._pending_anomalies[key]["count"] >= self.require_persistence:
            # Remove do cache após confirmar
            del self._pending_anomalies[key]
            return True
        
        return False
    
    def update_extended(
        self,
        frame_number: int,
        face_detections: List,
        emotion_results: List,
        activity_detections: List,
        object_detections: Optional[List] = None,
        overlay_detections: Optional[List] = None,
        segment_results: Optional[List] = None
    ) -> List[AnomalyEvent]:
        """
        Versão estendida do update que inclui anomalias visuais.
        
        Args:
            frame_number: Número do frame atual
            face_detections: Lista de detecções de rostos
            emotion_results: Lista de resultados de análise emocional
            activity_detections: Lista de atividades detectadas
            object_detections: Lista de detecções de objetos (opcional)
            overlay_detections: Lista de overlays detectados (opcional)
            segment_results: Lista de resultados de segmentação (opcional)
            
        Returns:
            Lista de todas as anomalias detectadas neste frame
        """
        # Processa anomalias comportamentais (método original)
        anomalies = self.update(frame_number, face_detections, emotion_results, activity_detections)
        
        # Processa anomalias de objetos
        if object_detections:
            anomalies.extend(self.process_object_detections(frame_number, object_detections))
        
        # Processa anomalias de overlays
        if overlay_detections:
            anomalies.extend(self.process_overlay_detections(frame_number, overlay_detections))
        
        # Processa anomalias de segmentação
        if segment_results:
            anomalies.extend(self.process_segment_validation(frame_number, segment_results))
        
        return anomalies


    def update_with_context(
        self,
        frame_number: int,
        face_detections: List,
        emotion_results: List,
        activity_detections: List,
        scene_context: Optional[object] = None,
        detected_objects: Optional[List] = None
    ) -> List[AnomalyEvent]:
        """
        Wrapper que inclui validação contextual de cena.
        Combina detecção comportamental padrão com regras de contexto.
        """
        # Executa detecção padrão (comportamental)
        anomalies = self.update(frame_number, face_detections, emotion_results, activity_detections)
        
        # Validação Contextual (Cena vs Objetos)
        if self.enable_object_anomalies and scene_context and detected_objects:
            ctx_anomalies = self._check_context_anomalies(frame_number, scene_context, detected_objects)
            anomalies.extend(ctx_anomalies)
            
        return anomalies

    def _check_context_anomalies(self, frame_number: int, scene_context, objects: List) -> List[AnomalyEvent]:
        """Verifica se objetos presentes são consistentes com a cena detectada."""
        anomalies = []
        scene_type = scene_context.scene_type
        
        if scene_type not in SCENE_CONTEXT_RULES:
            return anomalies
            
        rules = SCENE_CONTEXT_RULES[scene_type]
        anomalous_objs = rules.get("anomalous", [])
        
        for obj in objects:
            # Verifica se o nome do objeto está na lista de proibidos
            if obj.class_name.lower() in anomalous_objs:
                # Gera evento
                event = AnomalyEvent(
                    anomaly_type=AnomalyType.SCENE_INCONSISTENCY,
                    timestamp=frame_number / self.fps,
                    frame_number=frame_number,
                    person_id=None,
                    severity=0.85,
                    description=f"Objeto indevido ({obj.class_name}) em {scene_type}",
                    bbox=obj.bbox,
                    details={
                        "scene": scene_type, 
                        "object": obj.class_name, 
                        "confidence": float(obj.confidence)
                    }
                )
                anomalies.append(event)
                
        return anomalies


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
