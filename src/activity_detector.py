"""
Tech Challenge - Fase 4: Detector de Atividades
Usa YOLOv8-pose para detecção de pessoas e análise de poses/atividades.
"""

import cv2
import numpy as np
import logging

# Importa torch após numpy para evitar bug de compatibilidade
import torch

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum

from .config import (ACTIVITY_CATEGORIES, get_device, YOLO_MODEL_SIZE,
                       ACTIVITY_POSE_THRESHOLDS)

logger = logging.getLogger(__name__)


class ActivityType(Enum):
    """Tipos de atividades detectáveis."""
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"  # Deitado
    WALKING = "walking"
    RUNNING = "running"
    WAVING = "waving"
    POINTING = "pointing"
    DANCING = "dancing"
    CROUCHING = "crouching"
    ARMS_RAISED = "arms_raised"
    GREETING = "greeting"  # Cumprimentando (Handshake)
    UNKNOWN = "unknown"


@dataclass
class PoseKeypoints:
    """Keypoints de pose (formato COCO - 17 pontos)."""
    nose: Optional[Tuple[float, float]] = None
    left_eye: Optional[Tuple[float, float]] = None
    right_eye: Optional[Tuple[float, float]] = None
    left_ear: Optional[Tuple[float, float]] = None
    right_ear: Optional[Tuple[float, float]] = None
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    left_elbow: Optional[Tuple[float, float]] = None
    right_elbow: Optional[Tuple[float, float]] = None
    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None
    left_hip: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    left_knee: Optional[Tuple[float, float]] = None
    right_knee: Optional[Tuple[float, float]] = None
    left_ankle: Optional[Tuple[float, float]] = None
    right_ankle: Optional[Tuple[float, float]] = None


@dataclass
class ActivityDetection:
    """Resultado da detecção de atividade."""
    person_id: int
    activity: ActivityType
    activity_pt: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    keypoints: Optional[PoseKeypoints] = None
    velocity: float = 0.0


class ActivityDetector:
    """Detector de atividades usando YOLOv8-pose."""
    
    def __init__(
        self, 
        model_size: str = None,
        min_confidence: float = 0.5,
        history_size: int = 10,
        device: Optional[str] = None
    ):
        """
        Args:
            model_size: Tamanho do modelo ('n', 's', 'm', 'l', 'x'). None usa config.
            min_confidence: Confiança mínima para detecção
            history_size: Frames de histórico para análise temporal
            device: Device para inferência ('cuda', 'cpu' ou None para auto)
        """
        self.min_confidence = min_confidence
        self.history_size = history_size
        self.person_counter = 0
        self.position_history: Dict[int, deque] = {}
        self.pose_history: Dict[int, deque] = {}
        self.device = device if device is not None else get_device()
        
        self._init_yolo(model_size or YOLO_MODEL_SIZE)
    
    def _init_yolo(self, model_size: str):
        """Inicializa YOLO11-pose (mais preciso que YOLOv8)."""
        try:
            from ultralytics import YOLO
            # Usa YOLO11 (melhor precisão)
            model_name = f"yolo11{model_size}-pose.pt"
            self.model = YOLO(model_name)
            self.model.to(self.device)
            self.model_loaded = True
            logger.info(f"Modelo carregado: {model_name} (device: {self.device})")
        except Exception as e:
            logger.warning(f"YOLO11 não disponível, tentando YOLOv8: {e}")
            try:
                from ultralytics import YOLO
                model_name = f"yolov8{model_size}-pose.pt"
                self.model = YOLO(model_name)
                self.model.to(self.device)
                self.model_loaded = True
                logger.info(f"Modelo carregado: {model_name} (device: {self.device})")
            except Exception as e2:
                logger.error(f"Falha ao carregar modelo: {e2}")
                self.model = None
                self.model_loaded = False
    
    def detect(self, frame: np.ndarray, oriented_detections: Optional[List] = None) -> List[ActivityDetection]:
        """
        Detecta pessoas e suas atividades no frame.
        
        Args:
            frame: Frame BGR
            oriented_detections: Lista de OrientedDetection (opcional) para refinar postura
        """
        if not self.model_loaded:
            return []
        
        results = self.model(frame, verbose=False, conf=self.min_confidence, device=self.device)
        detections = []
        
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            
            keypoints_data = result.keypoints.xy.cpu().numpy()
            confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            boxes = result.boxes
            
            for i, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                # Filtra por confiança do box
                box_conf = float(box.conf[0])
                if box_conf < self.min_confidence:
                    continue
                
                # Bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                bbox = (
                    int(xyxy[0]), int(xyxy[1]),
                    int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                )
                
                # Keypoints
                kpt_conf = confidences[i] if confidences is not None else None
                keypoints = self._extract_keypoints(kpts, kpt_conf)
                
                # ID da pessoa (tracking simples por posição)
                person_id = self._assign_person_id(bbox)
                
                # Analisa atividade
                activity, act_conf = self._analyze_activity(person_id, keypoints)
                
                # Integração com Oriented Detection (OBB)
                if oriented_detections:
                    is_lying_obb = self._match_orientation(bbox, oriented_detections)
                    if is_lying_obb and activity != ActivityType.LYING:
                        activity = ActivityType.LYING
                        act_conf = max(act_conf, 0.85)

                # Velocidade
                velocity = self._calculate_velocity(person_id, bbox)
                
                detections.append(ActivityDetection(
                    person_id=person_id,
                    activity=activity,
                    activity_pt=ACTIVITY_CATEGORIES.get(activity.value, "Desconhecido"),
                    confidence=act_conf * box_conf,
                    bbox=bbox,
                    keypoints=keypoints,
                    velocity=velocity
                ))
        
        # Detecta interações sociais (ex: aperto de mão)
        detections = self._detect_social_interactions(detections)
        
        return detections

    def _detect_social_interactions(self, detections: List[ActivityDetection]) -> List[ActivityDetection]:
        """Detecta interações entre pessoas (ex: aperto de mão)."""
        if len(detections) < 2:
            return detections
            
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                p1 = detections[i]
                p2 = detections[j]
                
                # IMPORTANTE: Não reclassifica se uma das pessoas está deitada
                # (médico tocando paciente NÃO é cumprimento)
                if p1.activity == ActivityType.LYING or p2.activity == ActivityType.LYING:
                    continue
                
                # Precisa que AMBOS tenham pulso direito E ambos estejam de pé ou similar
                if not (p1.keypoints.right_wrist and p2.keypoints.right_wrist):
                    continue
                
                # Verifica se ambas as pessoas estão em postura vertical (não deitadas)
                p1_standing = self._is_person_upright(p1.keypoints)
                p2_standing = self._is_person_upright(p2.keypoints)
                
                if not (p1_standing and p2_standing):
                    continue
                
                rw1 = p1.keypoints.right_wrist
                rw2 = p2.keypoints.right_wrist
                
                # Distância entre pulsos
                wrist_dist = np.sqrt((rw1[0] - rw2[0])**2 + (rw1[1] - rw2[1])**2)
                
                # Usa limiares configuráveis
                max_wrist_dist = ACTIVITY_POSE_THRESHOLDS.get("greeting_wrist_distance_max", 60)
                min_shoulder_dist = ACTIVITY_POSE_THRESHOLDS.get("greeting_shoulder_distance_min", 150)
                max_wrist_height_diff = ACTIVITY_POSE_THRESHOLDS.get("greeting_wrist_height_diff_max", 50)
                
                # Se pulsos estão muito próximos (< 60px)
                if wrist_dist >= max_wrist_dist:
                    continue
                
                # Verifica também a distância dos ombros para garantir que são pessoas diferentes
                s1 = p1.keypoints.right_shoulder or p1.keypoints.left_shoulder
                s2 = p2.keypoints.right_shoulder or p2.keypoints.left_shoulder
                
                if not (s1 and s2):
                    continue
                
                shoulder_dist = np.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)
                
                # Pulsos muito próximos mas corpos separados
                # E pulsos devem estar aproximadamente na mesma ALTURA
                wrist_height_diff = abs(rw1[1] - rw2[1])
                
                if wrist_dist < max_wrist_dist and shoulder_dist > min_shoulder_dist and wrist_height_diff < max_wrist_height_diff:
                    # Verifica se os pulsos estão entre os dois corpos (região central)
                    # Isso evita detectar toques laterais como cumprimento
                    mid_x = (s1[0] + s2[0]) / 2
                    wrist_mid_x = (rw1[0] + rw2[0]) / 2
                    
                    # Pulsos devem estar próximos do centro entre as duas pessoas
                    if abs(wrist_mid_x - mid_x) < shoulder_dist * 0.3:
                        # Classifica ambos como cumprimento
                        p1.activity = ActivityType.GREETING
                        p1.activity_pt = ACTIVITY_CATEGORIES.get("greeting", "Cumprimentando")
                        p2.activity = ActivityType.GREETING
                        p2.activity_pt = ACTIVITY_CATEGORIES.get("greeting", "Cumprimentando")
                            
        return detections
    
    def _is_person_upright(self, kp: PoseKeypoints) -> bool:
        """Verifica se a pessoa está em postura vertical (não deitada)."""
        if not (kp.left_shoulder and kp.right_shoulder):
            return True  # Sem dados suficientes, assume vertical
        
        if not (kp.left_hip and kp.right_hip):
            return True
        
        # Calcula orientação do torso
        shoulder_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
        shoulder_x = (kp.left_shoulder[0] + kp.right_shoulder[0]) / 2
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        hip_x = (kp.left_hip[0] + kp.right_hip[0]) / 2
        
        vertical_diff = abs(shoulder_y - hip_y)
        horizontal_diff = abs(shoulder_x - hip_x)
        
        # Se corpo está mais vertical que horizontal = em pé
        return vertical_diff > horizontal_diff
                            
        return detections

    def _match_orientation(self, person_bbox: Tuple[int, int, int, int], oriented_detections: List) -> bool:
        """Verifica se há um OrientedDetection correspondente que indica 'Deitado'."""
        px, py, pw, ph = person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3]
        pcx, pcy = px + pw/2, py + ph/2
        
        for obb in oriented_detections:
            # Proximidade de centro
            dist = np.sqrt((pcx - obb.center[0])**2 + (pcy - obb.center[1])**2)
            
            # Se centros estão próximos (menos de metade da maior dimensão)
            if dist < max(pw, ph) * 0.5:
                if obb.is_lying_down():
                    return True
        return False
    
    def _extract_keypoints(
        self, 
        kpts: np.ndarray, 
        conf: Optional[np.ndarray] = None,
        min_kpt_conf: float = 0.3
    ) -> PoseKeypoints:
        """Extrai keypoints do formato YOLO (COCO 17 pontos)."""
        def get_point(idx: int) -> Optional[Tuple[float, float]]:
            if idx >= len(kpts):
                return None
            x, y = kpts[idx]
            if x == 0 and y == 0:
                return None
            if conf is not None and idx < len(conf) and conf[idx] < min_kpt_conf:
                return None
            return (float(x), float(y))
        
        return PoseKeypoints(
            nose=get_point(0),
            left_eye=get_point(1),
            right_eye=get_point(2),
            left_ear=get_point(3),
            right_ear=get_point(4),
            left_shoulder=get_point(5),
            right_shoulder=get_point(6),
            left_elbow=get_point(7),
            right_elbow=get_point(8),
            left_wrist=get_point(9),
            right_wrist=get_point(10),
            left_hip=get_point(11),
            right_hip=get_point(12),
            left_knee=get_point(13),
            right_knee=get_point(14),
            left_ankle=get_point(15),
            right_ankle=get_point(16)
        )
    
    def _assign_person_id(self, bbox: Tuple[int, int, int, int]) -> int:
        """Atribui ID baseado em proximidade com detecções anteriores."""
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        
        # Busca pessoa próxima no histórico
        min_dist = float('inf')
        best_id = None
        
        for pid, history in self.position_history.items():
            if history:
                last_pos = history[-1]
                dist = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
                if dist < min_dist and dist < 100:
                    min_dist = dist
                    best_id = pid
        
        if best_id is None:
            self.person_counter += 1
            best_id = self.person_counter
            self.position_history[best_id] = deque(maxlen=self.history_size)
            self.pose_history[best_id] = deque(maxlen=self.history_size)
        
        self.position_history[best_id].append((cx, cy))
        return best_id
    
    def _calculate_velocity(self, person_id: int, bbox: Tuple[int, int, int, int]) -> float:
        """Calcula velocidade do movimento em pixels/frame."""
        history = self.position_history.get(person_id)
        if not history or len(history) < 2:
            return 0.0
        
        positions = list(history)
        if len(positions) >= 2:
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
            return np.sqrt(dx**2 + dy**2)
        return 0.0
    
    def _analyze_activity(
        self, 
        person_id: int, 
        keypoints: PoseKeypoints
    ) -> Tuple[ActivityType, float]:
        """Analisa atividade baseada em pose e histórico."""
        self.pose_history.setdefault(person_id, deque(maxlen=self.history_size))
        self.pose_history[person_id].append(keypoints)
        
        # Calcula velocidade uma vez
        velocity = self._get_avg_velocity(person_id)
        
        # === ORDEM DE PRIORIDADE ===
        # 1. Deitado (posição horizontal clara)
        # 2. Em movimento (correndo/caminhando) - se está se movendo, NÃO está sentado
        # 3. Em pé (claramente ou frontal)
        # 4. Agachado/Sentado (apenas se estático)
        
        # 1. Verifica postura deitada primeiro (alta prioridade)
        if self._is_lying(keypoints):
            return ActivityType.LYING, 0.85
        
        # 2. MOVIMENTO 
        # Se velocity > running -> RUNNING
        running_threshold = ACTIVITY_POSE_THRESHOLDS["running_velocity_threshold"]
        walking_threshold = ACTIVITY_POSE_THRESHOLDS["walking_velocity_threshold"]
        
        if velocity > running_threshold:
            return ActivityType.RUNNING, 0.85
        
        # Se velocity > walking, verifica pose antes de classificar
        if velocity > walking_threshold:
            # Se a pessoa está CLARAMENTE EM PÉ (pernas retas, vertical) e velocidade não é de corrida,
            # pode ser apenas movimento de câmera. Prioriza STANDING sobre WALKING em limiares baixos.
            is_standing_clear = self._is_clearly_standing(keypoints)
            if is_standing_clear and velocity < running_threshold * 0.8:
                # Retorna STANDING com confiança menor (incerteza devido à velocidade)
                # O detector seguirá para o bloco de STANDING abaixo
                pass
            else:
                return ActivityType.WALKING, 0.8
        
        # 3. Verifica SENTADO primeiro (prioridade para pernas ocultas + parado)
        # Isso captura pessoas em mesas de café/escritório
        if self._is_sitting(keypoints, velocity):
            return ActivityType.SITTING, 0.8
        
        # 4. Verifica se está EM PÉ (requer evidência de pernas)
        # 4a. Claramente em pé (pernas visíveis, alinhamento vertical)
        is_standing_clear = self._is_clearly_standing(keypoints)
        # 4b. Em pé de frente (torso vertical + pernas parcialmente visíveis)
        is_standing_frontal = self._is_frontal_standing(keypoints)
        
        if is_standing_clear or is_standing_frontal:
            # Verifica gestos mesmo em pé
            gesture_velocity = ACTIVITY_POSE_THRESHOLDS["gesture_velocity_threshold"]
            if velocity > gesture_velocity:
                if self._is_waving(keypoints):
                    return ActivityType.WAVING, 0.8
                if self._is_pointing(keypoints):
                    return ActivityType.POINTING, 0.75
            
            # Braços levantados mesmo parado
            if self._is_arms_raised(keypoints):
                return ActivityType.ARMS_RAISED, 0.85
            
            # Confiança maior se ambos critérios concordam
            conf = 0.85 if (is_standing_clear and is_standing_frontal) else 0.8
            return ActivityType.STANDING, conf
        
        # 5. Agachado (postura específica)
        if self._is_crouching(keypoints):
            return ActivityType.CROUCHING, 0.75
        
        # 5. Verifica braços levantados (ambos acima da cabeça)
        if self._is_arms_raised(keypoints):
            return ActivityType.ARMS_RAISED, 0.85
        
        # 6. Gestos específicos de mãos (MAS NÃO se está parado)
        gesture_velocity = ACTIVITY_POSE_THRESHOLDS["gesture_velocity_threshold"]
        if velocity > gesture_velocity:
            if self._is_waving(keypoints):
                return ActivityType.WAVING, 0.8
            
            if self._is_pointing(keypoints):
                return ActivityType.POINTING, 0.75
        
        # 7. Dança (movimento rítmico)
        if self._is_dancing(person_id, keypoints):
            return ActivityType.DANCING, 0.8
        
        # 7. Verifica movimento geral pela velocidade
        running_threshold = ACTIVITY_POSE_THRESHOLDS["running_velocity_threshold"]
        walking_threshold = ACTIVITY_POSE_THRESHOLDS["walking_velocity_threshold"]
        
        if velocity > running_threshold:
            return ActivityType.RUNNING, 0.8
        elif velocity > walking_threshold:
            return ActivityType.WALKING, 0.75
        else:
            return ActivityType.STANDING, 0.7
    
    def _get_avg_velocity(self, person_id: int) -> float:
        """Calcula velocidade média recente."""
        history = self.position_history.get(person_id)
        if not history or len(history) < 3:
            return 0.0
        
        positions = list(history)
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))
        
        return np.mean(velocities) if velocities else 0.0
    
    def _is_waving(self, kp: PoseKeypoints) -> bool:
        """Detecta gesto de acenar/cumprimentar (mão levantada, pessoa em pé)."""
        # IMPORTANTE: Não detectar waving se pessoa parece estar deitada
        # Verifica orientação do corpo primeiro
        if kp.left_shoulder and kp.right_shoulder and kp.left_hip and kp.right_hip:
            shoulder_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
            shoulder_x = (kp.left_shoulder[0] + kp.right_shoulder[0]) / 2
            hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
            hip_x = (kp.left_hip[0] + kp.right_hip[0]) / 2
            
            vertical_diff = abs(shoulder_y - hip_y)
            horizontal_diff = abs(shoulder_x - hip_x)
            
            # Se corpo está mais horizontal que vertical, não é waving
            if horizontal_diff > vertical_diff * 0.8:
                return False
        
        waving_hand_above_shoulder = ACTIVITY_POSE_THRESHOLDS["waving_hand_above_shoulder"]
        waving_angle_min = ACTIVITY_POSE_THRESHOLDS["waving_elbow_angle_min"]
        waving_angle_max = ACTIVITY_POSE_THRESHOLDS["waving_elbow_angle_max"]
        
        for wrist, elbow, shoulder in [
            (kp.left_wrist, kp.left_elbow, kp.left_shoulder),
            (kp.right_wrist, kp.right_elbow, kp.right_shoulder)
        ]:
            if all([wrist, elbow, shoulder]):
                # Mão SIGNIFICATIVAMENTE acima do ombro (critério configurável)
                if wrist[1] < shoulder[1] - waving_hand_above_shoulder:
                    # Aceita ângulo do cotovelo dentro dos limiares
                    elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
                    if waving_angle_min < elbow_angle < waving_angle_max:
                        return True
        return False
    
    def _is_pointing(self, kp: PoseKeypoints) -> bool:
        """Detecta gesto de apontar (braço estendido horizontalmente)."""
        pointing_angle_min = ACTIVITY_POSE_THRESHOLDS["pointing_arm_angle_min"]
        pointing_length = ACTIVITY_POSE_THRESHOLDS["pointing_horizontal_length"]
        pointing_variance = ACTIVITY_POSE_THRESHOLDS["pointing_vertical_variance"]
        
        for wrist, elbow, shoulder in [
            (kp.left_wrist, kp.left_elbow, kp.left_shoulder),
            (kp.right_wrist, kp.right_elbow, kp.right_shoulder)
        ]:
            if all([wrist, elbow, shoulder]):
                # Braço estendido (ângulo > threshold)
                arm_angle = self._calculate_angle(shoulder, elbow, wrist)
                if arm_angle > pointing_angle_min:
                    # Horizontalmente (variação vertical pequena)
                    arm_height_diff = abs(wrist[1] - shoulder[1])
                    arm_length = abs(wrist[0] - shoulder[0])
                    if arm_length > pointing_length and arm_height_diff < pointing_variance:
                        return True
        return False
    
    def _is_sitting(self, kp: PoseKeypoints, velocity: float = 0.0) -> bool:
        """Detecta postura sentada.
        
        Critérios para SENTADO:
        1. Pernas NÃO visíveis (ocultas por mesa) + torso visível + parado
        2. Quadril e joelhos na mesma altura (pernas dobradas)
        3. NÃO está em pé com pernas visíveis
        """
        
        # REGRA 1: Se está em movimento significativo, NÃO está sentado
        max_velocity = ACTIVITY_POSE_THRESHOLDS.get("sitting_max_velocity", 15)
        if velocity > max_velocity:
            return False
        
        # REGRA 2: Se está CLARAMENTE em pé (pernas VISÍVEIS abaixo), NÃO está sentado
        # Isso exige ver joelhos/tornozelos bem abaixo do quadril
        if self._is_clearly_standing(kp):
            return False
        
        # REGRA 3: Pernas NÃO visíveis + torso visível + parado = SENTADO
        # (típico de pessoa em mesa de café/escritório)
        has_knees = kp.left_knee or kp.right_knee
        has_ankles = kp.left_ankle or kp.right_ankle
        has_torso = (kp.left_shoulder and kp.right_shoulder and 
                     kp.left_hip and kp.right_hip)
        
        if has_torso and not has_knees and not has_ankles:
            # Torso visível mas sem pernas = sentado em mesa
            # (pessoa em pé teria pernas visíveis)
            return True
        
        # REGRA 4: Análise geométrica quando temos joelhos visíveis
        if not all([kp.left_hip, kp.right_hip, kp.left_knee, kp.right_knee]):
            return False
        
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        knee_y = (kp.left_knee[1] + kp.right_knee[1]) / 2
        
        # Joelhos devem estar APROXIMADAMENTE na mesma altura que quadril
        hip_knee_diff = abs(hip_y - knee_y)
        
        # Critério: quadril e joelhos próximos em Y (pessoa sentada dobra as pernas)
        if kp.left_shoulder and kp.right_shoulder:
            shoulder_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
            torso_length = abs(hip_y - shoulder_y)
            
            # Sentado: quadril-joelho < 50% do torso E joelhos NÃO estão muito abaixo
            if hip_knee_diff < torso_length * ACTIVITY_POSE_THRESHOLDS["sitting_torso_factor"]:
                # Confirma: joelhos não estão bem abaixo do quadril
                if knee_y - hip_y < ACTIVITY_POSE_THRESHOLDS["standing_hip_knee_diff_min"]:
                    return True
        
        # Fallback: diferença absoluta pequena
        sitting_threshold = ACTIVITY_POSE_THRESHOLDS["sitting_knee_hip_diff_max"]
        if hip_knee_diff < sitting_threshold:
            # Mas apenas se joelhos não estão bem abaixo
            if knee_y - hip_y < ACTIVITY_POSE_THRESHOLDS["standing_hip_knee_diff_min"]:
                return True
        
        return False
    
    def _is_clearly_standing(self, kp: PoseKeypoints) -> bool:
        """Verifica se a pessoa está claramente em pé (postura vertical completa)."""
        # Precisa de pelo menos: ombros, quadril e (joelhos OU tornozelos)
        if not (kp.left_shoulder and kp.right_shoulder):
            return False
        if not (kp.left_hip and kp.right_hip):
            return False
        
        # Precisa de pelo menos joelhos OU tornozelos
        has_knees = kp.left_knee and kp.right_knee
        has_ankles = kp.left_ankle and kp.right_ankle
        
        if not (has_knees or has_ankles):
            return False
        
        # Calcula centros
        shoulder_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        
        # Verifica alinhamento VERTICAL (ombros acima de quadril)
        if shoulder_y >= hip_y:  # Em imagens, Y cresce para baixo
            return False
        
        # Limiares configuráveis
        hip_knee_min = ACTIVITY_POSE_THRESHOLDS["standing_hip_knee_diff_min"]
        knee_ankle_min = ACTIVITY_POSE_THRESHOLDS["standing_knee_ankle_diff_min"]
        hip_ankle_min = ACTIVITY_POSE_THRESHOLDS["standing_hip_ankle_diff_min"]
        
        # Se temos joelhos, verifica se estão ABAIXO do quadril
        if has_knees:
            knee_y = (kp.left_knee[1] + kp.right_knee[1]) / 2
            hip_knee_diff = knee_y - hip_y  # Deve ser positivo (joelho abaixo)
            
            # Se joelho está bem abaixo do quadril = em pé
            if hip_knee_diff > hip_knee_min:
                # Verifica também se tornozelos estão abaixo dos joelhos
                if has_ankles:
                    ankle_y = (kp.left_ankle[1] + kp.right_ankle[1]) / 2
                    knee_ankle_diff = ankle_y - knee_y  # Deve ser positivo
                    
                    # Tornozelos bem abaixo dos joelhos = definitivamente em pé
                    if knee_ankle_diff > knee_ankle_min:
                        return True
                else:
                    # Sem tornozelos visíveis, mas joelhos bem abaixo do quadril
                    return True
        
        # Se temos apenas tornozelos (sem joelhos), verifica distância
        elif has_ankles:
            ankle_y = (kp.left_ankle[1] + kp.right_ankle[1]) / 2
            hip_ankle_diff = ankle_y - hip_y  # Deve ser grande se está em pé
            
            # Grande distância quadril-tornozelo = em pé
            if hip_ankle_diff > hip_ankle_min:
                return True
        
        return False
    
    def _is_frontal_standing(self, kp: PoseKeypoints, bbox: Tuple[int, int, int, int] = None) -> bool:
        """
        Detecta pessoa em pé DE FRENTE para a câmera.
        
        IMPORTANTE: Torso vertical NÃO é suficiente!
        Pessoa SENTADA também tem torso vertical.
        
        Para confirmar EM PÉ frontal, precisa:
        - Torso vertical E
        - Alguma evidência de pernas (joelhos OU tornozelos visíveis)
        """
        # Precisa de ombros e quadril
        if not (kp.left_shoulder and kp.right_shoulder):
            return False
        if not (kp.left_hip and kp.right_hip):
            return False
        
        # CRÍTICO: Precisa de ALGUMA evidência de pernas!
        # Sem pernas visíveis, não podemos distinguir em pé de sentado
        has_knees = kp.left_knee or kp.right_knee
        has_ankles = kp.left_ankle or kp.right_ankle
        
        if not has_knees and not has_ankles:
            # Sem pernas visíveis = NÃO podemos confirmar em pé
            # (provavelmente sentado com pernas ocultas)
            return False
        
        # Calcula centros
        shoulder_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        
        # Thresholds
        shoulder_hip_min = ACTIVITY_POSE_THRESHOLDS.get("frontal_shoulder_hip_min", 40)
        
        # Torso vertical (ombros acima do quadril)
        vertical_diff = hip_y - shoulder_y
        if vertical_diff < shoulder_hip_min:
            return False
        
        # Se temos joelhos, verificar se estão em posição de pé (abaixo do quadril)
        if has_knees:
            knee_y = 0
            count = 0
            if kp.left_knee:
                knee_y += kp.left_knee[1]
                count += 1
            if kp.right_knee:
                knee_y += kp.right_knee[1]
                count += 1
            knee_y /= count
            
            # Joelhos devem estar ABAIXO do quadril para pessoa em pé
            # Se joelhos estão na mesma altura ou acima = sentado
            knee_hip_diff = knee_y - hip_y
            if knee_hip_diff < 20:  # Joelhos muito próximos do quadril = sentado
                return False
            
            # Joelhos bem abaixo do quadril = em pé
            return True
        
        # Se temos tornozelos, verificar distância do quadril
        if has_ankles:
            ankle_y = 0
            count = 0
            if kp.left_ankle:
                ankle_y += kp.left_ankle[1]
                count += 1
            if kp.right_ankle:
                ankle_y += kp.right_ankle[1]
                count += 1
            ankle_y /= count
            
            # Tornozelos devem estar BEM abaixo do quadril
            ankle_hip_diff = ankle_y - hip_y
            if ankle_hip_diff > 80:  # Boa distância = em pé
                return True
        
        return False
    
    def _is_crouching(self, kp: PoseKeypoints) -> bool:
        """Detecta postura agachada (quadril muito baixo, joelhos dobrados, corpo comprimido)."""
        if not all([kp.left_hip, kp.right_hip, kp.left_knee, kp.right_knee]):
            return False
        
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        knee_y = (kp.left_knee[1] + kp.right_knee[1]) / 2
        
        # IMPORTANTE: Agachado requer verificação mais rigorosa
        # Uma pessoa sentada também pode ter quadril e joelho próximos!
        # A diferença é que AGACHADO tem tornozelos visíveis abaixo dos joelhos
        
        # Tornozelos disponíveis para confirmação (critério principal)
        if kp.left_ankle and kp.right_ankle:
            ankle_y = (kp.left_ankle[1] + kp.right_ankle[1]) / 2
            
            # Agachado: quadril muito próximo aos joelhos (< threshold) E
            # joelhos CLARAMENTE acima dos tornozelos
            hip_knee_diff = abs(hip_y - knee_y)
            knee_ankle_diff = abs(ankle_y - knee_y)
            
            # Critério: quadril próximo joelho (corpo comprimido) E joelho acima tornozelo
            if (hip_knee_diff < ACTIVITY_POSE_THRESHOLDS["crouching_hip_knee_diff"] and
                knee_ankle_diff > ACTIVITY_POSE_THRESHOLDS["crouching_ankle_margin"]):
                # Verificação adicional: se temos ombros, confirma que corpo está comprimido
                if kp.left_shoulder and kp.right_shoulder:
                    shoulder_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
                    shoulder_hip_diff = abs(shoulder_y - hip_y)
                    
                    # Agachado: ombro-quadril é pequeno (corpo muito comprimido)
                    # Sentado: ombro-quadril é grande (torso esticado)
                    if shoulder_hip_diff < 150:  # Corpo comprimido
                        return True
                else:
                    # Sem ombros visíveis, usa apenas análise de pernas
                    return True
        
        return False
    
    def _is_lying(self, kp: PoseKeypoints) -> bool:
        """
        Detecta postura deitada com critérios rigorosos.
        
        Usa múltiplas evidências para evitar falsos positivos:
        1. Orientação horizontal do torso (ombros-quadril)
        2. Orientação da cabeça (olhos/nariz alinhados horizontalmente)
        3. Corpo inteiro horizontal (quando tornozelos visíveis)
        
        IMPORTANTE: Pessoa acenando/dançando com braços levantados NÃO é deitada.
        """
        # Precisa de ombros E quadril para análise confiável
        if not (kp.left_shoulder and kp.right_shoulder):
            return False
        if not (kp.left_hip and kp.right_hip):
            return False
        
        # === VERIFICAÇÃO DE FACE (novo critério) ===
        # Se a face está VERTICAL (olhos acima do nariz), pessoa NÃO está deitada
        face_is_vertical = self._is_face_vertical(kp)
        if face_is_vertical:
            # Face claramente vertical = pessoa em pé ou sentada, não deitada
            return False
        
        # Calcula centros do torso
        shoulder_center_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
        shoulder_center_x = (kp.left_shoulder[0] + kp.right_shoulder[0]) / 2
        hip_center_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        hip_center_x = (kp.left_hip[0] + kp.right_hip[0]) / 2
        
        # Diferença vertical e horizontal entre ombros e quadril
        vertical_diff = abs(shoulder_center_y - hip_center_y)
        horizontal_diff = abs(shoulder_center_x - hip_center_x)
        
        # Distância entre ombros (largura visível do corpo)
        shoulder_width = abs(kp.left_shoulder[0] - kp.right_shoulder[0])
        
        # === CRITÉRIO PRINCIPAL ===
        # Torso claramente horizontal: horizontal > X * vertical E distância significativa
        lying_ratio = ACTIVITY_POSE_THRESHOLDS.get("lying_horizontal_ratio", 2.0)
        lying_min_dist = ACTIVITY_POSE_THRESHOLDS.get("lying_min_horizontal_dist", 100)
        
        if horizontal_diff > vertical_diff * lying_ratio and horizontal_diff > lying_min_dist:
            # Confirma com face horizontal (se disponível)
            face_is_horizontal = self._is_face_horizontal(kp)
            if face_is_horizontal:
                return True
            # Se não temos dados de face, aceita apenas se torso é MUITO horizontal (margem de segurança + 1.0)
            if horizontal_diff > vertical_diff * (lying_ratio + 1.0):
                return True
        
        # === CRITÉRIO SECUNDÁRIO: Pessoa de lado ===
        # Ombros muito próximos em X (visto de lado) + torso horizontal
        shoulder_width_thresh = ACTIVITY_POSE_THRESHOLDS.get("lying_shoulder_width_threshold", 40)
        
        if shoulder_width < shoulder_width_thresh:
            hip_width = abs(kp.left_hip[0] - kp.right_hip[0])
            if hip_width < 40 and horizontal_diff > vertical_diff * 1.5:
                # Pessoa de lado: precisa confirmar com face ou torso muito horizontal
                face_is_horizontal = self._is_face_horizontal(kp)
                if face_is_horizontal or horizontal_diff > vertical_diff * 2.5:
                    return True
        
        # === CRITÉRIO COM TORNOZELOS ===
        # Verificação completa do corpo quando temos tornozelos
        if kp.left_ankle and kp.right_ankle:
            ankle_center_y = (kp.left_ankle[1] + kp.right_ankle[1]) / 2
            ankle_center_x = (kp.left_ankle[0] + kp.right_ankle[0]) / 2
            
            total_vertical = abs(shoulder_center_y - ankle_center_y)
            total_horizontal = abs(shoulder_center_x - ankle_center_x)
            
            # Corpo inteiro horizontal (2x mais largo que alto + distância mínima)
            if total_horizontal > total_vertical * 2.0 and total_horizontal > 150:
                return True
        
        return False
    
    def _is_face_vertical(self, kp: PoseKeypoints) -> bool:
        """
        Verifica se a face está em orientação VERTICAL (pessoa em pé/sentada).
        
        Critérios:
        - Olhos na mesma altura (variação Y pequena)
        - Nariz abaixo dos olhos
        - Boca/queixo abaixo do nariz (se disponível)
        """
        # Verifica olhos
        if kp.left_eye and kp.right_eye:
            eye_y_diff = abs(kp.left_eye[1] - kp.right_eye[1])
            eye_x_diff = abs(kp.left_eye[0] - kp.right_eye[0])
            
            # Olhos na mesma altura (variação Y < 20px) e separados horizontalmente
            if eye_y_diff < 20 and eye_x_diff > 15:
                # Verifica nariz abaixo dos olhos
                if kp.nose:
                    avg_eye_y = (kp.left_eye[1] + kp.right_eye[1]) / 2
                    # Nariz deve estar ABAIXO dos olhos (Y maior)
                    if kp.nose[1] > avg_eye_y + 5:
                        return True
        
        # Fallback: apenas nariz e ombros
        if kp.nose and kp.left_shoulder and kp.right_shoulder:
            shoulder_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
            # Nariz significativamente ACIMA dos ombros = face vertical
            if kp.nose[1] < shoulder_y - 30:
                return True
        
        return False
    
    def _is_face_horizontal(self, kp: PoseKeypoints) -> bool:
        """
        Verifica se a face está em orientação HORIZONTAL (pessoa deitada).
        
        Critérios:
        - Olhos em alturas diferentes (um acima do outro)
        - Ou nariz na mesma altura que olhos
        """
        if kp.left_eye and kp.right_eye:
            eye_y_diff = abs(kp.left_eye[1] - kp.right_eye[1])
            eye_x_diff = abs(kp.left_eye[0] - kp.right_eye[0])
            
            # Olhos em alturas DIFERENTES (Y > 20px) ou muito próximos em X
            if eye_y_diff > 25 or eye_x_diff < 10:
                return True
            
            # Olhos + nariz na mesma altura (face horizontal)
            if kp.nose:
                avg_eye_y = (kp.left_eye[1] + kp.right_eye[1]) / 2
                nose_eye_diff = abs(kp.nose[1] - avg_eye_y)
                # Nariz na mesma altura que olhos = face horizontal
                if nose_eye_diff < 15:
                    return True
        
        return False
    
    def _is_arms_raised(self, kp: PoseKeypoints) -> bool:
        """Detecta ambos os braços levantados acima da cabeça."""
        if not all([kp.left_wrist, kp.right_wrist, kp.nose]):
            return False
        
        # Ambos pulsos acima do nariz (usa limiar configurável)
        arm_above_head_threshold = ACTIVITY_POSE_THRESHOLDS["arms_raised_hand_above_head"]
        if (kp.left_wrist[1] < kp.nose[1] - arm_above_head_threshold and 
            kp.right_wrist[1] < kp.nose[1] - arm_above_head_threshold):
            return True
        
        return False
    
    def _is_dancing(self, person_id: int, kp: PoseKeypoints) -> bool:
        """Detecta dança baseada em movimento rítmico e variação de pose."""
        history = self.pose_history.get(person_id)
        if not history or len(history) < 8:  # Requer mais histórico (era 5)
            return False
            
        # Requer wrists (pulsos) visíveis e idealmente acima da cintura
        if not (kp.left_wrist and kp.right_wrist and kp.left_hip and kp.right_hip):
            return False
            
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        # Bailarinas geralmente mantêm braços mais altos (pelo menos na altura do quadril)
        if kp.left_wrist[1] > hip_y + 20 or kp.right_wrist[1] > hip_y + 20:
             # Se pulsos estão muito baixos (abaixo do quadril), provavel caminhada/parado
             return False
        
        # Calcula variação de posição dos braços ao longo do tempo
        wrist_variations = []
        # Analisa uma janela maior
        for i in range(len(history) - 5, len(history)):
             prev_kp = history[i-1]
             curr_kp = history[i]
             
             if prev_kp.left_wrist and curr_kp.left_wrist:
                dx = abs(prev_kp.left_wrist[0] - curr_kp.left_wrist[0])
                dy = abs(prev_kp.left_wrist[1] - curr_kp.left_wrist[1])
                wrist_variations.append(dx + dy)
             if prev_kp.right_wrist and curr_kp.right_wrist:
                dx = abs(prev_kp.right_wrist[0] - curr_kp.right_wrist[0])
                dy = abs(prev_kp.right_wrist[1] - curr_kp.right_wrist[1])
                wrist_variations.append(dx + dy)
        
        # Se há movimento constante dos braços = possível dança
        if wrist_variations:
            avg_variation = np.mean(wrist_variations)
            # Threshold aumentado (era 15) para evitar detectar gesticulação normal como dança
            if 30 < avg_variation < 120:
                # Verifica se torso também se move, mas não corre
                velocity = self._get_avg_velocity(person_id)
                # Velocidade mínima aumentada (era 5) para evitar dança parada
                if 10 < velocity < 60:
                    return True
        
        return False
    
    def _calculate_angle(
        self, 
        p1: Tuple[float, float], 
        p2: Tuple[float, float], 
        p3: Tuple[float, float]
    ) -> float:
        """Calcula ângulo entre três pontos (p2 é o vértice)."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        return np.degrees(angle)
    
    def reset(self):
        """Reseta estado do detector."""
        self.person_counter = 0
        self.position_history.clear()
        self.pose_history.clear()
