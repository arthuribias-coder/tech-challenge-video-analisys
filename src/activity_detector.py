"""
Tech Challenge - Fase 4: Detector de Atividades
Usa YOLOv8-pose para detecção de pessoas e análise de poses/atividades.
"""

import cv2
import numpy as np

# Importa torch após numpy para evitar bug de compatibilidade
import torch

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from enum import Enum

from .config import ACTIVITY_CATEGORIES, get_device, YOLO_MODEL_SIZE


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
        history_size: int = 10
    ):
        """
        Args:
            model_size: Tamanho do modelo ('n', 's', 'm', 'l', 'x'). None usa config.
            min_confidence: Confiança mínima para detecção
            history_size: Frames de histórico para análise temporal
        """
        self.min_confidence = min_confidence
        self.history_size = history_size
        self.person_counter = 0
        self.position_history: Dict[int, deque] = {}
        self.pose_history: Dict[int, deque] = {}
        self.device = get_device()
        
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
            print(f"[INFO] Modelo carregado: {model_name} (device: {self.device})")
        except Exception as e:
            print(f"[AVISO] YOLO11 não disponível, tentando YOLOv8: {e}")
            try:
                from ultralytics import YOLO
                model_name = f"yolov8{model_size}-pose.pt"
                self.model = YOLO(model_name)
                self.model.to(self.device)
                self.model_loaded = True
            except Exception as e2:
                print(f"[ERRO] Falha ao carregar modelo: {e2}")
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
        
        results = self.model(frame, verbose=False, conf=self.min_confidence)
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
                
                # Se ambos têm pulso direito detectado
                if p1.keypoints.right_wrist and p2.keypoints.right_wrist:
                    rw1 = p1.keypoints.right_wrist
                    rw2 = p2.keypoints.right_wrist
                    
                    # Distância entre pulsos
                    dist = np.sqrt((rw1[0] - rw2[0])**2 + (rw1[1] - rw2[1])**2)
                    
                    # Se pulsos estão muito próximos (< 50px) e pessoas próximas
                    # Verifica também a distância dos ombros para garantir que são pessoas diferentes próximas
                    s1 = p1.keypoints.right_shoulder or p1.keypoints.left_shoulder
                    s2 = p2.keypoints.right_shoulder or p2.keypoints.left_shoulder
                    
                    if s1 and s2:
                        shoulder_dist = np.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)
                        
                        # Pulsos unidos (<100px) mas corpos separados (>100px)
                        if dist < 100 and shoulder_dist > 100:
                            # Classifica ambos como cumprimento
                            p1.activity = ActivityType.GREETING
                            p1.activity_pt = ACTIVITY_CATEGORIES.get("greeting", "Cumprimentando")
                            p2.activity = ActivityType.GREETING
                            p2.activity_pt = ACTIVITY_CATEGORIES.get("greeting", "Cumprimentando")
                            
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
        
        # 1. Verifica postura deitada primeiro (alta prioridade)
        if self._is_lying(keypoints):
            return ActivityType.LYING, 0.85
        
        # 2. Posturas estáticas verificadas ANTES de gestos (prioridade)
        # Sentado/Agachado são mais comuns que gestos específicos
        if self._is_crouching(keypoints):
            return ActivityType.CROUCHING, 0.75
        
        if self._is_sitting(keypoints, velocity):
            return ActivityType.SITTING, 0.75
        
        # 3. Verifica braços levantados (ambos acima da cabeça)
        if self._is_arms_raised(keypoints):
            return ActivityType.ARMS_RAISED, 0.85
        
        # 4. Gestos específicos de mãos (MAS NÃO se está parado)
        # Evita confundir selfie/braço estendido com waving
        if velocity > 5:  # Só considera gestos se houver movimento mínimo
            if self._is_waving(keypoints):
                return ActivityType.WAVING, 0.8
            
            if self._is_pointing(keypoints):
                return ActivityType.POINTING, 0.75
        
        # 5. Dança (movimento rítmico)
        if self._is_dancing(person_id, keypoints):
            return ActivityType.DANCING, 0.8
        
        # 6. Verifica movimento geral pela velocidade
        if velocity > 80:
            return ActivityType.RUNNING, 0.8
        elif velocity > 25:
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
        
        for wrist, elbow, shoulder in [
            (kp.left_wrist, kp.left_elbow, kp.left_shoulder),
            (kp.right_wrist, kp.right_elbow, kp.right_shoulder)
        ]:
            if all([wrist, elbow, shoulder]):
                # Mão SIGNIFICATIVAMENTE acima do ombro (critério mais rigoroso)
                if wrist[1] < shoulder[1] - 40:
                    # Aceita ângulo do cotovelo entre 40 e 160 graus
                    elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
                    if 40 < elbow_angle < 160:
                        return True
        return False
    
    def _is_pointing(self, kp: PoseKeypoints) -> bool:
        """Detecta gesto de apontar (braço estendido horizontalmente)."""
        for wrist, elbow, shoulder in [
            (kp.left_wrist, kp.left_elbow, kp.left_shoulder),
            (kp.right_wrist, kp.right_elbow, kp.right_shoulder)
        ]:
            if all([wrist, elbow, shoulder]):
                # Braço estendido (ângulo > 150)
                arm_angle = self._calculate_angle(shoulder, elbow, wrist)
                if arm_angle > 150:
                    # Horizontalmente (variação vertical pequena)
                    arm_height_diff = abs(wrist[1] - shoulder[1])
                    arm_length = abs(wrist[0] - shoulder[0])
                    if arm_length > 80 and arm_height_diff < 60:
                        return True
        return False
    
    def _is_sitting(self, kp: PoseKeypoints, velocity: float = 0.0) -> bool:
        """Detecta postura sentada (análise melhorada de ângulos e oclusão)."""
        
        # Se está muito parado (velocidade baixa), prioriza sitting sobre standing
        # Cobre casos de selfie, trabalhando no computador, etc.
        if velocity < 10:
            # Verifica se tem torso visível (ombros + quadril)
            if kp.left_shoulder and kp.right_shoulder:
                # Caso 1: Tem quadril visível
                if kp.left_hip or kp.right_hip:
                    return True
                # Caso 2: Não tem quadril (ocluído por mesa/câmera) MAS está parado
                # Provavelmente sentado trabalhando/em videochamada
                else:
                    return True
        
        # 1. Fallback para "Sentado em mesa" (joelhos ocultos)
        # Se não vemos joelhos, mas vemos tronco superior + estático
        if not (kp.left_knee or kp.right_knee):
             # Precisa ter ombros e quadril visíveis (torso)
             if kp.left_shoulder and kp.right_shoulder and kp.left_hip and kp.right_hip:
                 # Se estamos vendo quadril mas não joelhos, e a pessoa está parada 
                 # (velocidade < 15), assumimos sentado (oclusão por mesa)
                 if velocity < 15:
                     return True
             
             # AJUSTE: Se nem sequer vemos o quadril (mesa alta/close-up), mas vemos ombros e rosto
             # e está muito estático, provavelmente está sentado trabalhando/lendo.
             # Evita classificar como "Standing" (padrão)
             elif kp.left_shoulder and kp.right_shoulder:
                 if velocity < 5: # Muito parado
                     return True
                     
        # 2. Precisa de quadril e joelho para análise geométrica padrão
        if not all([kp.left_hip, kp.right_hip, kp.left_knee, kp.right_knee]):
            return False
        
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        knee_y = (kp.left_knee[1] + kp.right_knee[1]) / 2
        
        # Se joelhos estão aproximadamente na mesma altura que quadril = sentado
        hip_knee_diff = abs(hip_y - knee_y)
        
        # Verifica também ângulo do tronco se tiver ombros
        if kp.left_shoulder and kp.right_shoulder:
            shoulder_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
            torso_length = abs(hip_y - shoulder_y)
            
            # Se diferença quadril-joelho é menor que 40% do tronco = sentado
            if hip_knee_diff < torso_length * 0.5:
                return True
        
        # Fallback: diferença absoluta pequena
        if hip_knee_diff < 80:
            return True
        
        return False
    
    def _is_crouching(self, kp: PoseKeypoints) -> bool:
        """Detecta postura agachada (quadril baixo, joelhos dobrados)."""
        if not all([kp.left_hip, kp.right_hip, kp.left_knee, kp.right_knee]):
            return False
        
        hip_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        knee_y = (kp.left_knee[1] + kp.right_knee[1]) / 2
        
        # Tornozelos disponíveis para melhor análise
        if kp.left_ankle and kp.right_ankle:
            ankle_y = (kp.left_ankle[1] + kp.right_ankle[1]) / 2
            
            # Agachado: quadril próximo aos joelhos, joelhos acima dos tornozelos
            if hip_y > knee_y - 30 and knee_y < ankle_y:
                return True
        
        return False
    
    def _is_lying(self, kp: PoseKeypoints) -> bool:
        """
        Detecta postura deitada com critérios rigorosos.
        Requer múltiplas evidências para evitar falsos positivos.
        """
        # Precisa de ombros E quadril para análise confiável
        if not (kp.left_shoulder and kp.right_shoulder):
            return False
        if not (kp.left_hip and kp.right_hip):
            return False
        
        # Calcula centros
        shoulder_center_y = (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2
        shoulder_center_x = (kp.left_shoulder[0] + kp.right_shoulder[0]) / 2
        hip_center_y = (kp.left_hip[1] + kp.right_hip[1]) / 2
        hip_center_x = (kp.left_hip[0] + kp.right_hip[0]) / 2
        
        # Diferença vertical e horizontal entre ombros e quadril
        vertical_diff = abs(shoulder_center_y - hip_center_y)
        horizontal_diff = abs(shoulder_center_x - hip_center_x)
        
        # Distância entre ombros (largura visível do corpo)
        shoulder_width = abs(kp.left_shoulder[0] - kp.right_shoulder[0])
        
        # CRITÉRIO PRINCIPAL: Corpo claramente mais horizontal que vertical
        # Exige que horizontal seja pelo menos 2x maior que vertical
        if horizontal_diff > vertical_diff * 2.0 and horizontal_diff > 100:
            return True
        
        # CRITÉRIO SECUNDÁRIO: Pessoa de lado (ombros muito próximos em X)
        # Mas APENAS se o torso também está horizontal
        if shoulder_width < 40:
            # Verifica se quadril também está alinhado horizontalmente
            hip_width = abs(kp.left_hip[0] - kp.right_hip[0])
            if hip_width < 40 and horizontal_diff > vertical_diff * 1.5:
                return True
        
        # CRITÉRIO COM TORNOZELOS: Verificação completa do corpo
        if kp.left_ankle and kp.right_ankle:
            ankle_center_y = (kp.left_ankle[1] + kp.right_ankle[1]) / 2
            ankle_center_x = (kp.left_ankle[0] + kp.right_ankle[0]) / 2
            
            # Distância total ombro->tornozelo
            total_vertical = abs(shoulder_center_y - ankle_center_y)
            total_horizontal = abs(shoulder_center_x - ankle_center_x)
            
            # Corpo inteiro claramente horizontal (2x mais largo que alto)
            if total_horizontal > total_vertical * 2.0 and total_horizontal > 150:
                return True
        
        return False
    
    def _is_arms_raised(self, kp: PoseKeypoints) -> bool:
        """Detecta ambos os braços levantados acima da cabeça."""
        if not all([kp.left_wrist, kp.right_wrist, kp.nose]):
            return False
        
        # Ambos pulsos acima do nariz
        if kp.left_wrist[1] < kp.nose[1] and kp.right_wrist[1] < kp.nose[1]:
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
