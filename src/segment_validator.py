"""
Tech Challenge - Fase 4: Validador de Segmentação
Módulo responsável pela validação de silhuetas humanas usando YOLO11-seg.
Detecta se pessoas identificadas têm forma humana realista.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import get_device, YOLO_MODEL_SIZE


@dataclass
class SegmentValidation:
    """Resultado da validação de segmentação."""
    person_id: int
    bbox: Tuple[int, int, int, int]
    mask: Optional[np.ndarray]
    is_valid_human: bool
    confidence: float
    aspect_ratio: float
    fill_ratio: float  # % da bbox preenchida pela máscara
    contour_complexity: float
    is_anomalous: bool = False
    anomaly_reason: Optional[str] = None


# Parâmetros típicos de silhueta humana
HUMAN_ASPECT_RATIO_MIN = 0.25  # Muito largo = provavelmente não é pessoa
HUMAN_ASPECT_RATIO_MAX = 0.9   # Muito alto e fino = provavelmente não é pessoa
HUMAN_FILL_RATIO_MIN = 0.3    # Muito vazio = máscara fragmentada
HUMAN_FILL_RATIO_MAX = 0.95   # Muito cheio = provavelmente bbox retangular, não silhueta
HUMAN_CONTOUR_COMPLEXITY_MIN = 3  # Muito simples = forma geométrica
HUMAN_CONTOUR_COMPLEXITY_MAX = 50 # Muito complexo = ruído


class SegmentValidator:
    """
    Validador de silhuetas humanas usando YOLO11-seg.
    Verifica se detecções de pessoas têm formato realista.
    """
    
    def __init__(
        self,
        model_size: str = None,
        min_confidence: float = 0.4,
        validate_pose_consistency: bool = True
    ):
        """
        Args:
            model_size: Tamanho do modelo ('n', 's', 'm'). None usa config.
            min_confidence: Confiança mínima para detecção
            validate_pose_consistency: Se True, valida consistência com pose YOLO
        """
        self.min_confidence = min_confidence
        self.validate_pose_consistency = validate_pose_consistency
        self.person_counter = 0
        self.device = get_device()
        
        # Cache de validações anteriores para tracking
        self.validation_cache: Dict[int, SegmentValidation] = {}
        
        self._init_yolo(model_size or YOLO_MODEL_SIZE)
    
    def _init_yolo(self, model_size: str):
        """Inicializa YOLO11-seg."""
        try:
            from ultralytics import YOLO
            model_name = f"yolo11{model_size}-seg.pt"
            self.model = YOLO(model_name)
            self.model.to(self.device)
            self.model_loaded = True
            print(f"[INFO] SegmentValidator carregado: {model_name} (device: {self.device})")
            print(f"[INFO] SegmentValidator carregado: {model_name}")
        except Exception as e:
            print(f"[AVISO] Falha ao carregar SegmentValidator: {e}")
            self.model = None
            self.model_loaded = False
    
    def validate(
        self,
        frame: np.ndarray,
        pose_detections: Optional[List] = None,
        frame_number: int = 0
    ) -> List[SegmentValidation]:
        """
        Valida silhuetas humanas no frame.
        
        Args:
            frame: Imagem BGR do OpenCV
            pose_detections: Detecções de pose do ActivityDetector (para cross-validation)
            frame_number: Número do frame atual
            
        Returns:
            Lista de validações de segmentação
        """
        if not self.model_loaded:
            return []
        
        h_frame, w_frame = frame.shape[:2]
        
        # Executa segmentação (apenas classe person = 0)
        results = self.model(
            frame,
            verbose=False,
            conf=self.min_confidence,
            classes=[0]  # Apenas pessoas
        )
        
        validations = []
        
        for result in results:
            if result.boxes is None or result.masks is None:
                continue
            
            boxes = result.boxes
            masks = result.masks
            
            for i in range(len(boxes)):
                confidence = float(boxes.conf[i].item())
                
                # Bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = (
                    int(xyxy[0]), int(xyxy[1]),
                    int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                )
                
                # Máscara de segmentação
                mask = masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (w_frame, h_frame))
                mask = (mask > 0.5).astype(np.uint8)
                
                # Atribui ID
                person_id = self._assign_person_id(bbox)
                
                # Calcula métricas de silhueta
                metrics = self._analyze_silhouette(bbox, mask)
                
                # Valida se é humano realista
                is_valid, reason = self._validate_human_shape(metrics, bbox, h_frame, w_frame)
                
                # Cross-validation com pose (se disponível)
                if self.validate_pose_consistency and pose_detections:
                    pose_match = self._match_with_pose(bbox, pose_detections)
                    if not pose_match and is_valid:
                        # Segmentação OK mas sem pose correspondente
                        is_valid = True  # Mantém válido, apenas nota
                    elif pose_match and not is_valid:
                        # Pose OK mas segmentação ruim - possível oclusão
                        reason = f"{reason} (pose detectada, possível oclusão)"
                
                validation = SegmentValidation(
                    person_id=person_id,
                    bbox=bbox,
                    mask=mask,
                    is_valid_human=is_valid,
                    confidence=confidence,
                    aspect_ratio=metrics["aspect_ratio"],
                    fill_ratio=metrics["fill_ratio"],
                    contour_complexity=metrics["contour_complexity"],
                    is_anomalous=not is_valid,
                    anomaly_reason=reason if not is_valid else None
                )
                
                validations.append(validation)
                self.validation_cache[person_id] = validation
        
        return validations
    
    def _assign_person_id(self, bbox: Tuple[int, int, int, int]) -> int:
        """Atribui ID baseado em proximidade com detecções anteriores."""
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        
        # Busca pessoa próxima no cache
        min_dist = float('inf')
        best_id = None
        
        for pid, val in self.validation_cache.items():
            prev_cx = val.bbox[0] + val.bbox[2] // 2
            prev_cy = val.bbox[1] + val.bbox[3] // 2
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            
            if dist < min_dist and dist < 100:
                min_dist = dist
                best_id = pid
        
        if best_id is None:
            self.person_counter += 1
            best_id = self.person_counter
        
        return best_id
    
    def _analyze_silhouette(
        self,
        bbox: Tuple[int, int, int, int],
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Analisa características da silhueta.
        
        Returns:
            Dict com métricas da silhueta
        """
        x, y, w, h = bbox
        
        # Extrai região da máscara
        mask_region = mask[y:y+h, x:x+w]
        
        if mask_region.size == 0:
            return {
                "aspect_ratio": 0,
                "fill_ratio": 0,
                "contour_complexity": 0
            }
        
        # Aspect ratio
        aspect_ratio = w / max(h, 1)
        
        # Fill ratio (% do bbox preenchido)
        fill_ratio = np.sum(mask_region) / max(mask_region.size, 1)
        
        # Contour complexity (número de pontos no contorno)
        contours, _ = cv2.findContours(
            mask_region, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Aproxima contorno para simplificar
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            contour_complexity = len(approx)
        else:
            contour_complexity = 0
        
        return {
            "aspect_ratio": aspect_ratio,
            "fill_ratio": fill_ratio,
            "contour_complexity": contour_complexity
        }
    
    def _validate_human_shape(
        self,
        metrics: Dict[str, float],
        bbox: Tuple[int, int, int, int],
        frame_height: int,
        frame_width: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Valida se a silhueta tem forma humana.
        
        Returns:
            Tuple (is_valid, reason_if_invalid)
        """
        aspect_ratio = metrics["aspect_ratio"]
        fill_ratio = metrics["fill_ratio"]
        contour_complexity = metrics["contour_complexity"]
        
        reasons = []
        
        # Verifica aspect ratio
        if aspect_ratio < HUMAN_ASPECT_RATIO_MIN:
            reasons.append(f"muito largo (ratio={aspect_ratio:.2f})")
        elif aspect_ratio > HUMAN_ASPECT_RATIO_MAX:
            reasons.append(f"muito estreito (ratio={aspect_ratio:.2f})")
        
        # Verifica fill ratio
        if fill_ratio < HUMAN_FILL_RATIO_MIN:
            reasons.append(f"máscara fragmentada (fill={fill_ratio:.2f})")
        elif fill_ratio > HUMAN_FILL_RATIO_MAX:
            reasons.append(f"forma muito sólida/retangular (fill={fill_ratio:.2f})")
        
        # Verifica complexidade do contorno
        if contour_complexity < HUMAN_CONTOUR_COMPLEXITY_MIN and contour_complexity > 0:
            reasons.append(f"forma muito simples (pontos={contour_complexity})")
        elif contour_complexity > HUMAN_CONTOUR_COMPLEXITY_MAX:
            reasons.append(f"contorno muito complexo/ruidoso (pontos={contour_complexity})")
        
        # Verifica tamanho relativo ao frame
        x, y, w, h = bbox
        bbox_area = w * h
        frame_area = frame_height * frame_width
        area_ratio = bbox_area / frame_area
        
        if area_ratio > 0.8:
            reasons.append(f"ocupa muito do frame ({area_ratio*100:.0f}%)")
        elif area_ratio < 0.001:
            reasons.append(f"muito pequeno ({area_ratio*100:.3f}%)")
        
        if reasons:
            return False, "Silhueta não-humana: " + "; ".join(reasons)
        
        return True, None
    
    def _match_with_pose(
        self,
        seg_bbox: Tuple[int, int, int, int],
        pose_detections: List
    ) -> bool:
        """
        Verifica se há uma detecção de pose correspondente.
        Usa IoU para matching.
        """
        sx, sy, sw, sh = seg_bbox
        seg_area = sw * sh
        
        for pose in pose_detections:
            px, py, pw, ph = pose.bbox
            
            # Calcula IoU
            ix = max(sx, px)
            iy = max(sy, py)
            iw = min(sx + sw, px + pw) - ix
            ih = min(sy + sh, py + ph) - iy
            
            if iw > 0 and ih > 0:
                intersection = iw * ih
                pose_area = pw * ph
                union = seg_area + pose_area - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > 0.5:
                    return True
        
        return False
    
    def get_anomaly_results(self, validations: List[SegmentValidation]) -> List[Dict]:
        """
        Converte validações em formato para o AnomalyDetector.
        
        Returns:
            Lista de dicts com anomalias para process_segment_validation()
        """
        results = []
        
        for val in validations:
            if val.is_anomalous:
                results.append({
                    "person_id": val.person_id,
                    "bbox": val.bbox,
                    "is_anomalous": True,
                    "severity": 0.5 + (1 - val.fill_ratio) * 0.3,  # Maior severidade para máscaras piores
                    "reason": val.anomaly_reason,
                    "metrics": {
                        "aspect_ratio": val.aspect_ratio,
                        "fill_ratio": val.fill_ratio,
                        "contour_complexity": val.contour_complexity
                    }
                })
        
        return results
    
    def reset(self):
        """Reseta estado do validador."""
        self.person_counter = 0
        self.validation_cache.clear()
