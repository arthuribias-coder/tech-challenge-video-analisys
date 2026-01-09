"""
Tech Challenge - Fase 4: Detector de Objetos
Módulo responsável pela detecção de objetos usando YOLO11.
Detecta objetos que podem indicar anomalias visuais ou contextuais.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .config import get_device, YOLO_MODEL_SIZE

logger = logging.getLogger(__name__)


class ObjectCategory(Enum):
    """Categorias de objetos relevantes para análise."""
    ELECTRONIC = "electronic"      # TV, laptop, celular
    FURNITURE = "furniture"        # Cadeira, mesa, sofá
    VEHICLE = "vehicle"            # Carro, moto, bicicleta
    ACCESSORY = "accessory"        # Bolsa, mochila, guarda-chuva
    SPORTS = "sports"              # Bola, raquete
    ANIMAL = "animal"              # Cachorro, gato
    FOOD = "food"                  # Comida, garrafa
    OTHER = "other"


@dataclass
class ObjectDetection:
    """Representa uma detecção de objeto."""
    object_id: int
    class_name: str
    class_id: int
    category: ObjectCategory
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    is_anomalous: bool = False
    anomaly_reason: Optional[str] = None


# Mapeamento de classes COCO para categorias
COCO_CATEGORIES = {
    # Eletrônicos (úteis para detectar overlays/telas)
    62: ("tv", ObjectCategory.ELECTRONIC),
    63: ("laptop", ObjectCategory.ELECTRONIC),
    67: ("cell phone", ObjectCategory.ELECTRONIC),
    72: ("refrigerator", ObjectCategory.ELECTRONIC),
    
    # Móveis (contexto de ambiente)
    56: ("chair", ObjectCategory.FURNITURE),
    57: ("couch", ObjectCategory.FURNITURE),
    59: ("bed", ObjectCategory.FURNITURE),
    60: ("dining table", ObjectCategory.FURNITURE),
    
    # Veículos (anomalia se aparecer em ambiente interno)
    2: ("car", ObjectCategory.VEHICLE),
    3: ("motorcycle", ObjectCategory.VEHICLE),
    1: ("bicycle", ObjectCategory.VEHICLE),
    5: ("bus", ObjectCategory.VEHICLE),
    7: ("truck", ObjectCategory.VEHICLE),
    
    # Acessórios
    24: ("backpack", ObjectCategory.ACCESSORY),
    25: ("umbrella", ObjectCategory.ACCESSORY),
    26: ("handbag", ObjectCategory.ACCESSORY),
    27: ("tie", ObjectCategory.ACCESSORY),
    28: ("suitcase", ObjectCategory.ACCESSORY),
    
    # Esportes
    32: ("sports ball", ObjectCategory.SPORTS),
    33: ("kite", ObjectCategory.SPORTS),
    34: ("baseball bat", ObjectCategory.SPORTS),
    38: ("tennis racket", ObjectCategory.SPORTS),
    35: ("baseball glove", ObjectCategory.SPORTS),
    36: ("skateboard", ObjectCategory.SPORTS),
    37: ("surfboard", ObjectCategory.SPORTS),
    
    # Animais
    15: ("bird", ObjectCategory.ANIMAL),
    16: ("cat", ObjectCategory.ANIMAL),
    17: ("dog", ObjectCategory.ANIMAL),
    18: ("horse", ObjectCategory.ANIMAL),
    
    # Livros e itens de leitura (contexto)
    73: ("book", ObjectCategory.OTHER),
    74: ("clock", ObjectCategory.OTHER),
    75: ("vase", ObjectCategory.OTHER),
    76: ("scissors", ObjectCategory.OTHER),
    
    # Comida/bebida
    39: ("bottle", ObjectCategory.FOOD),
    40: ("wine glass", ObjectCategory.FOOD),
    41: ("cup", ObjectCategory.FOOD),
    42: ("fork", ObjectCategory.FOOD),
    43: ("knife", ObjectCategory.FOOD),
    44: ("spoon", ObjectCategory.FOOD),
    45: ("bowl", ObjectCategory.FOOD),
    46: ("banana", ObjectCategory.FOOD),
    47: ("apple", ObjectCategory.FOOD),
    48: ("sandwich", ObjectCategory.FOOD),
    49: ("orange", ObjectCategory.FOOD),
    50: ("broccoli", ObjectCategory.FOOD),
    51: ("carrot", ObjectCategory.FOOD),
    52: ("hot dog", ObjectCategory.FOOD),
    53: ("pizza", ObjectCategory.FOOD),
    54: ("donut", ObjectCategory.FOOD),
    55: ("cake", ObjectCategory.FOOD),
}

# Classes que indicam potenciais anomalias visuais (overlays, edições)
OVERLAY_INDICATOR_CLASSES = {62, 63, 67}  # tv, laptop, cell phone (podem ser overlays)

# Objetos suspeitos para o contexto de análise social/escritório
# A presença destes objetos geralmente indica erro de detecção ou imagens fora de contexto
SUSPICIOUS_CONTEXT_OBJECTS = {
    "umbrella", "kite", "skateboard", "skis", "snowboard", "surfboard", "baseball bat",
    "zebra", "giraffe", "elephant", "bear", "sheep", "cow", "horse", 
    "airplane", "boat", "train", "stop sign", "fire hydrant"
}


class ObjectDetector:
    """
    Detector de objetos usando YOLO11.
    Complementa o ActivityDetector (pose) com detecção de objetos do ambiente.
    """
    
    def __init__(
        self, 
        model_size: str = None,
        min_confidence: float = 0.5,
        classes_of_interest: Optional[Set[int]] = None
    ):
        """
        Args:
            model_size: Tamanho do modelo ('n', 's', 'm', 'l', 'x'). None usa config.
            min_confidence: Confiança mínima para detecção (50% padrão)
            classes_of_interest: Classes COCO específicas para detectar (None = todas mapeadas)
        """
        self.min_confidence = min_confidence
        self.object_counter = 0
        self.tracked_objects: Dict[int, Tuple[int, int]] = {}  # object_id -> (cx, cy)
        self.device = get_device()
        
        # Classes de interesse (default: todas mapeadas)
        self.classes_of_interest = classes_of_interest or set(COCO_CATEGORIES.keys())
        
        # Histórico para análise temporal
        self.object_history: Dict[int, List[str]] = {}  # frame -> lista de classes
        self.history_window = 30  # frames
        
        self._init_yolo(model_size or YOLO_MODEL_SIZE)
    
    def _init_yolo(self, model_size: str):
        """Inicializa YOLO11 para detecção de objetos."""
        try:
            from ultralytics import YOLO
            model_name = f"yolo11{model_size}.pt"
            self.model = YOLO(model_name)
            self.model.to(self.device)
            self.model_loaded = True
            logger.info(f"ObjectDetector carregado: {model_name} (device: {self.device})")
        except Exception as e:
            logger.error(f"Falha ao carregar ObjectDetector: {e}")
            self.model = None
            self.model_loaded = False
    
    def detect(
        self, 
        frame: np.ndarray,
        frame_number: int = 0
    ) -> List[ObjectDetection]:
        """
        Detecta objetos no frame.
        
        Args:
            frame: Imagem BGR do OpenCV
            frame_number: Número do frame atual
            
        Returns:
            Lista de detecções de objetos
        """
        if not self.model_loaded:
            return []
        
        # Executa detecção
        results = self.model(
            frame, 
            verbose=False, 
            conf=self.min_confidence,
            classes=list(self.classes_of_interest)
        )
        
        detections = []
        frame_classes = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                
                # Verifica se é uma classe de interesse
                if class_id not in COCO_CATEGORIES:
                    continue
                
                class_name, category = COCO_CATEGORIES[class_id]
                frame_classes.append(class_name)
                
                # Bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = (
                    int(xyxy[0]), int(xyxy[1]),
                    int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                )
                
                # ID do objeto (tracking simples)
                object_id = self._assign_object_id(bbox)
                
                # Verifica se é potencialmente anômalo
                is_anomalous, reason = self._check_anomaly(
                    class_id, class_name, category, bbox, frame, frame_number
                )
                
                detections.append(ObjectDetection(
                    object_id=object_id,
                    class_name=class_name,
                    class_id=class_id,
                    category=category,
                    confidence=confidence,
                    bbox=bbox,
                    is_anomalous=is_anomalous,
                    anomaly_reason=reason
                ))
        
        # Atualiza histórico
        self.object_history[frame_number] = frame_classes
        self._cleanup_history(frame_number)
        
        return detections
    
    def _assign_object_id(self, bbox: Tuple[int, int, int, int]) -> int:
        """Atribui ID baseado em proximidade."""
        cx = bbox[0] + bbox[2] // 2
        cy = bbox[1] + bbox[3] // 2
        
        # Busca objeto próximo
        min_dist = float('inf')
        best_id = None
        
        for oid, (prev_cx, prev_cy) in self.tracked_objects.items():
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            if dist < min_dist and dist < 100:
                min_dist = dist
                best_id = oid
        
        if best_id is None:
            self.object_counter += 1
            best_id = self.object_counter
        
        self.tracked_objects[best_id] = (cx, cy)
        return best_id
    
    def _check_anomaly(
        self,
        class_id: int,
        class_name: str,
        category: ObjectCategory,
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray,
        frame_number: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifica se o objeto detectado é anômalo no contexto.
        
        Returns:
            Tuple (is_anomalous, reason)
        """
        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = bbox
        
        # 1. Objeto muito grande (pode ser overlay/watermark)
        object_area = w * h
        frame_area = w_frame * h_frame
        area_ratio = object_area / frame_area
        
        if area_ratio > 0.4:
            return True, f"Objeto '{class_name}' ocupa >40% do frame (possível overlay)"
        
        # 1.5. Objeto suspeito para o contexto (provável erro ou imagem inserida)
        if class_name in SUSPICIOUS_CONTEXT_OBJECTS:
            return True, f"Objeto '{class_name}' atípico para o contexto (possível erro/gráfico)"

        # 2. TV/Laptop em posição estranha (canto superior = possível logo/watermark)
        if class_id in OVERLAY_INDICATOR_CLASSES:
            # Se está no canto superior direito ou esquerdo
            if y < h_frame * 0.15 and (x < w_frame * 0.15 or x + w > w_frame * 0.85):
                return True, f"'{class_name}' em posição de overlay/watermark"
        
        # 3. Objeto apareceu subitamente sem contexto prévio
        if frame_number > self.history_window:
            recent_classes = []
            for fn in range(frame_number - self.history_window, frame_number):
                if fn in self.object_history:
                    recent_classes.extend(self.object_history[fn])
            
            # Se objeto nunca apareceu e surge de repente
            if class_name not in recent_classes and len(recent_classes) > 10:
                return True, f"'{class_name}' apareceu subitamente no frame {frame_number}"
        
        # 4. Veículo em ambiente interno (baseado em outros objetos)
        if category == ObjectCategory.VEHICLE:
            # Se há móveis no histórico recente, provavelmente é ambiente interno
            furniture_count = 0
            for fn in range(max(0, frame_number - 10), frame_number + 1):
                if fn in self.object_history:
                    for cls in self.object_history[fn]:
                        if cls in ["chair", "couch", "bed", "dining table"]:
                            furniture_count += 1
            
            if furniture_count > 3:
                return True, f"Veículo '{class_name}' detectado em ambiente interno"
        
        return False, None
    
    def _cleanup_history(self, current_frame: int):
        """Remove histórico antigo."""
        cutoff = current_frame - self.history_window * 2
        frames_to_remove = [f for f in self.object_history if f < cutoff]
        for f in frames_to_remove:
            del self.object_history[f]
    
    def get_context_summary(self, frame_number: int) -> Dict[str, int]:
        """
        Retorna resumo do contexto visual baseado nos objetos detectados.
        
        Returns:
            Dict com contagem de objetos por categoria
        """
        category_counts = {cat.value: 0 for cat in ObjectCategory}
        
        for fn in range(max(0, frame_number - 10), frame_number + 1):
            if fn in self.object_history:
                for class_name in self.object_history[fn]:
                    for class_id, (name, cat) in COCO_CATEGORIES.items():
                        if name == class_name:
                            category_counts[cat.value] += 1
                            break
        
        return category_counts
    
    def reset(self):
        """Reseta estado do detector."""
        self.object_counter = 0
        self.tracked_objects.clear()
        self.object_history.clear()
