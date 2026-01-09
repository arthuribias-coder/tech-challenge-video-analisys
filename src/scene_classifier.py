"""
Tech Challenge - Fase 4: Classificador de Cenas
Usa YOLOv8-cls para identificar o contexto do ambiente (ex: escritório, cozinha, rua).
Isso permite validação contextual de objetos e atividades.
"""

import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

from ultralytics import YOLO
from .config import get_device, SCENE_CONTEXT_RULES, DEBUG_LOGGING

logger = logging.getLogger(__name__)

@dataclass
class SceneContext:
    """Representa o contexto de cena detectado."""
    scene_class: str          # Classe raw do ImageNet (ex: "desk")
    scene_type: str           # Categoria mapeada (ex: "office")
    confidence: float
    is_indoor: bool
    top_probs: List[Tuple[str, float]] = field(default_factory=list)

class SceneClassifier:
    """Classificador de cenas usando YOLO11-cls."""
    
    def __init__(self, model_size: str = "n", conf_threshold: float = 0.2):
        """
        Inicializa o classificador de cenas.
        
        Args:
            model_size: Tamanho do modelo ('n', 's', 'm', 'l')
        """
        self.device = get_device()
        self.conf_threshold = conf_threshold
        
        # Carrega modelo
        model_path = f"yolo11{model_size}-cls.pt"
        logger.info(f"Carregando SceneClassifier: {model_path} ({self.device})")
        self.model = YOLO(model_path)
        
        # Cache de contexto (para não rodar a cada frame, pois cena não muda rápido)
        self.last_context: Optional[SceneContext] = None
        self.last_update_time = 0
        self.update_interval = 2.0  # Segundos entre atualizações de cena
        
    def classify(self, frame: np.ndarray, force_update: bool = False) -> SceneContext:
        """
        Classifica o quadro atual para determinar o contexto.
        
        Args:
            frame: Frame BGR
            force_update: Força reclassificação ignorando cache
            
        Returns:
            SceneContext com informações da cena
        """
        now = time.time()
        if not force_update and self.last_context and (now - self.last_update_time < self.update_interval):
            return self.last_context
            
        # Inferência
        results = self.model(frame, verbose=False, device=self.device)
        
        if not results:
            return self._get_unknown_context()
            
        r = results[0]
        
        # Acessa probabilidades (YOLO11 API)
        # r.probs.top1 -> index
        # r.probs.top1conf -> confidence
        # r.names -> dict
        
        top1_idx = r.probs.top1
        top1_conf = float(r.probs.top1conf)
        top1_name = r.names[top1_idx]
        
        # Pega top 5 para debug
        top5_indices = r.probs.top5
        top_probs = []
        for i in top5_indices:
            name = r.names[i]
            conf = float(r.probs.data[i])
            top_probs.append((name, conf))
            
        if DEBUG_LOGGING:
                logger.debug(f"Cena top5: {top_probs}")
        # Mapeia para categoria de cena (heurstica simples baseada em keywords)
        scene_type = "unknown"
        is_indoor = False # Default
        
        # Tenta casar com regras definidas em config
        matched_category = self._match_scene_category(top1_name, top_probs)
        
        if matched_category:
            scene_type = matched_category
            # Assumimos indoor exceto se for explicitamente outdoor
            is_indoor = (scene_type != "outdoor")
        else:
            # Fallback genérico: se tem "room", "hall", "shop" no nome -> indoor
            indoor_keywords = ["room", "hall", "shop", "store", "office", "home", "house", "bar", "restaurant"]
            if any(k in top1_name.lower() for k in indoor_keywords):
                is_indoor = True
                scene_type = "generic_indoor"
            
        context = SceneContext(
            scene_class=top1_name,
            scene_type=scene_type,
            confidence=top1_conf,
            is_indoor=is_indoor,
            top_probs=top_probs
        )
        
        self.last_context = context
        self.last_update_time = now
        
        return context
        
    def _match_scene_category(self, top_class: str, top_probs: List[Tuple[str, float]]) -> Optional[str]:
        """Tenta encontrar uma categoria de cena compatível."""
        
        # Verifica a classe top 1
        for category, rules in SCENE_CONTEXT_RULES.items():
            keywords = rules.get("keywords", [])
            if any(k in top_class.lower() for k in keywords):
                return category
                
        # Se não casou a top 1, verifica se alguma das top 3 tem match forte
        # Isso ajuda se a top 1 for ambígua (ex: "spotlight" pode ser palco ou estúdio)
        for i in range(1, min(len(top_probs), 3)):
            cls_name, conf = top_probs[i]
            if conf < 0.1: continue
            
            for category, rules in SCENE_CONTEXT_RULES.items():
                keywords = rules.get("keywords", [])
                if any(k in cls_name.lower() for k in keywords):
                    return category
                    
        return None

    def _get_unknown_context(self) -> SceneContext:
        """Retorna contexto padrão desconhecido."""
        return SceneContext(
            scene_class="unknown",
            scene_type="unknown",
            confidence=0.0,
            is_indoor=False
        )
