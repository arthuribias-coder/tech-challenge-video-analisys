"""
Tech Challenge - Fase 4: Detector de Rostos
Módulo responsável pelo reconhecimento e rastreamento de rostos no vídeo.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class FaceDetection:
    """Representa uma detecção de rosto em um frame."""
    face_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[Dict] = None
    embedding: Optional[np.ndarray] = None


class FaceDetector:
    """
    Detector de rostos usando OpenCV DNN ou Haar Cascades.
    Suporta rastreamento básico de identidades entre frames.
    """
    
    def __init__(self, method: str = "haar", confidence_threshold: float = 0.5):
        """
        Inicializa o detector de rostos.
        
        Args:
            method: Método de detecção ('haar', 'dnn', 'mediapipe')
            confidence_threshold: Limiar mínimo de confiança
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.face_counter = 0
        self.tracked_faces: Dict[int, np.ndarray] = {}
        
        self._init_detector()
    
    def _init_detector(self):
        """Inicializa o detector baseado no método escolhido."""
        if self.method == "haar":
            # Haar Cascade - usa caminho explícito para evitar problemas de cache
            import os
            from pathlib import Path
            
            # Tenta múltiplos caminhos possíveis
            possible_paths = [
                # Caminho do cv2 atual
                str(Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"),
                # Caminho do venv Python 3.12
                "/home/aineto/workspaces/POS/TC-4/.venv/lib/python3.12/site-packages/cv2/data/haarcascade_frontalface_default.xml",
                # Caminho do FER
                "/home/aineto/workspaces/POS/TC-4/.venv/lib/python3.12/site-packages/fer/data/haarcascade_frontalface_default.xml",
                # Caminhos do sistema
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.detector = cv2.CascadeClassifier(path)
                    if not self.detector.empty():
                        break
            
            if self.detector.empty():
                print("[AVISO] Haar Cascade não encontrado, usando método simplificado")
        elif self.method == "dnn":
            # DNN (mais preciso, requer modelo)
            self._init_dnn_detector()
        elif self.method == "mediapipe":
            self._init_mediapipe_detector()
        else:
            raise ValueError(f"Método desconhecido: {self.method}")
    
    def _init_dnn_detector(self):
        """Inicializa detector DNN do OpenCV."""
        # Usa modelo Caffe pré-treinado do OpenCV
        model_path = cv2.data.haarcascades.replace(
            "haarcascades/", ""
        ) + "deploy.prototxt"
        weights_path = cv2.data.haarcascades.replace(
            "haarcascades/", ""
        ) + "res10_300x300_ssd_iter_140000.caffemodel"
        
        try:
            self.detector = cv2.dnn.readNetFromCaffe(model_path, weights_path)
        except Exception:
            print("[AVISO] Modelo DNN não encontrado, usando Haar Cascade")
            self.method = "haar"
            self._init_detector()
    
    def _init_mediapipe_detector(self):
        """Inicializa detector MediaPipe Face Detection."""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 para detecção de longa distância
                min_detection_confidence=self.confidence_threshold
            )
        except ImportError:
            print("[AVISO] MediaPipe não instalado, usando Haar Cascade")
            self.method = "haar"
            self._init_detector()
    
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detecta rostos em um frame.
        
        Args:
            frame: Imagem BGR do OpenCV
            
        Returns:
            Lista de detecções de rostos
        """
        if self.method == "haar":
            return self._detect_haar(frame)
        elif self.method == "dnn":
            return self._detect_dnn(frame)
        elif self.method == "mediapipe":
            return self._detect_mediapipe(frame)
        return []
    
    def _detect_haar(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detecção usando Haar Cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            face_id = self._assign_face_id(frame, (x, y, w, h))
            detections.append(FaceDetection(
                face_id=face_id,
                bbox=(x, y, w, h),
                confidence=1.0  # Haar não fornece confiança
            ))
        
        return detections
    
    def _detect_dnn(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detecção usando DNN do OpenCV."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.detector.setInput(blob)
        detections_raw = self.detector.forward()
        
        detections = []
        for i in range(detections_raw.shape[2]):
            confidence = detections_raw[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                face_id = self._assign_face_id(frame, (x1, y1, x2-x1, y2-y1))
                detections.append(FaceDetection(
                    face_id=face_id,
                    bbox=(x1, y1, x2-x1, y2-y1),
                    confidence=float(confidence)
                ))
        
        return detections
    
    def _detect_mediapipe(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detecção usando MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        detections = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                face_id = self._assign_face_id(frame, (x, y, width, height))
                detections.append(FaceDetection(
                    face_id=face_id,
                    bbox=(x, y, width, height),
                    confidence=detection.score[0]
                ))
        
        return detections
    
    def _assign_face_id(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """
        Atribui um ID ao rosto baseado em rastreamento simples por posição.
        Para rastreamento mais robusto, usar embeddings faciais.
        """
        x, y, w, h = bbox
        center = np.array([x + w/2, y + h/2])
        
        # Procura rosto mais próximo já rastreado
        min_dist = float('inf')
        matched_id = None
        
        for face_id, prev_center in self.tracked_faces.items():
            dist = np.linalg.norm(center - prev_center)
            if dist < min_dist and dist < max(w, h) * 2:  # Threshold de proximidade
                min_dist = dist
                matched_id = face_id
        
        if matched_id is not None:
            self.tracked_faces[matched_id] = center
            return matched_id
        
        # Novo rosto detectado
        self.face_counter += 1
        self.tracked_faces[self.face_counter] = center
        return self.face_counter
    
    def reset_tracking(self):
        """Reseta o rastreamento de rostos."""
        self.face_counter = 0
        self.tracked_faces.clear()
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[FaceDetection],
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Desenha as detecções de rostos no frame.
        
        Args:
            frame: Imagem BGR
            detections: Lista de detecções
            color: Cor das bounding boxes (BGR)
            
        Returns:
            Frame com anotações
        """
        annotated = frame.copy()
        
        for det in detections:
            x, y, w, h = det.bbox
            
            # Bounding box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Label com ID e confiança
            label = f"Face #{det.face_id}"
            if det.confidence < 1.0:
                label += f" ({det.confidence:.0%})"
            
            # Fundo do texto
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                annotated, 
                (x, y - text_h - 10), 
                (x + text_w + 4, y), 
                color, 
                -1
            )
            
            # Texto
            cv2.putText(
                annotated, label,
                (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA
            )
        
        return annotated
