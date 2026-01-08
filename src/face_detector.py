"""
Tech Challenge - Fase 4: Detector de Rostos
Módulo responsável pelo reconhecimento e rastreamento de rostos no vídeo.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class FaceDetection:
    """Representa uma detecção de rosto em um frame."""
    face_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[Dict] = None

class FaceDetector:
    """
    Detector de rostos usando Haar Cascades.
    Otimizado para detecção de perfis e verificação de realismo (anti-spoofing básico).
    """
    
    def __init__(self):
        """Inicializa o detector."""
        self.face_counter = 0
        self.tracked_faces: Dict[int, np.ndarray] = {}
        self._init_haar_detector()
    
    def _init_haar_detector(self):
        """Carrega classificadores Haar Cascade."""
        import os
        from pathlib import Path
        
        # Caminhos padrão
        cv2_base = Path(cv2.__file__).parent
        
        paths = [
            str(cv2_base / "data" / "haarcascade_frontalface_default.xml"),
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        ]
        
        self.detector = None
        for p in paths:
            if os.path.exists(p):
                try:
                    self.detector = cv2.CascadeClassifier(p)
                    if not self.detector.empty():
                        break
                except:
                    continue
                    
        if self.detector is None or self.detector.empty():
             # Fallback: tenta carregar do diretório local se existir
             local_path = "models/haarcascade_frontalface_default.xml"
             if os.path.exists(local_path):
                 self.detector = cv2.CascadeClassifier(local_path)

        # Carrega detector de perfil (opcional, mas útil)
        profile_paths = [
            str(cv2_base / "data" / "haarcascade_profileface.xml"),
            cv2.data.haarcascades + "haarcascade_profileface.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml"
        ]
        self.profile_detector = None
        for p in profile_paths:
            if os.path.exists(p):
                try:
                    self.profile_detector = cv2.CascadeClassifier(p)
                    if not self.profile_detector.empty():
                        break
                except:
                    continue

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detecta rostos no frame usando estratégia híbrida."""
        if self.detector is None or self.detector.empty():
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_frame, w_frame = frame.shape[:2]
        
        min_dim = int(min(h_frame, w_frame) * 0.05)
        
        all_faces = []
        
        # 1. Frontal (Padrão)
        faces_frontal = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_dim, min_dim)
        )
        for (x, y, w, h) in faces_frontal:
            all_faces.append((x, y, w, h))
            
        # 2. Perfil (se disponível)
        if self.profile_detector and not self.profile_detector.empty():
            # Esquerda
            faces_left = self.profile_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_dim, min_dim)
            )
            for f in faces_left: all_faces.append(tuple(f))
            
            # Direita (flip)
            gray_flipped = cv2.flip(gray, 1)
            faces_right = self.profile_detector.detectMultiScale(
                gray_flipped, scaleFactor=1.1, minNeighbors=5, minSize=(min_dim, min_dim)
            )
            for (x, y, w, h) in faces_right:
                all_faces.append((w_frame - x - w, y, w, h))
        
        # Remove duplicatas (NMS simplificado)
        final_faces = self._non_max_suppression(all_faces, 0.4)
        
        detections = []
        for (x, y, w, h) in final_faces:
            # VALIDAÇÃO DE REALISMO (Anti-Wireframe/Anti-CGI)
            if not self._is_real_face(frame, (x, y, w, h)):
                continue
                
            face_id = self._assign_face_id(frame, (x, y, w, h))
            detections.append(FaceDetection(
                face_id=face_id,
                bbox=(x, y, w, h),
                confidence=1.0 # Haar não retorna score
            ))
            
        return detections

    def detect_in_regions(self, frame: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[FaceDetection]:
        """
        Detecta em regiões específicas (usado para pessoas deitadas/inclinadas).
        Aplica rotação local na região de interesse.
        """
        if not regions or not self.detector:
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        h_frame, w_frame = frame.shape[:2]
        
        for (rx, ry, rw, rh) in regions:
            rx, ry = max(0, rx), max(0, ry)
            rw = min(rw, w_frame - rx)
            rh = min(rh, h_frame - ry)
            
            if rw < 20 or rh < 20: continue
            
            roi = gray[ry:ry+rh, rx:rx+rw]
            
            # Tenta detectar rotacionando a ROI (90 e -90 graus)
            for angle in [90, -90]:
                try:
                    rotated, M_inv = self._rotate_roi(roi, angle)
                    faces = self.detector.detectMultiScale(
                        rotated, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
                    )
                    
                    for (fx, fy, fw, fh) in faces:
                        # Mapeia de volta para coordenadas originais
                        center_rot = np.array([fx + fw/2, fy + fh/2, 1])
                        center_roi = M_inv @ center_rot
                        
                        size = max(fw, fh)
                        gx = int(center_roi[0] - size/2) + rx
                        gy = int(center_roi[1] - size/2) + ry
                        
                        # Check realismo
                        if self._is_real_face(frame, (gx, gy, size, size)):
                             face_id = self._assign_face_id(frame, (gx, gy, size, size))
                             detections.append(FaceDetection(face_id, (gx, gy, size, size), 0.9))
                except:
                    continue
                    
        return detections

    def _is_real_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Filtra rostos artificiais (wireframes, hologramas, desenhos).
        """
        x, y, w, h = bbox
        
        # Validação geométrica
        if w < 10 or h < 10: return False
        
        # Garante crop seguro
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x+w), min(h_img, y+h)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return False
        
        # 1. Filtro de Cor (Pele Humana - YCrCb)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        # Faixas de pele humana
        min_ycrcb = np.array([0, 133, 77], np.uint8)
        max_ycrcb = np.array([255, 173, 127], np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, min_ycrcb, max_ycrcb)
        skin_ratio = cv2.countNonZero(skin_mask) / (roi.shape[0] * roi.shape[1])
        
        # 2. Filtro de "Azul Artificial" (Wireframe)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Azul/Ciano (H: 80-130)
        blue_mask = cv2.inRange(hsv, np.array([80, 50, 50]), np.array([130, 255, 255]))
        blue_ratio = cv2.countNonZero(blue_mask) / (roi.shape[0] * roi.shape[1])
        
        # REGRAS DE REJEIÇÃO:
        # A) Muito azul (>20%) = Wireframe/Holograma
        if blue_ratio > 0.20:
            return False
            
        # B) Pouca pele (<5%) = Falso positivo ou desenho sem cor de pele
        # (Relaxado para 5% para aceitar P&B ou iluminação ruim, mas rejeitar wireframe puro)
        if skin_ratio < 0.05:
            return False
            
        return True

    def _rotate_roi(self, image: np.ndarray, angle: float):
        """Rotaciona imagem e retorna matriz inversa."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Recalcula dimensões
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated, cv2.invertAffineTransform(M)

    def _non_max_suppression(self, boxes: List[Tuple], thresh: float):
        """NMS simples."""
        if not boxes: return []
        boxes = np.array(boxes).astype(float)
        pick = []
        
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,0] + boxes[:,2]
        y2 = boxes[:,1] + boxes[:,3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > thresh)[0])))
            
        return [tuple(boxes[i].astype(int)) for i in pick]

    def _assign_face_id(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """Tracking simples baseado em distância Euclidiana."""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        
        min_dist = float('inf')
        matched_id = None
        
        # Procura face próxima no histórico
        for fid, last_pos in self.tracked_faces.items():
            last_cx, last_cy = last_pos
            dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
            
            if dist < min_dist and dist < 100: # Max drift
                min_dist = dist
                matched_id = fid
        
        if matched_id is None:
            self.face_counter += 1
            matched_id = self.face_counter
            
        self.tracked_faces[matched_id] = np.array([cx, cy])
        return matched_id
