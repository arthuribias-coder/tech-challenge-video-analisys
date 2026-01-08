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
            
            # Inicializar cascata de perfil (opcional)
            self.profile_detector = None
            profile_paths = [
                str(Path(cv2.__file__).parent / "data" / "haarcascade_profileface.xml"),
                "/home/aineto/workspaces/POS/TC-4/.venv/lib/python3.12/site-packages/cv2/data/haarcascade_profileface.xml",
                "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml",
                "/usr/share/opencv/haarcascades/haarcascade_profileface.xml",
            ]
            
            for path in profile_paths:
                if os.path.exists(path):
                    self.profile_detector = cv2.CascadeClassifier(path)
                    if not self.profile_detector.empty():
                        break
                    
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
        """Detecção usando Haar Cascades com estratégia híbrida (Rigorosa + Permissiva + Perfil + Rotação)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h_frame, w_frame = frame.shape[:2]
        min_dim = int(min(h_frame, w_frame) * 0.05)  # 5% da tela (pequenos)
        large_dim = int(min(h_frame, w_frame) * 0.15) # 15% da tela (grandes)
        
        all_faces = []
        
        # PASS 1: Detecção RIGOROSA (para rostos pequenos/médios e fundos complexos)
        # Evita "rostos em objetos inanimados"
        faces_strict = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(min_dim, min_dim)
        )
        if len(faces_strict) > 0:
            all_faces.extend([(x, y, w, h, 0) for (x, y, w, h) in faces_strict])  # 0 = sem rotação
        
        # PASS 2: Detecção PERMISSIVA (apenas para rostos GRANDES)
        # Permite detectar rostos ocluídos (mão no rosto, perfil, inclinado)
        # Como o tamanho é grande, a chance de falso positivo em textura de fundo é baixa.
        faces_loose = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,  # Menos vizinhos exigidos
            minSize=(large_dim, large_dim)
        )
        if len(faces_loose) > 0:
            all_faces.extend([(x, y, w, h, 0) for (x, y, w, h) in faces_loose])
        
        # PASS 3: Detecção de PERFIL (esquerdo e direito)
        if self.profile_detector is not None:
            # Perfil esquerdo
            faces_profile_left = self.profile_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_dim, min_dim)
            )
            if len(faces_profile_left) > 0:
                all_faces.extend([(x, y, w, h, 0) for (x, y, w, h) in faces_profile_left])
            
            # Perfil direito (espelhar imagem)
            gray_flipped = cv2.flip(gray, 1)
            faces_profile_right = self.profile_detector.detectMultiScale(
                gray_flipped,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_dim, min_dim)
            )
            if len(faces_profile_right) > 0:
                # Converter coordenadas para imagem original (espelhada)
                for (x, y, w, h) in faces_profile_right:
                    x_original = w_frame - x - w
                    all_faces.append((x_original, y, w, h, 0))
        
        # NOTA: Rotação global removida por questões de performance.
        # Use detect_in_regions com rotação habilitada para áreas específicas (ex: pessoas deitadas).
        
        # Remover duplicatas usando NMS (Non-Maximum Suppression)
        final_faces = self._non_max_suppression(all_faces, 0.4)
        
        detections = []
        for (x, y, w, h, angle) in final_faces:
            # Validação de "Realismo" (filtra CGI/Hologramas)
            if not self._is_real_face(frame, (x, y, w, h)):
                continue

            # face_id será atribuído externamente ou precisa ser gerenciado aqui
            # No fluxo atual, _assign_face_id é chamado
            face_id = self._assign_face_id(frame, (x, y, w, h))
            
            detections.append(FaceDetection(
                face_id=face_id,
                bbox=(x, y, w, h),
                confidence=1.0  # Haar não fornece confiança simples
            ))
        
        return detections

    def detect_in_regions(self, frame: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[FaceDetection]:
        """
        Detecta rostos apenas em regiões específicas (otimizado).
        Rotaciona as regiões (90/-90) para tentar encontrar rostos deitados.
        """
        if not regions:
            return []

        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = []
        
        for (rx, ry, rw, rh) in regions:
            # Garante limites
            rx, ry = max(0, rx), max(0, ry)
            rw = min(rw, w_frame - rx)
            rh = min(rh, h_frame - ry)
            
            if rw < 20 or rh < 20: continue
            
            # Recorta ROI
            roi_gray = gray[ry:ry+rh, rx:rx+rw]
            
            # Tenta rotações (90 e -90)
            local_faces = []
            
            for angle in [90, -90]:
                try:
                    rotated, inv_matrix = self._rotate_image(roi_gray, angle)
                    # Detecta
                    faces = self.detector.detectMultiScale(
                         rotated, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
                    )
                    
                    if len(faces) > 0:
                        for (fx, fy, fw, fh) in faces:
                            # Converte coords: ROI Rotacionado -> ROI Original
                            center_rot = np.array([fx + fw/2, fy + fh/2, 1])
                            center_roi = inv_matrix @ center_rot
                            
                            # ROI Original -> Frame Global
                            size = max(fw, fh)
                            gx = int(center_roi[0] - size/2) + rx
                            gy = int(center_roi[1] - size/2) + ry
                            
                            # Adiciona (com ângulo para NMS se necessário)
                            local_faces.append((gx, gy, size, size, angle))
                except Exception:
                    continue

            # Processa faces encontradas nesta região
            for (x, y, w, h, _) in local_faces:
                if not self._is_real_face(frame, (x, y, w, h)):
                    continue
                    
                face_id = self._assign_face_id(frame, (x, y, w, h))
                detections.append(FaceDetection(
                    face_id=face_id,
                    bbox=(x, y, w, h),
                    confidence=0.9
                ))

        return detections
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rotaciona imagem e retorna matriz inversa para converter coordenadas."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Matriz de rotação
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calcular novo tamanho para não cortar
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Ajustar translação
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        
        # Matriz inversa para converter de volta
        M_inv = cv2.invertAffineTransform(M)
        
        return rotated, M_inv
    
    def _non_max_suppression(
        self, 
        faces: List[Tuple[int, int, int, int, int]], 
        overlap_thresh: float
    ) -> List[Tuple[int, int, int, int, int]]:
        """Remove detecções duplicadas usando Non-Maximum Suppression."""
        if len(faces) == 0:
            return []
        
        # Converter para arrays
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h, _) in faces])
        angles = [a for (_, _, _, _, a) in faces]
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        # Ordenar por área (manter os maiores)
        idxs = np.argsort(areas)[::-1]
        
        pick = []
        while len(idxs) > 0:
            i = idxs[0]
            pick.append(i)
            
            # Calcular IoU com os restantes
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / areas[idxs[1:]]
            
            # Remover os que sobrepõem demais
            idxs = idxs[1:][overlap <= overlap_thresh]
        
        result = []
        for i in pick:
            x, y, w, h, a = faces[i]
            result.append((x, y, w, h, a))
        
        return result

    def _is_real_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Valida se o rosto detectado parece real (filtra CGI/Wireframes).
        Usa análise de cor da pele e saturação.
        """
        x, y, w, h = bbox
        
        # Garante limites
        h_img, w_img = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w < 10 or h < 10:
            return False
            
        roi = frame[y:y+h, x:x+w]
        
        # 1. Verificação de Cor de Pele (YCbCr é robusto para pele real)
        try:
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            
            # Intervalos típicos de pele humana em YCrCb
            # Cr (Red-diff): 133-173 (Pele é sempre avermelhada)
            # Cb (Blue-diff): 77-127 (Pouco azul)
            min_YCrCb = np.array([0, 133, 77], np.uint8)
            max_YCrCb = np.array([255, 173, 127], np.uint8)
            
            skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
            non_zero = cv2.countNonZero(skin_mask)
            skin_ratio = non_zero / (w * h) if w * h > 0 else 0
            
            # 2. Verifica saturação (para distinguir P&B de Colorido Incorreto)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            avg_sat = np.mean(saturation)
            
            # Lógica de decisão:
            # - Se tem cor (Saturação > 20):
            #    - Rejeita se tiver pouquíssima cor de pele (< 10%)
            #      (Ex: Rosto azul de holograma, rosto verde, etc)
            # - Se é P&B (Saturação <= 20):
            #    - Aceita (não podemos usar cor para validar)
            
            if avg_sat > 20 and skin_ratio < 0.10:
                # Caso do Holograma Azul: Saturação alta (azul), Skin ratio ~ 0
                return False
                
            return True
            
        except Exception as e:
            # Em caso de erro na conversão, aceita por segurança
            print(f"[AVISO] Erro na validação de rosto: {e}")
            return True
    
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
