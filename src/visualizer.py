"""
Tech Challenge - Fase 4: Módulo de Visualização
Funções para desenhar detecções e anotações em frames.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image as PILImage, ImageDraw, ImageFont

from .face_detector import FaceDetection
from .emotion_analyzer import EmotionResult
from .activity_detector import ActivityDetection
from .anomaly_detector import AnomalyEvent
try:
    from .object_detector import ObjectDetection
except ImportError:
    # Caso o módulo não esteja disponível ainda
    ObjectDetection = None
from .config import ANOMALY_LABELS, OBJECT_LABELS


# Cores padrão (RGB para PIL)
COLORS = {
    "face": (0, 255, 0),       # Verde
    "emotion": (0, 255, 255),  # Ciano
    "activity": (255, 165, 0), # Laranja
    "anomaly": (255, 0, 0),    # Vermelho
    "text": (255, 255, 255),   # Branco
    "object": (180, 0, 180),   # Roxo para objetos
}


def _get_font(size: int = 20) -> ImageFont.FreeTypeFont:
    """Obtém fonte com suporte a UTF-8."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def put_text(
    img: np.ndarray, 
    text: str, 
    position: Tuple[int, int], 
    font_size: int = 20, 
    color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Adiciona texto com suporte a UTF-8 usando PIL."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    
    x, y = position
    # Borda preta para contraste
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text(position, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_detections(
    frame: np.ndarray,
    faces: List[FaceDetection],
    emotions: List[Optional[EmotionResult]],
    activities: List[ActivityDetection],
    anomalies: List[AnomalyEvent],
    objects: List[object] = None,  # Lista de ObjectDetection opcional
    overlays: List[object] = None, # Lista de OverlayDetection opcional
    min_face_size: int = 40,
    min_emotion_conf: float = 0.5,  # Fallback para emoções sem threshold específico
    min_activity_conf: float = 0.5,
    min_object_conf: float = 0.5,
    use_adaptive_threshold: bool = True  # Usa thresholds adaptativos por emoção
) -> np.ndarray:
    """
    Desenha todas as detecções no frame.
    
    Args:
        frame: Frame BGR
        faces: Lista de detecções de faces
        emotions: Lista de resultados de emoções (pode ter None)
        activities: Lista de detecções de atividades
        anomalies: Lista de anomalias detectadas
        objects: Lista de objetos detectados (opcional)
        overlays: Lista de overlays detectados (opcional)
        min_face_size: Tamanho mínimo de face para exibir
        min_emotion_conf: Confiança mínima para exibir emoção (fallback)
        min_activity_conf: Confiança mínima para exibir atividade
        use_adaptive_threshold: Se True, usa thresholds específicos por emoção
    
    Returns:
        Frame anotado
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Desenha objetos gerais (filtra por confiança mínima)
    if objects:
        for obj in objects:
            if hasattr(obj, 'bbox') and hasattr(obj, 'class_name'):
                # Verifica confiança mínima
                obj_conf = getattr(obj, 'confidence', 1.0)
                if obj_conf < min_object_conf:
                    continue
                    
                ox, oy, ow, oh = obj.bbox
                # Roxo para objetos
                cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), (180, 0, 180), 1)
                
                # Texto com background (traduzido para português)
                class_name_pt = OBJECT_LABELS.get(obj.class_name, obj.class_name)
                label = f"{class_name_pt} {obj_conf:.0%}"
                
                annotated = put_text(annotated, label, (ox, max(0, oy - 15)), 14, COLORS["object"])
    
    # Filtra faces válidas
    valid_faces = [f for f in faces if _is_valid_face(f, w, h, min_face_size)]
    
    # Desenha faces
    for i, face in enumerate(valid_faces):
        x, y, fw, fh = face.bbox
        cv2.rectangle(annotated, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        annotated = put_text(annotated, f"ID:{face.face_id}", (x, max(0, y - 25)), 18, COLORS["face"])
        
        # Emoção correspondente com threshold adaptativo
        if i < len(emotions) and emotions[i] is not None:
            emotion = emotions[i]
            # Usa threshold adaptativo por emoção (mais sensível para neutral/sad)
            from .config import EMOTION_THRESHOLDS
            emotion_threshold = EMOTION_THRESHOLDS.get(
                emotion.dominant_emotion, 
                min_emotion_conf  # fallback para emoções não mapeadas
            )
            
            if emotion.confidence >= emotion_threshold:
                text = f"{emotion.emotion_pt}: {emotion.confidence:.0%}"
                annotated = put_text(annotated, text, (x, y + fh + 5), 16, COLORS["emotion"])
    
    # Desenha atividades (apenas de pessoas detectadas pelo YOLO)
    for activity in activities:
        if activity.confidence < min_activity_conf:
            continue
        if activity.bbox:
            ax, ay, aw, ah = activity.bbox
            # Desenha bbox da pessoa (azul)
            cv2.rectangle(annotated, (ax, ay), (ax + aw, ay + ah), (255, 100, 0), 1)
            annotated = put_text(annotated, activity.activity_pt, (ax, max(0, ay - 10)), 18, COLORS["activity"])
    
    # Desenha anomalias com detalhes
    if anomalies:
        # Contador de anomalias no canto superior
        annotated = put_text(annotated, f"⚠ {len(anomalies)} ANOMALIA(S)", (10, 10), 24, COLORS["anomaly"])
        
        # Desenha cada anomalia que tem bbox
        for anomaly in anomalies:
            if anomaly.bbox:
                ax, ay, aw, ah = anomaly.bbox
                # Retângulo vermelho para anomalias
                cv2.rectangle(annotated, (ax, ay), (ax + aw, ay + ah), (0, 0, 255), 2)
                # Nome traduzido da anomalia
                anomaly_name = anomaly.anomaly_type.value if hasattr(anomaly.anomaly_type, 'value') else str(anomaly.anomaly_type)
                anomaly_label = ANOMALY_LABELS.get(anomaly_name, anomaly_name)
                severity_pct = f"{anomaly.severity:.0%}" if anomaly.severity else ""
                text = f"⚠ {anomaly_label} {severity_pct}"
                # Posiciona texto: acima se houver espaço, senão abaixo
                text_y = ay - 10 if ay > 30 else ay + ah + 20
                annotated = put_text(annotated, text, (ax, max(5, text_y)), 16, COLORS["anomaly"])
    
    return annotated


def _is_valid_face(
    face: FaceDetection, 
    frame_w: int, 
    frame_h: int, 
    min_size: int = 40
) -> bool:
    """Valida se uma detecção de face é plausível."""
    x, y, w, h = face.bbox
    
    # Tamanho mínimo
    if w < min_size or h < min_size:
        return False
    
    # Proporção válida (rostos são aproximadamente quadrados)
    aspect = w / h if h > 0 else 0
    if aspect < 0.5 or aspect > 2.0:
        return False
    
    # Dentro dos limites
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        return False
    
    return True


def show_frame(frame: np.ndarray, title: str = "Frame", figsize: Tuple[int, int] = (12, 8)):
    """Exibe um frame no notebook usando matplotlib."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_frame)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
