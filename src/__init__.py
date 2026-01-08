# Tech Challenge - Fase 4
# Análise de Vídeo com Reconhecimento Facial, Emoções e Atividades
from .config import VIDEO_PATH, OUTPUT_DIR, REPORTS_DIR, INPUT_DIR
from .face_detector import FaceDetector, FaceDetection
from .emotion_analyzer import EmotionAnalyzer, EmotionResult
from .activity_detector import ActivityDetector, ActivityDetection, ActivityType
from .anomaly_detector import AnomalyDetector, AnomalyEvent, AnomalyType
from .report_generator import ReportGenerator
from .visualizer import draw_detections, put_text, show_frame

# Novos módulos da Fase 4 (detecção avançada de anomalias)
from .object_detector import ObjectDetector, ObjectDetection, ObjectCategory

__all__ = [
    "VIDEO_PATH", "OUTPUT_DIR", "REPORTS_DIR", "INPUT_DIR",
    "FaceDetector", "FaceDetection",
    "EmotionAnalyzer", "EmotionResult", 
    "ActivityDetector", "ActivityDetection", "ActivityType",
    "AnomalyDetector", "AnomalyEvent", "AnomalyType",
    "ReportGenerator",
    "draw_detections", "put_text", "show_frame",
    # Novos exports
    "ObjectDetector", "ObjectDetection", "ObjectCategory",
]