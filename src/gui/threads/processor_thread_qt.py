"""
Thread de processamento com QThread
"""

from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
import time
import cv2
import numpy as np
from collections import Counter

from ...face_detector import FaceDetector
from ...emotion_analyzer import EmotionAnalyzer
from ...activity_detector import ActivityDetector
from ...anomaly_detector import AnomalyDetector
from ...visualizer import draw_detections
from ...config import (
    ENABLE_OBJECT_DETECTION, ENABLE_OVERLAY_DETECTION, ENABLE_SEGMENT_VALIDATION,
    should_use_gpu, get_device
)

# Novos detectores para anomalias avançadas
try:
    from ...object_detector import ObjectDetector
    OBJECT_DETECTOR_AVAILABLE = True
except ImportError:
    OBJECT_DETECTOR_AVAILABLE = False
    print("[INFO] ObjectDetector não disponível")

try:
    from ...overlay_detector import OverlayDetector
    OVERLAY_DETECTOR_AVAILABLE = True
except ImportError:
    OVERLAY_DETECTOR_AVAILABLE = False
    print("[INFO] OverlayDetector não disponível")

try:
    from ...segment_validator import SegmentValidator
    SEGMENT_VALIDATOR_AVAILABLE = True
except ImportError:
    SEGMENT_VALIDATOR_AVAILABLE = False
    print("[INFO] SegmentValidator não disponível")


class ProcessorThreadQt(QThread):
    """Thread Qt para processamento de vídeo."""
    
    # Signals
    progress = pyqtSignal(int, int, float, dict)  # frame_idx, total, fps, stats
    finished_signal = pyqtSignal(dict, float)  # stats, elapsed_time
    error = pyqtSignal(str)  # error_msg
    frame_processed = pyqtSignal(int, object, dict)  # frame_idx, processed_frame (ndarray), metadata
    
    def __init__(self, video_path, output_path, frame_skip=2, target_fps=30, 
                 enable_preview=True, preview_fps=10,
                 # Novas opções para detectores avançados (usa config como padrão)
                 enable_object_detection=None,
                 enable_overlay_detection=None,
                 enable_segment_validation=None,
                 # Configurações de hardware
                 use_gpu=None,
                 model_size=None
                 ):
        super().__init__()
        
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.frame_skip = frame_skip
        self.target_fps = target_fps
        self.enable_preview = enable_preview
        self.preview_fps = preview_fps
        
        # Armazena configurações de hardware para os detectores
        self.use_gpu = use_gpu  # "auto", "true", "false" ou None
        self.model_size = model_size  # "n", "s", "m", "l" ou None
        
        # Opções de detectores avançados (usa config se None)
        obj_enabled = enable_object_detection if enable_object_detection is not None else ENABLE_OBJECT_DETECTION
        overlay_enabled = enable_overlay_detection if enable_overlay_detection is not None else ENABLE_OVERLAY_DETECTION
        seg_enabled = enable_segment_validation if enable_segment_validation is not None else ENABLE_SEGMENT_VALIDATION
        
        self.enable_object_detection = obj_enabled and OBJECT_DETECTOR_AVAILABLE
        self.enable_overlay_detection = overlay_enabled and OVERLAY_DETECTOR_AVAILABLE
        self.enable_segment_validation = seg_enabled and SEGMENT_VALIDATOR_AVAILABLE
        
        self.is_paused = False
        self.should_stop = False
        
        # Controle de preview
        self._last_preview_time = 0
        self._preview_interval = 1.0 / preview_fps if preview_fps > 0 else 0.1
    
    def run(self):
        """Executa processamento."""
        try:
            start_time = time.time()
            
            # Log de configuração
            device = get_device()
            model_size = self.model_size if self.model_size else "n"
            print(f"[INFO] Inicializando componentes (device: {device}, model_size: {model_size})...")
            print(f"[INFO] Detectores: object={self.enable_object_detection}, overlay={self.enable_overlay_detection}, segment={self.enable_segment_validation}")
            
            # Inicializa componentes principais
            face_detector = FaceDetector()
            emotion_analyzer = EmotionAnalyzer()  # Usa DeepFace por padrão (via config)
            activity_detector = ActivityDetector(model_size=model_size)
            anomaly_detector = AnomalyDetector(
                enable_object_anomalies=self.enable_object_detection,
                enable_overlay_anomalies=self.enable_overlay_detection
            )
            
            # Inicializa componentes avançados (opcionais)
            object_detector = None
            overlay_detector = None
            segment_validator = None
            
            if self.enable_object_detection:
                try:
                    object_detector = ObjectDetector(model_size=model_size, min_confidence=0.5)
                    print("[INFO] ObjectDetector habilitado")
                except Exception as e:
                    print(f"[WARN] ObjectDetector falhou: {e}")
                    self.enable_object_detection = False
            
            if self.enable_overlay_detection:
                try:
                    overlay_detector = OverlayDetector(use_easyocr=True, min_text_confidence=0.5)
                    print("[INFO] OverlayDetector habilitado")
                except Exception as e:
                    print(f"[WARN] OverlayDetector falhou: {e}")
                    self.enable_overlay_detection = False
            
            if self.enable_segment_validation:
                try:
                    segment_validator = SegmentValidator(model_size=model_size, min_confidence=0.5)
                    print("[INFO] SegmentValidator habilitado")
                except Exception as e:
                    print(f"[WARN] SegmentValidator falhou: {e}")
                    self.enable_segment_validation = False
            
            print(f"[INFO] Abrindo vídeo: {self.video_path}")
            
            # Abre vídeo
            cap = cv2.VideoCapture(str(self.video_path))
            
            if not cap.isOpened():
                self.error.emit(f"Erro ao abrir vídeo: {self.video_path}")
                return
            
            # Configurações
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"[INFO] Vídeo: {width}x{height} @ {fps}fps, {total_frames} frames")
            print(f"[INFO] Configurações: frame_skip={self.frame_skip}, target_fps={self.target_fps}")
            print(f"[INFO] Preview: {'habilitado' if self.enable_preview else 'desabilitado'} @ {self.preview_fps} FPS")
            
            # Writer (usa target_fps configurado)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(self.output_path), fourcc, self.target_fps, (width, height))
            
            if not out.isOpened():
                self.error.emit("Erro ao criar arquivo de saída")
                cap.release()
                return
            
            # Estatísticas
            stats = {
                'faces': 0,
                'emotions': Counter(),
                'activities': Counter(),
                'anomalies': Counter(),
                'objects': Counter(),  # Novo: contagem de objetos
                'overlays': 0  # Novo: contagem de overlays
            }
            
            frame_idx = 0
            process_start = time.time()
            last_progress_update = 0
            
            # Variáveis para detectores opcionais
            objects = []
            overlays = []
            segment_results = []
            
            print("[INFO] Iniciando processamento...")
            
            # Cache para persistir detecções entre frames (para bbox fluído)
            last_faces = []
            last_emotions = []
            last_activities = []
            last_anomalies = []
            last_objects = []
            
            while cap.isOpened() and not self.should_stop:
                # Pausa
                while self.is_paused and not self.should_stop:
                    self.msleep(100)
                
                if self.should_stop:
                    break
                
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Processa frame
                processed_frame = frame.copy()
                
                # Variáveis para este frame (inicia com cache para persistência)
                faces = last_faces
                emotions = last_emotions
                activities = last_activities
                anomalies = last_anomalies
                objects = []
                overlays = []
                segment_results = []
                
                if frame_idx % self.frame_skip == 0:
                    # Reseta para novo processamento
                    faces = []
                    emotions = []
                    activities = []
                    anomalies = []
                    
                    try:
                        # 1. Detecta ATIVIDADES primeiro (para otimização de rostos)
                        activities = activity_detector.detect(frame)
                        for activity in activities:
                            activity_name = activity.activity_pt if hasattr(activity, 'activity_pt') else str(activity)
                            stats['activities'][activity_name] = stats['activities'].get(activity_name, 0) + 1
                        
                        # Prepara regiões para detecção rotacionada (pessoas deitadas)
                        lying_regions = []
                        # Import local para evitar circular, ou usa string
                        # Assumindo que act.activity é Enum, mas debug mostra que activity names são string em stats
                        # Mas act.activity é o Enum.
                        
                        for act in activities:
                            # Verifica se é "lying" (pelo value do enum ou string)
                            is_lying = False
                            if hasattr(act.activity, 'value'):
                                is_lying = (act.activity.value == "lying")
                            else:
                                is_lying = (str(act.activity) == "lying")
                                
                            if is_lying:
                                # Usa keypoints para focar na cabeça (muito mais rápido que rotacionar bbox inteiro)
                                if act.keypoints and act.keypoints.nose:
                                    nx, ny = act.keypoints.nose
                                    # Cria região ao redor do nariz (tamanho depende do bbox da pessoa ou fixo)
                                    # Estima tamanho da cabeça baseado na altura/largura da pessoa
                                    px, py, pw, ph = act.bbox
                                    person_dim = min(pw, ph) # Dimensão menor (largura se deitado)
                                    head_size = int(max(100, person_dim * 0.4)) # 40% da dimensão menor
                                    
                                    hx = int(nx - head_size/2)
                                    hy = int(ny - head_size/2)
                                    lying_regions.append((hx, hy, head_size, head_size))
                                else:
                                    # Fallback: bbox inteiro da pessoa
                                    lying_regions.append(act.bbox)

                        # 2. Detecta faces (Normal + Regiões Rotacionadas)
                        faces = face_detector.detect(frame)
                        
                        # Adiciona detecção otimizada nas regiões de pessoas deitadas
                        if lying_regions:
                            rotated_faces = face_detector.detect_in_regions(frame, lying_regions)
                            faces.extend(rotated_faces)

                        stats['faces'] += len(faces)
                        
                        # 3. Analisa emoções para cada face
                        for face in faces:
                            x, y, w, h = face.bbox
                            
                            # EmotionAnalyzer.analyze() precisa de frame completo, bbox e face_id
                            emotion = emotion_analyzer.analyze(frame, face.bbox, face.face_id)
                            emotions.append(emotion)
                            if emotion:
                                emotion_name = emotion.emotion_pt if hasattr(emotion, 'emotion_pt') else str(emotion)
                                stats['emotions'][emotion_name] = stats['emotions'].get(emotion_name, 0) + 1
                        
                        # === NOVOS DETECTORES ===
                        
                        # Detecta objetos (contexto visual)
                        if object_detector:
                            try:
                                objects = object_detector.detect(frame, frame_idx)
                                for obj in objects:
                                    stats['objects'][obj.class_name] = stats['objects'].get(obj.class_name, 0) + 1
                            except Exception as e:
                                print(f"[WARN] ObjectDetector erro: {e}")
                        
                        # Detecta overlays/texto (a cada 10 frames para performance)
                        if overlay_detector and frame_idx % 10 == 0:
                            try:
                                overlays = overlay_detector.detect(frame, frame_idx)
                                stats['overlays'] += len(overlays)
                            except Exception as e:
                                print(f"[WARN] OverlayDetector erro: {e}")
                        
                        # Valida segmentação (a cada 5 frames para performance)
                        if segment_validator and frame_idx % 5 == 0:
                            try:
                                validations = segment_validator.validate(frame, activities, frame_idx)
                                segment_results = segment_validator.get_anomaly_results(validations)
                            except Exception as e:
                                print(f"[WARN] SegmentValidator erro: {e}")
                        
                        # Detecta anomalias usando o método estendido
                        anomalies = anomaly_detector.update_extended(
                            frame_idx, 
                            faces, 
                            emotions, 
                            activities,
                            object_detections=objects if objects else None,
                            overlay_detections=overlays if overlays else None,
                            segment_results=segment_results if segment_results else None
                        )
                        
                        for anomaly in anomalies:
                            # AnomalyEvent tem anomaly_type (enum), não .type
                            anomaly_name = anomaly.anomaly_type.value if hasattr(anomaly, 'anomaly_type') else str(anomaly)
                            stats['anomalies'][anomaly_name] = stats['anomalies'].get(anomaly_name, 0) + 1
                        
                        # Visualiza (inclui objects)
                        processed_frame = draw_detections(frame, faces, emotions, activities, anomalies, objects=objects)
                        
                        # Atualiza cache para frames intermediários (persistência de bbox)
                        last_faces = faces
                        last_emotions = emotions
                        last_activities = activities
                        last_anomalies = anomalies
                        last_objects = objects
                    
                    except Exception as e:
                        print(f"[WARN] Erro ao processar frame {frame_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Frame intermediário: usa detecções cacheadas para manter bbox visível
                    if last_faces or last_activities or last_anomalies or last_objects:
                        processed_frame = draw_detections(frame, last_faces, last_emotions, last_activities, last_anomalies, objects=last_objects)
                
                # Escreve frame
                out.write(processed_frame)
                
                # Emite preview se habilitado
                current_time = time.time()
                if self.enable_preview and (current_time - self._last_preview_time) >= self._preview_interval:
                    # Downsample frame para preview (50% resolução)
                    preview_frame = cv2.resize(processed_frame, (width // 2, height // 2))
                    
                    # Metadata do frame
                    metadata = {
                        'faces_count': len(faces) if frame_idx % self.frame_skip == 0 else 0,
                        'activities_count': len(activities) if frame_idx % self.frame_skip == 0 else 0,
                        'anomalies_count': len(anomalies) if frame_idx % self.frame_skip == 0 else 0,
                        'objects_count': len(objects) if objects else 0,
                        'overlays_count': len(overlays) if overlays else 0
                    }
                    
                    self.frame_processed.emit(frame_idx, preview_frame, metadata)
                    self._last_preview_time = current_time
                
                # Progresso
                frame_idx += 1
                elapsed = time.time() - process_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                
                # Emite progresso a cada 30 frames ou 1 segundo
                if frame_idx % 30 == 0 or (time.time() - last_progress_update) > 1.0:
                    self.progress.emit(frame_idx, total_frames, current_fps, stats.copy())
                    last_progress_update = time.time()
            
            # Libera recursos
            cap.release()
            out.release()
            
            elapsed_time = time.time() - start_time
            
            print(f"[INFO] Processamento concluído em {elapsed_time:.1f}s")
            print(f"[INFO] Total de faces: {stats['faces']}")
            print(f"[INFO] Total de objetos: {sum(stats['objects'].values())}")
            print(f"[INFO] Total de anomalias: {sum(stats['anomalies'].values())}")
            print(f"[INFO] Vídeo salvo: {self.output_path}")
            
            if not self.should_stop:
                self.finished_signal.emit(stats, elapsed_time)
            
        except Exception as e:
            import traceback
            error_msg = f"Erro no processamento: {str(e)}\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            self.error.emit(error_msg)
    
    def toggle_pause(self):
        """Pausa/retoma."""
        self.is_paused = not self.is_paused
    
    def stop(self):
        """Para thread."""
        self.should_stop = True
