"""
Thread de processamento com QThread
"""

from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
import time
import cv2
import numpy as np
import logging
from collections import Counter

from ...face_detector import FaceDetector, FaceDetection
from ...emotion_analyzer import EmotionAnalyzer
from ...activity_detector import ActivityDetector
from ...anomaly_detector import AnomalyDetector
from ...scene_classifier import SceneClassifier, SceneContext
from ...visualizer import draw_detections, put_text
from ...config import (
    ENABLE_OBJECT_DETECTION, 
    should_use_gpu, get_device, is_gpu_available,
    DEBUG_LOGGING, DEBUG_LOG_INTERVAL
)

logger = logging.getLogger(__name__)

# Novos detectores para anomalias avançadas
try:
    from ...object_detector import ObjectDetector
    OBJECT_DETECTOR_AVAILABLE = True
except ImportError:
    OBJECT_DETECTOR_AVAILABLE = False
    logger.info("ObjectDetector não disponível")

try:
    from ...oriented_detector import OrientedDetector
    ORIENTED_DETECTOR_AVAILABLE = True
except ImportError as e:
    ORIENTED_DETECTOR_AVAILABLE = False
    logger.info(f"OrientedDetector não disponível: {e}")


class ProcessorThreadQt(QThread):
    """Thread Qt para processamento de vídeo."""
    
    # Signals
    progress = pyqtSignal(int, int, float, dict)  # frame_idx, total, fps, stats
    finished_signal = pyqtSignal(dict, float)  # stats, elapsed_time
    error = pyqtSignal(str)  # error_msg
    frame_processed = pyqtSignal(int, object, dict)  # frame_idx, processed_frame (ndarray), metadata
    
    def __init__(self, video_path, output_path, frame_skip=2, target_fps=30, 
                 enable_preview=True, preview_fps=10,
                 enable_object_detection=None,
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
        
        self.use_gpu = use_gpu
        self.model_size = model_size
        
        # Log control
        self.debug_mode = DEBUG_LOGGING
        
        # Configuração simplificada
        self.enable_object_detection = enable_object_detection if enable_object_detection is not None else ENABLE_OBJECT_DETECTION
        
        self.is_paused = False
        self.should_stop = False
        
        # Controle de preview
        self._last_preview_time = 0
        self._preview_interval = 1.0 / preview_fps if preview_fps > 0 else 0.1

    def _get_configured_device(self) -> str:
        """Determina o device baseado na configuração use_gpu."""
        if self.use_gpu == "true":
            # Força GPU
            if is_gpu_available():
                return "cuda"
            else:
                logger.warning("GPU solicitada mas não disponível. Usando CPU.")
                return "cpu"
        elif self.use_gpu == "false":
            # Força CPU
            return "cpu"
        else:
            # Auto: usa GPU se disponível
            return "cuda" if is_gpu_available() else "cpu"

    def set_debug_mode(self, enabled: bool):
        """Atualiza modo de debug em tempo de execução."""
        self.debug_mode = enabled
        # Propaga para outros módulos se necessário
        # Nota: modules como scene_classifier usam DEBUG_LOGGING global, 
        # que não muda em runtime facilmente sem reload,
        # mas podemos definir em config se quisermos persistência global
        from ... import config
        config.DEBUG_LOGGING = enabled
    
    def run(self):
        """Executa processamento."""
        try:
            start_time = time.time()
            
            # Determina device baseado nas configurações
            device = self._get_configured_device()
            model_size = self.model_size if self.model_size else "n"
            logger.info(f"Inicializando componentes (device: {device}, model_size: {model_size})...")
            
            # Inicializa componentes principais
            face_detector = FaceDetector()
            emotion_analyzer = EmotionAnalyzer()
            activity_detector = ActivityDetector(model_size=model_size, device=device)
            anomaly_detector = AnomalyDetector(
                enable_object_anomalies=self.enable_object_detection,
                enable_overlay_anomalies=False # Overlay detector foi removido
            )
            
            # Inicializa componentes opcionais
            object_detector = None
            
            if self.enable_object_detection:
                try:
                    object_detector = ObjectDetector(model_size=model_size, min_confidence=0.5, device=device)
                    logger.info("ObjectDetector habilitado")
                except Exception as e:
                    logger.warning(f"ObjectDetector falhou: {e}")
                    self.enable_object_detection = False
            
            # Inicializa compontentes novos (Scene + OBB)
            scene_classifier = None
            oriented_detector = None

            try:
                scene_classifier = SceneClassifier(model_size=model_size, device=device)
                logger.info("SceneClassifier habilitado")
            except Exception as e:
                logger.warning(f"SceneClassifier falhou: {e}")
            
            if ORIENTED_DETECTOR_AVAILABLE:
                try:
                    oriented_detector = OrientedDetector(model_size=model_size, device=device)
                    logger.info("OrientedDetector habilitado")
                except Exception as e:
                    logger.warning(f"OrientedDetector falhou: {e}")
            
            logger.info(f"Abrindo vídeo: {self.video_path}")
            
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
            
            logger.info(f"Vídeo: {width}x{height} @ {fps}fps, {total_frames} frames")
            logger.info(f"Configurações: frame_skip={self.frame_skip}, target_fps={self.target_fps}")
            logger.info(f"Preview: {'habilitado' if self.enable_preview else 'desabilitado'} @ {self.preview_fps} FPS")
            
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
                'objects': Counter(),
                'scenes': Counter()  # Novo: contagem de tipos de cena
            }
            
            frame_idx = 0
            process_start = time.time()
            last_progress_update = 0
            
            # Variáveis para detectores opcionais
            objects = []
            
            logger.info("Iniciando processamento...")
            
            # Cache para persistir detecções entre frames (para bbox fluído)
            last_faces = []
            last_emotions = []
            last_activities = []
            last_anomalies = []
            last_objects = []
            last_scene_ctx = None
            
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
                scene_ctx = last_scene_ctx
                objects = []
                
                if frame_idx % self.frame_skip == 0:
                    # Reseta para novo processamento
                    faces = []
                    emotions = []
                    activities = []
                    anomalies = []
                    
                    try:
                        # 0. Contexto de Cena (YOLO-cls)
                        if scene_classifier:
                            # Atualiza a cada 30 frames para otimização
                            force = (frame_idx % 30 == 0)
                            try:
                                scene_ctx = scene_classifier.classify(frame, force_update=force)
                                if scene_ctx:
                                    last_scene_ctx = scene_ctx
                                    stats['scenes'][scene_ctx.scene_type] += 1
                                    
                                    # Debug ocasional
                                    if force and self.debug_mode:
                                        logger.debug(f"Cena: {scene_ctx.scene_type} ({scene_ctx.confidence:.2f})")
                            except Exception as e:
                                logger.error(f"Erro na classificação de cena: {e}")
                            
                        # 0.5 Orientação (YOLO-obb)
                        obb_results = []
                        if oriented_detector:
                            obb_results = oriented_detector.detect(frame)

                        # 1. Detecta ATIVIDADES primeiro (passando OBB para refinar lying vs standing)
                        activities = activity_detector.detect(frame, oriented_detections=obb_results)
                        
                        if self.debug_mode and activities and (frame_idx % DEBUG_LOG_INTERVAL == 0):
                            logger.debug(f"Atividades ({len(activities)}): {[a.activity_pt for a in activities]}")

                        for activity in activities:
                            activity_name = activity.activity_pt if hasattr(activity, 'activity_pt') else str(activity)
                            stats['activities'][activity_name] = stats['activities'].get(activity_name, 0) + 1

                        # 2. Detecta faces (Top-Down: Extrai de Pessoas/Atividades)
                        # Removemos detecção global (Haar/DNN) para evitar falsos positivos no cenário
                        # Agora o rosto é extraído sempre baseado nos Keypoints do YOLO-pose
                        
                        faces = []
                        if activities:
                            for act in activities:
                                # Verifica se há keypoints essenciais para estimar rosto
                                if act.keypoints and act.keypoints.nose:
                                    # Pontos chave
                                    nx, ny = act.keypoints.nose
                                    
                                    # Tenta usar olhos para largura
                                    face_size = 0
                                    cx, cy = int(nx), int(ny)
                                    
                                    if act.keypoints.left_eye and act.keypoints.right_eye:
                                        lx, ly = act.keypoints.left_eye
                                        rx, ry = act.keypoints.right_eye
                                        # Distância entre olhos
                                        eye_dist = np.sqrt((lx-rx)**2 + (ly-ry)**2)
                                        # Rosto é aprox 2.5x a distância interpupilar (margem segura)
                                        face_size = int(eye_dist * 3.0) 
                                        
                                        # Ajusta centro para ser entre olhos e nariz
                                        mid_eye_x = (lx + rx) / 2
                                        mid_eye_y = (ly + ry) / 2
                                        cx = int((cx + mid_eye_x) / 2)
                                        cy = int((cy + mid_eye_y) / 2)
                                        
                                    elif act.keypoints.left_ear and act.keypoints.right_ear:
                                         # Fallback: orelhas (mais largas que olhos)
                                        lx, ly = act.keypoints.left_ear
                                        rx, ry = act.keypoints.right_ear
                                        ear_dist = np.sqrt((lx-rx)**2 + (ly-ry)**2)
                                        face_size = int(ear_dist * 1.8)
                                    else:
                                        # Fallback final: Proporção da altura da pessoa
                                        # Cabeça é aprox 1/7 ou 1/8 da altura
                                        px, py, pw, ph = act.bbox
                                        person_dim = max(ph, pw) # Usa dim maior
                                        face_size = int(person_dim / 7.0)
                                        
                                    # Tamanho mínimo de segurança (30px)
                                    face_size = max(30, face_size)
                                    
                                    # Calcula BBox do rosto (quadrado centrado)
                                    x = max(0, cx - face_size // 2)
                                    y = max(0, cy - face_size // 2)
                                    w = face_size
                                    h = face_size
                                    
                                    # Valida limites do frame
                                    if x+w > frame.shape[1]: w = frame.shape[1] - x
                                    if y+h > frame.shape[0]: h = frame.shape[0] - y
                                    
                                    # Cria detecção se válida
                                    if w > 10 and h > 10:
                                        # Usa ID da pessoa detectada pelo YOLO
                                        face_id = act.person_id 
                                        
                                        # Cria objeto FaceDetection
                                        faces.append(FaceDetection(
                                            face_id=face_id,
                                            bbox=(x, y, w, h),
                                            confidence=act.confidence,
                                            landmarks={
                                                'nose': act.keypoints.nose,
                                                'left_eye': act.keypoints.left_eye,
                                                'right_eye': act.keypoints.right_eye
                                            }
                                        ))

                        stats['faces'] += len(faces)
                        
                        # 3. Analisa emoções para cada face
                        for face in faces:
                            x, y, w, h = face.bbox
                            
                            # EmotionAnalyzer.analyze() precisa de frame completo, bbox e face_id
                            # Passamos o contexto da cena atual para calibrar pesos emocionais
                            current_scene = last_scene_ctx.scene_type if last_scene_ctx else "unknown"
                            emotion = emotion_analyzer.analyze(frame, face.bbox, face.face_id, scene_context=current_scene)
                            
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
                                logger.warning(f"ObjectDetector erro: {e}")
                        
                        # Detecta anomalias usando o método estendido
                        anomalies = anomaly_detector.update_extended(
                            frame_idx, 
                            faces, 
                            emotions, 
                            activities,
                            object_detections=objects if objects else None,
                            overlay_detections=None,
                            segment_results=None
                        )

                        if self.debug_mode and anomalies and (frame_idx % DEBUG_LOG_INTERVAL == 0):
                            logger.debug(f"Anomalias ({len(anomalies)}): {[a.anomaly_type.value for a in anomalies]}")
                        
                        # Validação Contextual de Cena (Extra)
                        if scene_ctx and objects and anomaly_detector.enable_object_anomalies:
                            # Chama verificação de contexto se disponível
                            if hasattr(anomaly_detector, '_check_context_anomalies'):
                                ctx_anomalies = anomaly_detector._check_context_anomalies(frame_idx, scene_ctx, objects)
                                anomalies.extend(ctx_anomalies)
                        
                        for anomaly in anomalies:
                            # AnomalyEvent tem anomaly_type (enum), não .type
                            anomaly_name = anomaly.anomaly_type.value if hasattr(anomaly, 'anomaly_type') else str(anomaly)
                            stats['anomalies'][anomaly_name] = stats['anomalies'].get(anomaly_name, 0) + 1
                        
                        # Visualiza (inclui objects)
                        processed_frame = draw_detections(frame, faces, emotions, activities, anomalies, objects=objects)
                        
                        # Desenha Info de Cena - REMOVIDO
                        # if scene_ctx:
                        #      scene_pt = SCENE_LABELS.get(scene_ctx.scene_type, scene_ctx.scene_type).upper()
                        #      text = f"AMB: {scene_pt} ({scene_ctx.confidence:.1f})"
                        #      # Usa put_text para suportar acentos UTF-8 (visualizer.py)
                        #      processed_frame = put_text(processed_frame, text, (10, 50), 24, (0, 255, 255))
                        
                        # Atualiza cache para frames intermediários (persistência de bbox)
                        last_faces = faces
                        last_emotions = emotions
                        last_activities = activities
                        last_anomalies = anomalies
                        last_objects = objects
                    
                    except Exception as e:
                        logger.warning(f"Erro ao processar frame {frame_idx}: {e}")
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
                        'overlays_count': 0
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
            
            logger.info(f"Processamento concluído em {elapsed_time:.1f}s")
            logger.info(f"Total de faces: {stats['faces']}")
            logger.info(f"Total de objetos: {sum(stats['objects'].values())}")
            logger.info(f"Total de anomalias: {sum(stats['anomalies'].values())}")
            logger.info(f"Vídeo salvo: {self.output_path}")
            
            if not self.should_stop:
                self.finished_signal.emit(stats, elapsed_time)
            
        except Exception as e:
            import traceback
            error_msg = f"Erro no processamento: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error.emit(error_msg)
    
    def toggle_pause(self):
        """Pausa/retoma."""
        self.is_paused = not self.is_paused
    
    def stop(self):
        """Para thread."""
        self.should_stop = True
