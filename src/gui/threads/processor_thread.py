"""
Thread de processamento de vídeo
"""

import threading
import time
import cv2
from pathlib import Path
from collections import Counter

from ...face_detector import FaceDetector
from ...emotion_analyzer import EmotionAnalyzer
from ...activity_detector import ActivityDetector
from ...anomaly_detector import AnomalyDetector
from ...visualizer import draw_detections


class ProcessorThread(threading.Thread):
    """Thread para processar vídeo sem bloquear a UI."""
    
    def __init__(self, video_path, output_path, frame_skip=2, min_face_size=40):
        super().__init__(daemon=True)
        
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.frame_skip = frame_skip
        self.min_face_size = min_face_size
        
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        
        # Callbacks
        self.on_progress = None  # callback(frame_idx, total_frames, fps, stats)
        self.on_complete = None  # callback(stats, elapsed_time)
        self.on_error = None     # callback(error_message)
        self.on_frame_processed = None  # callback(frame)
        
        # Estatísticas
        self.stats = {
            'faces': 0,
            'emotions': Counter(),
            'activities': Counter(),
            'anomalies': Counter()
        }
    
    def run(self):
        """Executa processamento."""
        self.is_running = True
        
        try:
            self._process_video()
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
        finally:
            self.is_running = False
    
    def pause(self):
        """Pausa processamento."""
        self.is_paused = True
    
    def resume(self):
        """Retoma processamento."""
        self.is_paused = False
    
    def stop(self):
        """Para processamento."""
        self.should_stop = True
    
    def _process_video(self):
        """Processa vídeo frame a frame."""
        # Inicializa detectores
        face_detector = FaceDetector(method="haar")
        emotion_analyzer = EmotionAnalyzer(method="fer")
        activity_detector = ActivityDetector(model_size="s")
        anomaly_detector = AnomalyDetector()
        
        # Abre vídeo
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {self.video_path}")
        
        # Propriedades
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configura gravador
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        
        # Cache de detecções
        cache = {
            'faces': [],
            'emotions': [],
            'activities': [],
            'anomalies': []
        }
        
        # Processamento
        frame_idx = 0
        start_time = time.time()
        last_update = 0
        
        while not self.should_stop:
            # Pausa
            while self.is_paused and not self.should_stop:
                time.sleep(0.1)
            
            if self.should_stop:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detecção a cada N frames
            if frame_idx % self.frame_skip == 0:
                # Faces
                cache['faces'] = face_detector.detect(frame)
                self.stats['faces'] += len(cache['faces'])
                
                # Emoções
                cache['emotions'] = []
                for face in cache['faces']:
                    if face.bbox[2] >= self.min_face_size:
                        emotion = emotion_analyzer.analyze(frame, face.bbox, face.face_id)
                        cache['emotions'].append(emotion)
                        if emotion:
                            self.stats['emotions'][emotion.emotion_pt] += 1
                    else:
                        cache['emotions'].append(None)
                
                # Atividades
                cache['activities'] = activity_detector.detect(frame)
                for activity in cache['activities']:
                    self.stats['activities'][activity.activity_pt] += 1
                
                # Anomalias
                cache['anomalies'] = anomaly_detector.update(
                    frame_idx,
                    cache['faces'],
                    [e for e in cache['emotions'] if e],
                    cache['activities']
                )
                for anomaly in cache['anomalies']:
                    self.stats['anomalies'][anomaly.anomaly_type] += 1
            
            # Desenha anotações
            annotated = draw_detections(
                frame,
                cache['faces'],
                cache['emotions'],
                cache['activities'],
                cache['anomalies'],
                self.min_face_size
            )
            out.write(annotated)
            
            # Callback de frame processado
            if self.on_frame_processed:
                self.on_frame_processed(annotated)
            
            # Callback de progresso (a cada 1 segundo)
            current_time = time.time()
            if current_time - last_update >= 1.0:
                elapsed = current_time - start_time
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                
                if self.on_progress:
                    self.on_progress(frame_idx, total_frames, fps_proc, self.stats.copy())
                
                last_update = current_time
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Callback de conclusão
        if not self.should_stop and self.on_complete:
            elapsed = time.time() - start_time
            self.on_complete(self.stats, elapsed)
