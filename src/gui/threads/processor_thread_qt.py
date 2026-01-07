"""
Thread de processamento com QThread
"""

from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
import time
import cv2
from collections import Counter

from ...face_detector import FaceDetector
from ...emotion_analyzer import EmotionAnalyzer
from ...activity_detector import ActivityDetector
from ...anomaly_detector import AnomalyDetector
from ...visualizer import draw_detections


class ProcessorThreadQt(QThread):
    """Thread Qt para processamento de vídeo."""
    
    # Signals
    progress = pyqtSignal(int, int, float, dict)  # frame_idx, total, fps, stats
    finished_signal = pyqtSignal(dict, float)  # stats, elapsed_time
    error = pyqtSignal(str)  # error_msg
    
    def __init__(self, video_path, output_path, frame_skip=2):
        super().__init__()
        
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.frame_skip = frame_skip
        
        self.is_paused = False
        self.should_stop = False
    
    def run(self):
        """Executa processamento."""
        try:
            start_time = time.time()
            
            print("[INFO] Inicializando componentes...")
            
            # Inicializa componentes
            face_detector = FaceDetector()
            emotion_analyzer = EmotionAnalyzer()
            activity_detector = ActivityDetector()
            anomaly_detector = AnomalyDetector()
            
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
            
            # Writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                self.error.emit("Erro ao criar arquivo de saída")
                cap.release()
                return
            
            # Estatísticas
            stats = {
                'faces': 0,
                'emotions': Counter(),
                'activities': Counter(),
                'anomalies': Counter()
            }
            
            frame_idx = 0
            process_start = time.time()
            last_progress_update = 0
            
            print("[INFO] Iniciando processamento...")
            
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
                if frame_idx % self.frame_skip == 0:
                    try:
                        # Detecta faces
                        faces = face_detector.detect(frame)
                        stats['faces'] += len(faces)
                        
                        # Analisa emoções para cada face
                        emotions = []
                        for face in faces:
                            x, y, w, h = face.bbox
                            face_img = frame[y:y+h, x:x+w]
                            
                            if face_img.size > 0:
                                emotion = emotion_analyzer.analyze(face_img)
                                emotions.append(emotion)
                                if emotion:
                                    emotion_name = emotion.emotion_pt if hasattr(emotion, 'emotion_pt') else str(emotion)
                                    stats['emotions'][emotion_name] = stats['emotions'].get(emotion_name, 0) + 1
                            else:
                                emotions.append(None)
                        
                        # Detecta atividades
                        activities = activity_detector.detect(frame)
                        for activity in activities:
                            activity_name = activity.activity_pt if hasattr(activity, 'activity_pt') else str(activity)
                            stats['activities'][activity_name] = stats['activities'].get(activity_name, 0) + 1
                        
                        # Detecta anomalias
                        anomalies = anomaly_detector.detect(frame, faces, stats)
                        for anomaly in anomalies:
                            anomaly_name = anomaly.type if hasattr(anomaly, 'type') else str(anomaly)
                            stats['anomalies'][anomaly_name] = stats['anomalies'].get(anomaly_name, 0) + 1
                        
                        # Visualiza
                        processed_frame = draw_detections(frame, faces, emotions, activities, anomalies)
                    
                    except Exception as e:
                        print(f"[WARN] Erro ao processar frame {frame_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Escreve frame
                out.write(processed_frame)
                
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
