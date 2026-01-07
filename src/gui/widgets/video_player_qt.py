"""
Player de vídeo com PyQt6 usando OpenCV
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSlider
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2


class VideoPlayerQt(QWidget):
    """Player de vídeo usando OpenCV."""
    
    frame_changed = pyqtSignal(int)  # Signal para mudança de frame
    
    def __init__(self):
        super().__init__()
        
        self.video_capture = None
        self.current_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30
        self.is_playing = False
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display
        self.video_label = QLabel("🎥 Carregue um vídeo para começar")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #0d0d0d;
                color: #666;
                border: 1px solid #333;
                font-size: 14px;
            }
        """)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label, stretch=1)
        
        # Controles
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Botão Play/Pause
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)
        
        # Seek bar
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(100)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderMoved.connect(self._seek)
        self.seek_slider.setEnabled(False)
        controls_layout.addWidget(self.seek_slider, stretch=1)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(120)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.time_label)
        
        layout.addLayout(controls_layout)
    
    def load_video(self, video_path):
        """Carrega vídeo."""
        if self.video_capture:
            self.video_capture.release()
        
        self.video_capture = cv2.VideoCapture(video_path)
        
        if not self.video_capture.isOpened():
            return False
        
        # Obtém informações do vídeo
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0
        
        # Atualiza controles
        self.play_btn.setEnabled(True)
        self.seek_slider.setEnabled(True)
        self.seek_slider.setMaximum(self.total_frames - 1)
        
        # Exibe primeiro frame
        self._update_frame()
        
        return True
    
    def _toggle_play(self):
        """Play/Pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """Inicia reprodução."""
        if not self.video_capture:
            return
        
        self.is_playing = True
        self.play_btn.setText("⏸️ Pausar")
        
        # Calcula intervalo do timer (ms)
        interval = int(1000 / self.fps) if self.fps > 0 else 33
        self.timer.start(interval)
    
    def pause(self):
        """Pausa reprodução."""
        self.is_playing = False
        self.play_btn.setText("▶️ Play")
        self.timer.stop()
    
    def stop(self):
        """Para reprodução."""
        self.pause()
        self.current_frame_idx = 0
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._update_frame()
    
    def _seek(self, position):
        """Pula para posição."""
        if not self.video_capture:
            return
        
        self.current_frame_idx = position
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        self._update_frame()
    
    def _update_frame(self):
        """Atualiza frame."""
        if not self.video_capture or not self.video_capture.isOpened():
            return
        
        ret, frame = self.video_capture.read()
        
        if ret:
            self.current_frame = frame
            self.current_frame_idx = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            
            self._display_frame(frame)
            self._update_time_label()
            
            # Atualiza seek bar
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.current_frame_idx)
            self.seek_slider.blockSignals(False)
            
            self.frame_changed.emit(self.current_frame_idx)
        else:
            # Fim do vídeo
            self.pause()
            self.current_frame_idx = 0
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def _display_frame(self, frame):
        """Exibe frame."""
        # Converte BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Cria QImage
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Escala para caber no label mantendo aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def _update_time_label(self):
        """Atualiza label de tempo."""
        if not self.video_capture or self.fps == 0:
            return
        
        current_sec = int(self.current_frame_idx / self.fps)
        total_sec = int(self.total_frames / self.fps)
        
        current_time = f"{current_sec // 60:02d}:{current_sec % 60:02d}"
        total_time = f"{total_sec // 60:02d}:{total_sec % 60:02d}"
        
        self.time_label.setText(f"{current_time} / {total_time}")
    
    def resizeEvent(self, event):
        """Redimensiona frame ao redimensionar janela."""
        super().resizeEvent(event)
        if self.current_frame is not None:
            self._display_frame(self.current_frame)
