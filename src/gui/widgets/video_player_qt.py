"""
Player de vídeo com PyQt6 usando OpenCV
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from collections import deque
from enum import Enum


class PlayerMode(Enum):
    """Modos de operação do player."""
    IDLE = "idle"
    READY = "ready"
    PROCESSING = "processing"
    PLAYBACK = "playback"


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
        self.playback_speed = 1.0  # Velocidade de reprodução (1.0 = normal)
        self.is_seeking = False    # Flag para detectar se está fazendo seek
        
        # Preview mode
        self.mode = PlayerMode.IDLE
        self.preview_buffer = deque(maxlen=30)
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._show_next_preview_frame)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display
        self.video_label = QLabel("[○] Carregue um vídeo para começar")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #0d0d0d;
                color: #666;
                border: 1px solid #333;
                font-size: 14px;
            }
        """)
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setScaledContents(False)
        layout.addWidget(self.video_label, stretch=1)
        
        # Status overlay (modo de processamento)
        self.status_overlay = QLabel()
        self.status_overlay.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.status_overlay.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: #4CAF50;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.status_overlay.hide()
        layout.addWidget(self.status_overlay)
        
        # Controles
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Botão Play/Pause
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)
        
        # Seek bar
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(100)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        self.seek_slider.setEnabled(False)
        controls_layout.addWidget(self.seek_slider, stretch=1)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(120)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self.time_label)
        
        # Controle de Velocidade
        speed_label = QLabel("Velocidade:")
        controls_layout.addWidget(speed_label)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "0.75x", "1.0x", "1.25x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.speed_combo.setEnabled(False)
        self.speed_combo.setMaximumWidth(80)
        controls_layout.addWidget(self.speed_combo)
        
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
        
        # Atualiza modo
        self.mode = PlayerMode.READY
        self.status_overlay.hide()
        
        # Atualiza controles
        self.play_btn.setEnabled(True)
        self.seek_slider.setEnabled(True)
        self.speed_combo.setEnabled(True)
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
        self.play_btn.setText("Pausar")
        
        # Calcula intervalo do timer considerando a velocidade
        interval = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 33
        self.timer.start(interval)
    
    def pause(self):
        """Pausa reprodução."""
        self.is_playing = False
        self.play_btn.setText("Play")
        self.timer.stop()
    
    def stop(self):
        """Para reprodução."""
        self.pause()
        self.current_frame_idx = 0
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._update_frame()
    
    def _on_slider_moved(self, position):
        """Callback quando o slider é movido manualmente (pausa automática)."""
        # Define flag para indicar que estamos fazendo seek
        self.is_seeking = True
        
        # Pausa reprodução automaticamente
        was_playing = self.is_playing
        if self.is_playing:
            self.pause()
        
        # Realiza o seek
        self._seek(position)
        
        # Reset flag
        self.is_seeking = False
    
    def _seek(self, position):
        """Pula para posição."""
        if not self.video_capture:
            return
        
        self.current_frame_idx = position
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        self._update_frame()
    
    def _on_speed_changed(self, index):
        """Callback quando a velocidade é alterada."""
        speed_text = self.speed_combo.currentText()
        # Extrai o valor numérico (ex: "1.5x" -> 1.5)
        speed_value = float(speed_text.replace("x", ""))
        self.playback_speed = speed_value
        
        # Se está reproduzindo, reinicia o timer com novo intervalo
        if self.is_playing:
            self.timer.stop()
            interval = int(1000 / (self.fps * self.playback_speed)) if self.fps > 0 else 33
            self.timer.start(interval)
    
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
    
    # ===== Métodos de Preview em Tempo Real =====
    
    def enable_preview_mode(self, preview_fps=10, total_frames=0):
        """Ativa modo preview durante processamento."""
        self.mode = PlayerMode.PROCESSING
        self.preview_buffer.clear()
        
        # Pausa reprodução normal se estiver ativa
        if self.is_playing:
            self.pause()
        
        # Configura total de frames para o slider
        if total_frames > 0:
            self.total_frames = total_frames
            self.seek_slider.setMaximum(total_frames - 1)
        
        # Mantém slider habilitado para visualização do progresso (mas não seek)
        self.play_btn.setEnabled(False)
        self.seek_slider.setEnabled(True)  # Habilitado para mostrar progresso
        
        # Mostra overlay
        self.status_overlay.setText("PREVIEW TEMPO REAL")
        self.status_overlay.show()
        
        # Inicia timer de preview
        interval = int(1000 / preview_fps)
        self.preview_timer.start(interval)
    
    def disable_preview_mode(self):
        """Desativa modo preview."""
        self.preview_timer.stop()
        self.preview_buffer.clear()
        self.status_overlay.hide()
        
        if self.mode == PlayerMode.PROCESSING:
            self.mode = PlayerMode.READY
    
    def add_preview_frame(self, frame_idx, frame):
        """Adiciona frame processado ao buffer de preview."""
        if self.mode != PlayerMode.PROCESSING:
            return
        
        # Adiciona ao buffer (deque descarta automaticamente os mais antigos)
        self.preview_buffer.append((frame_idx, frame.copy()))
    
    def _show_next_preview_frame(self):
        """Mostra próximo frame do buffer de preview."""
        if not self.preview_buffer:
            return
        
        # Pega frame mais recente
        frame_idx, frame = self.preview_buffer[-1]
        
        # Atualiza display
        self.current_frame_idx = frame_idx
        self._display_frame(frame)
        
        # Atualiza slider para mostrar progresso
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_idx)
        self.seek_slider.blockSignals(False)
        
        # Atualiza time label
        self._update_time_label()
        
        # Atualiza overlay
        buffer_size = len(self.preview_buffer)
        self.status_overlay.setText(
            f"PREVIEW TEMPO REAL\n"
            f"Frame: {frame_idx}\n"
            f"Buffer: {buffer_size}/30"
        )
    
    def switch_to_playback_mode(self):
        """Muda para modo de reprodução após processamento."""
        self.disable_preview_mode()
        self.mode = PlayerMode.PLAYBACK
        
        # Reabilita controles
        self.play_btn.setEnabled(True)
        self.seek_slider.setEnabled(True)
