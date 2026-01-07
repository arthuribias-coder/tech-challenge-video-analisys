"""
Widget de reprodução de vídeo usando OpenCV e CustomTkinter
"""

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
from pathlib import Path


class VideoPlayer(ctk.CTkFrame):
    """Widget para exibir e controlar vídeo processado."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30.0
        self.playback_thread = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura interface do player."""
        # Canvas para vídeo
        self.canvas = ctk.CTkCanvas(
            self,
            width=640,
            height=480,
            bg='#2b2b2b',
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Texto placeholder
        self.placeholder_text = self.canvas.create_text(
            320, 240,
            text="Nenhum vídeo carregado\n\nClique em 'Abrir Vídeo' para começar",
            fill="#888888",
            font=("Arial", 14),
            justify="center"
        )
    
    def load_video(self, video_path):
        """
        Carrega vídeo no player.
        
        Args:
            video_path: Caminho do arquivo de vídeo
        """
        self.stop()
        
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame_idx = 0
        
        # Remove placeholder
        if self.placeholder_text:
            self.canvas.delete(self.placeholder_text)
            self.placeholder_text = None
        
        # Mostra primeiro frame
        self._show_frame(0)
        
        return True
    
    def play(self):
        """Inicia reprodução."""
        if not self.cap or self.is_playing:
            return
        
        self.is_playing = True
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
    
    def pause(self):
        """Pausa reprodução."""
        self.is_playing = False
    
    def stop(self):
        """Para reprodução e reseta."""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.current_frame_idx = 0
    
    def seek(self, frame_idx):
        """
        Pula para frame específico.
        
        Args:
            frame_idx: Índice do frame
        """
        if not self.cap:
            return
        
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.current_frame_idx = frame_idx
        self._show_frame(frame_idx)
    
    def _playback_loop(self):
        """Loop de reprodução em thread separada."""
        delay = 1.0 / self.fps
        
        while self.is_playing and self.cap:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                # Fim do vídeo - volta ao início
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_idx = 0
                continue
            
            self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._display_frame(frame)
            
            # Controle de FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, delay - elapsed)
            time.sleep(sleep_time)
    
    def _show_frame(self, frame_idx):
        """Mostra frame específico sem reproduzir."""
        if not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self._display_frame(frame)
    
    def _display_frame(self, frame):
        """Renderiza frame no canvas."""
        # Converte BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensiona para caber no canvas mantendo proporção
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 640, 480
        
        h, w = frame_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Converte para ImageTk
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=image)
        
        # Atualiza canvas
        self.canvas.delete("all")
        x = (canvas_width - new_w) // 2
        y = (canvas_height - new_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=photo)
        
        # Mantém referência para evitar garbage collection
        self.canvas.image = photo
    
    def get_current_position(self):
        """Retorna posição atual (frame, tempo)."""
        if not self.cap:
            return 0, 0.0
        
        current_time = self.current_frame_idx / self.fps if self.fps > 0 else 0
        return self.current_frame_idx, current_time
    
    def get_duration(self):
        """Retorna duração total (frames, tempo)."""
        if not self.cap:
            return 0, 0.0
        
        duration = self.total_frames / self.fps if self.fps > 0 else 0
        return self.total_frames, duration
