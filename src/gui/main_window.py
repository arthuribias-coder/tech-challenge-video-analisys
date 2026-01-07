"""
Janela principal da aplicação GUI
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
import os

from .widgets import VideoPlayer, StatsPanel, ChartsPanel
from .threads import ProcessorThread
from ..config import OUTPUT_DIR


class VideoAnalyzerGUI(ctk.CTk):
    """Janela principal da aplicação."""
    
    def __init__(self):
        super().__init__()
        
        # Configurações da janela
        self.title("Tech Challenge - Fase 4: Análise de Vídeo com IA")
        self.geometry("1400x900")
        
        # Define tema
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Estado
        self.video_path = None
        self.output_path = None
        self.processor_thread = None
        self.is_processing = False
        
        # Setup UI
        self._setup_ui()
        
        # Configura protocolo de fechamento
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_ui(self):
        """Configura interface."""
        # Menu bar (simulado com frame)
        menubar = self._create_menubar()
        menubar.pack(fill="x", side="top")
        
        # Container principal
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Layout: Video + Stats (top) | Charts (bottom)
        top_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        top_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Video Player (esquerda)
        video_frame = ctk.CTkFrame(top_frame)
        video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.video_player = VideoPlayer(video_frame)
        self.video_player.pack(fill="both", expand=True)
        
        # Stats Panel (direita)
        self.stats_panel = StatsPanel(top_frame, width=300)
        self.stats_panel.pack(side="right", fill="y")
        
        # Controles
        controls_frame = self._create_controls()
        controls_frame.pack(fill="x", pady=(0, 10))
        
        # Charts Panel (bottom)
        self.charts_panel = ChartsPanel(main_container, height=350)
        self.charts_panel.pack(fill="both")
        
        # Status bar
        self.status_bar = ctk.CTkLabel(
            self,
            text="Pronto | Aguardando vídeo...",
            anchor="w",
            height=25
        )
        self.status_bar.pack(fill="x", side="bottom", padx=10, pady=5)
    
    def _create_menubar(self):
        """Cria barra de menu."""
        menubar = ctk.CTkFrame(self, height=40, fg_color="#1f1f1f")
        
        # Botões de menu
        menu_items = [
            ("[+] Abrir Vídeo", self._open_video),
            ("[>] Processar", self._start_processing),
            ("[||] Pausar", self._pause_processing),
            ("[■] Parar", self._stop_processing),
            ("[*] Salvar Vídeo", self._save_video),
            ("[?] Sobre", self._show_about)
        ]
        
        for text, command in menu_items:
            btn = ctk.CTkButton(
                menubar,
                text=text,
                width=120,
                height=30,
                command=command
            )
            btn.pack(side="left", padx=5, pady=5)
        
        return menubar
    
    def _create_controls(self):
        """Cria painel de controles."""
        frame = ctk.CTkFrame(self.master)
        
        # Barra de progresso
        self.progress_bar = ctk.CTkProgressBar(frame, width=400)
        self.progress_bar.pack(side="left", padx=10, fill="x", expand=True)
        self.progress_bar.set(0)
        
        # Label de progresso
        self.progress_label = ctk.CTkLabel(
            frame,
            text="0 / 0 frames (0%)",
            width=200
        )
        self.progress_label.pack(side="left", padx=10)
        
        # FPS
        self.fps_label = ctk.CTkLabel(frame, text="FPS: --", width=100)
        self.fps_label.pack(side="left", padx=10)
        
        # Frame skip
        skip_frame = ctk.CTkFrame(frame, fg_color="transparent")
        skip_frame.pack(side="left", padx=10)
        
        skip_label = ctk.CTkLabel(skip_frame, text="Skip:")
        skip_label.pack(side="left", padx=(0, 5))
        
        self.skip_var = ctk.StringVar(value="2")
        self.skip_entry = ctk.CTkEntry(skip_frame, width=50, textvariable=self.skip_var)
        self.skip_entry.pack(side="left")
        
        return frame
    
    def _open_video(self):
        """Abre diálogo para selecionar vídeo."""
        filetypes = [
            ("Arquivos de Vídeo", "*.mp4 *.avi *.mov *.mkv"),
            ("Todos os arquivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Selecione o vídeo",
            filetypes=filetypes
        )
        
        if filename:
            self.video_path = Path(filename)
            self._load_video()
    
    def _load_video(self):
        """Carrega vídeo no player."""
        if not self.video_path or not self.video_path.exists():
            messagebox.showerror("Erro", "Vídeo não encontrado!")
            return
        
        success = self.video_player.load_video(self.video_path)
        
        if success:
            self.status_bar.configure(
                text=f"Vídeo carregado: {self.video_path.name}"
            )
            # Reseta estatísticas
            self.stats_panel.reset()
        else:
            messagebox.showerror("Erro", "Não foi possível carregar o vídeo!")
    
    def _start_processing(self):
        """Inicia processamento do vídeo."""
        if not self.video_path:
            messagebox.showwarning(
                "Aviso",
                "Selecione um vídeo primeiro!"
            )
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Processamento já em andamento!")
            return
        
        # Define caminho de saída
        OUTPUT_DIR.mkdir(exist_ok=True)
        self.output_path = OUTPUT_DIR / "video_analisado.mp4"
        
        # Frame skip
        try:
            frame_skip = int(self.skip_var.get())
        except ValueError:
            frame_skip = 2
        
        # Cria e inicia thread
        self.processor_thread = ProcessorThread(
            self.video_path,
            self.output_path,
            frame_skip=frame_skip
        )
        
        # Configura callbacks
        self.processor_thread.on_progress = self._on_progress
        self.processor_thread.on_complete = self._on_complete
        self.processor_thread.on_error = self._on_error
        
        self.processor_thread.start()
        self.is_processing = True
        
        self.status_bar.configure(text="Processando vídeo...")
    
    def _pause_processing(self):
        """Pausa/retoma processamento."""
        if not self.processor_thread or not self.is_processing:
            return
        
        if self.processor_thread.is_paused:
            self.processor_thread.resume()
            self.status_bar.configure(text="Processando vídeo...")
        else:
            self.processor_thread.pause()
            self.status_bar.configure(text="Processamento pausado")
    
    def _stop_processing(self):
        """Para processamento."""
        if self.processor_thread and self.is_processing:
            self.processor_thread.stop()
            self.processor_thread.join(timeout=2.0)
            self.is_processing = False
            self.status_bar.configure(text="Processamento cancelado")
    
    def _save_video(self):
        """Salva vídeo processado."""
        if not self.output_path or not self.output_path.exists():
            messagebox.showwarning(
                "Aviso",
                "Nenhum vídeo processado disponível!"
            )
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Salvar vídeo como",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("Todos", "*.*")]
        )
        
        if save_path:
            import shutil
            shutil.copy(self.output_path, save_path)
            messagebox.showinfo("Sucesso", f"Vídeo salvo em:\n{save_path}")
    
    def _show_about(self):
        """Mostra janela sobre."""
        messagebox.showinfo(
            "Sobre",
            "Tech Challenge - Fase 4\n"
            "Análise de Vídeo com IA\n\n"
            "Desenvolvido com:\n"
            "• OpenCV\n"
            "• FER (Facial Expression Recognition)\n"
            "• YOLO11-pose (Ultralytics)\n"
            "• CustomTkinter"
        )
    
    def _on_progress(self, frame_idx, total_frames, fps, stats):
        """Callback de progresso."""
        # Atualiza barra de progresso
        progress = frame_idx / total_frames if total_frames > 0 else 0
        self.progress_bar.set(progress)
        
        # Atualiza labels
        percent = int(progress * 100)
        self.progress_label.configure(
            text=f"{frame_idx} / {total_frames} frames ({percent}%)"
        )
        self.fps_label.configure(text=f"FPS: {fps:.1f}")
        
        # Atualiza estatísticas
        self.stats_panel.update_stats(stats)
        self.charts_panel.update_data(stats)
        
        # Atualiza status
        self.status_bar.configure(
            text=f"Processando... {percent}% | FPS: {fps:.1f}"
        )
    
    def _on_complete(self, stats, elapsed_time):
        """Callback de conclusão."""
        self.is_processing = False
        
        # Atualiza UI
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="Concluído!")
        self.stats_panel.update_stats(stats)
        self.charts_panel.update_data(stats)
        
        # Carrega vídeo processado no player
        if self.output_path.exists():
            self.video_player.load_video(self.output_path)
        
        # Status
        self.status_bar.configure(
            text=f"Processamento concluído em {elapsed_time:.1f}s | "
                 f"Vídeo salvo: {self.output_path.name}"
        )
        
        messagebox.showinfo(
            "Concluído",
            f"Processamento finalizado!\n\n"
            f"Tempo: {elapsed_time:.1f}s\n"
            f"Faces: {stats['faces']}\n"
            f"Vídeo salvo em:\n{self.output_path}"
        )
    
    def _on_error(self, error_msg):
        """Callback de erro."""
        self.is_processing = False
        self.status_bar.configure(text=f"Erro: {error_msg}")
        messagebox.showerror("Erro no Processamento", error_msg)
    
    def _on_closing(self):
        """Handler de fechamento da janela."""
        if self.is_processing:
            if messagebox.askokcancel(
                "Sair",
                "Processamento em andamento. Deseja realmente sair?"
            ):
                self._stop_processing()
                self.destroy()
        else:
            self.destroy()


def main():
    """Função principal para executar a GUI."""
    app = VideoAnalyzerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
