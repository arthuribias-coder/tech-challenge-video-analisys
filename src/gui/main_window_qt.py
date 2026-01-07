"""
Janela principal da aplicação GUI com PyQt6
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QProgressBar,
    QFileDialog, QMessageBox, QLabel, QSpinBox,
    QPushButton, QToolBar
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QAction, QIcon, QFontDatabase
from pathlib import Path

from .widgets import VideoPlayerQt, StatsPanelQt, ChartsPanelQt
from .threads import ProcessorThreadQt
from ..config import OUTPUT_DIR


class MainWindow(QMainWindow):
    """Janela principal da aplicação Qt."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tech Challenge - Fase 4: Análise de Vídeo com IA")
        self.resize(1400, 900)
        
        # Configurar suporte a emojis (Qt 6.9+)
        try:
            # Tenta usar fontes do sistema que suportam emojis
            emoji_fonts = ["Noto Color Emoji", "Apple Color Emoji", "Segoe UI Emoji", "Twitter Color Emoji"]
            for font in emoji_fonts:
                if font in QFontDatabase.families():
                    QFontDatabase.addApplicationEmojiFontFamily(font)
                    break
        except AttributeError:
            # Fallback para Qt < 6.9
            pass
        
        # Estado
        self.video_path = None
        self.output_path = None
        self.processor_thread = None
        
        # Aplica estilo dark
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: 1px solid #555;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        
        self._setup_ui()
        self._setup_toolbar()
        self._setup_statusbar()
    
    def _setup_ui(self):
        """Configura interface."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top: Video + Stats
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)
        
        self.video_player = VideoPlayerQt()
        top_layout.addWidget(self.video_player, stretch=7)
        
        self.stats_panel = StatsPanelQt()
        top_layout.addWidget(self.stats_panel, stretch=3)
        
        main_layout.addLayout(top_layout, stretch=7)
        
        # Bottom: Charts
        self.charts_panel = ChartsPanelQt()
        main_layout.addWidget(self.charts_panel, stretch=3)
    
    def _setup_toolbar(self):
        """Cria toolbar com botões."""
        toolbar = QToolBar("Controles Principais")
        toolbar.setIconSize(QSize(32, 32))
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2d2d2d;
                border-bottom: 2px solid #3d3d3d;
                spacing: 5px;
                padding: 5px;
            }
            QToolButton {
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
                min-width: 120px;
            }
            QToolButton:hover {
                background-color: #4d4d4d;
                border: 1px solid #777;
            }
            QToolButton:pressed {
                background-color: #2d2d2d;
            }
            QToolButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
        """)
        self.addToolBar(toolbar)
        
        # Botão Abrir Vídeo
        open_action = QAction("📂 Abrir Vídeo", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_video)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # Botão Processar
        self.start_action = QAction("▶️ Processar", self)
        self.start_action.triggered.connect(self._start_processing)
        toolbar.addAction(self.start_action)
        
        # Botão Pausar
        self.pause_action = QAction("⏸️ Pausar", self)
        self.pause_action.triggered.connect(self._pause_processing)
        self.pause_action.setEnabled(False)
        toolbar.addAction(self.pause_action)
        
        # Botão Parar
        self.stop_action = QAction("⏹️ Parar", self)
        self.stop_action.triggered.connect(self._stop_processing)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)
        
        toolbar.addSeparator()
        
        # Botão Salvar
        save_action = QAction("💾 Salvar Vídeo", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_video)
        toolbar.addAction(save_action)
        
        # Botão Exportar
        export_action = QAction("📊 Exportar Relatório", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_report)
        toolbar.addAction(export_action)
        
        # Espaçador
        spacer = QWidget()
        spacer.setSizePolicy(toolbar.sizePolicy().MinimumExpanding, toolbar.sizePolicy().Preferred)
        toolbar.addWidget(spacer)
        
        # Botão Sobre
        about_action = QAction("ℹ️ Sobre", self)
        about_action.triggered.connect(self._show_about)
        toolbar.addAction(about_action)
    
    def _setup_statusbar(self):
        """Cria barra de status."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Label de status
        self.status_label = QLabel("Pronto | Aguardando vídeo...")
        self.statusbar.addWidget(self.status_label, stretch=1)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        # FPS
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setMinimumWidth(80)
        self.statusbar.addPermanentWidget(self.fps_label)
    
    def _open_video(self):
        """Abre vídeo."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione o vídeo",
            "",
            "Vídeos (*.mp4 *.avi *.mov *.mkv);;Todos (*.*)"
        )
        
        if filename:
            self.video_path = Path(filename)
            if self.video_player.load_video(str(filename)):
                self.status_label.setText(f"Vídeo carregado: {self.video_path.name}")
                self.stats_panel.reset()
                self.charts_panel.clear_data()
            else:
                QMessageBox.critical(self, "Erro", "Não foi possível carregar o vídeo!")
    
    def _start_processing(self):
        """Inicia processamento."""
        if not self.video_path:
            QMessageBox.warning(self, "Aviso", "Selecione um vídeo primeiro!")
            return
        
        if self.processor_thread and self.processor_thread.isRunning():
            QMessageBox.information(self, "Info", "Processamento já em andamento!")
            return
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        self.output_path = OUTPUT_DIR / f"analisado_{self.video_path.name}"
        
        self.processor_thread = ProcessorThreadQt(
            str(self.video_path),
            str(self.output_path),
            frame_skip=2
        )
        
        # Conecta signals
        self.processor_thread.progress.connect(self._on_progress)
        self.processor_thread.finished_signal.connect(self._on_complete)
        self.processor_thread.error.connect(self._on_error)
        
        self.processor_thread.start()
        
        # Atualiza UI
        self.start_action.setEnabled(False)
        self.pause_action.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.status_label.setText("Processando vídeo...")
        self.progress_bar.setValue(0)
    
    def _pause_processing(self):
        """Pausa/retoma processamento."""
        if self.processor_thread:
            self.processor_thread.toggle_pause()
            if self.processor_thread.is_paused:
                self.status_label.setText("⏸️ Processamento pausado")
                self.pause_action.setText("▶️ Retomar")
            else:
                self.status_label.setText("⚙️ Processando vídeo...")
                self.pause_action.setText("⏸️ Pausar")
    
    def _stop_processing(self):
        """Para processamento."""
        if self.processor_thread:
            self.processor_thread.stop()
            self.processor_thread.wait()
            
            # Atualiza UI
            self.start_action.setEnabled(True)
            self.pause_action.setEnabled(False)
            self.stop_action.setEnabled(False)
            self.status_label.setText("Processamento cancelado")
            self.progress_bar.setValue(0)
    
    def _save_video(self):
        """Salva vídeo."""
        if not self.output_path or not self.output_path.exists():
            QMessageBox.warning(self, "Aviso", "Nenhum vídeo processado disponível!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar vídeo como",
            str(self.output_path.name),
            "MP4 (*.mp4);;Todos (*.*)"
        )
        
        if filename:
            import shutil
            shutil.copy(self.output_path, filename)
            QMessageBox.information(self, "Sucesso", f"Vídeo salvo em:\n{filename}")
    
    def _export_report(self):
        """Exporta relatório."""
        if not hasattr(self, 'last_stats') or not self.last_stats:
            QMessageBox.warning(self, "Aviso", "Processe um vídeo primeiro!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar relatório",
            "relatorio.txt",
            "Texto (*.txt);;Todos (*.*)"
        )
        
        if filename:
            from ..report_generator import ReportGenerator
            generator = ReportGenerator()
            report = generator.generate(self.last_stats)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            QMessageBox.information(self, "Sucesso", f"Relatório exportado:\n{filename}")
    
    def _show_about(self):
        """Mostra sobre."""
        QMessageBox.about(
            self,
            "Sobre",
            "Tech Challenge - Fase 4\n"
            "Análise de Vídeo com IA\n\n"
            "Desenvolvido com:\n"
            "- OpenCV\n"
            "- FER (Facial Expression Recognition)\n"
            "- YOLO11-pose (Ultralytics)\n"
            "- PyQt6\n\n"
            "Versão 3.0.0"
        )
    
    def _on_progress(self, frame_idx, total_frames, fps, stats):
        """Callback de progresso."""
        progress = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
        self.progress_bar.setValue(progress)
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.status_label.setText(f"Processando... {progress}% | FPS: {fps:.1f}")
        
        # Atualiza painéis
        self.stats_panel.update_stats(stats)
        self.charts_panel.update_data(stats)
        
        # Armazena stats
        self.last_stats = stats
    
    def _on_complete(self, stats, elapsed_time):
        """Callback de conclusão."""
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Processamento concluído em {elapsed_time:.1f}s")
        
        # Atualiza painéis
        self.stats_panel.update_stats(stats)
        self.charts_panel.update_data(stats)
        self.last_stats = stats
        
        # Carrega vídeo processado
        if self.output_path.exists():
            self.video_player.load_video(str(self.output_path))
        
        # Atualiza UI
        self.start_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        
        QMessageBox.information(
            self,
            "Concluído",
            f"Processamento finalizado!\n\n"
            f"Tempo: {elapsed_time:.1f}s\n"
            f"Faces: {stats.get('faces', 0)}\n"
            f"Vídeo salvo em:\n{self.output_path}"
        )
    
    def _on_error(self, error_msg):
        """Callback de erro."""
        self.status_label.setText(f"Erro: {error_msg}")
        
        # Atualiza UI
        self.start_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        
        QMessageBox.critical(self, "Erro no Processamento", error_msg)
    
    def closeEvent(self, event):
        """Handler de fechamento."""
        if self.processor_thread and self.processor_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Sair",
                "Processamento em andamento. Deseja realmente sair?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._stop_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
