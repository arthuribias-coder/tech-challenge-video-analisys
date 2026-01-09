"""
Janela principal da aplicação GUI com PyQt6
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QProgressBar,
    QFileDialog, QMessageBox, QLabel, QSpinBox,
    QPushButton, QToolBar, QSizePolicy, QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QAction, QIcon, QFontDatabase
from pathlib import Path

from .widgets import VideoPlayerQt, StatsPanelQt, ChartsPanelQt, SettingsDialog
from .threads import ProcessorThreadQt
from .icon_provider import IconProvider
from ..config import (
    OUTPUT_DIR, FRAME_SKIP, TARGET_FPS, ENABLE_PREVIEW, PREVIEW_FPS,
    ENABLE_OBJECT_DETECTION, VIDEO_PATH, USE_GPU, YOLO_MODEL_SIZE
)


class MainWindow(QMainWindow):
    """Janela principal da aplicação Qt."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tech Challenge - Fase 4: Análise de Vídeo com IA")
        self.resize(1600, 950)
        self.setMinimumSize(1200, 700)
        
        # Estado
        self.video_path = None
        self.output_path = None
        self.processor_thread = None
        
        # Configurações de processamento padrão
        self.processing_settings = {
            'frame_skip': FRAME_SKIP,
            'target_fps': TARGET_FPS,
            'enable_preview': ENABLE_PREVIEW,
            'preview_fps': PREVIEW_FPS,
            'enable_object_detection': ENABLE_OBJECT_DETECTION,
            'use_gpu': USE_GPU,
            'model_size': YOLO_MODEL_SIZE
        }
        
        # Define vídeo padrão se disponível
        if VIDEO_PATH and Path(VIDEO_PATH).exists():
            self.video_path = Path(VIDEO_PATH)
        
        # Aplica estilo dark
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QSplitter::handle {
                background-color: #3d3d3d;
                margin: 2px;
            }
            QSplitter::handle:horizontal {
                width: 4px;
            }
            QSplitter::handle:vertical {
                height: 4px;
            }
            QSplitter::handle:hover {
                background-color: #4CAF50;
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
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Splitter horizontal principal (video à esquerda / painel direito)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Esquerda: Video Player
        self.video_player = VideoPlayerQt()
        self.video_player.setMinimumWidth(500)
        main_splitter.addWidget(self.video_player)
        
        # Direita: Painel com Stats + Charts (em scroll area)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Stats Panel
        self.stats_panel = StatsPanelQt()
        self.stats_panel.setMinimumHeight(300)
        right_layout.addWidget(self.stats_panel)
        
        # Charts Panel (abaixo das estatísticas)
        self.charts_panel = ChartsPanelQt()
        self.charts_panel.setMinimumHeight(350)
        right_layout.addWidget(self.charts_panel)
        
        # Adiciona espaçador para empurrar conteúdo para cima
        right_layout.addStretch()
        
        # Scroll area para o painel direito
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(right_panel)
        scroll_area.setMinimumWidth(350)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        main_splitter.addWidget(scroll_area)
        
        # Define proporções iniciais do splitter
        main_splitter.setStretchFactor(0, 6)  # Video (60%)
        main_splitter.setStretchFactor(1, 4)  # Stats + Charts (40%)
        
        main_layout.addWidget(main_splitter)
    
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
        open_action = QAction(IconProvider.document_open(), "Abrir Vídeo", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setToolTip("Abrir vídeo para análise (Ctrl+O)")
        open_action.triggered.connect(self._open_video)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # Botão Configurações
        settings_action = QAction("Configuracoes", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.setToolTip("Configurar processamento (Ctrl+,)")
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)
        
        toolbar.addSeparator()
        
        # Botão Processar
        self.start_action = QAction(IconProvider.media_play(), "Processar", self)
        self.start_action.setToolTip("Iniciar processamento do vídeo")
        self.start_action.triggered.connect(self._start_processing)
        toolbar.addAction(self.start_action)
        
        # Botão Pausar
        self.pause_action = QAction(IconProvider.media_pause(), "Pausar", self)
        self.pause_action.setToolTip("Pausar processamento")
        self.pause_action.triggered.connect(self._pause_processing)
        self.pause_action.setEnabled(False)
        toolbar.addAction(self.pause_action)
        
        # Botão Parar
        self.stop_action = QAction(IconProvider.media_stop(), "Parar", self)
        self.stop_action.setToolTip("Parar processamento")
        self.stop_action.triggered.connect(self._stop_processing)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)
        
        toolbar.addSeparator()
        
        # Botão Salvar
        save_action = QAction(IconProvider.document_save(), "Salvar Vídeo", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setToolTip("Salvar vídeo processado (Ctrl+S)")
        save_action.triggered.connect(self._save_video)
        toolbar.addAction(save_action)
        
        # Botão Exportar
        export_action = QAction(IconProvider.chart_bar(), "Exportar Relatório", self)
        export_action.setShortcut("Ctrl+E")
        export_action.setToolTip("Exportar relatório de análise (Ctrl+E)")
        export_action.triggered.connect(self._export_report)
        toolbar.addAction(export_action)
        
        # Espaçador
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        # Botão Sobre
        about_action = QAction(IconProvider.help_about(), "Sobre", self)
        about_action.setToolTip("Informações sobre o aplicativo")
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
    def _open_settings(self):
        """Abre dialog de configurações."""
        dialog = SettingsDialog(self, self.processing_settings)
        dialog.settings_applied.connect(self._on_settings_changed)
        dialog.exec()
    
    def _on_settings_changed(self, settings):
        """Callback quando configurações são alteradas."""
        self.processing_settings = settings
        self.status_label.setText(
            f"Configuracoes atualizadas: Skip={settings['frame_skip']}, "
            f"FPS={settings['target_fps']}, Preview={'ON' if settings['enable_preview'] else 'OFF'}"
        )    
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
        
        # Usa configuracoes armazenadas
        settings = self.processing_settings
        
        self.processor_thread = ProcessorThreadQt(
            str(self.video_path),
            str(self.output_path),
            frame_skip=settings['frame_skip'],
            target_fps=settings['target_fps'],
            enable_preview=settings['enable_preview'],
            preview_fps=settings['preview_fps'],
            # Detectores avançados
            enable_object_detection=settings.get('enable_object_detection'),
            enable_overlay_detection=settings.get('enable_overlay_detection'),
            enable_segment_validation=settings.get('enable_segment_validation'),
            # Configurações de hardware
            use_gpu=settings.get('use_gpu'),
            model_size=settings.get('model_size')
        )
        
        # Conecta signals
        self.processor_thread.progress.connect(self._on_progress)
        self.processor_thread.finished_signal.connect(self._on_complete)
        self.processor_thread.error.connect(self._on_error)
        self.processor_thread.frame_processed.connect(self._on_frame_processed)
        
        # Ativa modo preview no player se habilitado
        if settings['enable_preview']:
            # Passa total de frames para o slider funcionar durante preview
            total_frames = self.video_player.total_frames if self.video_player.total_frames > 0 else 0
            self.video_player.enable_preview_mode(settings['preview_fps'], total_frames)
        
        self.processor_thread.start()
        
        # Atualiza UI
        self.start_action.setEnabled(False)
        self.pause_action.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.status_label.setText("Processando video...")
        self.progress_bar.setValue(0)
    
    def _pause_processing(self):
        """Pausa/retoma processamento."""
        if self.processor_thread:
            self.processor_thread.toggle_pause()
            if self.processor_thread.is_paused:
                self.status_label.setText("[❚❚] Processamento pausado")
                self.pause_action.setText("[▶] Retomar")
            else:
                self.status_label.setText("[⚙] Processando vídeo...")
                self.pause_action.setText("[❚❚] Pausar")
    
    def _stop_processing(self):
        """Para processamento."""
        if self.processor_thread:
            self.processor_thread.stop()
            self.processor_thread.wait()
            
            # Desativa preview
            self.video_player.disable_preview_mode()
            
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
        
        # Desativa preview e carrega vídeo processado
        self.video_player.disable_preview_mode()
        if self.output_path.exists():
            self.video_player.load_video(str(self.output_path))
            self.video_player.switch_to_playback_mode()
        
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
        
        # Desativa preview
        self.video_player.disable_preview_mode()
        
        # Atualiza UI
        self.start_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        
        QMessageBox.critical(self, "Erro no Processamento", error_msg)
    
    def _on_frame_processed(self, frame_idx, frame, metadata):
        """Callback para frame processado (preview em tempo real)."""
        # Adiciona frame ao buffer de preview do player
        self.video_player.add_preview_frame(frame_idx, frame)
    
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
