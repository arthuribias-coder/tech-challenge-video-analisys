"""
Janela principal da aplicação GUI com PyQt6
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QProgressBar,
    QFileDialog, QMessageBox, QLabel, QSpinBox,
    QPushButton, QToolBar, QSizePolicy, QSplitter, QScrollArea,
    QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QAction, QIcon, QFontDatabase
from pathlib import Path
import logging

from .widgets import VideoPlayerQt, StatsPanelQt, ChartsPanelQt
# removed SettingsDialog import as we are moving settings to header
from .widgets.error_dialog_qt import ErrorDialog
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
        
        # Layout horizontal (Video | Painel Fixo)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(5)
        
        # Esquerda: Video Player (expansível)
        self.video_player = VideoPlayerQt()
        self.video_player.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_player.setMinimumWidth(500)
        content_layout.addWidget(self.video_player, stretch=1)
        
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
        self.charts_panel.setMinimumHeight(400)
        right_layout.addWidget(self.charts_panel)
        
        # Adiciona espaçador para empurrar conteúdo para cima
        right_layout.addStretch()
        
        # Scroll area para o painel direito
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(right_panel)
        
        # Tamanho fixo de 406px
        scroll_area.setFixedWidth(406) 
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content_layout.addWidget(scroll_area)
        
        main_layout.addLayout(content_layout)
    
    def _setup_toolbar(self):
        """Cria toolbar com botões compactos e configurações no header."""
        toolbar = QToolBar("Controles Principais")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #1e1e1e;
                border-bottom: 1px solid #333;
                spacing: 6px;
                padding: 4px;
            }
            QToolButton {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
                min-width: 50px;
            }
            QToolButton:hover {
                background-color: #3d3d3d;
                border: 1px solid #666;
            }
            QToolButton:pressed {
                background-color: #1a1a1a;
            }
            QToolButton:disabled {
                background-color: #222;
                color: #555;
                border: 1px solid #2a2a2a;
            }
            QLabel {
                color: #bbb;
                font-size: 11px;
                margin-left: 2px;
            }
            QComboBox {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 2px 5px;
                font-size: 11px;
                min-width: 60px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QCheckBox {
                color: #e0e0e0;
                font-size: 11px;
                spacing: 4px;
            }
        """)
        self.addToolBar(toolbar)
        
        # Botão Abrir Vídeo
        open_action = QAction(IconProvider.document_open(), "Abrir", self)
        open_action.setToolTip("Abrir vídeo (Ctrl+O)")
        open_action.triggered.connect(self._open_video)
        toolbar.addAction(open_action)
        
        # Botão Processar
        self.start_action = QAction(IconProvider.media_play(), "Iniciar", self)
        self.start_action.setToolTip("Iniciar processamento")
        self.start_action.triggered.connect(self._start_processing)
        toolbar.addAction(self.start_action)
        
        # Botão Pausar
        self.pause_action = QAction(IconProvider.media_pause(), "Pausar", self)
        self.pause_action.setToolTip("Pausar")
        self.pause_action.triggered.connect(self._pause_processing)
        self.pause_action.setEnabled(False)
        toolbar.addAction(self.pause_action)
        
        # Botão Parar
        self.stop_action = QAction(IconProvider.media_stop(), "Parar", self)
        self.stop_action.setToolTip("Parar")
        self.stop_action.triggered.connect(self._stop_processing)
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)
        
        # Botão Salvar
        save_action = QAction(IconProvider.document_save(), "Salvar", self)
        save_action.setToolTip("Salvar vídeo")
        save_action.triggered.connect(self._save_video)
        toolbar.addAction(save_action)

        toolbar.addSeparator()

        # === Configurações no Header ===
        
        # 1. Preset (Combina FrameSkip e PreviewFPS)
        toolbar.addWidget(QLabel("Modo:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["Balanceado", "Rapido", "Alta Qualidade"])
        self.combo_preset.setToolTip("Define velocidade vs precisão")
        # Default: Balanceado
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        toolbar.addWidget(self.combo_preset)
        
        toolbar.addSeparator()

        # 2. Modelo YOLO
        toolbar.addWidget(QLabel("Modelo:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Nano", "Small", "Medium", "Large"])
        
        # Set initial value
        size_map = {"n": 0, "s": 1, "m": 2, "l": 3}
        current_idx = size_map.get(self.processing_settings.get('model_size', 'n'), 0)
        self.combo_model.setCurrentIndex(current_idx)
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        toolbar.addWidget(self.combo_model)

        # 3. Dispositivo
        toolbar.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["Auto", "GPU", "CPU"])
        
        current_gpu = self.processing_settings.get('use_gpu', 'auto')
        dev_map = {"auto": 0, "true": 1, "false": 2}
        self.combo_device.setCurrentIndex(dev_map.get(current_gpu, 0))
        self.combo_device.currentIndexChanged.connect(self._on_device_changed)
        toolbar.addWidget(self.combo_device)

        toolbar.addSeparator()

        # 4. Checkboxes
        self.chk_preview = QCheckBox("Preview Vídeo")
        self.chk_preview.setChecked(self.processing_settings.get('enable_preview', True))
        self.chk_preview.stateChanged.connect(self._on_preview_changed)
        toolbar.addWidget(self.chk_preview)

        self.chk_obj = QCheckBox("Rastrear Objetos")
        self.chk_obj.setToolTip("Detectar objetos fora de contexto")
        self.chk_obj.setChecked(self.processing_settings.get('enable_object_detection', True))
        self.chk_obj.stateChanged.connect(self._on_obj_det_changed)
        toolbar.addWidget(self.chk_obj)

        # Checkbox Debug
        self.chk_debug = QCheckBox("Debug")
        self.chk_debug.setToolTip("Ativar modo de depuração e logs")
        self.chk_debug.stateChanged.connect(self._toggle_debug)
        toolbar.addWidget(self.chk_debug)

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

    # === Métodos de Configuração (Header) ===

    def _on_preset_changed(self, index):
        """Atualiza configurações baseado no preset."""
        # Itens: 0=Balanceado, 1=Rapido, 2=Alta Qualidade
        text = self.combo_preset.currentText()
        if "Rapido" in text:
            self.processing_settings['frame_skip'] = 5
            self.processing_settings['preview_fps'] = 5
        elif "Alta" in text:
            self.processing_settings['frame_skip'] = 1
            self.processing_settings['preview_fps'] = 15
        else: # Balanceado ou outros
            self.processing_settings['frame_skip'] = 2
            self.processing_settings['preview_fps'] = 10
        
        self.status_label.setText(f"Modo alterado para: {text}")

    def _on_model_changed(self, index):
        """Atualiza tamanho do modelo YOLO."""
        # Index: 0=n, 1=s, 2=m, 3=l (mesma ordem do combo)
        sizes = ['n', 's', 'm', 'l']
        if 0 <= index < len(sizes):
            self.processing_settings['model_size'] = sizes[index]
            self.status_label.setText(f"Modelo definido: {sizes[index].upper()}")

    def _on_device_changed(self, index):
        """Atualiza dispositivo de processamento."""
        # Combo: 0=Auto, 1=GPU, 2=CPU
        vals = ["auto", "true", "false"]
        if 0 <= index < len(vals):
            self.processing_settings['use_gpu'] = vals[index]
            self.status_label.setText(f"Device: {self.combo_device.currentText()}")

    def _on_preview_changed(self, state):
        """Habilita/desabilita preview."""
        enabled = (state != 0) # Checked or PartiallyChecked
        self.processing_settings['enable_preview'] = enabled
        self.status_label.setText(f"Preview {'ativado' if enabled else 'desativado'}")

    def _on_obj_det_changed(self, state):
        """Habilita/desabilita detecção de objetos."""
        enabled = (state != 0)
        self.processing_settings['enable_object_detection'] = enabled
        self.status_label.setText(f"Obj. Det. {'ativada' if enabled else 'desativada'}")

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
            # Configurações de hardware
            use_gpu=settings.get('use_gpu'),
            model_size=settings.get('model_size')
        )
        
        # Conecta signals
        self.processor_thread.progress.connect(self._on_progress)
        self.processor_thread.finished_signal.connect(self._on_complete)
        self.processor_thread.error.connect(self._on_error)
        self.processor_thread.frame_processed.connect(self._on_frame_processed)
        
        # Aplica estado inicial do debug mode
        if hasattr(self, 'chk_debug'):
             self.processor_thread.set_debug_mode(self.chk_debug.isChecked())
        
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
            "Sobre - Tech Challenge Fase 4",
            "<h2>Tech Challenge - Fase 4</h2>"
            "<h3>Análise de Vídeo com IA</h3>"
            "<p><b>Análise Inteligente de Vídeos</b> utilizando técnicas modernas de "
            "Visão Computacional e Inteligência Artificial para extrair insights "
            "comportamentais, contextuais e emocionais.</p>"
            "<hr>"
            "<p><b>Tecnologias Utilizadas:</b></p>"
            "<ul>"
            "<li><b>YOLO11-pose</b> - Detecção de atividades e poses humanas</li>"
            "<li><b>YOLO11-cls</b> - Classificação de contexto/cena</li>"
            "<li><b>YOLO11-obb</b> - Detecção orientada (pessoas deitadas)</li>"
            "<li><b>YOLO11</b> - Detecção de objetos no ambiente</li>"
            "<li><b>DeepFace</b> - Análise facial e reconhecimento de emoções</li>"
            "<li><b>OpenCV</b> - Processamento de imagens e vídeo</li>"
            "<li><b>PyQt6</b> - Interface gráfica profissional</li>"
            "<li><b>PyTorch</b> - Framework de Deep Learning</li>"
            "</ul>"
            "<hr>"
            "<p><b>Funcionalidades:</b></p>"
            "<ul>"
            "<li>Detecção e rastreamento de pessoas</li>"
            "<li>Análise de emoções em tempo real</li>"
            "<li>Classificação de atividades (caminhando, sentado, acenando, etc.)</li>"
            "<li>Detecção de anomalias comportamentais</li>"
            "<li>Contexto de cena (escritório, sala, rua, etc.)</li>"
            "<li>Relatórios automáticos detalhados</li>"
            "</ul>"
            "<hr>"
            "<p><b>Versão:</b> 4.0.0<br>"
            "<b>Pós Tech Data Analytics - FIAP</b><br>"
            "© 2026 - Tech Challenge</p>"
        )
    
    def _on_progress(self, frame_idx, total_frames, fps, stats):
        """Callback de progresso."""
        progress = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
        self.progress_bar.setValue(progress)
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # Mostra status com indicador [PREVIEW] se ativado
        status_prefix = "[PREVIEW ON] " if self.processing_settings.get('enable_preview') else ""
        self.status_label.setText(f"{status_prefix}Processando... Frame {frame_idx}/{total_frames} ({progress}%) | FPS: {fps:.1f}")
        
        # Atualiza painéis
        self.stats_panel.update_stats(stats)
        self.charts_panel.update_data(stats)
        
        # Armazena stats
        self.last_stats = stats

    def _toggle_debug(self, state):
        """Alterna modo debug."""
        enabled = (state == 2) # Qt.CheckState.Checked
        
        # Atualiza nível de log global (afeta saída do console)
        level = logging.DEBUG if enabled else logging.INFO
        logging.getLogger().setLevel(level)
        
        # Log de confirmação
        if enabled:
            logging.debug("Modo de depuração ativado")
        else:
            logging.info("Modo de depuração desativado")
        
        # Atualiza thread de processamento
        if hasattr(self, 'processor_thread') and self.processor_thread:
            if self.processor_thread.isRunning():
                self.processor_thread.set_debug_mode(enabled)
    
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
        # Desativa preview
        self.video_player.disable_preview_mode()
        
        # Atualiza UI
        self.start_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        
        # Atualiza status
        self.status_label.setText("Erro no processamento")
        
        # Mostra dialogo customizado
        dlg = ErrorDialog(
            self,
            title="Erro no Processamento",
            message="Ocorreu um erro durante o processamento do vídeo.",
            details=f"Mensagem de Erro: {error_msg}\n\nVerifique o console para traceback completo se necessário." 
        )
        dlg.exec()
    
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
