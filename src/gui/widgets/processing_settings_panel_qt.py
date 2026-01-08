"""
Painel de configurações de processamento
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSpinBox, QComboBox, QCheckBox,
    QRadioButton, QButtonGroup, QFormLayout, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt

# Importa configurações padrão
from ...config import (
    ENABLE_OBJECT_DETECTION,
    YOLO_MODEL_SIZE, USE_GPU, is_gpu_available,
    FRAME_SKIP, TARGET_FPS, PREVIEW_FPS, ENABLE_PREVIEW
)


class ProcessingSettingsPanel(QWidget):
    """Painel de configurações de processamento."""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Define tamanho do painel
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        
        # Grupo: Configurações Básicas
        basic_group = QGroupBox("Configuracoes de Processamento")
        basic_layout = QFormLayout(basic_group)
        basic_layout.setSpacing(8)
        basic_layout.setContentsMargins(10, 15, 10, 10)
        
        # Frame Skip
        self.frame_skip_spinbox = QSpinBox()
        self.frame_skip_spinbox.setMinimum(1)
        self.frame_skip_spinbox.setMaximum(10)
        self.frame_skip_spinbox.setValue(FRAME_SKIP)
        self.frame_skip_spinbox.setToolTip("Pula N frames durante processamento (maior = mais rápido, menor precisão)")
        self.frame_skip_spinbox.valueChanged.connect(self._emit_settings)
        basic_layout.addRow("Frame Skip:", self.frame_skip_spinbox)
        
        # FPS Alvo
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["15", "30", "60"])
        self.fps_combo.setCurrentText(str(TARGET_FPS))
        self.fps_combo.setToolTip("Quadros por segundo do vídeo de saída")
        self.fps_combo.currentTextChanged.connect(self._emit_settings)
        basic_layout.addRow("FPS Alvo:", self.fps_combo)
        
        layout.addWidget(basic_group)
        
        # Grupo: Presets de Qualidade
        quality_group = QGroupBox("Presets de Qualidade")
        quality_layout = QVBoxLayout(quality_group)
        quality_layout.setSpacing(5)
        quality_layout.setContentsMargins(10, 15, 10, 10)
        
        self.quality_group = QButtonGroup(self)
        
        # Radio buttons para presets
        presets = [
            ("fast", "Rapida", "Skip=5, Preview 5 FPS (melhor para testes)"),
            ("balanced", "Balanceada", "Skip=2, Preview 10 FPS (recomendado)"),
            ("high", "Alta", "Skip=1, Preview 15 FPS (maxima precisao)")
        ]
        
        for idx, (preset_id, label, tooltip) in enumerate(presets):
            radio = QRadioButton(label)
            radio.setToolTip(tooltip)
            radio.setProperty("preset_id", preset_id)
            self.quality_group.addButton(radio, idx)
            quality_layout.addWidget(radio)
            
            if preset_id == "balanced":
                radio.setChecked(True)
        
        # Conecta signals dos radio buttons DEPOIS de criar todos os widgets
        for button in self.quality_group.buttons():
            preset_id = button.property("preset_id")
            if preset_id:
                button.toggled.connect(lambda checked, pid=preset_id: self._on_preset_changed(pid) if checked else None)
        
        layout.addWidget(quality_group)
        
        # Grupo: Preview em Tempo Real
        preview_group = QGroupBox("Preview Durante Processamento")
        preview_layout = QFormLayout(preview_group)
        preview_layout.setSpacing(8)
        preview_layout.setContentsMargins(10, 15, 10, 10)
        
        # Checkbox para habilitar preview
        self.preview_checkbox = QCheckBox("Mostrar preview em tempo real")
        self.preview_checkbox.setChecked(ENABLE_PREVIEW)
        self.preview_checkbox.setToolTip("Mostra frames processados durante análise (usa mais memória)")
        self.preview_checkbox.stateChanged.connect(self._on_preview_toggled)
        preview_layout.addRow(self.preview_checkbox)
        
        # FPS do preview
        preview_fps_layout = QHBoxLayout()
        self.preview_fps_label = QLabel("Taxa do Preview:")
        self.preview_fps_combo = QComboBox()
        self.preview_fps_combo.addItems(["5 FPS", "10 FPS", "15 FPS"])
        self.preview_fps_combo.setCurrentText(f"{PREVIEW_FPS} FPS")  # Usa valor da config
        self.preview_fps_combo.setToolTip("Frequência de atualização do preview (menor = menos overhead)")
        self.preview_fps_combo.currentTextChanged.connect(self._emit_settings)
        preview_fps_layout.addWidget(self.preview_fps_label)
        preview_fps_layout.addWidget(self.preview_fps_combo, stretch=1)
        preview_layout.addRow(preview_fps_layout)
        
        layout.addWidget(preview_group)
        
        # Grupo: Configurações Avançadas (GPU e Detectores)
        advanced_group = QGroupBox("Configuracoes Avancadas")
        advanced_layout = QFormLayout(advanced_group)
        advanced_layout.setSpacing(8)
        advanced_layout.setContentsMargins(10, 15, 10, 10)
        
        # GPU
        gpu_layout = QHBoxLayout()
        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["Auto", "GPU (CUDA)", "CPU"])
        # Define valor inicial baseado na config
        if USE_GPU == "true":
            self.gpu_combo.setCurrentIndex(1)
        elif USE_GPU == "false":
            self.gpu_combo.setCurrentIndex(2)
        else:
            self.gpu_combo.setCurrentIndex(0)
        
        gpu_available = is_gpu_available()
        gpu_status = "disponivel" if gpu_available else "nao disponivel"
        self.gpu_combo.setToolTip(f"Seleciona dispositivo de processamento (GPU {gpu_status})")
        self.gpu_combo.currentTextChanged.connect(self._emit_settings)
        
        # Indicador de GPU
        self.gpu_status_label = QLabel(f"({'GPU OK' if gpu_available else 'CPU only'})")
        self.gpu_status_label.setStyleSheet(f"color: {'#4CAF50' if gpu_available else '#FF9800'}; font-size: 10px;")
        
        gpu_layout.addWidget(self.gpu_combo, stretch=1)
        gpu_layout.addWidget(self.gpu_status_label)
        advanced_layout.addRow("Dispositivo:", gpu_layout)
        
        # Tamanho do modelo YOLO
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["Nano (n)", "Small (s)", "Medium (m)", "Large (l)"])
        # Define valor inicial
        size_map = {"n": 0, "s": 1, "m": 2, "l": 3}
        self.model_size_combo.setCurrentIndex(size_map.get(YOLO_MODEL_SIZE, 0))
        self.model_size_combo.setToolTip("Tamanho dos modelos YOLO (maior = mais preciso, mais lento)")
        self.model_size_combo.currentTextChanged.connect(self._emit_settings)
        advanced_layout.addRow("Modelo YOLO:", self.model_size_combo)
        
        layout.addWidget(advanced_group)
        
        # Grupo: Detectores de Anomalias
        detectors_group = QGroupBox("Detectores de Anomalias")
        detectors_layout = QVBoxLayout(detectors_group)
        detectors_layout.setSpacing(5)
        detectors_layout.setContentsMargins(10, 15, 10, 10)
        
        # Checkbox: Object Detection
        self.object_detection_checkbox = QCheckBox("Detectar objetos fora de contexto")
        self.object_detection_checkbox.setChecked(ENABLE_OBJECT_DETECTION)
        self.object_detection_checkbox.setToolTip("Usa YOLO11 para detectar objetos e identificar anomalias contextuais")
        self.object_detection_checkbox.stateChanged.connect(self._emit_settings)
        detectors_layout.addWidget(self.object_detection_checkbox)
        

        
        layout.addWidget(detectors_group)
        
        # Espaçador final
        layout.addStretch(1)
        
        # Estilo
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 11px;
                border: 2px solid #3d3d3d;
                border-radius: 5px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2d2d2d;
                color: #e0e0e0;
            }
            QSpinBox:hover, QComboBox:hover {
                border: 1px solid #777;
            }
            QRadioButton {
                spacing: 5px;
                padding: 3px;
                font-size: 11px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QCheckBox {
                spacing: 5px;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
    
    def _on_preset_changed(self, preset_id):
        """Aplica preset de qualidade."""
        self._apply_quality_preset(preset_id)
        self._emit_settings()
    
    def _apply_quality_preset(self, preset_id):
        """Aplica configurações do preset."""
        presets = {
            'fast': {
                'frame_skip': 5,
                'fps': '30',
                'preview_fps': '5 FPS',
                'enable_preview': True
            },
            'balanced': {
                'frame_skip': 2,
                'fps': '30',
                'preview_fps': '10 FPS',
                'enable_preview': True
            },
            'high': {
                'frame_skip': 1,
                'fps': '60',
                'preview_fps': '15 FPS',
                'enable_preview': True
            }
        }
        
        preset = presets.get(preset_id, presets['balanced'])
        
        # Bloqueia signals temporariamente
        self.frame_skip_spinbox.blockSignals(True)
        self.fps_combo.blockSignals(True)
        self.preview_fps_combo.blockSignals(True)
        
        # Aplica valores
        self.frame_skip_spinbox.setValue(preset['frame_skip'])
        self.fps_combo.setCurrentText(preset['fps'])
        self.preview_fps_combo.setCurrentText(preset['preview_fps'])
        self.preview_checkbox.setChecked(preset['enable_preview'])
        
        # Desbloqueia signals
        self.frame_skip_spinbox.blockSignals(False)
        self.fps_combo.blockSignals(False)
        self.preview_fps_combo.blockSignals(False)
    
    def _on_preview_toggled(self, state):
        """Habilita/desabilita controles de preview."""
        enabled = state == Qt.CheckState.Checked.value
        self.preview_fps_label.setEnabled(enabled)
        self.preview_fps_combo.setEnabled(enabled)
        self._emit_settings()
    
    def _emit_settings(self):
        """Emite signal com configurações atuais."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)
    
    def get_settings(self):
        """Retorna configurações atuais."""
        # Extrai FPS do preview (remove " FPS" do texto)
        preview_fps_text = self.preview_fps_combo.currentText()
        preview_fps = int(preview_fps_text.split()[0])
        
        # Extrai tamanho do modelo YOLO
        model_size_text = self.model_size_combo.currentText()
        model_size_map = {"Nano (n)": "n", "Small (s)": "s", "Medium (m)": "m", "Large (l)": "l"}
        model_size = model_size_map.get(model_size_text, "n")
        
        # Extrai configuração de GPU
        gpu_text = self.gpu_combo.currentText()
        gpu_map = {"Auto": "auto", "GPU (CUDA)": "true", "CPU": "false"}
        use_gpu = gpu_map.get(gpu_text, "auto")
        
        return {
            'frame_skip': self.frame_skip_spinbox.value(),
            'target_fps': int(self.fps_combo.currentText()),
            'enable_preview': self.preview_checkbox.isChecked(),
            'preview_fps': preview_fps,
            # Configurações avançadas
            'use_gpu': use_gpu,
            'model_size': model_size,
            # Detectores
            'enable_object_detection': self.object_detection_checkbox.isChecked()
        }
    
    def set_enabled_all(self, enabled):
        """Habilita/desabilita todos os controles."""
        self.frame_skip_spinbox.setEnabled(enabled)
        self.fps_combo.setEnabled(enabled)
        self.preview_checkbox.setEnabled(enabled)
        self.preview_fps_combo.setEnabled(enabled and self.preview_checkbox.isChecked())
        
        # Desabilita radio buttons
        for button in self.quality_group.buttons():
            button.setEnabled(enabled)
        
        # Controles avançados
        self.gpu_combo.setEnabled(enabled)
        self.model_size_combo.setEnabled(enabled)
        self.object_detection_checkbox.setEnabled(enabled)
