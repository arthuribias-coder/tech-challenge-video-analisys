"""
Dialog de configurações de processamento
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QDialogButtonBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from .processing_settings_panel_qt import ProcessingSettingsPanel


class SettingsDialog(QDialog):
    """Dialog para configurações de processamento."""
    
    settings_applied = pyqtSignal(dict)
    
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Configuracoes de Processamento")
        self.setModal(True)
        self.resize(450, 650)  # Aumentado para acomodar novos controles
        
        self._setup_ui()
        
        # Aplica configurações existentes se fornecidas
        if current_settings:
            self._apply_current_settings(current_settings)
    
    def _setup_ui(self):
        """Configura interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Painel de configurações
        self.settings_panel = ProcessingSettingsPanel()
        layout.addWidget(self.settings_panel, stretch=1)
        
        # Botões
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.apply_btn = QPushButton("Aplicar")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(self.apply_btn)
        
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Estilo
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #e0e0e0;
                border: 1px solid #555;
                padding: 8px 20px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
                border: 1px solid #777;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QPushButton[default="true"] {
                background-color: #4CAF50;
                border: 1px solid #45a049;
            }
            QPushButton[default="true"]:hover {
                background-color: #5CBF60;
            }
        """)
    
    def _apply_current_settings(self, settings):
        """Aplica configurações atuais ao painel."""
        # Bloqueia signals temporariamente
        self.settings_panel.frame_skip_spinbox.blockSignals(True)
        self.settings_panel.fps_combo.blockSignals(True)
        self.settings_panel.preview_fps_combo.blockSignals(True)
        self.settings_panel.preview_checkbox.blockSignals(True)
        self.settings_panel.gpu_combo.blockSignals(True)
        self.settings_panel.model_size_combo.blockSignals(True)
        self.settings_panel.object_detection_checkbox.blockSignals(True)

        
        # Aplica valores básicos
        self.settings_panel.frame_skip_spinbox.setValue(settings.get('frame_skip', 2))
        self.settings_panel.fps_combo.setCurrentText(str(settings.get('target_fps', 30)))
        self.settings_panel.preview_checkbox.setChecked(settings.get('enable_preview', True))
        
        preview_fps = settings.get('preview_fps', 10)
        self.settings_panel.preview_fps_combo.setCurrentText(f"{preview_fps} FPS")
        
        # Aplica configurações avançadas (GPU)
        use_gpu = settings.get('use_gpu', 'auto')
        gpu_map = {"auto": 0, "true": 1, "false": 2}
        self.settings_panel.gpu_combo.setCurrentIndex(gpu_map.get(use_gpu, 0))
        
        # Aplica tamanho do modelo YOLO
        model_size = settings.get('model_size', 'n')
        size_map = {"n": 0, "s": 1, "m": 2, "l": 3}
        self.settings_panel.model_size_combo.setCurrentIndex(size_map.get(model_size, 0))
        
        # Aplica configurações de detectores
        self.settings_panel.object_detection_checkbox.setChecked(
            settings.get('enable_object_detection', True)
        )

        
        # Desbloqueia signals
        self.settings_panel.frame_skip_spinbox.blockSignals(False)
        self.settings_panel.fps_combo.blockSignals(False)
        self.settings_panel.preview_fps_combo.blockSignals(False)
        self.settings_panel.preview_checkbox.blockSignals(False)
        self.settings_panel.gpu_combo.blockSignals(False)
        self.settings_panel.model_size_combo.blockSignals(False)
        self.settings_panel.object_detection_checkbox.blockSignals(False)

    
    def _on_apply(self):
        """Aplica configurações e fecha dialog."""
        settings = self.settings_panel.get_settings()
        self.settings_applied.emit(settings)
        self.accept()
    
    def get_settings(self):
        """Retorna configurações atuais."""
        return self.settings_panel.get_settings()
