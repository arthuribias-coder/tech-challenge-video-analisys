
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton, QHBoxLayout, QApplication
from PyQt6.QtCore import Qt

class ErrorDialog(QDialog):
    """Dialogo customizado para erros com opcao de copiar."""
    
    def __init__(self, parent=None, title="Erro", message="Ocorreu um erro", details=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 400)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #e0e0e0; }
            QLabel { color: #ff6b6b; font-weight: bold; font-size: 14px; }
            QTextEdit { background-color: #2d2d2d; color: #aaa; border: 1px solid #555; }
            QPushButton { background-color: #3d3d3d; color: #e0e0e0; padding: 5px 15px; border: 1px solid #555; }
            QPushButton:hover { background-color: #4d4d4d; }
        """)
        
        layout = QVBoxLayout(self)
        
        # Mensagem principal
        lbl_msg = QLabel(message)
        lbl_msg.setWordWrap(True)
        layout.addWidget(lbl_msg)
        
        # Detalhes (Log)
        if details:
            lbl_details = QLabel("Detalhes do Erro:")
            lbl_details.setStyleSheet("color: #ccc; font-size: 12px; font-weight: normal;")
            layout.addWidget(lbl_details)
            
            self.text_edit = QTextEdit()
            self.text_edit.setPlainText(details)
            self.text_edit.setReadOnly(True)
            layout.addWidget(self.text_edit)
            
        # Botoes
        btn_layout = QHBoxLayout()
        
        if details:
            btn_copy = QPushButton("Copiar Log")
            btn_copy.clicked.connect(self._copy_to_clipboard)
            btn_layout.addWidget(btn_copy)
            
        btn_layout.addStretch()
        
        btn_close = QPushButton("Fechar")
        btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        
    def _copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())
        
        # Feedback visual rapido no botao
        sender = self.sender()
        if sender:
            sender.setText("Copiado!")
            sender.setEnabled(False)
