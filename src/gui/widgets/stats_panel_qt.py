"""
Painel de estatísticas em tempo real com PyQt6
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox,
    QDialog, QTextEdit, QDialogButtonBox
)
from PyQt6.QtCore import Qt
from collections import Counter


class StatsPanelQt(QWidget):
    """Painel lateral com estatísticas da análise."""
    
    def __init__(self):
        super().__init__()
        
        self.stats = {
            'faces': 0,
            'emotions': Counter(),
            'activities': Counter(),
            'anomalies': Counter()
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura interface do painel."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Título
        title = QLabel("ESTATÍSTICAS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                background-color: #2d2d2d;
                border-radius: 5px;
            }
        """)
        layout.addWidget(title)
        
        # Faces
        faces_group = self._create_stat_group("Faces Detectadas")
        self.faces_label = QLabel("0")
        self.faces_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.faces_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
                padding: 10px;
            }
        """)
        faces_group.layout().addWidget(self.faces_label)
        layout.addWidget(faces_group)
        
        # Emoção
        emotion_group = self._create_stat_group("Emoção Principal")
        self.emotion_label = QLabel("--")
        self.emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.emotion_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #2196F3;
                padding: 5px;
            }
        """)
        self.emotion_count_label = QLabel("(0 ocorrências)")
        self.emotion_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.emotion_count_label.setStyleSheet("color: #888;")
        
        emotion_group.layout().addWidget(self.emotion_label)
        emotion_group.layout().addWidget(self.emotion_count_label)
        layout.addWidget(emotion_group)
        
        # Atividade
        activity_group = self._create_stat_group("Atividade Principal")
        self.activity_label = QLabel("--")
        self.activity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.activity_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #FF9800;
                padding: 5px;
            }
        """)
        self.activity_count_label = QLabel("(0 ocorrências)")
        self.activity_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.activity_count_label.setStyleSheet("color: #888;")
        
        activity_group.layout().addWidget(self.activity_label)
        activity_group.layout().addWidget(self.activity_count_label)
        layout.addWidget(activity_group)
        
        # Anomalias
        anomaly_group = self._create_stat_group("Anomalias")
        self.anomaly_label = QLabel("0")
        self.anomaly_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.anomaly_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #F44336;
                padding: 10px;
            }
        """)
        anomaly_group.layout().addWidget(self.anomaly_label)
        layout.addWidget(anomaly_group)
        
        # Botão detalhes
        self.details_btn = QPushButton("Ver Detalhes Completos")
        self.details_btn.clicked.connect(self._show_details)
        self.details_btn.setEnabled(False)
        self.details_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 13px;
            }
        """)
        layout.addWidget(self.details_btn)
        
        layout.addStretch()
    
    def _create_stat_group(self, title):
        """Cria grupo de estatística."""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        group.setLayout(layout)
        
        return group
    
    def reset(self):
        """Reseta estatísticas."""
        self.stats = {
            'faces': 0,
            'emotions': Counter(),
            'activities': Counter(),
            'anomalies': Counter()
        }
        self.update_stats(self.stats)
        self.details_btn.setEnabled(False)
    
    def update_stats(self, stats):
        """Atualiza estatísticas."""
        self.stats = stats
        
        # Faces
        self.faces_label.setText(str(stats.get('faces', 0)))
        
        # Emoção dominante
        emotions = stats.get('emotions', Counter())
        if isinstance(emotions, dict):
            emotions = Counter(emotions)
        
        if emotions:
            dominant_emotion = emotions.most_common(1)[0]
            self.emotion_label.setText(dominant_emotion[0].title())
            total = sum(emotions.values())
            percent = (dominant_emotion[1] / total * 100) if total > 0 else 0
            self.emotion_count_label.setText(f"({dominant_emotion[1]} - {percent:.1f}%)")
        else:
            self.emotion_label.setText("--")
            self.emotion_count_label.setText("(0 ocorrências)")
        
        # Atividade dominante
        activities = stats.get('activities', Counter())
        if isinstance(activities, dict):
            activities = Counter(activities)
        
        if activities:
            dominant_activity = activities.most_common(1)[0]
            self.activity_label.setText(dominant_activity[0].title())
            total = sum(activities.values())
            percent = (dominant_activity[1] / total * 100) if total > 0 else 0
            self.activity_count_label.setText(f"({dominant_activity[1]} - {percent:.1f}%)")
        else:
            self.activity_label.setText("--")
            self.activity_count_label.setText("(0 ocorrências)")
        
        # Anomalias
        anomalies = stats.get('anomalies', Counter())
        if isinstance(anomalies, dict):
            anomalies = Counter(anomalies)
        
        total_anomalies = sum(anomalies.values())
        self.anomaly_label.setText(str(total_anomalies))
        
        # Habilita botão de detalhes se há dados
        self.details_btn.setEnabled(bool(stats.get('faces', 0) > 0))
    
    def _show_details(self):
        """Mostra detalhes completos."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Estatísticas Completas")
        dialog.resize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        text = QTextEdit()
        text.setReadOnly(True)
        text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                padding: 10px;
            }
        """)
        
        # Gera texto
        details = "ESTATÍSTICAS COMPLETAS\n"
        details += "=" * 50 + "\n\n"
        
        details += f"[FACES] FACES DETECTADAS: {self.stats['faces']}\n\n"
        
        details += "[EMOÇÕES]:\n"
        emotions = self.stats.get('emotions', Counter())
        if isinstance(emotions, dict):
            emotions = Counter(emotions)
        
        for emotion, count in emotions.most_common():
            total = sum(emotions.values())
            percent = (count / total * 100) if total > 0 else 0
            details += f"   - {emotion}: {count} ({percent:.1f}%)\n"
        
        details += "\n[ATIVIDADES]:\n"
        activities = self.stats.get('activities', Counter())
        if isinstance(activities, dict):
            activities = Counter(activities)
        
        for activity, count in activities.most_common():
            total = sum(activities.values())
            percent = (count / total * 100) if total > 0 else 0
            details += f"   - {activity}: {count} ({percent:.1f}%)\n"
        
        anomalies = self.stats.get('anomalies', Counter())
        if isinstance(anomalies, dict):
            anomalies = Counter(anomalies)
        
        total_anomalies = sum(anomalies.values())
        details += f"\n[!] ANOMALIAS: {total_anomalies}\n"
        for anomaly, count in anomalies.most_common():
            details += f"   - {anomaly}: {count}\n"
        
        text.setText(details)
        layout.addWidget(text)
        
        # Botões
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        
        dialog.exec()
