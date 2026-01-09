"""
Painel de estatísticas em tempo real com PyQt6
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox,
    QDialog, QTextEdit, QDialogButtonBox, QHBoxLayout, QFrame
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
            'anomalies': Counter(),
            'scenes': Counter()
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura interface minimalista ('dashboard')."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Estilos globais para o painel
        self.setStyleSheet("""
            QLabel#title_lbl { 
                color: #888; 
                font-size: 11px; 
                font-weight: bold; 
                text-transform: uppercase; 
                letter-spacing: 1px;
            }
            QLabel#value_lbl { 
                font-size: 22px; 
                font-weight: bold; 
            }
            QLabel#subtext_lbl { 
                color: #666; 
                font-size: 11px; 
            }
            QFrame#line { 
                background-color: #333; 
                max-height: 1px; 
                border: none;
            }
        """)

        # -- LINHA 1: Faces e Anomalias (Lado a Lado) --
        row1 = QHBoxLayout()
        
        # Coluna Faces
        col_faces = QVBoxLayout()
        col_faces.setSpacing(2)
        col_faces.addWidget(QLabel("FACES", objectName="title_lbl"))
        self.faces_label = QLabel("0", objectName="value_lbl")
        self.faces_label.setStyleSheet("color: #4CAF50;") # Verde
        col_faces.addWidget(self.faces_label)
        row1.addLayout(col_faces)

        # Coluna Anomalias
        col_anom = QVBoxLayout()
        col_anom.setSpacing(2)
        col_anom.addWidget(QLabel("ANOMALIAS", objectName="title_lbl"))
        self.anomaly_label = QLabel("0", objectName="value_lbl")
        self.anomaly_label.setStyleSheet("color: #F44336;") # Vermelho
        col_anom.addWidget(self.anomaly_label)
        row1.addLayout(col_anom)
        
        layout.addLayout(row1)
        
        # Separador
        line1 = QFrame(objectName="line")
        line1.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line1)

        # -- SEÇÃO DE EMOÇÃO --
        layout.addWidget(QLabel("EMOÇÃO PREDOMINANTE", objectName="title_lbl"))
        self.emotion_label = QLabel("--", objectName="value_lbl")
        self.emotion_label.setStyleSheet("color: #2196F3;") # Azul
        layout.addWidget(self.emotion_label)
        self.emotion_count_label = QLabel("-", objectName="subtext_lbl")
        layout.addWidget(self.emotion_count_label) # Detalhe %

        # Separador
        line2 = QFrame(objectName="line")
        line2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line2)

        # -- SEÇÃO DE ATIVIDADE --
        layout.addWidget(QLabel("ATIVIDADE PRINCIPAL", objectName="title_lbl"))
        self.activity_label = QLabel("--", objectName="value_lbl")
        self.activity_label.setStyleSheet("color: #FF9800;") # Laranja
        layout.addWidget(self.activity_label)
        self.activity_count_label = QLabel("-", objectName="subtext_lbl")
        layout.addWidget(self.activity_count_label)
        
        # Separador
        line3 = QFrame(objectName="line")
        line3.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line3)

        # -- SEÇÃO DE CENA --
        layout.addWidget(QLabel("CONTEXTO / CENA", objectName="title_lbl"))
        self.scene_label = QLabel("--", objectName="value_lbl")
        self.scene_label.setStyleSheet("color: #9C27B0;") # Roxo
        layout.addWidget(self.scene_label)
        self.scene_conf_label = QLabel("Aguardando detecção...", objectName="subtext_lbl")
        layout.addWidget(self.scene_conf_label)

        layout.addStretch()

        # Botão Rodapé
        self.details_btn = QPushButton("RELATÓRIO DETALHADO")
        self.details_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.details_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                border: 1px solid #555;
                font-weight: bold;
                padding: 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #444;
                border: 1px solid #666;
            }
            QPushButton:disabled {
                background-color: #222;
                color: #555;
                border: 1px solid #333;
            }
        """)
        self.details_btn.clicked.connect(self._show_details)
        self.details_btn.setEnabled(False)
        layout.addWidget(self.details_btn)

    def reset(self):
        """Reseta estatísticas."""
        self.stats = {
            'faces': 0,
            'emotions': Counter(),
            'activities': Counter(),
            'anomalies': Counter(),
            'scenes': Counter()
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
            total_emotions = sum(emotions.values())
            dominant_emotion = emotions.most_common(1)[0]
            self.emotion_label.setText(dominant_emotion[0].title())
            percent = (dominant_emotion[1] / total_emotions * 100) if total_emotions > 0 else 0
            self.emotion_count_label.setText(f"({percent:.1f}%)")
        else:
            self.emotion_label.setText("--")
            self.emotion_count_label.setText("(0.0%)")
        
        # Atividade dominante
        activities = stats.get('activities', Counter())
        if isinstance(activities, dict):
            activities = Counter(activities)
        
        if activities:
            total_activities = sum(activities.values())
            dominant_activity = activities.most_common(1)[0]
            if isinstance(dominant_activity[0], str):
                self.activity_label.setText(dominant_activity[0].title())
            else:
                 self.activity_label.setText(str(dominant_activity[0]).title())

            percent = (dominant_activity[1] / total_activities * 100) if total_activities > 0 else 0
            self.activity_count_label.setText(f"({percent:.1f}%)")
        else:
            self.activity_label.setText("--")
            self.activity_count_label.setText("(0.0%)")
        
        # Cena (Novo)
        scenes = stats.get('scenes', Counter())
        if scenes:
            total_scenes = sum(scenes.values())
            # Pega a cena mais comum
            top_scene = scenes.most_common(1)[0]
            
            scene_map = {
                'office': 'Escritório',
                'home': 'Residência',
                'outdoors': 'Ambiente Externo',
                'unknown': 'Desconhecido'
            }
            
            raw_name = top_scene[0]
            scene_name = scene_map.get(raw_name, raw_name.replace("_", " ").title())
            
            count = top_scene[1]
            percent = (count / total_scenes * 100) if total_scenes > 0 else 0
            
            self.scene_label.setText(scene_name)
            self.scene_conf_label.setText(f"({percent:.1f}%)")
            
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

        details += "\n[CONTEXTO / CENAS]:\n"
        scenes = self.stats.get('scenes', Counter())
        if isinstance(scenes, dict):
            scenes = Counter(scenes)
            
        scene_map = {
            'office': 'Escritório',
            'home': 'Residência',
            'outdoors': 'Ambiente Externo',
            'unknown': 'Desconhecido'
        }
            
        for scene, count in scenes.most_common():
            raw_name = scene
            scene_name = scene_map.get(raw_name, raw_name.replace("_", " ").title())
            
            total = sum(scenes.values())
            percent = (count / total * 100) if total > 0 else 0
            details += f"   - {scene_name}: {count} ({percent:.1f}%)\n"
        
        anomalies = self.stats.get('anomalies', Counter())
        if isinstance(anomalies, dict):
            anomalies = Counter(anomalies)
        
        anomaly_map = {
            'unusual_activity': 'Atividade Atípica',
            'scene_inconsistency': 'Inconsistência com o Ambiente',
            'prolonged_inactivity': 'Inatividade Prolongada',
            'sudden_movement': 'Movimento Brusco',
            'abnormal_pose': 'Postura Anormal',
            'unauthorized_object': 'Objeto Não Autorizado',
            'emotion_spike': 'Pico Emocional',
            'crowd_anomaly': 'Aglomeração Anômala',
            'visual_overlay': 'Sobreposição Visual',
            'sudden_object_appear': 'Aparição Súbita de Objeto',
            'silhouette_anomaly': 'Anomalia de Silhueta'
        }
        
        total_anomalies = sum(anomalies.values())
        details += f"\n[!] ANOMALIAS: {total_anomalies}\n"
        for anomaly, count in anomalies.most_common():
            anomaly_name = anomaly_map.get(anomaly, anomaly)
            total_anom = sum(anomalies.values())
            percent = (count / total_anom * 100) if total_anom > 0 else 0
            details += f"   - {anomaly_name}: {count} ({percent:.1f}%)\n"
        
        text.setText(details)
        layout.addWidget(text)
        
        # Botões
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.close)
        layout.addWidget(buttons)
        
        dialog.exec()
