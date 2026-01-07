"""
Painel de gráficos com PyQt6 e Matplotlib
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from collections import Counter


class ChartsPanelQt(QWidget):
    """Painel com abas de gráficos."""
    
    def __init__(self):
        super().__init__()
        
        self.stats = {
            'emotions': Counter(),
            'activities': Counter(),
            'anomalies': Counter()
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 8px 20px;
                border: 1px solid #444;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #3d3d3d;
            }
            QTabBar::tab:hover {
                background-color: #3d3d3d;
            }
        """)
        
        # Cria abas
        self.emotion_canvas = self._create_chart_canvas()
        self.tabs.addTab(self.emotion_canvas, "😊 Emoções")
        
        self.activity_canvas = self._create_chart_canvas()
        self.tabs.addTab(self.activity_canvas, "🏃 Atividades")
        
        self.anomaly_canvas = self._create_chart_canvas()
        self.tabs.addTab(self.anomaly_canvas, "⚠️ Anomalias")
        
        layout.addWidget(self.tabs)
        
        # Desenha gráficos vazios iniciais
        self._draw_empty_charts()
    
    def _create_chart_canvas(self):
        """Cria canvas matplotlib."""
        figure = Figure(figsize=(8, 3), facecolor='#1e1e1e')
        canvas = FigureCanvas(figure)
        canvas.setStyleSheet("background-color: #1e1e1e;")
        return canvas
    
    def _draw_empty_charts(self):
        """Desenha gráficos vazios."""
        # Emoções
        fig = self.emotion_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Sem dados ainda\nProcesse um vídeo primeiro',
                ha='center', va='center', fontsize=14, color='#666',
                transform=ax.transAxes)
        ax.set_facecolor('#1e1e1e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        self.emotion_canvas.draw()
        
        # Atividades
        fig = self.activity_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Sem dados ainda\nProcesse um vídeo primeiro',
                ha='center', va='center', fontsize=14, color='#666',
                transform=ax.transAxes)
        ax.set_facecolor('#1e1e1e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        self.activity_canvas.draw()
        
        # Anomalias
        fig = self.anomaly_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Sem dados ainda\nProcesse um vídeo primeiro',
                ha='center', va='center', fontsize=14, color='#666',
                transform=ax.transAxes)
        ax.set_facecolor('#1e1e1e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        self.anomaly_canvas.draw()
    
    def clear_data(self):
        """Limpa dados."""
        self.stats = {
            'emotions': Counter(),
            'activities': Counter(),
            'anomalies': Counter()
        }
        self._draw_empty_charts()
    
    def update_data(self, stats):
        """Atualiza gráficos com novos dados."""
        self.stats = stats
        
        # Atualiza cada gráfico
        self._draw_emotions_chart()
        self._draw_activities_chart()
        self._draw_anomalies_chart()
    
    def _draw_emotions_chart(self):
        """Desenha gráfico de emoções."""
        emotions = self.stats.get('emotions', Counter())
        if isinstance(emotions, dict):
            emotions = Counter(emotions)
        
        if not emotions:
            return
        
        fig = self.emotion_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Dados
        items = emotions.most_common(7)  # Top 7
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        
        # Gráfico de barras horizontal
        bars = ax.barh(labels, values, color='#2196F3', edgecolor='#333')
        
        # Estilo
        ax.set_facecolor('#1e1e1e')
        ax.set_xlabel('Ocorrências', color='#e0e0e0')
        ax.set_title('😊 Distribuição de Emoções', color='#e0e0e0', fontsize=12, pad=10)
        ax.tick_params(colors='#e0e0e0')
        
        for spine in ax.spines.values():
            spine.set_color('#444')
        
        # Valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {int(width)}',
                   va='center', color='#e0e0e0', fontsize=9)
        
        fig.tight_layout()
        self.emotion_canvas.draw()
    
    def _draw_activities_chart(self):
        """Desenha gráfico de atividades."""
        activities = self.stats.get('activities', Counter())
        if isinstance(activities, dict):
            activities = Counter(activities)
        
        if not activities:
            return
        
        fig = self.activity_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Dados
        items = activities.most_common(7)  # Top 7
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        
        # Gráfico de barras horizontal
        bars = ax.barh(labels, values, color='#FF9800', edgecolor='#333')
        
        # Estilo
        ax.set_facecolor('#1e1e1e')
        ax.set_xlabel('Ocorrências', color='#e0e0e0')
        ax.set_title('🏃 Distribuição de Atividades', color='#e0e0e0', fontsize=12, pad=10)
        ax.tick_params(colors='#e0e0e0')
        
        for spine in ax.spines.values():
            spine.set_color('#444')
        
        # Valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {int(width)}',
                   va='center', color='#e0e0e0', fontsize=9)
        
        fig.tight_layout()
        self.activity_canvas.draw()
    
    def _draw_anomalies_chart(self):
        """Desenha gráfico de anomalias."""
        anomalies = self.stats.get('anomalies', Counter())
        if isinstance(anomalies, dict):
            anomalies = Counter(anomalies)
        
        if not anomalies or sum(anomalies.values()) == 0:
            fig = self.anomaly_canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Nenhuma anomalia detectada',
                    ha='center', va='center', fontsize=14, color='#4CAF50',
                    transform=ax.transAxes)
            ax.set_facecolor('#1e1e1e')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            self.anomaly_canvas.draw()
            return
        
        fig = self.anomaly_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        # Dados
        labels = list(anomalies.keys())
        values = list(anomalies.values())
        colors = ['#F44336', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5']
        
        # Gráfico de pizza
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors[:len(labels)],
            startangle=90,
            textprops={'color': '#e0e0e0', 'fontsize': 10}
        )
        
        # Estilo
        ax.set_facecolor('#1e1e1e')
        ax.set_title('⚠️ Distribuição de Anomalias', color='#e0e0e0', fontsize=12, pad=10)
        
        for autotext in autotexts:
            autotext.set_color('#1e1e1e')
            autotext.set_fontweight('bold')
        
        fig.tight_layout()
        self.anomaly_canvas.draw()
