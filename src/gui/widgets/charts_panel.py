"""
Painel de gráficos com matplotlib
"""

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ChartsPanel(ctk.CTkFrame):
    """Painel inferior com gráficos."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.current_tab = "emotions"
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura interface do painel."""
        # Título e tabs
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        title = ctk.CTkLabel(
            header_frame,
            text="📈 GRÁFICOS",
            font=("Arial", 14, "bold")
        )
        title.pack(side="left")
        
        # Tabs
        tabs_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        tabs_frame.pack(side="right")
        
        self.tab_buttons = {}
        tabs = [
            ("emotions", "Emoções"),
            ("activities", "Atividades"),
            ("timeline", "Timeline"),
            ("anomalies", "Anomalias")
        ]
        
        for tab_id, tab_name in tabs:
            btn = ctk.CTkButton(
                tabs_frame,
                text=tab_name,
                width=100,
                height=30,
                command=lambda t=tab_id: self._switch_tab(t)
            )
            btn.pack(side="left", padx=2)
            self.tab_buttons[tab_id] = btn
        
        # Container para gráficos
        self.chart_container = ctk.CTkFrame(self)
        self.chart_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Figura matplotlib
        self.figure = Figure(figsize=(10, 4), facecolor='#2b2b2b')
        self.canvas = FigureCanvasTkAgg(self.figure, self.chart_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Mostra tab inicial
        self._switch_tab("emotions")
    
    def _switch_tab(self, tab_id):
        """Muda tab ativa."""
        self.current_tab = tab_id
        
        # Atualiza aparência dos botões
        for tid, btn in self.tab_buttons.items():
            if tid == tab_id:
                btn.configure(fg_color="#1f6aa5")
            else:
                btn.configure(fg_color="#2b2b2b")
        
        # Redesenha gráfico
        self._draw_chart()
    
    def update_data(self, stats):
        """
        Atualiza dados e redesenha gráfico.
        
        Args:
            stats: Dicionário com estatísticas
        """
        self.stats = stats
        self._draw_chart()
    
    def _draw_chart(self):
        """Desenha gráfico baseado na tab ativa."""
        if not hasattr(self, 'stats'):
            return
        
        self.figure.clear()
        
        if self.current_tab == "emotions":
            self._draw_emotions_chart()
        elif self.current_tab == "activities":
            self._draw_activities_chart()
        elif self.current_tab == "timeline":
            self._draw_timeline_chart()
        elif self.current_tab == "anomalies":
            self._draw_anomalies_chart()
        
        self.canvas.draw()
    
    def _draw_emotions_chart(self):
        """Gráfico de barras das emoções."""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        
        if not self.stats['emotions']:
            ax.text(0.5, 0.5, 'Nenhum dado disponível',
                   ha='center', va='center', fontsize=14, color='#888888')
            return
        
        emotions = self.stats['emotions'].most_common(5)
        labels = [e[0] for e in emotions]
        values = [e[1] for e in emotions]
        
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']
        bars = ax.barh(labels, values, color=colors[:len(labels)])
        
        ax.set_xlabel('Contagem', color='white')
        ax.set_title('Top 5 Emoções Detectadas', color='white', pad=20)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {int(width)}',
                   va='center', color='white', fontweight='bold')
    
    def _draw_activities_chart(self):
        """Gráfico de barras das atividades."""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        
        if not self.stats['activities']:
            ax.text(0.5, 0.5, 'Nenhum dado disponível',
                   ha='center', va='center', fontsize=14, color='#888888')
            return
        
        activities = self.stats['activities'].most_common(5)
        labels = [a[0] for a in activities]
        values = [a[1] for a in activities]
        
        colors = ['#FF9800', '#FF5722', '#FFC107', '#795548', '#607D8B']
        bars = ax.barh(labels, values, color=colors[:len(labels)])
        
        ax.set_xlabel('Contagem', color='white')
        ax.set_title('Top 5 Atividades Detectadas', color='white', pad=20)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f' {int(width)}',
                   va='center', color='white', fontweight='bold')
    
    def _draw_timeline_chart(self):
        """Placeholder para timeline."""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.text(0.5, 0.5, 'Timeline (Em desenvolvimento)',
               ha='center', va='center', fontsize=14, color='#888888')
    
    def _draw_anomalies_chart(self):
        """Gráfico de anomalias."""
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        
        if not self.stats['anomalies'] or sum(self.stats['anomalies'].values()) == 0:
            ax.text(0.5, 0.5, 'Nenhuma anomalia detectada',
                   ha='center', va='center', fontsize=14, color='#4CAF50')
            return
        
        anomalies = list(self.stats['anomalies'].items())
        labels = [a[0].name if hasattr(a[0], 'name') else str(a[0]) for a in anomalies]
        values = [a[1] for a in anomalies]
        
        colors = ['#F44336', '#E91E63', '#9C27B0']
        ax.pie(values, labels=labels, autopct='%1.1f%%',
              colors=colors[:len(labels)], textprops={'color': 'white'})
        ax.set_title('Distribuição de Anomalias', color='white', pad=20)
