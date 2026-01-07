"""
Painel de estatísticas em tempo real
"""

import customtkinter as ctk
from collections import Counter


class StatsPanel(ctk.CTkFrame):
    """Painel lateral com estatísticas da análise."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.stats = {
            'faces': 0,
            'emotions': Counter(),
            'activities': Counter(),
            'anomalies': Counter()
        }
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura interface do painel."""
        # Título
        title = ctk.CTkLabel(
            self,
            text="[=] ESTATÍSTICAS",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(10, 20), padx=10)
        
        # Faces
        self.faces_frame = self._create_stat_section("[#] Faces Detectadas")
        self.faces_label = ctk.CTkLabel(
            self.faces_frame,
            text="0",
            font=("Arial", 24, "bold"),
            text_color="#4CAF50"
        )
        self.faces_label.pack(pady=5)
        
        # Emoção Dominante
        self.emotion_frame = self._create_stat_section("[:)] Emoção Principal")
        self.emotion_label = ctk.CTkLabel(
            self.emotion_frame,
            text="--",
            font=("Arial", 18),
            text_color="#2196F3"
        )
        self.emotion_label.pack(pady=5)
        self.emotion_count_label = ctk.CTkLabel(
            self.emotion_frame,
            text="(0 ocorrências)",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.emotion_count_label.pack()
        
        # Atividade Dominante
        self.activity_frame = self._create_stat_section("[>] Atividade Principal")
        self.activity_label = ctk.CTkLabel(
            self.activity_frame,
            text="--",
            font=("Arial", 18),
            text_color="#FF9800"
        )
        self.activity_label.pack(pady=5)
        self.activity_count_label = ctk.CTkLabel(
            self.activity_frame,
            text="(0 ocorrências)",
            font=("Arial", 12),
            text_color="#888888"
        )
        self.activity_count_label.pack()
        
        # Anomalias
        self.anomaly_frame = self._create_stat_section("[!] Anomalias")
        self.anomaly_label = ctk.CTkLabel(
            self.anomaly_frame,
            text="0",
            font=("Arial", 24, "bold"),
            text_color="#F44336"
        )
        self.anomaly_label.pack(pady=5)
        
        # Botão Ver Detalhes
        self.details_btn = ctk.CTkButton(
            self,
            text="Ver Detalhes Completos",
            command=self._show_details,
            state="disabled"
        )
        self.details_btn.pack(pady=20, padx=10, fill="x")
    
    def _create_stat_section(self, title):
        """Cria seção de estatística."""
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(pady=10, padx=10, fill="x")
        
        label = ctk.CTkLabel(
            frame,
            text=title,
            font=("Arial", 13)
        )
        label.pack(anchor="w", pady=(0, 5))
        
        return frame
    
    def update_stats(self, stats):
        """
        Atualiza estatísticas exibidas.
        
        Args:
            stats: Dicionário com estatísticas atuais
        """
        self.stats = stats
        
        # Faces
        self.faces_label.configure(text=str(stats['faces']))
        
        # Emoção dominante
        if stats['emotions']:
            top_emotion, count = stats['emotions'].most_common(1)[0]
            self.emotion_label.configure(text=top_emotion)
            self.emotion_count_label.configure(text=f"({count} ocorrências)")
        else:
            self.emotion_label.configure(text="--")
            self.emotion_count_label.configure(text="(nenhuma)")
        
        # Atividade dominante
        if stats['activities']:
            top_activity, count = stats['activities'].most_common(1)[0]
            self.activity_label.configure(text=top_activity)
            self.activity_count_label.configure(text=f"({count} ocorrências)")
        else:
            self.activity_label.configure(text="--")
            self.activity_count_label.configure(text="(nenhuma)")
        
        # Anomalias
        total_anomalies = sum(stats['anomalies'].values())
        self.anomaly_label.configure(text=str(total_anomalies))
        
        # Habilita botão de detalhes se houver dados
        if stats['faces'] > 0:
            self.details_btn.configure(state="normal")
    
    def reset(self):
        """Reseta estatísticas."""
        self.stats = {
            'faces': 0,
            'emotions': Counter(),
            'activities': Counter(),
            'anomalies': Counter()
        }
        self.update_stats(self.stats)
        self.details_btn.configure(state="disabled")
    
    def _show_details(self):
        """Abre janela com detalhes completos."""
        # TODO: Implementar janela de detalhes
        dialog = ctk.CTkToplevel(self)
        dialog.title("Detalhes da Análise")
        dialog.geometry("600x400")
        
        # Texto com estatísticas
        text = ctk.CTkTextbox(dialog, width=580, height=380)
        text.pack(padx=10, pady=10)
        
        details = f"""ESTATÍSTICAS COMPLETAS

[FACES] FACES DETECTADAS: {self.stats['faces']}

[EMOÇÕES]:
"""
        for emotion, count in self.stats['emotions'].most_common():
            details += f"   - {emotion}: {count}\n"
        
        details += "\n[ATIVIDADES]:\n"
        for activity, count in self.stats['activities'].most_common():
            details += f"   - {activity}: {count}\n"
        
        details += f"\n[!] ANOMALIAS: {sum(self.stats['anomalies'].values())}\n"
        for anomaly, count in self.stats['anomalies'].most_common():
            details += f"   - {anomaly}: {count}\n"
        
        text.insert("1.0", details)
        text.configure(state="disabled")
