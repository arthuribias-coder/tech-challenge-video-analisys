"""
Tech Challenge - Fase 4: Configurações do Projeto
Centraliza todas as configurações e constantes utilizadas na aplicação.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Diretórios
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

# Criar diretórios se não existirem
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Vídeo de entrada
VIDEO_PATH = os.getenv(
    "VIDEO_PATH", 
    str(INPUT_DIR / "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4")
)

# Configurações de processamento
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# OpenAI (opcional para geração de resumo)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Configurações de visualização
COLORS = {
    "face": (0, 255, 0),       # Verde para rostos
    "emotion": (255, 255, 0),  # Amarelo para emoções
    "activity": (0, 165, 255), # Laranja para atividades
    "anomaly": (0, 0, 255),    # Vermelho para anomalias
    "text_bg": (0, 0, 0),      # Fundo preto para texto
}

# Mapeamento de emoções (português)
EMOTION_LABELS = {
    "angry": "Raiva",
    "disgust": "Nojo",
    "fear": "Medo",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Surpreso",
    "neutral": "Neutro"
}

# Categorias de atividades detectáveis
ACTIVITY_CATEGORIES = {
    "walking": "Caminhando",
    "running": "Correndo",
    "sitting": "Sentado",
    "standing": "Em pé",
    "talking": "Conversando",
    "gesturing": "Gesticulando",
    "waving": "Acenando",
    "pointing": "Apontando",
    "dancing": "Dançando",
    "crouching": "Agachado",
    "arms_raised": "Braços Levantados",
    "unknown": "Desconhecido"
}

# Limiares para detecção de anomalias
ANOMALY_THRESHOLDS = {
    "sudden_movement": 50,      # Pixels de movimento brusco
    "emotion_change_rate": 0.5, # Taxa de mudança emocional
    "activity_duration": 2.0,   # Segundos mínimos para atividade válida
}
