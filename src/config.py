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

# ===== CONFIGURAÇÕES DE GPU =====
# Define se deve usar GPU (CUDA) quando disponível
# Valores: "auto" (detecta automaticamente), "true" (força GPU), "false" (força CPU)
USE_GPU = os.getenv("USE_GPU", "auto").lower()

def is_gpu_available() -> bool:
    """Verifica se GPU CUDA está disponível."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def should_use_gpu() -> bool:
    """Retorna True se deve usar GPU baseado na configuração."""
    if USE_GPU == "false":
        return False
    if USE_GPU == "true":
        return True
    # auto: usa se disponível
    return is_gpu_available()

def get_device() -> str:
    """Retorna o device PyTorch a ser usado ('cuda' ou 'cpu')."""
    return "cuda" if should_use_gpu() else "cpu"

# ===== CONFIGURAÇÕES DE PROCESSAMENTO =====
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "2"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# ===== CONFIGURAÇÕES DO ANALISADOR DE EMOÇÕES =====
# Método principal: deepface (2-3x mais rápido que FER)
EMOTION_ANALYZER_METHOD = os.getenv("EMOTION_ANALYZER_METHOD", "deepface")

# DeepFace backbone: 'VGG-Face', 'FaceNet', 'ArcFace', 'Facenet512'
DEEPFACE_BACKBONE = os.getenv("DEEPFACE_BACKBONE", "ArcFace")

# Thresholds adaptativos por emoção (DeepFace scores são maiores que FER)
# Emoções passivas (neutral, sad) requerem threshold menor pois são expressões sutis
EMOTION_THRESHOLDS = {
    'neutral': 0.25,   # Expressões neutras são sutis e comuns
    'sad': 0.30,       # Tristeza pode ser sutil
    'happy': 0.35,     # Sorrisos leves precisam threshold menor
    'surprise': 0.40,  # Surpresa geralmente é mais óbvia
    'fear': 0.40,      # Medo tende a ser pronunciado
    'angry': 0.40,     # Raiva é geralmente clara
    'disgust': 0.45,   # Nojo é expressão complexa
    'grimace': 0.35    # Caretas variam em intensidade
}

# Configurações de preview em tempo real
ENABLE_PREVIEW = os.getenv("ENABLE_PREVIEW", "true").lower() == "true"
PREVIEW_FPS = int(os.getenv("PREVIEW_FPS", "10"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "30"))

# ===== CONFIGURAÇÕES DOS DETECTORES AVANÇADOS =====
# Habilita/desabilita detectores (pode impactar performance)
ENABLE_OBJECT_DETECTION = os.getenv("ENABLE_OBJECT_DETECTION", "true").lower() == "true"
ENABLE_OVERLAY_DETECTION = os.getenv("ENABLE_OVERLAY_DETECTION", "false").lower() == "true"
ENABLE_SEGMENT_VALIDATION = os.getenv("ENABLE_SEGMENT_VALIDATION", "false").lower() == "true"

# Tamanho dos modelos YOLO ('n'=nano, 's'=small, 'm'=medium, 'l'=large)
# Modelos maiores são mais precisos mas mais lentos
YOLO_MODEL_SIZE = os.getenv("YOLO_MODEL_SIZE", "n")

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
    "neutral": "Neutro",
    "grimace": "Careta"  # Expressão exagerada/engraçada
}

# Categorias de atividades detectáveis
ACTIVITY_CATEGORIES = {
    "walking": "Caminhando",
    "running": "Correndo",
    "sitting": "Sentado",
    "lying": "Deitado",
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

# Mapeamento de anomalias (português)
ANOMALY_LABELS = {
    "sudden_movement": "Movimento Brusco",
    "emotion_spike": "Mudança Emocional",
    "unusual_activity": "Atividade Incomum",
    "crowd_anomaly": "Anomalia de Grupo",
    "prolonged_inactivity": "Inatividade Prolongada",
    "visual_overlay": "Texto/Overlay",
    "scene_inconsistency": "Inconsistência de Cena",
    "sudden_object_appear": "Objeto Súbito",
    "silhouette_anomaly": "Silhueta Anômala",
}

# Limiares para detecção de anomalias
ANOMALY_THRESHOLDS = {
    "sudden_movement": 50,      # Pixels de movimento brusco
    "emotion_change_rate": 0.5, # Taxa de mudança emocional
    "activity_duration": 2.0,   # Segundos mínimos para atividade válida
}

# Mapeamento de objetos COCO para português
OBJECT_LABELS = {
    # Eletrônicos
    "tv": "TV",
    "laptop": "Notebook",
    "cell phone": "Celular",
    "remote": "Controle Remoto",
    "keyboard": "Teclado",
    "mouse": "Mouse",
    "refrigerator": "Geladeira",
    "microwave": "Micro-ondas",
    "oven": "Forno",
    "toaster": "Torradeira",
    # Móveis
    "chair": "Cadeira",
    "couch": "Sofá",
    "bed": "Cama",
    "dining table": "Mesa",
    "toilet": "Vaso Sanitário",
    "sink": "Pia",
    # Veículos
    "car": "Carro",
    "motorcycle": "Moto",
    "bicycle": "Bicicleta",
    "bus": "Ônibus",
    "truck": "Caminhão",
    "airplane": "Avião",
    "train": "Trem",
    "boat": "Barco",
    # Acessórios
    "backpack": "Mochila",
    "umbrella": "Guarda-chuva",
    "handbag": "Bolsa",
    "tie": "Gravata",
    "suitcase": "Mala",
    # Esportes
    "sports ball": "Bola",
    "kite": "Pipa",
    "baseball bat": "Taco de Baseball",
    "baseball glove": "Luva de Baseball",
    "skateboard": "Skate",
    "surfboard": "Prancha de Surf",
    "tennis racket": "Raquete de Tênis",
    "frisbee": "Frisbee",
    "skis": "Esquis",
    "snowboard": "Snowboard",
    # Animais
    "bird": "Pássaro",
    "cat": "Gato",
    "dog": "Cachorro",
    "horse": "Cavalo",
    "sheep": "Ovelha",
    "cow": "Vaca",
    "elephant": "Elefante",
    "bear": "Urso",
    "zebra": "Zebra",
    "giraffe": "Girafa",
    # Itens diversos
    "book": "Livro",
    "clock": "Relógio",
    "vase": "Vaso",
    "scissors": "Tesoura",
    "teddy bear": "Ursinho de Pelúcia",
    "hair drier": "Secador de Cabelo",
    "toothbrush": "Escova de Dentes",
    # Comida/Bebida
    "bottle": "Garrafa",
    "wine glass": "Taça de Vinho",
    "cup": "Xícara",
    "fork": "Garfo",
    "knife": "Faca",
    "spoon": "Colher",
    "bowl": "Tigela",
    "banana": "Banana",
    "apple": "Maçã",
    "sandwich": "Sanduíche",
    "orange": "Laranja",
    "broccoli": "Brócolis",
    "carrot": "Cenoura",
    "hot dog": "Cachorro-Quente",
    "pizza": "Pizza",
    "donut": "Rosquinha",
    "cake": "Bolo",
    # Outros
    "person": "Pessoa",
    "traffic light": "Semáforo",
    "fire hydrant": "Hidrante",
    "stop sign": "Placa de Pare",
    "parking meter": "Parquímetro",
    "bench": "Banco",
    "potted plant": "Planta",
}
