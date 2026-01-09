"""
Tech Challenge - Fase 4: Configurações do Projeto
Centraliza todas as configurações e constantes utilizadas na aplicação.
"""

from pathlib import Path
import os

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

# Vídeo de entrada padrão
VIDEO_PATH = str(INPUT_DIR / "Unlocking Facial Recognition_ Diverse Activities Analysis.mp4")

# ===== CONFIGURAÇÕES DE GPU =====
# Valores: "auto" (detecta automaticamente), "true" (força GPU), "false" (força CPU)
USE_GPU = "auto"

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
FRAME_SKIP = 2
CONFIDENCE_THRESHOLD = 0.5

# ===== CONFIGURAÇÕES DO ANALISADOR DE EMOÇÕES =====
EMOTION_ANALYZER_METHOD = "deepface"
DEEPFACE_BACKBONE = "ArcFace"
DEEPFACE_CACHE_DIR = str(MODELS_DIR / "deepface")

# Thresholds adaptativos por emoção
EMOTION_THRESHOLDS = {
    'neutral': 0.25,
    'sad': 0.30,
    'happy': 0.35,
    'surprise': 0.40,
    'fear': 0.40,
    'angry': 0.40,
    'disgust': 0.45,
    'grimace': 0.35
}

# Configurações de preview em tempo real
ENABLE_PREVIEW = True
PREVIEW_FPS = 10
TARGET_FPS = 30

# ===== CONFIGURAÇÕES DOS DETECTORES AVANÇADOS =====
ENABLE_OBJECT_DETECTION = True

# Tamanho dos modelos YOLO ('n'=nano, 's'=small, 'm'=medium, 'l'=large)
YOLO_MODEL_SIZE = "n"

# ===== LIMIARES DE DETECÇÃO DE ATIVIDADES =====
# Estes valores controlam como as poses são classificadas
ACTIVITY_POSE_THRESHOLDS = {
    # Em Pé (Standing) - verificações de postura vertical
    "standing_hip_knee_diff_min": 50,     # Diferença mínima Y entre quadril e joelho para em pé (px)
    "standing_knee_ankle_diff_min": 30,   # Diferença mínima Y entre joelho e tornozelo (px)
    "standing_hip_ankle_diff_min": 100,   # Diferença mínima Y quadril-tornozelo (px)
    
    # Sentado vs Agachado
    "sitting_knee_hip_diff_max": 80,      # Máxima diferença Y entre quadril e joelho (px)
    "sitting_torso_factor": 0.5,          # Fator para comparar diferença com comprimento do torso
    "crouching_hip_knee_diff": 30,        # Máxima diferença Y entre quadril e joelho para agachado (px)
    "crouching_ankle_margin": 10,         # Margem entre joelho e tornozelo para agachado (px)
    
    # Deitado
    "lying_horizontal_ratio": 2.0,        # Razão horizontal/vertical para considerar deitado
    "lying_min_horizontal_dist": 100,     # Distância horizontal mínima (px)
    "lying_min_total_horizontal": 150,    # Distância horizontal total ombro-tornozelo
    "lying_shoulder_width_threshold": 40, # Largura dos ombros (pessoa de lado)
    "lying_hip_width_threshold": 40,      # Largura do quadril (pessoa de lado)
    
    # Braços Levantados
    "arms_raised_hand_above_head": 80,    # Mão deve estar acima da cabeça por este valor (px)
    
    # Acenando/Waving
    "waving_hand_above_shoulder": 40,     # Mão deve estar acima do ombro por este valor (px)
    "waving_elbow_angle_min": 40,         # Ângulo mínimo do cotovelo
    "waving_elbow_angle_max": 160,        # Ângulo máximo do cotovelo
    
    # Apontando
    "pointing_arm_angle_min": 150,        # Ângulo mínimo do braço estendido
    "pointing_horizontal_length": 80,     # Comprimento mínimo horizontal
    "pointing_vertical_variance": 60,     # Variação máxima vertical
    
    # Movimento
    "running_velocity_threshold": 80,     # Velocidade mínima para correr (px/frame)
    "walking_velocity_threshold": 25,     # Velocidade mínima para caminhar (px/frame)
    "gesture_velocity_threshold": 5,      # Velocidade mínima para considerar gestos
    "sitting_max_velocity": 15,           # Velocidade MÁXIMA para considerar sentado (px/frame)
    
    # Em Pé Frontal (pessoa de frente para câmera)
    "frontal_shoulder_hip_min": 40,       # Distância Y mínima ombro-quadril (px) para postura vertical
    "frontal_torso_vertical_ratio": 1.5,  # Torso deve ser mais vertical que horizontal (ratio)
    "frontal_head_above_shoulders": 20,   # Cabeça deve estar acima dos ombros (px)
    "frontal_bbox_aspect_ratio": 1.2,     # BBox mais alto que largo indica pessoa em pé
    
    # Cumprimento/Greeting - detecção de aperto de mão
    "greeting_wrist_distance_max": 60,    # Distância máxima entre pulsos (px)
    "greeting_shoulder_distance_min": 150, # Distância mínima entre ombros (px)
    "greeting_wrist_height_diff_max": 50, # Variação máxima de altura entre pulsos (px)
}

# OpenAI (opcional para geração de resumo)
OPENAI_API_KEY = None
OPENAI_MODEL = "gpt-4o-mini"

# Configurações de visualização
COLORS = {
    "face": (0, 255, 0),       # Verde para rostos
    "emotion": (255, 255, 0),  # Amarelo para emoções
    "activity": (0, 165, 255), # Laranja para atividades
    "anomaly": (0, 0, 255),    # Vermelho para anomalias
    "object": (180, 0, 180),   # Roxo para objetos
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
    "grimace": "Careta"
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
    "greeting": "Cumprimentando",
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

# Regras de contexto de cena
SCENE_CONTEXT_RULES = {
    "office": {
        "keywords": ["office", "conference_room", "cubicle", "library", "classroom", "studio"],
        "expected": ["person", "chair", "laptop", "tv", "cell phone", "book", "keyboard", "mouse", "desk"],
        "anomalous": ["umbrella", "sports ball", "skateboard", "baseball bat", "bed", "toilet", "bicycle", "car", "motorcycle"]
    },
    "home": {
        "keywords": ["living_room", "bedroom", "kitchen", "dining_room"],
        "expected": ["person", "chair", "sofa", "tv", "bed", "refrigerator", "microwave", "cup", "bottle"],
        "anomalous": ["bicycle", "car", "bus", "truck", "traffic light", "fire hydrant"]
    },
    "outdoors": {
        "keywords": ["park", "street", "playground", "parking", "forest"],
        "expected": ["person", "bicycle", "car", "dog", "bird", "umbrella", "bench"],
        "anomalous": ["tv", "mouse", "keyboard", "microwave", "refrigerator", "couch", "bed"]
    }
}

# Mapeamento de objetos COCO para português
OBJECT_LABELS = {
    # Eletrônicos
    "tv": "TV", "laptop": "Notebook", "cell phone": "Celular", "remote": "Controle Remoto",
    "keyboard": "Teclado", "mouse": "Mouse", "refrigerator": "Geladeira",
    "microwave": "Micro-ondas", "oven": "Forno", "toaster": "Torradeira",
    # Móveis
    "chair": "Cadeira", "couch": "Sofá", "bed": "Cama", "dining table": "Mesa",
    "toilet": "Vaso Sanitário", "sink": "Pia",
    # Veículos
    "car": "Carro", "motorcycle": "Moto", "bicycle": "Bicicleta", "bus": "Ônibus",
    "truck": "Caminhão", "airplane": "Avião", "train": "Trem", "boat": "Barco",
    # Acessórios
    "backpack": "Mochila", "umbrella": "Guarda-chuva", "handbag": "Bolsa",
    "tie": "Gravata", "suitcase": "Mala",
    # Esportes
    "sports ball": "Bola", "kite": "Pipa", "baseball bat": "Taco de Baseball",
    "baseball glove": "Luva de Baseball", "skateboard": "Skate", "surfboard": "Prancha de Surf",
    "tennis racket": "Raquete de Tênis", "frisbee": "Frisbee", "skis": "Esquis", "snowboard": "Snowboard",
    # Animais
    "bird": "Pássaro", "cat": "Gato", "dog": "Cachorro", "horse": "Cavalo",
    "sheep": "Ovelha", "cow": "Vaca", "elephant": "Elefante", "bear": "Urso",
    "zebra": "Zebra", "giraffe": "Girafa",
    # Itens diversos
    "book": "Livro", "clock": "Relógio", "vase": "Vaso", "scissors": "Tesoura",
    "teddy bear": "Ursinho de Pelúcia", "hair drier": "Secador de Cabelo",
    "toothbrush": "Escova de Dentes",
    # Comida/Bebida
    "bottle": "Garrafa", "wine glass": "Taça de Vinho", "cup": "Xícara",
    "fork": "Garfo", "knife": "Faca", "spoon": "Colher", "bowl": "Tigela",
    "banana": "Banana", "apple": "Maçã", "sandwich": "Sanduíche", "orange": "Laranja",
    "broccoli": "Brócolis", "carrot": "Cenoura", "hot dog": "Cachorro-Quente",
    "pizza": "Pizza", "donut": "Rosquinha", "cake": "Bolo",
    # Outros
    "person": "Pessoa", "traffic light": "Semáforo", "fire hydrant": "Hidrante",
    "stop sign": "Placa de Pare", "parking meter": "Parquímetro", "bench": "Banco",
    "potted plant": "Planta",
}
