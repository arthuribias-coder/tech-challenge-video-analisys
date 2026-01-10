"""
Tech Challenge - Fase 4: Configurações do Projeto
Centraliza todas as configurações e constantes utilizadas na aplicação.
"""

from pathlib import Path
import logging

# Configuração de logging
logger = logging.getLogger(__name__)

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
VIDEO_PATH = None

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
DEBUG_LOGGING = False  # Habilita logs detalhados de análise (Padrão: False)

# Constantes para intervalos de debug
DEBUG_LOG_INTERVAL = 30  # Loga a cada N frames quando debug está ativo

# ===== CONFIGURAÇÕES DO ANALISADOR DE EMOÇÕES =====
EMOTION_ANALYZER_METHOD = "deepface"
DEEPFACE_BACKBONE = "ArcFace"
DEEPFACE_CACHE_DIR = str(MODELS_DIR / "deepface")

# Thresholds adaptativos por emoção
EMOTION_THRESHOLDS = {
    'neutral': 0.25,
    'sad': 0.60,
    'happy': 0.35,
    'surprise': 0.50,
    'fear': 0.70,
    'angry': 0.50,
    'disgust': 0.55,
    'grimace': 0.35
}

# Configurações de preview em tempo real
ENABLE_PREVIEW = True
PREVIEW_FPS = 10
TARGET_FPS = 30

# ===== PERSISTÊNCIA DE DETECÇÕES (BBOX) =====
# Número de frames para manter detecções visíveis mesmo sem re-detecção
DETECTION_PERSISTENCE_FRAMES = 5  # Mantém bbox por 5 frames (~0.16s a 30fps)

# ===== CONFIGURAÇÕES DOS DETECTORES AVANÇADOS =====
ENABLE_OBJECT_DETECTION = True

# Tamanho dos modelos YOLO ('n'=nano, 's'=small, 'm'=medium, 'l'=large)
YOLO_MODEL_SIZE = "n"

# ===== LIMIARES DE DETECÇÃO DE ATIVIDADES =====
# Estes valores controlam como as poses são classificadas
ACTIVITY_POSE_THRESHOLDS = {
    # Em Pé (Standing) - verificações de postura vertical
    "standing_hip_knee_diff_min": 50,
    "standing_knee_ankle_diff_min": 30,
    "standing_hip_ankle_diff_min": 100,
    
    # Sentado vs Agachado
    "sitting_knee_hip_diff_max": 80,
    "sitting_torso_factor": 0.5,
    "crouching_hip_knee_diff": 30,
    "crouching_ankle_margin": 10,
    
    # Deitado
    "lying_horizontal_ratio": 2.0,
    "lying_min_horizontal_dist": 100,
    "lying_min_total_horizontal": 150,
    "lying_shoulder_width_threshold": 40,
    "lying_hip_width_threshold": 40,
    
    # Braços Levantados
    "arms_raised_hand_above_head": 80,
    
    # Acenando/Waving
    "waving_hand_above_shoulder": 40,
    "waving_elbow_angle_min": 40,
    "waving_elbow_angle_max": 160,
    
    # Apontando
    "pointing_arm_angle_min": 150,
    "pointing_horizontal_length": 80,
    "pointing_vertical_variance": 60,
    
    # Movimento
    "running_velocity_threshold": 80,
    "walking_velocity_threshold": 25,
    "gesture_velocity_threshold": 5,
    "sitting_max_velocity": 15,
    
    # Em Pé Frontal (pessoa de frente para câmera)
    "frontal_shoulder_hip_min": 40,
    "frontal_torso_vertical_ratio": 1.5,
    "frontal_head_above_shoulders": 20,
    "frontal_bbox_aspect_ratio": 1.2,
    
    # Cumprimento/Greeting - detecção de aperto de mão
    "greeting_wrist_distance_max": 60,
    "greeting_shoulder_distance_min": 150,
    "greeting_wrist_height_diff_max": 50,
}

# ===== AJUSTES CONTEXTUAIS DE EMOÇÃO =====
# Define pesos multiplicadores para emoções baseados no contexto da cena.
# Útil per corrigir vieses do modelo (ex: pessoa lendo = "triste"/"medo").
# Pesos < 1.0 penalizam, > 1.0 favorecem.
SCENE_EMOTION_WEIGHTS = {
    "office": {
        "neutral": 1.3,     # Favorece neutro (trabalho/concentração)
        "happy": 1.0,
        "sad": 0.4,         # Penaliza fortemente (leitura/cabeça baixa confunde modelo)
        "fear": 0.4,        # Penaliza medo (comum falso positivo em escritório)
        "disgust": 0.4,     # Penaliza nojo (sobrancelhas franzidas ao ler)
        "angry": 0.6,       # Penaliza raiva (concentração)
        "surprise": 0.8,
        "grimace": 0.5
    },
    "home": {
        "neutral": 1.1,
        "sad": 0.6,         # Relaxamento pode parecer tristeza
        "fear": 0.6,
        "disgust": 0.6
    },
    "outdoors": {
        "neutral": 1.0
        # Externo assume menos premissas
    }
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
        "keywords": [
            "office", "conference", "cubicle", "library", "classroom", "studio", 
            "lab", "school", "shop", "store", "barber", "restaurant", "hospital",
            "computer", "desk", "notebook", "laptop", "suit", "tie", "gown", "mortarboard",
            "turnstile", "monitor", "screen", "uniform", "accordion", "violin", "cello"
        ],
        "expected": ["person", "chair", "laptop", "tv", "cell phone", "book", "keyboard", "mouse", "desk", "tie", "suit"],
        "anomalous": ["sports ball", "skateboard", "baseball bat", "bed", "toilet", "bicycle", "car", "motorcycle", "surfboard"]
    },
    "home": {
        "keywords": [
            "living_room", "bedroom", "kitchen", "dining", "room", "home", "house", 
            "bath", "shower", "toilet", "theater", "couch", "sofa", "bed", 
            "tissue", "towel", "cap", "nipple", "cradle", "crib", "diaper",
            "washer", "dryer", "refrigerator", "microwave", "oven"
        ],
        "expected": ["person", "chair", "sofa", "tv", "bed", "refrigerator", "microwave", "cup", "bottle", "dining table", "sink", "toilet"],
        "anomalous": ["bus", "truck", "traffic light", "fire hydrant", "airplane"]
    },
    "outdoors": {
        "keywords": [
            "park", "street", "playground", "parking", "forest", "road", "vehicle", 
            "transport", "plane", "boat", "sport", "stadium", "jersey", "ski", 
            "mask", "parachute", "traffic", "sand", "sea", "mountain", "crutch"
        ],
        "expected": ["person", "bicycle", "car", "dog", "bird", "umbrella", "bench", "bus", "truck", "airplane", "traffic light"],
        "anomalous": ["tv", "mouse", "keyboard", "microwave", "refrigerator", "couch", "bed", "sink"]
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


def load_settings() -> dict:
    """Carrega configurações padrão ou de arquivo settings.json."""
    import json
    
    # Configurações padrão
    default_settings = {
        'frame_skip': FRAME_SKIP,
        'target_fps': TARGET_FPS,
        'enable_preview': ENABLE_PREVIEW,
        'preview_fps': PREVIEW_FPS,
        'enable_object_detection': ENABLE_OBJECT_DETECTION,
        'use_gpu': USE_GPU,
        'model_size': YOLO_MODEL_SIZE,
        'debug_logging': DEBUG_LOGGING
    }
    
    # Tenta carregar settings.json se existir
    settings_file = BASE_DIR / "settings.json"
    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                custom_settings = json.load(f)
                default_settings.update(custom_settings)
                logger.info(f"Configurações carregadas de {settings_file}")
        except Exception as e:
            logger.warning(f"Erro ao carregar settings.json: {e}")
    
    return default_settings
