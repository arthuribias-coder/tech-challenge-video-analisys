#!/usr/bin/env python3
"""
Tech Challenge - Fase 4: Script Principal
Ponto de entrada para execução da análise de vídeo.

Uso:
    python main.py                          # Análise com visualização
    python main.py --no-display             # Análise sem visualização
    python main.py --save                   # Salva vídeo processado
    python main.py --video caminho/video.mp4  # Vídeo específico
"""

import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.video_analyzer import main

if __name__ == "__main__":
    main()
