#!/usr/bin/env python3
"""
Tech Challenge - Fase 4: Entry Point GUI
Inicia a aplicação gráfica com PyQt6.

Uso:
    python gui_app.py
"""

import sys
import logging
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent))

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

from PyQt6.QtWidgets import QApplication
from src.gui import MainWindow


def main():
    """Função principal."""
    print("=" * 70)
    print(" " * 15 + "TECH CHALLENGE - FASE 4")
    print(" " * 10 + "Análise de Vídeo com IA - GUI PyQt6")
    print("=" * 70)
    print("\nIniciando interface gráfica...\n")
    
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')  # Estilo moderno
        
        window = MainWindow()
        window.show()
        
        sys.exit(app.exec())
        
    except KeyboardInterrupt:
        print("\n\nAplicação encerrada pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nErro ao iniciar aplicação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
