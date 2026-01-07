#!/usr/bin/env python3
"""
Tech Challenge - Fase 4: Entry Point GUI
Inicia a aplicação gráfica.

Uso:
    python gui_app.py
"""

import sys
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui import VideoAnalyzerGUI


def main():
    """Função principal."""
    print("=" * 70)
    print(" " * 15 + "TECH CHALLENGE - FASE 4")
    print(" " * 10 + "Análise de Vídeo com IA - GUI")
    print("=" * 70)
    print("\nIniciando interface gráfica...\n")
    
    try:
        app = VideoAnalyzerGUI()
        app.mainloop()
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
