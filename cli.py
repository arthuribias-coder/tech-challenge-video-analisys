#!/usr/bin/env python3
"""
CLI para processamento de vídeo
Tech Challenge - Fase 4
"""

import argparse
import sys
import shutil
import cv2
from pathlib import Path

# Adiciona diretório raiz ao path para imports
sys.path.append(str(Path(__file__).parent))

from src.config import load_settings
from src.gui.threads.processor_thread_qt import ProcessorThreadQt
from PyQt6.QtCore import QCoreApplication

def main():
    parser = argparse.ArgumentParser(description="Processador de Vídeo CLI - Tech Challenge")
    
    parser.add_argument("video", help="Caminho do arquivo de vídeo")
    parser.add_argument("--config", help="Caminho do arquivo de configuração JSON (opcional)")
    parser.add_argument("--debug", action="store_true", help="Habilitar logs de debug")
    parser.add_argument("--output", help="Caminho de saída (opcional)")
    parser.add_argument("--no-gpu", action="store_true", help="Forçar uso de CPU")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {video_path}")
        sys.exit(1)
        
    # Carrega configurações
    # Se arquivo passado, mescla. Se não, usa padrão/settings.json
    settings = load_settings()
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            import json
            try:
                with open(config_path, 'r') as f:
                    custom_settings = json.load(f)
                    settings.update(custom_settings)
            except Exception as e:
                print(f"[ERRO] Falha ao ler config: {e}")
        else:
            print(f"[AVISO] Config não encontrada: {config_path}")
            
    # Sobrescreve debug se flag passada
    if args.debug:
        # Nota: ProcessorThreadQt usa DEBUG_LOGGING do config module ou self.debug_mode
        # Aqui vamos usar o self.debug_mode da thread
        pass

    # Setup output
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"processed_{video_path.name}"
        
    print("="*60)
    print("PROCESSADOR DE VÍDEO CLI")
    print("="*60)
    print(f"Vídeo: {video_path}")
    print(f"Saída: {output_path}")
    print(f"Debug: {'Sim' if args.debug else 'Não'}")
    print(f"GPU: {'Não' if args.no_gpu else 'Auto'}")
    
    # Necessário para QThread
    app = QCoreApplication(sys.argv)
    
    # Instancia processador
    # Reutilizamos ProcessorThreadQt para manter consistência lógica
    # embora seja acoplado ao Qt Signals
    
    use_gpu = False if args.no_gpu else settings.get('use_gpu')
    
    processor = ProcessorThreadQt(
        video_path=str(video_path),
        output_path=str(output_path),
        frame_skip=settings.get('frame_skip', 2),
        target_fps=settings.get('target_fps', 30),
        enable_preview=False, # Sem preview gráfico no CLI
        preview_fps=0,
        enable_object_detection=settings.get('enable_object_detection'),
        use_gpu=use_gpu,
        model_size=settings.get('model_size')
    )
    
    # Configura debug
    processor.set_debug_mode(args.debug)
    
    # Callbacks
    def on_progress(frame, total, fps, stats):
        percent = (frame / total * 100) if total > 0 else 0
        sys.stdout.write(f"\rProcessando: {percent:3.1f}% | Frame: {frame}/{total} | FPS: {fps:.1f}")
        sys.stdout.flush()
        
    def on_finished(stats, elapsed):
        print(f"\n\nConcluído em {elapsed:.1f}s")
        print("\nEstatísticas Finais:")
        print(f"Faces: {stats.get('faces', 0)}")
        print(f"Objetos: {sum(stats.get('objects', {}).values())}")
        print(f"Anomalias: {sum(stats.get('anomalies', {}).values())}")
        app.quit()
        
    def on_error(msg):
        print(f"\n[ERRO FATAL] {msg}")
        app.quit()
        
    processor.progress.connect(on_progress)
    processor.finished_signal.connect(on_finished)
    processor.error.connect(on_error)
    
    # Inicia
    print("\nIniciando análise...")
    processor.start()
    
    # Loop Qt
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        processor.stop()
        processor.wait()
        sys.exit(0)

if __name__ == "__main__":
    main()
