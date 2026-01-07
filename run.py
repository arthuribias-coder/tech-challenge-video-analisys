#!/usr/bin/env python3
"""
Tech Challenge - Fase 4: Script de Execução Rápida
Versão simplificada para demonstração.

Uso:
    python run.py              # Executa sem janela (salva vídeo)
    python run.py --display    # Tenta mostrar janela (requer X11)
    python run.py --no-save    # Não salva vídeo de saída
"""

import cv2
import sys
import os
from pathlib import Path

# Adiciona src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import VIDEO_PATH, OUTPUT_DIR, REPORTS_DIR
from src.video_analyzer import VideoAnalyzer
from src.report_generator import ReportGenerator


def check_display_available():
    """Verifica se o display X11 está disponível."""
    display = os.environ.get('DISPLAY')
    if not display:
        return False
    try:
        # Tenta criar uma janela de teste
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("test")
        return True
    except:
        return False


def run_analysis(show_display: bool = False, save_video: bool = True):
    """Executa análise completa do vídeo."""
    print("="*60)
    print("TECH CHALLENGE - FASE 4")
    print("Análise de Vídeo com IA")
    print("="*60)
    
    # Verifica se vídeo existe
    video_path = VIDEO_PATH
    if not Path(video_path).exists():
        print(f"\n[ERRO] Vídeo não encontrado: {video_path}")
        print("\nCertifique-se de que o vídeo está no diretório correto.")
        return None
    
    # Verifica display se solicitado
    if show_display:
        if not check_display_available():
            print("\n[AVISO] Display X11 não disponível. Desabilitando visualização.")
            print("[AVISO] Use --no-display ou execute em ambiente com servidor gráfico.")
            show_display = False
    
    print(f"\n[INFO] Vídeo: {video_path}")
    print(f"[INFO] Visualização: {'Habilitada' if show_display else 'Desabilitada'}")
    print(f"[INFO] Salvar vídeo: {'Sim' if save_video else 'Não'}")
    print("[INFO] Iniciando análise...\n")
    
    # Cria analisador
    analyzer = VideoAnalyzer(
        video_path=video_path,
        frame_skip=2,  # Processa 1 a cada 2 frames para performance
        show_visualization=show_display,
        save_output=save_video
    )
    
    # Executa análise
    result = analyzer.analyze()
    
    # Gera relatório
    print("\n[INFO] Gerando relatório...")
    generator = ReportGenerator(use_llm=False)  # Template mode (não requer API key)
    
    video_stem = Path(video_path).stem
    report_md = REPORTS_DIR / f"relatorio_{video_stem}.md"
    report_json = REPORTS_DIR / f"relatorio_{video_stem}.json"
    
    report = generator.generate(result, str(report_md))
    generator.save_json_report(result, str(report_json))
    
    # Exibe resumo
    print("\n" + "="*60)
    print("RESUMO DA ANÁLISE")
    print("="*60)
    print(f"• Frames analisados: {result.total_frames}")
    print(f"• Pessoas detectadas: {result.unique_faces}")
    print(f"• Anomalias encontradas: {result.total_anomalies}")
    print(f"• Tempo de processamento: {result.processing_time_seconds:.1f}s")
    
    # Mostra emoções detectadas
    if result.emotions_summary:
        print("\n[EMOÇÕES DETECTADAS]")
        for emotion, count in sorted(result.emotions_summary.items(), key=lambda x: -x[1])[:5]:
            print(f"  • {emotion}: {count}")
    
    # Mostra atividades detectadas
    if result.activities_summary:
        print("\n[ATIVIDADES DETECTADAS]")
        for activity, count in sorted(result.activities_summary.items(), key=lambda x: -x[1])[:5]:
            print(f"  • {activity}: {count}")
    
    print(f"\n[ARQUIVOS GERADOS]")
    print(f"  • Relatório MD: {report_md}")
    print(f"  • Relatório JSON: {report_json}")
    if save_video:
        output_video = OUTPUT_DIR / f"analyzed_{video_stem}.mp4"
        print(f"  • Vídeo análise: {output_video}")
    
    print("\n" + "="*60)
    print("Análise concluída com sucesso!")
    print("="*60)
    
    return result


if __name__ == "__main__":
    # Parse argumentos simples
    show_display = "--display" in sys.argv
    save_video = "--no-save" not in sys.argv
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        sys.exit(0)
    
    run_analysis(show_display=show_display, save_video=save_video)
