#!/usr/bin/env python3
"""
Tech Challenge - Fase 4: Análise de Vídeo
Script principal para processamento via linha de comando.

Uso:
    python main.py [caminho_video] [--output OUTPUT] [--skip N] [--show]
"""

import sys
import argparse
import cv2
import time
from pathlib import Path
from collections import Counter

from src import (
    FaceDetector, EmotionAnalyzer, ActivityDetector, 
    AnomalyDetector, draw_detections
)
from src.config import FRAME_SKIP, VIDEO_PATH, OUTPUT_DIR


def print_banner():
    """Exibe banner do programa."""
    print("=" * 70)
    print(" " * 15 + "TECH CHALLENGE - FASE 4")
    print(" " * 10 + "Análise de Vídeo com IA")
    print("=" * 70)
    print()


def print_stats(stats, elapsed, total_frames, fps_processing):
    """Imprime estatísticas da análise no console."""
    print("\n" + "=" * 70)
    print(" " * 25 + "RESULTADOS DA ANÁLISE")
    print("=" * 70)
    
    print(f"\n⏱️  DESEMPENHO:")
    print(f"   • Tempo total: {elapsed:.1f}s")
    print(f"   • FPS processamento: {fps_processing:.1f} fps")
    print(f"   • Frames processados: {total_frames}")
    
    print(f"\n👤 DETECÇÃO DE FACES:")
    print(f"   • Total de faces detectadas: {stats['faces']}")
    
    if stats['emotions']:
        print(f"\n😊 ANÁLISE DE EMOÇÕES (Top 5):")
        for i, (emotion, count) in enumerate(stats['emotions'].most_common(5), 1):
            bar = "█" * int(count / max(stats['emotions'].values()) * 30)
            print(f"   {i}. {emotion:15s} │ {bar} {count}")
    
    if stats['activities']:
        print(f"\n🏃 ATIVIDADES DETECTADAS (Top 5):")
        for i, (activity, count) in enumerate(stats['activities'].most_common(5), 1):
            bar = "█" * int(count / max(stats['activities'].values()) * 30)
            print(f"   {i}. {activity:20s} │ {bar} {count}")
    
    total_anomalies = sum(stats['anomalies'].values())
    if total_anomalies > 0:
        print(f"\n⚠️  ANOMALIAS DETECTADAS:")
        print(f"   • Total: {total_anomalies}")
        for anom_type, count in stats['anomalies'].most_common():
            print(f"     - {anom_type}: {count}")
    else:
        print(f"\n✅ Nenhuma anomalia detectada")
    
    print("\n" + "=" * 70)


def process_video(video_path, output_path, frame_skip=2, min_face_size=40):
    """
    Processa o vídeo completo.
    
    Args:
        video_path: Caminho do vídeo de entrada
        output_path: Caminho do vídeo de saída
        frame_skip: Intervalo de frames para detecção
        min_face_size: Tamanho mínimo de face em pixels
    
    Returns:
        dict: Estatísticas da análise
    """
    # Inicializa detectores
    print("🔧 Inicializando modelos de IA...")
    face_detector = FaceDetector(method="haar")
    emotion_analyzer = EmotionAnalyzer(method="fer")
    activity_detector = ActivityDetector(model_size="s")
    anomaly_detector = AnomalyDetector()
    print("✅ Modelos carregados!\n")
    
    # Abre vídeo
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
    
    # Propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Vídeo de entrada:")
    print(f"   • Arquivo: {Path(video_path).name}")
    print(f"   • Resolução: {width}x{height}")
    print(f"   • FPS: {fps:.1f}")
    print(f"   • Frames totais: {total_frames}")
    print(f"   • Duração: {total_frames/fps:.1f}s\n")
    
    # Configura gravador
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Estatísticas
    stats = {
        'faces': 0,
        'emotions': Counter(),
        'activities': Counter(),
        'anomalies': Counter()
    }
    
    # Cache de detecções
    cache = {
        'faces': [],
        'emotions': [],
        'activities': [],
        'anomalies': []
    }
    
    # Processamento
    frame_idx = 0
    start_time = time.time()
    last_progress = 0
    
    print("🎬 Processando vídeo...")
    print("┌" + "─" * 68 + "┐")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detecção a cada N frames
        if frame_idx % frame_skip == 0:
            # Faces
            cache['faces'] = face_detector.detect(frame)
            stats['faces'] += len(cache['faces'])
            
            # Emoções
            cache['emotions'] = []
            for face in cache['faces']:
                if face.bbox[2] >= min_face_size:
                    emotion = emotion_analyzer.analyze(frame, face.bbox, face.face_id)
                    cache['emotions'].append(emotion)
                    if emotion:
                        stats['emotions'][emotion.emotion_pt] += 1
                else:
                    cache['emotions'].append(None)
            
            # Atividades
            cache['activities'] = activity_detector.detect(frame)
            for activity in cache['activities']:
                stats['activities'][activity.activity_pt] += 1
            
            # Anomalias
            cache['anomalies'] = anomaly_detector.update(
                frame_idx,
                cache['faces'],
                [e for e in cache['emotions'] if e],
                cache['activities']
            )
            for anomaly in cache['anomalies']:
                stats['anomalies'][anomaly.anomaly_type] += 1
        
        # Desenha anotações
        annotated = draw_detections(
            frame,
            cache['faces'],
            cache['emotions'],
            cache['activities'],
            cache['anomalies'],
            min_face_size
        )
        out.write(annotated)
        
        # Barra de progresso
        progress = int((frame_idx / total_frames) * 100)
        if progress > last_progress:
            bar_length = 50
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps_proc if fps_proc > 0 else 0
            
            print(f"\r│ {bar} │ {progress:3d}% │ {fps_proc:5.1f} fps │ ETA: {eta:5.0f}s ", end="", flush=True)
            last_progress = progress
        
        frame_idx += 1
    
    print(f"\r│ {'█' * 50} │ 100% │                    ")
    print("└" + "─" * 68 + "┘\n")
    
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    fps_processing = frame_idx / elapsed
    
    return stats, elapsed, total_frames, fps_processing, output_path


def play_video(video_path):
    """
    Reproduz o vídeo usando OpenCV (player embutido).
    
    Args:
        video_path: Caminho do vídeo a ser reproduzido
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Vídeo não encontrado: {video_path}")
        return
    
    print(f"\n▶️  Reproduzindo: {video_path.name}")
    print("   Controles: [Q] Sair | [ESPAÇO] Pausar | [←/→] -10s/+10s")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Não foi possível abrir o vídeo")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(1000 / fps) if fps > 0 else 33  # ms entre frames
    
    window_name = "Tech Challenge - Video Analisado"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Fim do vídeo - volta ao início
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Mostra informações no frame
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = current_frame / fps
            total_time = total_frames / fps
            info = f"[{current_time:.1f}s / {total_time:.1f}s]"
            
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(delay if not paused else 100) & 0xFF
        
        if key == ord('q') or key == 27:  # Q ou ESC
            break
        elif key == ord(' '):  # Espaço - pausar
            paused = not paused
            status = "PAUSADO" if paused else "REPRODUZINDO"
            print(f"\r   Status: {status}          ", end="", flush=True)
        elif key == 81 or key == ord('a'):  # Seta esquerda ou A - voltar 10s
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - fps * 10))
        elif key == 83 or key == ord('d'):  # Seta direita ou D - avançar 10s
            current = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames, current + fps * 10))
        
        # Verifica se a janela foi fechada
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Reprodução encerrada")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Tech Challenge Fase 4 - Análise de Vídeo com IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                                    # Usa vídeo padrão
  python main.py input/meu_video.mp4                # Vídeo específico
  python main.py input/video.mp4 --skip 3           # Processa a cada 3 frames
  python main.py input/video.mp4 --show             # Reproduz após processar
        """
    )
    
    parser.add_argument(
        'video',
        nargs='?',
        default=VIDEO_PATH,
        help='Caminho do vídeo de entrada (padrão: definido em config)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Caminho do vídeo de saída (padrão: output/video_analisado.mp4)'
    )
    parser.add_argument(
        '--skip', '-s',
        type=int,
        default=FRAME_SKIP,
        help=f'Intervalo de frames para detecção (padrão: {FRAME_SKIP})'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Reproduz o vídeo após processamento'
    )
    parser.add_argument(
        '--min-face-size',
        type=int,
        default=40,
        help='Tamanho mínimo de face em pixels (padrão: 40)'
    )
    
    args = parser.parse_args()
    
    # Valida entrada
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Erro: Vídeo não encontrado: {video_path}")
        sys.exit(1)
    
    # Define saída
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / "video_analisado.mp4"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Banner
    print_banner()
    
    try:
        # Processa vídeo
        stats, elapsed, total_frames, fps_proc, output_file = process_video(
            video_path,
            output_path,
            frame_skip=args.skip,
            min_face_size=args.min_face_size
        )
        
        # Mostra resultados
        print_stats(stats, elapsed, total_frames, fps_proc)
        
        # Info do arquivo de saída
        output_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n💾 Vídeo processado salvo:")
        print(f"   • Arquivo: {output_path}")
        print(f"   • Tamanho: {output_size:.1f} MB\n")
        
        # Reproduz se solicitado
        if args.show:
            play_video(output_path)
        else:
            print(f"💡 Para reproduzir o vídeo, execute:")
            print(f"   python main.py --show\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processamento cancelado pelo usuário")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Erro durante processamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
