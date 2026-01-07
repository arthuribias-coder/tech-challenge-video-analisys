"""
Tech Challenge - Fase 4: Processador de Vídeo
Módulo principal que integra todos os componentes para análise de vídeo em tempo real.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

from .config import (
    VIDEO_PATH, FRAME_SKIP, CONFIDENCE_THRESHOLD,
    OUTPUT_DIR, REPORTS_DIR, COLORS
)
from .face_detector import FaceDetector, FaceDetection
from .emotion_analyzer import EmotionAnalyzer, EmotionResult
from .activity_detector import ActivityDetector, ActivityDetection
from .anomaly_detector import AnomalyDetector, AnomalyEvent, draw_anomaly
from .report_generator import ReportGenerator, VideoAnalysisResult


@dataclass
class FrameAnalysis:
    """Resultado da análise de um frame."""
    frame_number: int
    faces: List[FaceDetection]
    emotions: List[EmotionResult]
    activities: List[ActivityDetection]
    anomalies: List[AnomalyEvent]


class VideoAnalyzer:
    """
    Analisador de vídeo em tempo real.
    Integra reconhecimento facial, análise de emoções, detecção de atividades e anomalias.
    """
    
    def __init__(
        self,
        video_path: str = VIDEO_PATH,
        frame_skip: int = FRAME_SKIP,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        show_visualization: bool = True,
        save_output: bool = False
    ):
        """
        Inicializa o analisador de vídeo.
        
        Args:
            video_path: Caminho do vídeo para análise
            frame_skip: Processar 1 a cada N frames
            confidence_threshold: Limiar de confiança
            show_visualization: Mostrar visualização em tempo real
            save_output: Salvar vídeo processado
        """
        self.video_path = video_path
        self.frame_skip = frame_skip
        self.confidence_threshold = confidence_threshold
        self.show_visualization = show_visualization
        self.save_output = save_output
        
        # Inicializa componentes
        print("[INFO] Inicializando detectores...")
        self.face_detector = FaceDetector(
            method="haar",
            confidence_threshold=confidence_threshold
        )
        self.emotion_analyzer = EmotionAnalyzer(method="fer")
        self.activity_detector = ActivityDetector(method="mediapipe")
        self.anomaly_detector = AnomalyDetector()
        
        # Estatísticas
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "total_faces": 0,
            "unique_faces": set(),
            "emotions": defaultdict(int),
            "activities": defaultdict(int),
            "anomalies": []
        }
        
        # Video writer
        self.video_writer = None
        self.fps = 30.0
    
    def analyze(self) -> VideoAnalysisResult:
        """
        Executa a análise completa do vídeo.
        
        Returns:
            Resultado consolidado da análise
        """
        print(f"[INFO] Abrindo vídeo: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {self.video_path}")
        
        # Propriedades do vídeo
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / self.fps
        
        print(f"[INFO] Vídeo: {width}x{height} @ {self.fps:.1f}fps, {total_frames} frames ({duration:.1f}s)")
        
        # Configura FPS no detector de anomalias
        self.anomaly_detector.fps = self.fps
        
        # Inicializa video writer se necessário
        if self.save_output:
            output_path = str(OUTPUT_DIR / f"analyzed_{Path(self.video_path).stem}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            print(f"[INFO] Salvando vídeo processado em: {output_path}")
        
        start_time = time.time()
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.stats["total_frames"] = frame_number + 1
                
                # Processa apenas frames selecionados
                if frame_number % self.frame_skip == 0:
                    analysis = self._process_frame(frame, frame_number)
                    self.stats["processed_frames"] += 1
                    
                    # Desenha visualização
                    if self.show_visualization or self.save_output:
                        annotated_frame = self._draw_annotations(frame, analysis)
                        
                        if self.show_visualization:
                            # Redimensiona para visualização se muito grande
                            display_frame = annotated_frame
                            if width > 1280:
                                scale = 1280 / width
                                display_frame = cv2.resize(
                                    annotated_frame,
                                    (int(width * scale), int(height * scale))
                                )
                            
                            cv2.imshow("Tech Challenge - Análise de Vídeo", display_frame)
                            
                            # Tecla 'q' para sair
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("[INFO] Análise interrompida pelo usuário")
                                break
                        
                        if self.save_output and self.video_writer:
                            self.video_writer.write(annotated_frame)
                else:
                    # Escreve frame original se não processado
                    if self.save_output and self.video_writer:
                        self.video_writer.write(frame)
                
                frame_number += 1
                
                # Progress log
                if frame_number % 100 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"[INFO] Progresso: {progress:.1f}% ({frame_number}/{total_frames})")
        
        finally:
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            if self.show_visualization:
                cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        print(f"[INFO] Análise concluída em {processing_time:.1f}s")
        
        # Compila resultado
        return VideoAnalysisResult(
            video_path=self.video_path,
            total_frames=self.stats["total_frames"],
            fps=self.fps,
            duration_seconds=duration,
            total_faces_detected=self.stats["total_faces"],
            unique_faces=len(self.stats["unique_faces"]),
            emotions_summary=dict(self.stats["emotions"]),
            activities_summary=dict(self.stats["activities"]),
            total_anomalies=len(self.stats["anomalies"]),
            anomalies_by_type=self._count_anomalies_by_type(),
            anomaly_events=self.anomaly_detector.get_anomalies_summary(),
            processing_time_seconds=processing_time
        )
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> FrameAnalysis:
        """Processa um frame individual."""
        # 1. Detecção de rostos
        faces = self.face_detector.detect(frame)
        self.stats["total_faces"] += len(faces)
        
        # 2. Análise de emoções para cada rosto
        emotions = []
        for face in faces:
            self.stats["unique_faces"].add(face.face_id)
            
            emotion = self.emotion_analyzer.analyze(
                frame, face.bbox, face.face_id
            )
            if emotion:
                emotions.append(emotion)
                self.stats["emotions"][emotion.dominant_emotion] += 1
        
        # 3. Detecção de atividades
        activities = self.activity_detector.detect(frame)
        for activity in activities:
            self.stats["activities"][activity.activity_pt] += 1
        
        # 4. Detecção de anomalias
        anomalies = self.anomaly_detector.update(
            frame_number, faces, emotions, activities
        )
        self.stats["anomalies"].extend(anomalies)
        
        return FrameAnalysis(
            frame_number=frame_number,
            faces=faces,
            emotions=emotions,
            activities=activities,
            anomalies=anomalies
        )
    
    def _draw_annotations(
        self, 
        frame: np.ndarray, 
        analysis: FrameAnalysis
    ) -> np.ndarray:
        """Desenha todas as anotações no frame."""
        annotated = frame.copy()
        
        # Desenha rostos
        annotated = self.face_detector.draw_detections(
            annotated, analysis.faces, COLORS["face"]
        )
        
        # Desenha emoções
        for face, emotion in zip(analysis.faces, analysis.emotions):
            if emotion:
                annotated = self.emotion_analyzer.draw_emotion(
                    annotated, face.bbox, emotion, COLORS["emotion"]
                )
        
        # Desenha atividades
        for activity in analysis.activities:
            annotated = self.activity_detector.draw_activity(
                annotated, activity, draw_skeleton=True, color=COLORS["activity"]
            )
        
        # Desenha anomalias
        for anomaly in analysis.anomalies:
            annotated = draw_anomaly(annotated, anomaly, COLORS["anomaly"])
        
        # HUD com estatísticas
        annotated = self._draw_hud(annotated, analysis)
        
        return annotated
    
    def _draw_hud(self, frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        """Desenha HUD com estatísticas no frame."""
        h, w = frame.shape[:2]
        
        # Fundo semi-transparente no canto superior direito
        hud_w, hud_h = 250, 120
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (w - hud_w - 10, 10),
            (w - 10, hud_h + 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Textos
        texts = [
            f"Frame: {analysis.frame_number}",
            f"Rostos: {len(analysis.faces)}",
            f"Atividades: {len(analysis.activities)}",
            f"Anomalias: {len(self.stats['anomalies'])}",
        ]
        
        y = 35
        for text in texts:
            cv2.putText(
                frame, text,
                (w - hud_w, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )
            y += 25
        
        return frame
    
    def _count_anomalies_by_type(self) -> Dict[str, int]:
        """Conta anomalias por tipo."""
        counts = defaultdict(int)
        for anomaly in self.stats["anomalies"]:
            counts[anomaly.anomaly_type.value] += 1
        return dict(counts)
    
    def reset(self):
        """Reseta o estado do analisador."""
        self.face_detector.reset_tracking()
        self.emotion_analyzer.reset_history()
        self.activity_detector.reset()
        self.anomaly_detector.reset()
        
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "total_faces": 0,
            "unique_faces": set(),
            "emotions": defaultdict(int),
            "activities": defaultdict(int),
            "anomalies": []
        }


def main():
    """Função principal para execução standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Tech Challenge Fase 4 - Análise de Vídeo"
    )
    parser.add_argument(
        "--video", "-v",
        default=VIDEO_PATH,
        help="Caminho do vídeo para análise"
    )
    parser.add_argument(
        "--skip", "-s",
        type=int,
        default=FRAME_SKIP,
        help="Processar 1 a cada N frames"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Não mostrar visualização em tempo real"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salvar vídeo processado"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Não gerar relatório"
    )
    
    args = parser.parse_args()
    
    # Executa análise
    analyzer = VideoAnalyzer(
        video_path=args.video,
        frame_skip=args.skip,
        show_visualization=not args.no_display,
        save_output=args.save
    )
    
    try:
        result = analyzer.analyze()
        
        # Gera relatório
        if not args.no_report:
            generator = ReportGenerator(use_llm=True)
            
            # Relatório Markdown
            report_path = REPORTS_DIR / f"relatorio_{Path(args.video).stem}.md"
            report = generator.generate(result, str(report_path))
            
            # Relatório JSON
            json_path = REPORTS_DIR / f"relatorio_{Path(args.video).stem}.json"
            generator.save_json_report(result, str(json_path))
            
            print("\n" + "="*60)
            print("RELATÓRIO")
            print("="*60)
            print(report)
        
        return result
        
    except KeyboardInterrupt:
        print("\n[INFO] Análise cancelada pelo usuário")
    except Exception as e:
        print(f"[ERRO] {e}")
        raise


if __name__ == "__main__":
    main()
