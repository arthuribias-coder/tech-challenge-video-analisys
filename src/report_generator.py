"""
Tech Challenge - Fase 4: Gerador de Relatório
Módulo responsável pela geração automática do relatório de análise do vídeo.
Suporta geração com LLM (OpenAI) ou template local.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from .config import OPENAI_API_KEY, OPENAI_MODEL, REPORTS_DIR, EMOTION_LABELS


@dataclass
class VideoAnalysisResult:
    """Resultado consolidado da análise de vídeo."""
    video_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    total_faces_detected: int
    unique_faces: int
    emotions_summary: Dict[str, int]
    activities_summary: Dict[str, int]
    total_anomalies: int
    anomalies_by_type: Dict[str, int]
    anomaly_events: List[Dict]
    processing_time_seconds: float


class ReportGenerator:
    """
    Gerador de relatórios de análise de vídeo.
    Suporta geração com LLM para resumos mais elaborados.
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Inicializa o gerador de relatórios.
        
        Args:
            use_llm: Se True, usa LLM para gerar resumos (requer OPENAI_API_KEY)
        """
        self.use_llm = use_llm and bool(OPENAI_API_KEY)
        self.llm = None
        
        if self.use_llm:
            self._init_llm()
    
    def _init_llm(self):
        """Inicializa o LLM para geração de resumos."""
        try:
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate
            
            self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
            self.PromptTemplate = PromptTemplate
        except ImportError:
            print("[AVISO] LangChain não instalado, usando geração de relatório sem LLM")
            self.use_llm = False
    
    def generate(
        self,
        analysis_result: VideoAnalysisResult,
        output_path: Optional[str] = None
    ) -> str:
        """
        Gera o relatório de análise.
        
        Args:
            analysis_result: Resultado da análise do vídeo
            output_path: Caminho para salvar o relatório (opcional)
            
        Returns:
            Texto do relatório gerado
        """
        # Gera seções do relatório
        header = self._generate_header(analysis_result)
        statistics = self._generate_statistics(analysis_result)
        emotions_section = self._generate_emotions_section(analysis_result)
        activities_section = self._generate_activities_section(analysis_result)
        anomalies_section = self._generate_anomalies_section(analysis_result)
        
        # Gera resumo executivo
        if self.use_llm:
            summary = self._generate_llm_summary(analysis_result)
        else:
            summary = self._generate_template_summary(analysis_result)
        
        # Monta relatório completo
        report = f"""
{header}

## Resumo Executivo
{summary}

{statistics}

{emotions_section}

{activities_section}

{anomalies_section}

---
*Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*
"""
        
        # Salva se caminho fornecido
        if output_path:
            Path(output_path).write_text(report, encoding='utf-8')
            print(f"[INFO] Relatório salvo em: {output_path}")
        
        return report
    
    def _generate_header(self, result: VideoAnalysisResult) -> str:
        """Gera cabeçalho do relatório."""
        video_name = Path(result.video_path).name
        return f"""# Relatório de Análise de Vídeo
## Tech Challenge - Fase 4

**Arquivo**: `{video_name}`  
**Duração**: {result.duration_seconds:.1f} segundos ({result.total_frames} frames @ {result.fps:.1f} FPS)  
**Data da Análise**: {datetime.now().strftime('%d/%m/%Y %H:%M')}  
**Tempo de Processamento**: {result.processing_time_seconds:.1f} segundos
"""
    
    def _generate_statistics(self, result: VideoAnalysisResult) -> str:
        """Gera seção de estatísticas."""
        return f"""## Estatísticas Gerais

| Métrica | Valor |
|---------|-------|
| Total de Frames Analisados | {result.total_frames} |
| Rostos Detectados (total) | {result.total_faces_detected} |
| Pessoas Únicas Identificadas | {result.unique_faces} |
| Anomalias Detectadas | {result.total_anomalies} |
"""
    
    def _generate_emotions_section(self, result: VideoAnalysisResult) -> str:
        """Gera seção de análise de emoções."""
        if not result.emotions_summary:
            return "## Análise de Emoções\n\n*Nenhuma emoção detectada.*"
        
        # Traduz e ordena emoções
        emotions_translated = {}
        for emotion, count in result.emotions_summary.items():
            emotion_pt = EMOTION_LABELS.get(emotion, emotion)
            emotions_translated[emotion_pt] = count
        
        sorted_emotions = sorted(
            emotions_translated.items(), 
            key=lambda x: -x[1]
        )
        
        table = "| Emoção | Frequência |\n|--------|------------|\n"
        for emotion, count in sorted_emotions:
            table += f"| {emotion} | {count} |\n"
        
        # Emoção dominante
        dominant = sorted_emotions[0][0] if sorted_emotions else "N/A"
        
        return f"""## Análise de Emoções

**Emoção Predominante**: {dominant}

{table}
"""
    
    def _generate_activities_section(self, result: VideoAnalysisResult) -> str:
        """Gera seção de análise de atividades."""
        if not result.activities_summary:
            return "## Detecção de Atividades\n\n*Nenhuma atividade detectada.*"
        
        sorted_activities = sorted(
            result.activities_summary.items(),
            key=lambda x: -x[1]
        )
        
        table = "| Atividade | Frequência |\n|-----------|------------|\n"
        for activity, count in sorted_activities:
            table += f"| {activity} | {count} |\n"
        
        return f"""## Detecção de Atividades

{table}
"""
    
    def _generate_anomalies_section(self, result: VideoAnalysisResult) -> str:
        """Gera seção de anomalias."""
        if result.total_anomalies == 0:
            return "## Anomalias Detectadas\n\n*Nenhuma anomalia detectada durante a análise.*"
        
        # Tabela por tipo
        type_table = "| Tipo de Anomalia | Quantidade |\n|------------------|------------|\n"
        for atype, count in result.anomalies_by_type.items():
            type_table += f"| {atype} | {count} |\n"
        
        # Lista detalhada (limitada a 20 eventos)
        events_list = ""
        for i, event in enumerate(result.anomaly_events[:20], 1):
            events_list += f"\n### {i}. {event['tipo'].replace('_', ' ').title()}\n"
            events_list += f"- **Timestamp**: {event['timestamp']}\n"
            events_list += f"- **Frame**: {event['frame']}\n"
            events_list += f"- **Severidade**: {event['severidade']}\n"
            events_list += f"- **Descrição**: {event['descricao']}\n"
        
        more_events = ""
        if len(result.anomaly_events) > 20:
            more_events = f"\n*... e mais {len(result.anomaly_events) - 20} eventos.*"
        
        return f"""## Anomalias Detectadas

**Total**: {result.total_anomalies} anomalias

### Distribuição por Tipo

{type_table}

### Detalhamento dos Eventos
{events_list}{more_events}
"""
    
    def _generate_llm_summary(self, result: VideoAnalysisResult) -> str:
        """Gera resumo usando LLM."""
        if not self.llm:
            return self._generate_template_summary(result)
        
        # Prepara dados para o prompt
        data = {
            "duracao": f"{result.duration_seconds:.1f}s",
            "frames": result.total_frames,
            "pessoas": result.unique_faces,
            "emocoes": result.emotions_summary,
            "atividades": result.activities_summary,
            "anomalias": result.total_anomalies,
            "tipos_anomalia": result.anomalies_by_type
        }
        
        prompt = self.PromptTemplate.from_template("""
Você é um analista de segurança e comportamento. Com base nos dados de análise de vídeo abaixo,
escreva um resumo executivo em português brasileiro de 3-4 parágrafos.

Dados da análise:
- Duração do vídeo: {duracao}
- Total de frames: {frames}
- Pessoas identificadas: {pessoas}
- Emoções detectadas: {emocoes}
- Atividades detectadas: {atividades}
- Número de anomalias: {anomalias}
- Tipos de anomalia: {tipos_anomalia}

O resumo deve:
1. Descrever o cenário geral observado no vídeo
2. Destacar as emoções e atividades predominantes
3. Comentar sobre as anomalias detectadas (se houver)
4. Fornecer insights relevantes sobre o comportamento observado

Resumo:
""")
        
        try:
            chain = prompt | self.llm
            response = chain.invoke(data)
            return response.content.strip()
        except Exception as e:
            print(f"[ERRO] LLM: {e}")
            return self._generate_template_summary(result)
    
    def _generate_template_summary(self, result: VideoAnalysisResult) -> str:
        """Gera resumo usando template (sem LLM)."""
        # Emoção dominante
        dominant_emotion = "N/A"
        if result.emotions_summary:
            dominant_emotion = EMOTION_LABELS.get(
                max(result.emotions_summary, key=result.emotions_summary.get),
                "Desconhecida"
            )
        
        # Atividade dominante
        dominant_activity = "N/A"
        if result.activities_summary:
            dominant_activity = max(
                result.activities_summary, 
                key=result.activities_summary.get
            )
        
        # Monta resumo
        summary = f"""O vídeo analisado tem duração de {result.duration_seconds:.1f} segundos e contém {result.total_frames} frames.

Durante a análise, foram identificadas **{result.unique_faces} pessoa(s)** com um total de **{result.total_faces_detected} detecções de rostos**. A emoção predominante observada foi **{dominant_emotion}**, enquanto a atividade mais frequente foi **{dominant_activity}**.

"""
        
        if result.total_anomalies > 0:
            summary += f"""Foram detectadas **{result.total_anomalies} anomalias** comportamentais durante o vídeo. """
            
            if result.anomalies_by_type:
                types_desc = ", ".join([
                    f"{count} {atype.replace('_', ' ')}"
                    for atype, count in result.anomalies_by_type.items()
                ])
                summary += f"Os tipos identificados incluem: {types_desc}."
        else:
            summary += "Não foram detectadas anomalias significativas durante o período analisado."
        
        return summary
    
    def save_json_report(
        self,
        analysis_result: VideoAnalysisResult,
        output_path: str
    ):
        """Salva relatório em formato JSON."""
        data = {
            "video_path": analysis_result.video_path,
            "total_frames": analysis_result.total_frames,
            "fps": analysis_result.fps,
            "duration_seconds": analysis_result.duration_seconds,
            "total_faces_detected": analysis_result.total_faces_detected,
            "unique_faces": analysis_result.unique_faces,
            "emotions_summary": analysis_result.emotions_summary,
            "activities_summary": analysis_result.activities_summary,
            "total_anomalies": analysis_result.total_anomalies,
            "anomalies_by_type": analysis_result.anomalies_by_type,
            "anomaly_events": analysis_result.anomaly_events,
            "processing_time_seconds": analysis_result.processing_time_seconds,
            "generated_at": datetime.now().isoformat()
        }
        
        Path(output_path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        print(f"[INFO] Relatório JSON salvo em: {output_path}")
