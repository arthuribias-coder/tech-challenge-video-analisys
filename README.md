# Tech Challenge - Fase 4: Análise de Vídeo com IA

## Descrição

Aplicação de análise de vídeo que utiliza técnicas de **reconhecimento facial**, **análise de expressões emocionais**, **detecção de atividades** e **identificação de anomalias comportamentais**.

O sistema processa vídeos em **tempo real**, exibindo bounding boxes, labels e informações relevantes diretamente no vídeo, similar a sistemas de detecção de objetos como YOLO.

## Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| **Reconhecimento Facial** | Detecta e rastreia rostos no vídeo, atribuindo IDs únicos |
| **Análise de Emoções** | Classifica expressões faciais (feliz, triste, raiva, etc.) |
| **Detecção de Atividades** | Identifica ações (caminhando, sentado, gesticulando, etc.) |
| **Detecção de Anomalias** | Identifica comportamentos atípicos (movimentos bruscos, mudanças emocionais súbitas) |
| **Geração de Relatório** | Cria resumo automático com estatísticas e insights |

## Arquitetura

```
TC-4/
├── main.py                 # Ponto de entrada principal
├── requirements.txt        # Dependências do projeto
├── .env.example            # Exemplo de configuração
├── src/
│   ├── __init__.py         # Exporta módulos principais
│   ├── config.py           # Configurações centralizadas
│   ├── face_detector.py    # Detector de rostos
│   ├── emotion_analyzer.py # Analisador de emoções
│   ├── activity_detector.py# Detector de atividades (YOLO11-pose)
│   ├── anomaly_detector.py # Detector de anomalias
│   ├── visualizer.py       # Desenho de anotações nos frames
│   └── report_generator.py # Gerador de relatórios
├── input/                  # Vídeos de entrada
├── output/                 # Vídeos processados
├── reports/                # Relatórios gerados
└── models/                 # Modelos YOLO baixados
```

## Instalação

### 1. Clonar o repositório

```bash
git clone <repo-url>
cd TC-4
```

### 2. Criar ambiente virtual (Python 3.12+)

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### 3. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Colocar o vídeo na pasta `input/`

```bash
cp seu_video.mp4 input/
```

## Uso

### Processamento de Vídeo

```bash
# Ativar ambiente virtual (se ainda não ativou)
source .venv/bin/activate

# Processar vídeo padrão (definido em .env ou config.py)
python main.py

# Processar vídeo específico
python main.py input/seu_video.mp4

# Processar e reproduzir automaticamente (abre player OpenCV)
python main.py input/video.mp4 --show

# Ajustar intervalo de frames (mais rápido, menos preciso)
python main.py input/video.mp4 --skip 3

# Definir arquivo de saída customizado
python main.py input/video.mp4 --output meu_resultado.mp4

# Ver todas as opções disponíveis
python main.py --help
```

### Controles do Player (--show)

| Tecla | Ação |
| --- | --- |
| **Q** ou **ESC** | Sair do player |
| **Espaço** | Pausar/Continuar |
| **← / A** | Voltar 10 segundos |
| **→ / D** | Avançar 10 segundos |

### Saída no Console

O sistema exibe em tempo real:

- 🔧 Carregamento dos modelos de IA
- 📹 Informações do vídeo de entrada
- 🎬 Barra de progresso detalhada (%, FPS, ETA)
- 📊 Estatísticas completas da análise:
  - Total de faces detectadas
  - Top 5 emoções com gráfico ASCII
  - Top 5 atividades com gráfico ASCII
  - Anomalias detectadas
- 💾 Informações do arquivo gerado

## Tecnologias Utilizadas

| Categoria | Tecnologia |
| --- | --- |
| **Visão Computacional** | OpenCV, MediaPipe |
| **Reconhecimento Facial** | OpenCV Haar Cascades |
| **Análise de Emoções** | FER (Facial Expression Recognition) |
| **Detecção de Atividades** | YOLO11-pose (Ultralytics) |
| **Deep Learning** | PyTorch |

## Vídeo Processado

O vídeo de saída contém:

- ✅ Bounding boxes verdes para rostos detectados
- 😊 Labels de emoções com confiança (ciano)
- 🏃 Detecção de atividades das pessoas (laranja)
- ⚠️ Alertas visuais para anomalias (vermelho)

## Estrutura dos Módulos

### FaceDetector (`src/face_detector.py`)

- Método padrão: Haar Cascades
- Rastreamento de IDs entre frames
- Suporte para MediaPipe e DNN

### EmotionAnalyzer (`src/emotion_analyzer.py`)

- Baseado em FER (Facial Expression Recognition)
- Suavização temporal para reduzir ruído
- 7 emoções: feliz, triste, raiva, medo, surpresa, nojo, neutro

### ActivityDetector (`src/activity_detector.py`)

- Usa YOLO11-pose para detecção de pessoas
- Análise de keypoints (17 pontos COCO)
- Detecta 9 atividades: em pé, sentado, caminhando, correndo, acenando, apontando, dançando, agachado, braços levantados

### AnomalyDetector (`src/anomaly_detector.py`)

- Análise estatística de comportamento
- Detecção de: movimentos bruscos, mudanças emocionais súbitas, atividades incomuns
- Histórico temporal para baseline adaptativo

## Notas Importantes

- ✅ Projeto convertido de notebooks para aplicação CLI simples
- 🚀 Performance otimizada com `frame_skip` configurável
- 📹 Suporta qualquer formato de vídeo compatível com OpenCV
- 🎯 YOLO11-pose oferece melhor precisão que YOLOv8

## Autor

Desenvolvido para o Tech Challenge - Fase 4 do curso de Pós-Graduação.

## Licença

Este projeto é de uso educacional.
