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
├── run.py                  # Script de execução rápida
├── requirements.txt        # Dependências do projeto
├── .env.example            # Exemplo de configuração
├── src/
│   ├── config.py           # Configurações centralizadas
│   ├── face_detector.py    # Detector de rostos
│   ├── emotion_analyzer.py # Analisador de emoções
│   ├── activity_detector.py# Detector de atividades
│   ├── anomaly_detector.py # Detector de anomalias
│   ├── report_generator.py # Gerador de relatórios
│   └── video_analyzer.py   # Integrador principal
├── output/                 # Vídeos processados
└── reports/                # Relatórios gerados
```

## Instalação

### 1. Clonar o repositório

```bash
git clone <repo-url>
cd TC-4
```

### 2. Criar ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente (opcional)

```bash
cp .env.example .env
# Edite .env com sua chave da OpenAI (para resumos com LLM)
```

## Uso

### Execução Básica

```bash
python run.py
```

### Execução com Opções

```bash
# Sem visualização (apenas processamento)
python main.py --no-display

# Salvando vídeo processado
python main.py --save

# Vídeo específico
python main.py --video caminho/para/video.mp4

# Ajustar frame skip (performance)
python main.py --skip 3  # Processa 1 a cada 3 frames
```

### Controles Durante Execução

- **Q**: Encerrar análise
- O vídeo é exibido em tempo real com todas as anotações

## Tecnologias Utilizadas

| Categoria | Tecnologia |
|-----------|------------|
| **Visão Computacional** | OpenCV, MediaPipe |
| **Reconhecimento Facial** | OpenCV DNN, Haar Cascades |
| **Análise de Emoções** | FER (Facial Expression Recognition) |
| **Detecção de Pose** | MediaPipe Pose |
| **Deep Learning** | PyTorch, TensorFlow (backend) |
| **LLM (opcional)** | LangChain + OpenAI GPT-4 |

## Saída do Sistema

### Visualização em Tempo Real

- Bounding boxes coloridos para rostos (verde)
- Labels de emoções (amarelo)
- Esqueleto de pose para atividades (laranja)
- Alertas de anomalias (vermelho)
- HUD com estatísticas no canto

### Relatório Gerado

```markdown
# Relatório de Análise de Vídeo
## Tech Challenge - Fase 4

**Arquivo**: video.mp4
**Duração**: 120.0 segundos
**Frames Analisados**: 3600

## Resumo Executivo
[Resumo gerado automaticamente]

## Estatísticas Gerais
- Total de Rostos: 450
- Pessoas Únicas: 5
- Anomalias: 3

## Análise de Emoções
| Emoção | Frequência |
|--------|------------|
| Neutro | 280 |
| Feliz  | 120 |
...

## Anomalias Detectadas
1. Movimento brusco em 00:45
2. Mudança emocional súbita em 01:20
...
```

## Estrutura dos Módulos

### FaceDetector

- Métodos: `haar`, `dnn`, `mediapipe`
- Rastreamento de IDs entre frames
- Configurável para diferentes níveis de precisão/performance

### EmotionAnalyzer

- Suporte a FER e DeepFace
- Suavização temporal para reduzir ruído
- 7 emoções básicas: feliz, triste, raiva, medo, surpresa, nojo, neutro

### ActivityDetector

- Detecção de pose com MediaPipe
- Classificação de atividades por análise de keypoints
- Cálculo de velocidade e padrões de movimento

### AnomalyDetector

- Análise estatística de comportamento
- Detecção de outliers em movimento e emoção
- Histórico temporal para baseline adaptativo

## Observações

- O projeto foi desenvolvido para análise de vídeos pré-gravados
- A performance depende do hardware (GPU acelera significativamente)
- Recomenda-se `frame_skip >= 2` para vídeos longos
- A geração de resumo com LLM requer chave da OpenAI

## Requisitos do Sistema

- Python 3.9+
- 4GB+ RAM
- Webcam ou arquivo de vídeo
- GPU (opcional, melhora performance)

## Autor

Desenvolvido para o Tech Challenge - Fase 4 do curso de Pós-Graduação.

## Licença

Este projeto é de uso educacional.
