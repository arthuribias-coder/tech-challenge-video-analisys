# Tech Challenge - Fase 4: Análise de Vídeo com IA

## Descrição

Aplicação **GUI profissional** para análise de vídeo utilizando **PyQt6**, com reconhecimento facial, análise de expressões emocionais, detecção de atividades e identificação de anomalias comportamentais.

## Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| **Reconhecimento Facial** | Detecta e rastreia rostos no vídeo, atribuindo IDs únicos |
| **Análise de Emoções** | Classifica expressões faciais (feliz, triste, raiva, etc.) |
| **Detecção de Atividades** | Identifica ações (caminhando, sentado, gesticulando, etc.) |
| **Detecção de Anomalias** | Identifica comportamentos atípicos (movimentos bruscos, mudanças emocionais súbitas) |
| **Geração de Relatório** | Cria resumo automático com estatísticas e insights |
| **Interface GUI Profissional** | PyQt6 com visualização em tempo real, gráficos interativos e controles avançados |

## Arquitetura

```
TC-4/
├── gui_app.py              # Entry point - Interface gráfica
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
│   ├── report_generator.py # Gerador de relatórios
│   └── gui/                # Interface PyQt6
│       ├── main_window_qt.py  # Janela principal Qt
│       ├── widgets/        # Componentes da UI
│       │   ├── video_player_qt.py
│       │   ├── stats_panel_qt.py
│       │   └── charts_panel_qt.py
│       └── threads/        # Processamento em background
│           └── processor_thread_qt.py
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

### 4. (Linux apenas) Dependências do sistema (opcionais)

A maioria das distribuições Linux modernas já possui as bibliotecas necessárias. Caso encontre erros, instale:

```bash
# Oracle Linux / Red Hat / Fedora
sudo dnf install libxcb libxkbcommon fontconfig

# Ubuntu / Debian
sudo apt install libxcb-xinerama0 libxkbcommon-x11-0
```

## Uso

### Iniciar Aplicação

```bash
# Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Executar GUI
python gui_app.py
```

### Fluxo de Trabalho

1. **Abrir Vídeo**: Menu Arquivo → Abrir Vídeo (ou Ctrl+O)
2. **Processar**: Menu Processar → Iniciar (Player exibe progresso em tempo real)
3. **Visualizar Resultados**: Gráficos e estatísticas atualizados automaticamente
4. **Exportar**: 
   - Vídeo: Arquivo → Salvar Vídeo (Ctrl+S)
   - Relatório: Arquivo → Exportar Relatório (Ctrl+E)

### Funcionalidades da GUI

**Player de Vídeo:**
- Reprodução com controles (play, pause, seek)
- Visualização frame-a-frame
- Indicador de tempo atual/total

**Painel de Estatísticas:**
- Total de faces detectadas
- Emoção dominante com percentual
- Atividade dominante com percentual
- Contagem de anomalias
- Botão "Ver Detalhes Completos"

**Gráficos Interativos:**
- **Emoções**: Distribuição em gráfico de barras
- **Atividades**: Frequência de atividades detectadas
- **Anomalias**: Distribuição por tipo (pizza)

**Controles:**
- Processar, Pausar, Parar
- Barra de progresso visual
- Indicador de FPS em tempo real

### Atalhos de Teclado

| Atalho | Ação |
| --- | --- |
| `Ctrl+O` | Abrir vídeo |
| `Ctrl+S` | Salvar vídeo processado |
| `Ctrl+E` | Exportar relatório |
| `Ctrl+Q` | Sair da aplicação |
| `Espaço` | Play/Pause no player |

## Tecnologias Utilizadas

| Categoria | Tecnologia |
| --- | --- |
| **Interface** | PyQt6 (GUI profissional), PyQt6-Charts |
| **Visão Computacional** | OpenCV, MediaPipe |
| **Reconhecimento Facial** | OpenCV Haar Cascades |
| **Análise de Emoções** | FER (Facial Expression Recognition) |
| **Detecção de Atividades** | YOLO11-pose (Ultralytics) |
| **Deep Learning** | PyTorch |
| **Visualização** | Matplotlib + Qt Backend (FigureCanvas) |
| **Threading** | QThread com pyqtSignal (processamento assíncrono) |

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

- ✅ Aplicação profissional com GUI PyQt6
- 🚀 Processamento assíncrono com QThread (não bloqueia interface)
- 📹 Suporta qualquer formato de vídeo compatível com OpenCV
- 🎯 YOLO11-pose oferece melhor precisão que YOLOv8
- 🎨 Interface com tema dark e gráficos interativos

## Autor

Desenvolvido para o Tech Challenge - Fase 4 do curso de Pós-Graduação.

## Licença

Este projeto é de uso educacional.
