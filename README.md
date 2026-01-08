# Tech Challenge - Fase 4: Análise de Vídeo com IA

## Descrição

Aplicação **GUI profissional** para análise de vídeo utilizando **PyQt6**, com reconhecimento facial, análise de expressões emocionais, detecção de atividades e identificação de anomalias comportamentais.

**Versão 3.2.0** - Agora com **detecção avançada de anomalias** usando múltiplos modelos YOLO11!

## Funcionalidades

| Funcionalidade | Descrição |
|----------------|-----------|
| **Reconhecimento Facial** | Detecta e rastreia rostos no vídeo, atribuindo IDs únicos |
| **Análise de Emoções** | Classifica expressões faciais (feliz, triste, raiva, etc.) |
| **Detecção de Atividades** | Identifica ações (caminhando, sentado, gesticulando, etc.) |
| **Detecção de Anomalias** | Identifica comportamentos atípicos (movimentos bruscos, mudanças emocionais súbitas) |
| **🔍 Detecção de Objetos** | Identifica objetos fora de contexto usando YOLO11 (NOVO v3.2.0) |
| **📝 Detecção de Overlays** | OCR para watermarks, timestamps e textos sobrepostos (NOVO v3.2.0) |
| **👤 Validação de Silhuetas** | YOLO11-seg valida formas humanas realistas (NOVO v3.2.0) |
| **Geração de Relatório** | Cria resumo automático com estatísticas e insights |
| **Interface GUI Profissional** | PyQt6 com visualização em tempo real, gráficos interativos e controles avançados |
| **🎬 Preview em Tempo Real** | Visualize frames processados durante análise |
| **⚙️ Configurações Avançadas** | Controle FPS, frame skip e qualidade do processamento |

## Arquitetura

```text
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
│   ├── anomaly_detector.py # Detector de anomalias (comportamentais + visuais)
│   ├── object_detector.py  # Detector de objetos YOLO11 (NOVO v3.2.0)
│   ├── overlay_detector.py # Detector de overlays/OCR (NOVO v3.2.0)
│   ├── segment_validator.py# Validador de silhuetas YOLO11-seg (NOVO v3.2.0)
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
2. **Configurar Processamento** (NOVO v3.1.0):
   - Escolha preset de qualidade (Rápida/Balanceada/Alta)
   - Ajuste Frame Skip (1-10) e FPS alvo (15/30/60)
   - Habilite/desabilite preview em tempo real
3. **Processar**: Menu Processar → Iniciar
   - Preview exibe frames processados em tempo real
   - Estatísticas atualizam dinamicamente
4. **Visualizar Resultados**: Gráficos e estatísticas atualizados automaticamente
5. **Exportar**:
   - Vídeo: Arquivo → Salvar Vídeo (Ctrl+S)
   - Relatório: Arquivo → Exportar Relatório (Ctrl+E)

### Novas Funcionalidades (v3.1.0)

#### **Preview em Tempo Real**

- Visualize frames processados durante análise
- Buffer circular de 30 frames
- Taxa configurável (5, 10, 15 FPS)
- Overlay com informações de processamento

#### **Painel de Configurações**

- **Frame Skip (1-10)**: Controla quantos frames são pulados
- **FPS Alvo (15/30/60)**: Taxa de quadros do vídeo final
- **Presets de Qualidade**:
  - ⚡ Rápida: Skip=5, ideal para testes
  - ⚖️ Balanceada: Skip=2, recomendado
  - 💎 Alta: Skip=1, máxima precisão

### Novas Funcionalidades (v3.2.0)

#### **Detecção Avançada de Anomalias**

O sistema agora utiliza múltiplos modelos YOLO11 para detectar anomalias visuais e contextuais:

| Tipo de Anomalia | Descrição | Modelo |
|------------------|-----------|--------|
| `scene_inconsistency` | Objeto fora de contexto (ex: veículo em ambiente interno) | YOLO11n |
| `sudden_object_appear` | Objeto surge subitamente sem contexto prévio | YOLO11n |
| `visual_overlay` | Watermark, timestamp ou texto sobreposto detectado | OCR (pytesseract) |
| `silhouette_anomaly` | Silhueta detectada não tem forma humana realista | YOLO11n-seg |

#### **Novos Módulos**

1. **ObjectDetector** (`src/object_detector.py`)
   - Usa `yolo11n.pt` para detectar 80 classes COCO
   - Categoriza objetos (eletrônicos, móveis, veículos, etc.)
   - Identifica objetos fora de contexto automaticamente

2. **OverlayDetector** (`src/overlay_detector.py`)
   - OCR em regiões típicas de watermark (cantos)
   - Detecta timestamps, logos e banners promocionais
   - Requer `pytesseract` ou `easyocr` (opcionais)

3. **SegmentValidator** (`src/segment_validator.py`)
   - Usa `yolo11n-seg.pt` para segmentação de pessoas
   - Valida aspect ratio, fill ratio e complexidade do contorno
   - Cross-validation com detecção de pose

#### **Instalação de Dependências Opcionais**

Para habilitar detecção de overlays/texto:

```bash
# Opção 1: Pytesseract (mais leve)
pip install pytesseract
# + Instalar Tesseract OCR no sistema:
# Ubuntu/Debian: sudo apt install tesseract-ocr tesseract-ocr-por
# Fedora/RHEL: sudo dnf install tesseract tesseract-langpack-por

# Opção 2: EasyOCR (mais preciso, usa GPU se disponível)
pip install easyocr
```

📖 Veja [MELHORIAS_UI_UX.md](MELHORIAS_UI_UX.md) para documentação completa das novas funcionalidades.

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
