# Tech Challenge Fase 4 - AI Coding Instructions

## Arquitetura

**Pipeline de Processamento de Vídeo**: `gui_app.py` → `MainWindow` → `ProcessorThreadQt` → Detectores (face/emotion/activity/anomaly) → `Visualizer` → Saída

**Separação de Responsabilidades**:
- `src/*.py`: Módulos de detecção/análise (independentes de GUI)
- `src/gui/`: Interface PyQt6 (janela, widgets, threads)
- Detecção em memória → Visualização em frames → Escrita em arquivo de saída

## PyQt6 Threading Pattern

**CRITICAL**: Processamento de vídeo SEMPRE em `QThread`, nunca na main thread.

```python
# src/gui/threads/processor_thread_qt.py
class ProcessorThreadQt(QThread):
    progress = pyqtSignal(int, int, float, dict)  # Atualiza UI
    finished_signal = pyqtSignal(dict, float)
    error = pyqtSignal(str)
```

**Conectar signals em `main_window_qt.py`**:
```python
self.processor_thread.progress.connect(self._on_progress)
self.processor_thread.finished_signal.connect(self._on_complete)
```

## API Críticas dos Detectores

**FaceDetector**: `detect(frame) → List[FaceDetection]`
**EmotionAnalyzer**: `analyze(frame, bbox, face_id) → EmotionResult` (requer frame COMPLETO + bbox)
**ActivityDetector**: `detect(frame) → List[ActivityResult]`
**AnomalyDetector**: `update(frame_idx, faces, emotions, activities) → List[AnomalyEvent]` (não `detect()`!)

**Ordem de chamada obrigatória**:
1. `face_detector.detect()`
2. `emotion_analyzer.analyze()` para cada face
3. `activity_detector.detect()`
4. `anomaly_detector.update()` com todos resultados

## Configuração Central (`src/config.py`)

- Diretórios criados automaticamente: `INPUT_DIR`, `OUTPUT_DIR`, `REPORTS_DIR`, `MODELS_DIR`
- Variáveis de ambiente: `VIDEO_PATH`, `FRAME_SKIP`, `CONFIDENCE_THRESHOLD`, `OPENAI_API_KEY`
- `EMOTION_LABELS`: Mapeamento inglês → português (usado em `EmotionResult.emotion_pt`)
- `COLORS`: Cores BGR para visualização (verde=faces, amarelo=emoções, laranja=atividades, vermelho=anomalias)

## GUI Components Pattern

**Toolbar (não menus)**: Botões grandes com símbolos Unicode em `main_window_qt.py` (`_setup_toolbar()`)
- Emojis EVITAR (problemas de rendering) → usar `[+]`, `[▶]`, `[⏸⏸]`, `[■]`, `[:)]`, etc.

**Matplotlib no Qt**: 
```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
canvas = FigureCanvas(figure)
# Tema dark: figure.set_facecolor('#1e1e1e'), ax.set_facecolor('#1e1e1e')
```

**Abas de gráficos**: `QTabWidget` em `charts_panel_qt.py` (Emoções, Atividades, Anomalias)

## Estrutura de Dados

Todos detectores retornam **dataclasses**:
- `FaceDetection(face_id, bbox, confidence)`
- `EmotionResult(face_id, dominant_emotion, emotion_scores, confidence, emotion_pt)`
- `ActivityResult(person_id, activity, activity_pt, confidence, keypoints, velocity, bbox)`
- `AnomalyEvent(anomaly_type, timestamp, frame_number, person_id, severity, description)`

**Counter para stats**: `stats = {'faces': 0, 'emotions': Counter(), 'activities': Counter(), 'anomalies': Counter()}`

## Visualização (`src/visualizer.py`)

**Função principal**: `draw_detections(frame, faces, emotions, activities, anomalies) → annotated_frame`
- NÃO é uma classe (era anteriormente)
- Desenha boxes, labels e ícones diretamente no frame (em-place modification)
- Usa `cv2.rectangle()`, `cv2.putText()` com cores de `config.COLORS`

## Dependências YOLO

**Download automático**: YOLO11-pose baixa automaticamente na primeira execução
- Modelos em `models/yolo11n-pose.pt` ou `yolo11s-pose.pt`
- ActivityDetector usa `YOLO('yolo11n-pose.pt')`

## Execução e Debugging

**Comando único**: `python gui_app.py`
- Ativa venv primeiro: `source .venv/bin/activate`
- Python 3.12+ requerido

**Logs**: `print()` direto no terminal (sem logging framework)
- ProcessorThread: `[INFO]`, `[WARN]` prefixes nos prints
- Traceback completo em exceções

**Git workflow**: Feature v3.0.0 migrou para PyQt6 (tag `v3.0.0`)
- Commits recentes: correções de API e UX (toolbar + símbolos Unicode)

## Padrões Específicos do Projeto

**Imports relativos**: Usar `from ...module` (3 dots) de threads/widgets para acessar `src/`
**Suavização temporal**: `EmotionAnalyzer` mantém histórico por face_id (janela de 5 frames)
**IDs persistentes**: FaceDetector rastreia faces entre frames (mesmo person_id)
**Stats em tempo real**: `ProcessorThreadQt` emite `progress` a cada 30 frames ou 1 segundo

## Testes

Não há suite de testes automatizados. Testar manualmente:
1. Abrir vídeo de `input/`
2. Processar até completar
3. Verificar saída em `output/analisado_*.mp4`
4. Verificar gráficos populados
5. Exportar relatório em `reports/`

## Known Issues

- Emojis não renderizam em alguns sistemas Linux → usar símbolos ASCII/Unicode `[⏸⏸]` `[▶]`
- QSizePolicy: usar `QSizePolicy.Policy.Expanding` (não `MinimumExpanding`)
- AnomalyDetector.detect() não existe → usar `update()`
