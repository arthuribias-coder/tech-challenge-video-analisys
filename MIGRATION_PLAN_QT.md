# Plano de Migração: CustomTkinter → PyQt6

## Visão Geral

Migração completa da aplicação GUI de CustomTkinter para PyQt6, removendo CLI e mantendo apenas interface gráfica.

---

## 1. Decisões Técnicas

### Framework Escolhido: PyQt6

**Por quê PyQt6 (e não PySide6)?**
- ✅ Mais maduro e estável
- ✅ Melhor documentação
- ✅ Performance ligeiramente superior
- ✅ Comunidade maior
- ⚠️ Licença: GPL v3 (ok para projetos acadêmicos/open-source)

**Alternativa:** PySide6 (licença LGPL, código idêntico, trocar nos imports)

---

## 2. Estrutura de Arquivos

### 2.1 Arquivos a CRIAR

```
src/gui/
├── __init__.py                    # [MODIFICAR] Exportar MainWindow Qt
├── main_window_qt.py              # [CRIAR] Janela principal Qt
├── widgets/
│   ├── __init__.py                # [MODIFICAR]
│   ├── video_player_qt.py         # [CRIAR] Player com QMediaPlayer
│   ├── stats_panel_qt.py          # [CRIAR] QGroupBox com QLabels
│   └── charts_panel_qt.py         # [CRIAR] QTabWidget + matplotlib
└── threads/
    ├── __init__.py                # [MODIFICAR]
    └── processor_thread_qt.py     # [CRIAR] QThread para processamento
```

### 2.2 Arquivos a REMOVER

```
main.py                            # ❌ CLI removido
src/gui/main_window.py             # ❌ Versão CustomTkinter
src/gui/widgets/video_player.py   # ❌ Versão CustomTkinter
src/gui/widgets/stats_panel.py    # ❌ Versão CustomTkinter
src/gui/widgets/charts_panel.py   # ❌ Versão CustomTkinter
src/gui/threads/processor_thread.py # ❌ Versão threading padrão
```

### 2.3 Arquivos a MANTER (sem alteração)

```
src/
├── config.py                      # ✅ Configurações
├── face_detector.py               # ✅ Módulo core
├── emotion_analyzer.py            # ✅ Módulo core
├── activity_detector.py           # ✅ Módulo core
├── anomaly_detector.py            # ✅ Módulo core
├── visualizer.py                  # ✅ Módulo core
└── report_generator.py            # ✅ Módulo core
```

### 2.4 Arquivos a MODIFICAR

```
gui_app.py                         # [MODIFICAR] Entry point Qt
requirements.txt                   # [MODIFICAR] Trocar customtkinter por PyQt6
README.md                          # [MODIFICAR] Remover seções CLI
GUI_GUIDE.md                       # [MODIFICAR] Atualizar para Qt
```

---

## 3. Dependências

### 3.1 requirements.txt (NOVO)

```txt
# Core - Análise de Vídeo
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0

# Machine Learning
fer>=22.5.1
ultralytics>=8.0.0
torch>=2.0.0

# GUI - PyQt6
PyQt6>=6.6.0
PyQt6-Qt6>=6.6.0
PyQt6-Charts>=6.6.0         # Para gráficos nativos Qt
matplotlib>=3.8.0           # Alternativa para gráficos

# Utilitários
python-dotenv>=1.0.0
tqdm>=4.66.0
```

### 3.2 Instalação

**Todas as plataformas:**
```bash
pip install PyQt6 PyQt6-Charts
```

**Linux (dependências mínimas - geralmente já estão instaladas):**
```bash
# Oracle Linux / Red Hat / Fedora
sudo dnf install libxcb libxkbcommon fontconfig

# Ubuntu / Debian
sudo apt install libxcb-xinerama0 libxkbcommon-x11-0
```

---

## 4. Implementação dos Componentes

### 4.1 main_window_qt.py

```python
"""
Janela principal com PyQt6
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QProgressBar,
    QFileDialog, QMessageBox, QLabel, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from .widgets import VideoPlayerQt, StatsPanelQt, ChartsPanelQt
from .threads import ProcessorThreadQt
from ..config import OUTPUT_DIR


class MainWindow(QMainWindow):
    """Janela principal da aplicação Qt."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tech Challenge - Fase 4: Análise de Vídeo com IA")
        self.resize(1400, 900)
        
        # Estado
        self.video_path = None
        self.output_path = None
        self.processor_thread = None
        
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
    
    def _setup_ui(self):
        """Configura interface."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        
        # Top: Video + Stats
        top_layout = QHBoxLayout()
        
        self.video_player = VideoPlayerQt()
        top_layout.addWidget(self.video_player, stretch=7)
        
        self.stats_panel = StatsPanelQt()
        top_layout.addWidget(self.stats_panel, stretch=3)
        
        main_layout.addLayout(top_layout)
        
        # Bottom: Charts
        self.charts_panel = ChartsPanelQt()
        main_layout.addWidget(self.charts_panel)
    
    def _setup_menu(self):
        """Cria menu."""
        menubar = self.menuBar()
        
        # Arquivo
        file_menu = menubar.addMenu("&Arquivo")
        
        open_action = QAction("[+] &Abrir Vídeo", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_video)
        file_menu.addAction(open_action)
        
        save_action = QAction("[*] &Salvar Vídeo", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_video)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Sai&r", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Processar
        process_menu = menubar.addMenu("&Processar")
        
        start_action = QAction("[>] &Iniciar", self)
        start_action.triggered.connect(self._start_processing)
        process_menu.addAction(start_action)
        
        pause_action = QAction("[||] &Pausar", self)
        pause_action.triggered.connect(self._pause_processing)
        process_menu.addAction(pause_action)
        
        stop_action = QAction("[■] Pa&rar", self)
        stop_action.triggered.connect(self._stop_processing)
        process_menu.addAction(stop_action)
        
        # Ajuda
        help_menu = menubar.addMenu("A&juda")
        
        about_action = QAction("[?] &Sobre", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_statusbar(self):
        """Cria barra de status."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Label de status
        self.status_label = QLabel("Pronto | Aguardando vídeo...")
        self.statusbar.addWidget(self.status_label, stretch=1)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(300)
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        # FPS
        self.fps_label = QLabel("FPS: --")
        self.statusbar.addPermanentWidget(self.fps_label)
    
    def _open_video(self):
        """Abre vídeo."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione o vídeo",
            "",
            "Vídeos (*.mp4 *.avi *.mov *.mkv);;Todos (*.*)"
        )
        
        if filename:
            self.video_path = filename
            self.video_player.load_video(filename)
            self.status_label.setText(f"Vídeo carregado: {filename}")
    
    def _start_processing(self):
        """Inicia processamento."""
        if not self.video_path:
            QMessageBox.warning(self, "Aviso", "Selecione um vídeo primeiro!")
            return
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        self.output_path = OUTPUT_DIR / "video_analisado.mp4"
        
        self.processor_thread = ProcessorThreadQt(
            self.video_path,
            self.output_path
        )
        
        # Conecta signals
        self.processor_thread.progress.connect(self._on_progress)
        self.processor_thread.finished.connect(self._on_complete)
        self.processor_thread.error.connect(self._on_error)
        
        self.processor_thread.start()
        self.status_label.setText("Processando vídeo...")
    
    def _pause_processing(self):
        """Pausa processamento."""
        if self.processor_thread:
            self.processor_thread.toggle_pause()
    
    def _stop_processing(self):
        """Para processamento."""
        if self.processor_thread:
            self.processor_thread.stop()
            self.processor_thread.wait()
    
    def _save_video(self):
        """Salva vídeo."""
        if not self.output_path or not self.output_path.exists():
            QMessageBox.warning(self, "Aviso", "Nenhum vídeo processado!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar vídeo como",
            "",
            "MP4 (*.mp4);;Todos (*.*)"
        )
        
        if filename:
            import shutil
            shutil.copy(self.output_path, filename)
            QMessageBox.information(self, "Sucesso", f"Vídeo salvo em:\n{filename}")
    
    def _show_about(self):
        """Mostra sobre."""
        QMessageBox.about(
            self,
            "Sobre",
            "Tech Challenge - Fase 4\n"
            "Análise de Vídeo com IA\n\n"
            "Desenvolvido com:\n"
            "- OpenCV\n"
            "- FER (Facial Expression Recognition)\n"
            "- YOLO11-pose (Ultralytics)\n"
            "- PyQt6"
        )
    
    def _on_progress(self, frame_idx, total_frames, fps, stats):
        """Callback de progresso."""
        progress = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
        self.progress_bar.setValue(progress)
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.status_label.setText(f"Processando... {progress}% | FPS: {fps:.1f}")
        
        # Atualiza painéis
        self.stats_panel.update_stats(stats)
        self.charts_panel.update_data(stats)
    
    def _on_complete(self, stats, elapsed_time):
        """Callback de conclusão."""
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Concluído em {elapsed_time:.1f}s")
        
        QMessageBox.information(
            self,
            "Concluído",
            f"Processamento finalizado!\n\n"
            f"Tempo: {elapsed_time:.1f}s\n"
            f"Faces: {stats['faces']}\n"
            f"Vídeo salvo em:\n{self.output_path}"
        )
    
    def _on_error(self, error_msg):
        """Callback de erro."""
        self.status_label.setText(f"Erro: {error_msg}")
        QMessageBox.critical(self, "Erro", error_msg)
```

### 4.2 video_player_qt.py

```python
"""
Player de vídeo com QMediaPlayer ou OpenCV
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import cv2


class VideoPlayerQt(QWidget):
    """Player de vídeo usando OpenCV."""
    
    def __init__(self):
        super().__init__()
        
        self.video_capture = None
        self.current_frame = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura UI."""
        layout = QVBoxLayout(self)
        
        self.video_label = QLabel("Carregue um vídeo")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: #888;")
        self.video_label.setMinimumSize(640, 480)
        
        layout.addWidget(self.video_label)
    
    def load_video(self, video_path):
        """Carrega vídeo."""
        self.video_capture = cv2.VideoCapture(str(video_path))
        
        if self.video_capture.isOpened():
            self._update_frame()
            return True
        return False
    
    def _update_frame(self):
        """Atualiza frame."""
        if not self.video_capture or not self.video_capture.isOpened():
            return
        
        ret, frame = self.video_capture.read()
        
        if ret:
            self.current_frame = frame
            self._display_frame(frame)
    
    def _display_frame(self, frame):
        """Exibe frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Escala para caber no label
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)
```

### 4.3 processor_thread_qt.py

```python
"""
Thread de processamento com QThread
"""
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
import time

from ..face_detector import FaceDetector
from ..emotion_analyzer import EmotionAnalyzer
from ..activity_detector import ActivityDetector
from ..anomaly_detector import AnomalyDetector
from ..visualizer import Visualizer
import cv2


class ProcessorThreadQt(QThread):
    """Thread Qt para processamento de vídeo."""
    
    # Signals
    progress = pyqtSignal(int, int, float, dict)  # frame_idx, total, fps, stats
    finished = pyqtSignal(dict, float)  # stats, elapsed_time
    error = pyqtSignal(str)  # error_msg
    
    def __init__(self, video_path, output_path, frame_skip=2):
        super().__init__()
        
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.frame_skip = frame_skip
        
        self.is_paused = False
        self.should_stop = False
    
    def run(self):
        """Executa processamento."""
        try:
            start_time = time.time()
            
            # Inicializa componentes
            face_detector = FaceDetector()
            emotion_analyzer = EmotionAnalyzer()
            activity_detector = ActivityDetector()
            anomaly_detector = AnomalyDetector()
            visualizer = Visualizer()
            
            # Abre vídeo
            cap = cv2.VideoCapture(str(self.video_path))
            
            if not cap.isOpened():
                self.error.emit("Erro ao abrir vídeo")
                return
            
            # Configurações
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
            
            # Estatísticas
            stats = {
                'faces': 0,
                'emotions': {},
                'activities': {},
                'anomalies': {}
            }
            
            frame_idx = 0
            process_start = time.time()
            
            while cap.isOpened() and not self.should_stop:
                # Pausa
                while self.is_paused and not self.should_stop:
                    self.msleep(100)
                
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Processa frame
                if frame_idx % self.frame_skip == 0:
                    faces = face_detector.detect(frame)
                    stats['faces'] += len(faces)
                    
                    for face in faces:
                        emotion = emotion_analyzer.analyze(frame, face)
                        if emotion:
                            stats['emotions'][emotion] = stats['emotions'].get(emotion, 0) + 1
                    
                    activities = activity_detector.detect(frame)
                    for activity in activities:
                        stats['activities'][activity] = stats['activities'].get(activity, 0) + 1
                    
                    # Visualiza
                    frame = visualizer.draw(frame, faces, activities)
                
                out.write(frame)
                
                # Progresso
                frame_idx += 1
                elapsed = time.time() - process_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                
                if frame_idx % 30 == 0:  # Atualiza a cada 30 frames
                    self.progress.emit(frame_idx, total_frames, current_fps, stats)
            
            cap.release()
            out.release()
            
            elapsed_time = time.time() - start_time
            self.finished.emit(stats, elapsed_time)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def toggle_pause(self):
        """Pausa/retoma."""
        self.is_paused = not self.is_paused
    
    def stop(self):
        """Para thread."""
        self.should_stop = True
```

---

## 5. Passos de Migração

### Fase 1: Preparação (30 min)

```bash
# 1. Backup do projeto
cd /home/aineto/workspaces/POS/TC-4
git add .
git commit -m "Backup antes da migração Qt"
git branch backup-customtkinter

# 2. Desinstalar CustomTkinter
source .venv/bin/activate
pip uninstall customtkinter darkdetect -y

# 3. Instalar PyQt6
pip install PyQt6 PyQt6-Charts
```

### Fase 2: Remoção CLI (15 min)

```bash
# Remover CLI
rm main.py

# Remover widgets CustomTkinter
rm src/gui/main_window.py
rm src/gui/widgets/video_player.py
rm src/gui/widgets/stats_panel.py
rm src/gui/widgets/charts_panel.py
rm src/gui/threads/processor_thread.py
```

### Fase 3: Implementação Qt (2-3 horas)

```bash
# Criar novos arquivos Qt (copiar código acima)
# main_window_qt.py
# video_player_qt.py
# stats_panel_qt.py
# charts_panel_qt.py
# processor_thread_qt.py
```

### Fase 4: Atualização de Configuração (30 min)

```bash
# Atualizar requirements.txt
# Atualizar gui_app.py
# Atualizar __init__.py dos módulos
```

### Fase 5: Testes (1 hora)

```bash
# Testar aplicação
python gui_app.py

# Carregar vídeo
# Processar
# Verificar gráficos
# Testar exportação
```

### Fase 6: Documentação (1 hora)

```bash
# Atualizar README.md
# Atualizar GUI_GUIDE.md
# Criar MIGRATION_NOTES.md
```

---

## 6. Atualização da Documentação

### 6.1 README.md (NOVO)

```markdown
# Tech Challenge - Fase 4: Análise de Vídeo com IA

## Descrição

Aplicação GUI profissional para análise de vídeo utilizando **PyQt6**, com reconhecimento facial, análise de emoções, detecção de atividades e identificação de anomalias comportamentais.

## Funcionalidades

- Reconhecimento Facial
- Análise de Emoções
- Detecção de Atividades
- Detecção de Anomalias
- Interface Gráfica Profissional (PyQt6)
- Gráficos Interativos
- Exportação de Vídeos e Relatórios

## Arquitetura

```
TC-4/
├── gui_app.py              # Entry point GUI
├── requirements.txt
├── src/
│   ├── config.py
│   ├── face_detector.py
│   ├── emotion_analyzer.py
│   ├── activity_detector.py
│   ├── anomaly_detector.py
│   ├── visualizer.py
│   ├── report_generator.py
│   └── gui/                # Interface Qt
│       ├── main_window_qt.py
│       ├── widgets/
│       └── threads/
├── input/
├── output/
└── reports/
```

## Instalação

### 1. Criar ambiente virtual

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 2. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. (Linux apenas) Dependências do sistema (opcional)

```bash
# Oracle Linux / Red Hat / Fedora
sudo dnf install libxcb libxkbcommon fontconfig

# Ubuntu / Debian
sudo apt install libxcb-xinerama0 libxkbcommon-x11-0
```

## Uso

### Iniciar Aplicação GUI

```bash
source .venv/bin/activate
python gui_app.py
```

### Fluxo de Trabalho

1. **Abrir Vídeo**: Arquivo → Abrir Vídeo
2. **Processar**: Processar → Iniciar
3. **Visualizar**: Gráficos e estatísticas em tempo real
4. **Exportar**: Arquivo → Salvar Vídeo

## Tecnologias

- **Interface**: PyQt6 (framework profissional Qt)
- **Visão Computacional**: OpenCV
- **Reconhecimento Facial**: OpenCV Haar Cascades
- **Análise de Emoções**: FER
- **Detecção de Atividades**: YOLO11-pose (Ultralytics)
- **Deep Learning**: PyTorch
- **Visualização**: PyQt6 Charts + Matplotlib
```

### 6.2 GUI_GUIDE.md (Atualizar)

- Remover referências a CustomTkinter
- Adicionar instruções específicas do Qt
- Atualizar screenshots (se houver)
- Documentar diferenças de comportamento

---

## 7. Estimativa de Tempo

| Fase | Tempo | Descrição |
| --- | --- | --- |
| **Preparação** | 30 min | Backup, desinstalar/instalar pacotes |
| **Remoção CLI** | 15 min | Deletar arquivos antigos |
| **Implementação Qt** | 2-3 horas | Criar novos componentes Qt |
| **Configuração** | 30 min | Atualizar requirements, imports |
| **Testes** | 1 hora | Testar funcionalidades |
| **Documentação** | 1 hora | Atualizar docs |
| **TOTAL** | **5-6 horas** | Migração completa |

---

## 8. Riscos e Mitigações

### Riscos

1. **Incompatibilidade OpenCV + Qt**
   - Mitigação: Usar QImage para conversão de frames

2. **Threading diferente**
   - Mitigação: QThread é mais robusto que threading padrão

3. **Curva de aprendizado Qt**
   - Mitigação: Documentação oficial excelente

4. **Tamanho do pacote**
   - Mitigação: Aceitável para aplicação desktop

### Benefícios

- ✅ Interface mais profissional
- ✅ Melhor performance de threading
- ✅ Widgets nativos do OS
- ✅ Gráficos Qt Charts integrados
- ✅ Melhor portabilidade (Windows/macOS)
- ✅ Código mais limpo e estruturado

---

## 9. Checklist de Migração

### Antes de Começar
- [ ] Backup do projeto (git commit + branch)
- [ ] Ler documentação PyQt6
- [ ] Testar instalação PyQt6 no ambiente

### Durante Migração
- [ ] Desinstalar CustomTkinter
- [ ] Instalar PyQt6
- [ ] Remover main.py (CLI)
- [ ] Remover widgets CustomTkinter
- [ ] Criar main_window_qt.py
- [ ] Criar video_player_qt.py
- [ ] Criar stats_panel_qt.py
- [ ] Criar charts_panel_qt.py
- [ ] Criar processor_thread_qt.py
- [ ] Atualizar __init__.py dos módulos
- [ ] Atualizar gui_app.py
- [ ] Atualizar requirements.txt

### Testes
- [ ] Aplicação inicia sem erros
- [ ] Abrir vídeo funciona
- [ ] Processamento funciona
- [ ] Estatísticas atualizam
- [ ] Gráficos renderizam
- [ ] Exportar vídeo funciona
- [ ] Menu e atalhos funcionam

### Documentação
- [ ] Atualizar README.md
- [ ] Atualizar GUI_GUIDE.md
- [ ] Criar MIGRATION_NOTES.md
- [ ] Atualizar CHANGELOG.md

### Finalização
- [ ] Commit final
- [ ] Tag de versão (v3.0.0-qt)
- [ ] Testar em ambiente limpo
- [ ] Documentar lições aprendidas

---

## 10. Comandos Rápidos

### Iniciar Migração

```bash
cd /home/aineto/workspaces/POS/TC-4
git checkout -b migration-qt
source .venv/bin/activate
pip uninstall customtkinter darkdetect -y
pip install PyQt6 PyQt6-Charts
```

### Remover Arquivos Antigos

```bash
rm main.py
rm src/gui/main_window.py
rm src/gui/widgets/video_player.py
rm src/gui/widgets/stats_panel.py
rm src/gui/widgets/charts_panel.py
rm src/gui/threads/processor_thread.py
```

### Testar Nova Versão

```bash
python gui_app.py
```

---

## Conclusão

Migração bem planejada que transforma a aplicação em uma solução GUI profissional, removendo a CLI e focando em uma experiência de usuário superior com PyQt6.

**Pronto para começar? Siga o checklist passo a passo!**
