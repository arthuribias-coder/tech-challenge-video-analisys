# Guia da Interface Gráfica (GUI)

## Visão Geral

A interface gráfica oferece uma experiência completa para análise de vídeos com IA, incluindo:

- Player de vídeo integrado com controles completos
- Estatísticas em tempo real
- Gráficos interativos e analíticos
- Processamento em background (não trava a interface)
- Tema escuro moderno

## Instalação de Dependências

### Linux (Oracle Linux, Red Hat, Fedora, CentOS)

```bash
# Instalar Tkinter para Python 3.12
sudo dnf install python3.12-tkinter -y
```

### Linux (Ubuntu, Debian)

```bash
# Instalar Tkinter para Python 3.12
sudo apt install python3.12-tk -y
```

### Windows

Tkinter já vem incluído na instalação padrão do Python.

### macOS

Tkinter já vem incluído na instalação padrão do Python.

## Iniciando a GUI

```bash
# Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Executar GUI
python gui_app.py
```

## Layout da Janela Principal

```
┌─────────────────────────────────────────────────────────────┐
│ Menu: Arquivo | Editar | Processar | Ajuda                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────┐    ┌─────────────────────────┐    │
│  │                     │    │   Estatísticas          │    │
│  │   Player de Vídeo   │    │   ───────────────────   │    │
│  │                     │    │   Faces: 0              │    │
│  │   (Canvas OpenCV)   │    │   Emoção: -             │    │
│  │                     │    │   Atividade: -          │    │
│  │                     │    │   Anomalias: 0          │    │
│  └─────────────────────┘    │                         │    │
│                              │   [Ver Detalhes]        │    │
│  ┌─────────────────────────────────────────────┐      │    │
│  │ [◄◄] [►/❚❚] [■] ━━━━━━━━━━━━━━━━━━━━━━━━    │      │    │
│  │ 00:00 / 00:00                               │      │    │
│  └─────────────────────────────────────────────┘      └────┘
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Gráficos [Emoções] [Atividades] [Timeline] [Anomalias]│  │
│  │                                                         │  │
│  │   █████████████ happy (45%)                           │  │
│  │   ████████ neutral (30%)                              │  │
│  │   ████ sad (15%)                                      │  │
│  │   ██ angry (10%)                                      │  │
│  │                                                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│ Status: Pronto | FPS: 0 | Tempo: 00:00 | Frame: 0/0        │
└─────────────────────────────────────────────────────────────┘
```

## Componentes da Interface

### 1. Menu Superior

**Arquivo**

- `Abrir Vídeo...` - Selecionar vídeo para análise
- `Salvar Vídeo Processado...` - Exportar resultado
- `Exportar Relatório...` - Gerar relatório em texto
- `Sair` - Fechar aplicação

**Processar**

- `Iniciar Análise` - Começar processamento
- `Pausar` - Pausar processamento
- `Parar` - Cancelar processamento
- `Configurações...` - Ajustar parâmetros

**Ajuda**

- `Documentação` - Abrir guia
- `Atalhos de Teclado` - Ver comandos
- `Sobre` - Informações do sistema

### 2. Player de Vídeo

**Funcionalidades:**

- Reprodução em tempo real com OpenCV
- Renderização de anotações (faces, emoções, atividades)
- Controle de velocidade
- Seek bar para navegação

**Controles de Teclado:**

- `Espaço` - Play/Pause
- `Q` ou `ESC` - Fechar
- `←` ou `A` - Voltar 10 segundos
- `→` ou `D` - Avançar 10 segundos

### 3. Painel de Estatísticas

**Exibe em Tempo Real:**

- **Total de Faces**: Contador acumulado de detecções
- **Emoção Dominante**: Emoção mais frequente com percentual
- **Atividade Dominante**: Atividade mais detectada com percentual
- **Anomalias Detectadas**: Contagem de comportamentos atípicos

**Botão "Ver Detalhes":**

- Abre diálogo com estatísticas completas
- Lista todas as emoções com percentuais
- Lista todas as atividades detectadas
- Detalha anomalias encontradas

### 4. Barra de Controles

**Botões:**

- `[◄◄]` - Voltar 10 segundos
- `[►]` - Play (torna-se [❚❚] quando reproduzindo)
- `[■]` - Parar reprodução

**Seek Bar:**

- Barra de progresso interativa
- Clique para pular para posição específica
- Indicador de tempo (atual / total)

### 5. Painel de Gráficos

**Aba "Emoções":**

- Gráfico de barras horizontais
- Distribuição percentual de todas as emoções
- Cores distintas por emoção

**Aba "Atividades":**

- Gráfico de barras horizontais
- Frequência de cada atividade detectada
- Classificação por ocorrência

**Aba "Timeline":**

- Linha do tempo dos eventos
- Marcadores de mudanças emocionais
- Indicação de anomalias temporais

**Aba "Anomalias":**

- Gráfico de pizza
- Distribuição por tipo de anomalia
- Percentual de cada categoria

### 6. Barra de Status

**Informações Exibidas:**

- **Status**: Pronto / Processando / Pausado / Concluído
- **FPS**: Taxa de frames por segundo (processamento)
- **Tempo**: Tempo decorrido de processamento
- **Progresso**: Frame atual / Total de frames
- **Barra de Progresso**: Indicador visual 0-100%

## Fluxo de Trabalho Recomendado

### 1. Abrir Vídeo

```
Arquivo → Abrir Vídeo... → Selecionar arquivo
```

O player carrega e exibe o primeiro frame.

### 2. Iniciar Análise

```
Processar → Iniciar Análise
```

ou clique no botão `[Processar]` na barra de controles.

**Durante o Processamento:**

- Estatísticas atualizam em tempo real
- Barra de progresso mostra avanço
- Gráficos são atualizados automaticamente
- Player exibe frames com anotações

### 3. Pausar/Retomar (Opcional)

```
Processar → Pausar
Processar → Iniciar Análise (para retomar)
```

### 4. Visualizar Resultados

**Enquanto processa ou após concluir:**

- Navegue pelas abas de gráficos
- Clique em "Ver Detalhes" no painel de estatísticas
- Use o seek bar para revisar momentos específicos

### 5. Exportar Resultados

**Salvar Vídeo:**

```
Arquivo → Salvar Vídeo Processado...
```

**Exportar Relatório:**

```
Arquivo → Exportar Relatório...
```

## Atalhos de Teclado

| Atalho | Ação |
| --- | --- |
| `Ctrl+O` | Abrir vídeo |
| `Ctrl+S` | Salvar vídeo processado |
| `Ctrl+E` | Exportar relatório |
| `Ctrl+Q` | Sair da aplicação |
| `Espaço` | Play/Pause no player |
| `←` / `A` | Voltar 10 segundos |
| `→` / `D` | Avançar 10 segundos |
| `F11` | Tela cheia |
| `ESC` | Sair da tela cheia |

## Configurações Avançadas

Acesse via `Processar → Configurações...`

**Opções Disponíveis:**

- **Skip Frames**: Processar a cada N frames (performance)
- **Confiança Mínima**: Threshold para detecções (0.0-1.0)
- **Buffer de Suavização**: Janela temporal para emoções
- **Threshold de Anomalia**: Sensibilidade para detectar anomalias

## Solução de Problemas

### GUI não abre

**Erro: `ModuleNotFoundError: No module named 'tkinter'`**

```bash
# Linux
sudo dnf install python3.12-tkinter -y  # Red Hat/Oracle
sudo apt install python3.12-tk -y      # Debian/Ubuntu

# Verifique instalação
python -c "import tkinter; print('OK')"
```

### Player não exibe vídeo

**Verifique:**

1. Formato do vídeo (suportados: MP4, AVI, MOV, MKV)
2. Codecs instalados no OpenCV
3. Arquivo não corrompido

```bash
# Testar vídeo via CLI
python main.py input/seu_video.mp4 --show
```

### Gráficos não aparecem

**Verifique:**

1. Matplotlib instalado: `pip list | grep matplotlib`
2. Backend do Matplotlib configurado
3. Processe ao menos alguns frames antes de verificar

### Interface lenta

**Otimizações:**

1. Aumente o valor de skip frames (Configurações)
2. Reduza resolução do vídeo antes de processar
3. Feche outros aplicativos pesados
4. Desabilite visualização em tempo real temporariamente

### Processamento trava

**Possíveis causas:**

1. Vídeo muito longo (>30 min) - processe em partes
2. Memória insuficiente - feche outros apps
3. GPU não disponível - PyTorch usa CPU (mais lento)

**Solução:**

- Use CLI para vídeos longos: `python main.py input/video.mp4 --skip 5`

## Comparação CLI vs GUI

| Aspecto | CLI | GUI |
| --- | --- | --- |
| **Velocidade** | Mais rápido | Ligeiramente mais lento |
| **Interface** | Terminal | Gráfica moderna |
| **Gráficos** | ASCII (console) | Matplotlib interativo |
| **Controles** | Argumentos de linha | Botões e menus |
| **Multitarefa** | Bloqueia terminal | Background thread |
| **Visualização** | Player OpenCV simples | Player integrado avançado |
| **Ideal para** | Automação, scripts | Análise interativa |

## Dicas de Uso

1. **Processamento Rápido**: Use skip frames = 3-5 para preview rápido
2. **Máxima Qualidade**: Skip frames = 1 (processa todos os frames)
3. **Análise Detalhada**: Pause em momentos-chave usando seek bar
4. **Comparar Resultados**: Abra CLI e GUI lado a lado
5. **Exportação**: Sempre gere relatório após processar para documentação

## Recursos Futuros (Planejados)

- [ ] Exportação de relatórios em PDF
- [ ] Comparação lado a lado de múltiplos vídeos
- [ ] Anotações manuais e marcadores
- [ ] Detecção de objetos customizada (além de pessoas)
- [ ] Integração com webcam em tempo real
- [ ] Exportação de clipes de anomalias
- [ ] Dashboard de múltiplos vídeos
- [ ] API REST para integração

## Suporte

Para problemas, sugestões ou dúvidas:

1. Verifique este guia primeiro
2. Consulte o [README.md](README.md) principal
3. Abra issue no repositório
4. Contate o time de desenvolvimento
