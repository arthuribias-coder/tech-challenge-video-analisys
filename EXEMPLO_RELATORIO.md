# Relatório de Análise de Vídeo
## Tech Challenge - Fase 4

**Arquivo**: `video_exemplo.mp4`  
**Duração**: 45.3 segundos (1360 frames @ 30.0 FPS)  
**Data da Análise**: 09/01/2026 14:30  
**Tempo de Processamento**: 98.2 segundos

## Resumo Executivo

O vídeo analisado tem duração de 45.3 segundos e contém 1360 frames.

Durante a análise, foram identificadas **3 pessoa(s)** com um total de **847 detecções de rostos**. A emoção predominante observada foi **Neutro**, enquanto a atividade mais frequente foi **Em pé**.

Foram detectadas **12 anomalias** comportamentais durante o vídeo. Os tipos identificados incluem: 7 movimento_anomalo, 3 emocao_negativa, 2 inatividade_prolongada.

## Estatísticas Gerais

### Análise do Vídeo

| Métrica | Valor |
|---------|-------|
| **Total de Frames Analisados** | 1360 |
| Rostos Detectados (total) | 847 |
| Pessoas Únicas Identificadas | 3 |
| **Número de Anomalias Detectadas** | 12 |
| FPS (Frames por Segundo) | 30.0 |
| Duração Total | 45.3s |
| Tempo de Processamento | 98.2s |

### Observação Importante

> **Critério de Anomalia**: Movimento anômalo não segue o padrão geral de atividades observadas no vídeo.
> 
> A aplicação classifica como anômalo comportamentos que se desviam significativamente do padrão estabelecido,
> como gestos bruscos, mudanças súbitas de postura, inatividade prolongada, emoções negativas intensas,
> ou comportamentos atípicos para o contexto da cena detectado.
> 
> Movimentos normais e esperados (caminhando, sentado, conversando) não são considerados anômalos.

## Análise de Emoções

**Emoção Predominante**: Neutro (68.2% das detecções)

| Emoção | Frequência | Percentual |
|--------|------------|------------|
| Neutro | 578 | 68.2% █████████████ |
| Feliz | 156 | 18.4% ███ |
| Surpreso | 67 | 7.9% █ |
| Triste | 28 | 3.3% |
| Com raiva | 12 | 1.4% |
| Com medo | 6 | 0.7% |

*Total de detecções emocionais: 847*

## Detecção de Atividades

**Atividade Predominante**: Em pé (52.3% das detecções)

| Atividade | Frequência | Percentual |
|-----------|------------|------------|
| Em pé | 612 | 52.3% ██████████ |
| Caminhando | 287 | 24.5% ████ |
| Sentado | 189 | 16.1% ███ |
| Acenando | 45 | 3.8% |
| Correndo | 23 | 2.0% |
| Apontando | 15 | 1.3% |

*Total de detecções de atividades: 1171*

## Anomalias Detectadas

**Total**: 12 anomalias comportamentais ou contextuais

### Distribuição por Tipo

| Tipo de Anomalia | Quantidade | Percentual |
|------------------|------------|------------|
| Movimento Anômalo | 7 | 58.3% ███████████ |
| Emoção Negativa | 3 | 25.0% ████ |
| Inatividade Prolongada | 2 | 16.7% ███ |

### Análise Detalhada

As anomalias detectadas representam desvios do padrão normal de comportamento ou
inconsistências contextuais. Estes eventos requerem atenção especial, pois podem indicar
situações de risco, comportamentos suspeitos ou necessidade de intervenção.

### Detalhamento dos Principais Eventos

### 1. Movimento Anômalo
- **Timestamp**: 00:08.3s
- **Frame**: 250
- **Severidade**: Média
- **Descrição**: Gesto brusco detectado - pessoa 1 realizou movimento súbito com os braços

### 2. Movimento Anômalo
- **Timestamp**: 00:12.7s
- **Frame**: 381
- **Severidade**: Alta
- **Descrição**: Mudança súbita de postura - pessoa 2 passou de em pé para deitado rapidamente

### 3. Emoção Negativa
- **Timestamp**: 00:15.2s
- **Frame**: 456
- **Severidade**: Média
- **Descrição**: Emoção negativa persistente (tristeza) detectada por 3 frames consecutivos

### 4. Movimento Anômalo
- **Timestamp**: 00:19.8s
- **Frame**: 594
- **Severidade**: Baixa
- **Descrição**: Velocidade de movimento acima do padrão estabelecido

### 5. Inatividade Prolongada
- **Timestamp**: 00:25.1s
- **Frame**: 753
- **Severidade**: Média
- **Descrição**: Pessoa 3 imóvel por mais de 5 segundos na mesma posição

### Notas sobre Classificação de Anomalias

- **Movimento Anômalo**: Gestos bruscos, quedas ou movimentos atípicos
- **Emoção Negativa**: Manifestações de tristeza, raiva ou medo prolongadas
- **Inatividade**: Períodos de imobilidade superiores ao esperado
- **Contexto Suspeito**: Objetos ou comportamentos incompatíveis com o ambiente

## Metodologia e Tecnologias

### Modelos Utilizados

| Componente | Tecnologia | Propósito |
|------------|-----------|----------|
| Detecção de Atividades | YOLO11-pose | Identificação de pessoas e poses/atividades |
| Análise de Emoções | DeepFace | Reconhecimento facial e classificação emocional |
| Classificação de Cena | YOLO11-cls | Identificação do contexto/ambiente |
| Detecção Orientada | YOLO11-obb | Detecção de pessoas deitadas vs em pé |
| Detecção de Objetos | YOLO11 | Identificação de objetos no ambiente |

### Processo de Análise

1. **Pré-processamento**: Leitura e preparação dos frames do vídeo
2. **Detecção Multi-modal**: Análise simultânea de faces, atividades, objetos e contexto de cena
3. **Classificação Emocional**: Análise facial profunda para identificar estados emocionais
4. **Detecção de Anomalias**: Motor de regras que combina dados comportamentais e visuais
5. **Pós-processamento**: Geração de visualizações e relatórios

### Critérios de Detecção de Anomalias

- **Movimento Anômalo**: Gestos bruscos ou mudanças súbitas não condizentes com o padrão geral
- **Emoções Negativas**: Picos de tristeza, raiva ou medo mantidos por período prolongado
- **Inatividade Prolongada**: Ausência de movimento por tempo superior ao normal
- **Contexto Visual**: Objetos ou posturas incompatíveis com o ambiente detectado
- **Posturas Atípicas**: Pessoas deitadas em ambientes inadequados (escritórios, lojas, etc.)

---
**Tech Challenge - Fase 4: Análise de Vídeo com IA**  
*Relatório gerado automaticamente em 09/01/2026 às 14:30:45*
