# Diretório de Ícones

Este diretório contém ícones de fallback para sistemas que não possuem tema nativo configurado.

## Estrutura

Os ícones devem seguir a nomenclatura padrão do Freedesktop Icon Theme Specification:

- `document-open.png` - Ícone para abrir arquivos
- `document-save.png` - Ícone para salvar arquivos
- `media-playback-start.png` - Ícone para play/iniciar
- `media-playback-pause.png` - Ícone para pausar
- `media-playback-stop.png` - Ícone para parar
- `chart-bar.png` - Ícone para gráficos/relatórios
- `help-about.png` - Ícone para informações

## Suporte HiDPI

Para suporte a telas de alta resolução, adicione versões `@2x`:

- `document-open@2x.png` (48x48 pixels)

## Fontes de Ícones

Fontes recomendadas para ícones PNG (licenças livres):

1. **Breeze Icons** (KDE): <https://github.com/KDE/breeze-icons>
2. **Adwaita Icons** (GNOME): <https://gitlab.gnome.org/GNOME/adwaita-icon-theme>
3. **Material Icons**: <https://fonts.google.com/icons>
4. **Feather Icons**: <https://feathericons.com/>

## Tamanhos

- Padrão: 24x24 pixels
- Toolbar: 32x32 pixels
- HiDPI: 48x48 ou 64x64 pixels (com sufixo @2x)

## Nota

O sistema usa QStyle.standardIcon() como fallback principal, então estes arquivos são opcionais.
O Qt fornece ícones padrão que funcionam em todas as plataformas.
