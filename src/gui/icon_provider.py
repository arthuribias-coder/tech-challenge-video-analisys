"""
Provedor centralizado de ícones para a interface Qt.
Suporta temas nativos do sistema com fallback para QStyle.
"""

from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt6.QtWidgets import QApplication, QStyle
from PyQt6.QtCore import QSize, Qt
from pathlib import Path
from typing import Optional


class IconProvider:
    """
    Provedor centralizado de ícones com suporte a:
    - Temas nativos do sistema (Freedesktop, macOS, Windows)
    - Fallback para ícones padrão do Qt (QStyle)
    - Fallback para arquivos locais (PNG/SVG)
    """
    
    # Tamanho padrão dos ícones
    DEFAULT_SIZE = QSize(24, 24)
    TOOLBAR_SIZE = QSize(32, 32)
    
    # Diretório de recursos (caso necessário)
    RESOURCE_DIR = Path(__file__).parent.parent.parent / "resources" / "icons"
    
    # Cores para tema escuro
    ICON_COLOR_LIGHT = QColor(224, 224, 224)  # #e0e0e0
    ICON_COLOR_GREEN = QColor(76, 175, 80)    # #4CAF50
    ICON_COLOR_BLUE = QColor(33, 150, 243)    # #2196F3
    ICON_COLOR_ORANGE = QColor(255, 152, 0)   # #FF9800
    
    @staticmethod
    def _colorize_icon(icon: QIcon, color: QColor, size: QSize = None) -> QIcon:
        """
        Coloriza um ícone com uma cor específica.
        Útil para temas escuros onde ícones precisam ser visíveis.
        
        Args:
            icon: Ícone original
            color: Cor para aplicar
            size: Tamanho do ícone (padrão: TOOLBAR_SIZE)
            
        Returns:
            Ícone colorizado
        """
        if icon.isNull():
            return icon
            
        if size is None:
            size = IconProvider.TOOLBAR_SIZE
            
        # Cria pixmap do ícone
        pixmap = icon.pixmap(size)
        
        # Cria novo pixmap colorizado
        colored_pixmap = QPixmap(pixmap.size())
        colored_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(colored_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(colored_pixmap.rect(), color)
        painter.end()
        
        return QIcon(colored_pixmap)
    
    @staticmethod
    def _get_from_theme_with_fallback(
        theme_icon: QIcon.ThemeIcon,
        style_icon: QStyle.StandardPixmap,
        local_path: Optional[str] = None
    ) -> QIcon:
        """
        Obtém ícone do tema com fallback para arquivo local e QStyle.
        
        Args:
            theme_icon: Enum do tema (ex: QIcon.ThemeIcon.DocumentOpen)
            style_icon: Ícone padrão Qt (ex: QStyle.StandardPixmap.SP_DialogOpenButton)
            local_path: Caminho relativo do ícone local (opcional)
            
        Returns:
            QIcon configurado
        """
        # Tenta tema nativo primeiro
        icon = QIcon.fromTheme(theme_icon)
        
        # Se não encontrar, tenta arquivo local SVG (prioridade maior que QStyle)
        if icon.isNull() and local_path:
            full_path = IconProvider.RESOURCE_DIR / local_path
            if full_path.exists():
                icon = QIcon(str(full_path))
        
        # Se ainda não encontrar, usa QStyle padrão do Qt
        if icon.isNull():
            try:
                style = QApplication.style()
                if style:
                    icon = style.standardIcon(style_icon)
            except Exception:
                pass
        
        # Se tudo falhar, retorna ícone vazio (não é None)
        return icon if not icon.isNull() else QIcon()
    
    # ===== ÍCONES DE DOCUMENTO =====
    
    @classmethod
    def document_open(cls) -> QIcon:
        """Ícone para abrir documento/vídeo."""
        icon = cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.DocumentOpen,
            QStyle.StandardPixmap.SP_DialogOpenButton,
            "document-open.svg"
        )
        # Coloriza para melhor visibilidade em tema escuro
        return cls._colorize_icon(icon, cls.ICON_COLOR_BLUE)
    
    @classmethod
    def document_save(cls) -> QIcon:
        """Ícone para salvar documento."""
        icon = cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.DocumentSave,
            QStyle.StandardPixmap.SP_DialogSaveButton,
            "document-save.svg"
        )
        return cls._colorize_icon(icon, cls.ICON_COLOR_LIGHT)
    
    @classmethod
    def folder_new(cls) -> QIcon:
        """Ícone para criar nova pasta."""
        icon = cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.FolderNew,
            QStyle.StandardPixmap.SP_FileDialogNewFolder,
            "folder-new.svg"
        )
        return cls._colorize_icon(icon, cls.ICON_COLOR_LIGHT)
    
    # ===== ÍCONES DE MÍDIA =====
    
    @classmethod
    def media_play(cls) -> QIcon:
        """Ícone para reproduzir/processar (verde destaque)."""
        icon = cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.MediaPlaybackStart,
            QStyle.StandardPixmap.SP_MediaPlay,
            "media-playback-start.svg"
        )
        # Verde para destaque da ação principal
        return cls._colorize_icon(icon, cls.ICON_COLOR_GREEN)
    
    @classmethod
    def media_pause(cls) -> QIcon:
        """Ícone para pausar."""
        icon = cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.MediaPlaybackPause,
            QStyle.StandardPixmap.SP_MediaPause,
            "media-playback-pause.svg"
        )
        return cls._colorize_icon(icon, cls.ICON_COLOR_ORANGE)
    
    @classmethod
    def media_stop(cls) -> QIcon:
        """Ícone para parar."""
        icon = cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.MediaPlaybackStop,
            QStyle.StandardPixmap.SP_MediaStop,
            "media-playback-stop.svg"
        )
        return cls._colorize_icon(icon, QColor(244, 67, 54))  # Vermelho
    
    # ===== ÍCONES DE AÇÃO =====
    
    @classmethod
    def process_stop(cls) -> QIcon:
        """Ícone para parar processamento."""
        return cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.ProcessStop,
            QStyle.StandardPixmap.SP_BrowserStop,
            "process-stop.png"
        )
    
    @classmethod
    def view_refresh(cls) -> QIcon:
        """Ícone para atualizar/refresh."""
        return cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.ViewRefresh,
            QStyle.StandardPixmap.SP_BrowserReload,
            "view-refresh.png"
        )
    
    # ===== ÍCONES DE DIÁLOGO =====
    
    @classmethod
    def dialog_information(cls) -> QIcon:
        """Ícone de informação."""
        return cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.DialogInformation,
            QStyle.StandardPixmap.SP_MessageBoxInformation,
            "dialog-information.png"
        )
    
    @classmethod
    def dialog_question(cls) -> QIcon:
        """Ícone de questão."""
        return cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.DialogQuestion,
            QStyle.StandardPixmap.SP_MessageBoxQuestion,
            "dialog-question.png"
        )
    
    @classmethod
    def dialog_warning(cls) -> QIcon:
        """Ícone de aviso."""
        return cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.DialogWarning,
            QStyle.StandardPixmap.SP_MessageBoxWarning,
            "dialog-warning.png"
        )
    
    @classmethod
    def dialog_error(cls) -> QIcon:
        """Ícone de erro."""
        return cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.DialogError,
            QStyle.StandardPixmap.SP_MessageBoxCritical,
            "dialog-error.png"
        )
    
    # ===== ÍCONES PERSONALIZADOS PARA ANÁLISE =====
    
    @classmethod
    def chart_bar(cls) -> QIcon:
        """Ícone para gráficos/relatórios (usa documentos como fallback)."""
        # Não há ThemeIcon específico para gráficos, usa DocumentProperties
        icon = cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.DocumentProperties,
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
            "document-properties.svg"
        )
        return cls._colorize_icon(icon, cls.ICON_COLOR_GREEN)
    
    @classmethod
    def help_about(cls) -> QIcon:
        """Ícone para 'Sobre'."""
        icon = cls._get_from_theme_with_fallback(
            QIcon.ThemeIcon.HelpAbout,
            QStyle.StandardPixmap.SP_MessageBoxInformation,
            "help-about.svg"
        )
        return cls._colorize_icon(icon, cls.ICON_COLOR_LIGHT)
