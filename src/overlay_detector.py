"""
Tech Challenge - Fase 4: Detector de Overlays e Texto
Módulo responsável pela detecção de texto sobreposto, watermarks e overlays visuais.
Usa OCR (pytesseract ou EasyOCR) para identificar elementos visuais não-naturais.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re


class OverlayType(Enum):
    """Tipos de overlays detectados."""
    TEXT = "text"                    # Texto genérico
    WATERMARK = "watermark"          # Watermark/logo
    TIMESTAMP = "timestamp"          # Timestamp/data
    LOGO = "logo"                    # Logo corporativo
    SUBTITLE = "subtitle"            # Legenda/subtítulo
    BANNER = "banner"                # Banner promocional
    UI_ELEMENT = "ui_element"        # Elemento de UI (botões, etc)


@dataclass
class OverlayDetection:
    """Representa uma detecção de overlay/texto."""
    overlay_id: int
    overlay_type: OverlayType
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    position_zone: str               # "top_left", "bottom_center", etc
    is_anomalous: bool = False
    anomaly_reason: Optional[str] = None


# Padrões regex para identificar tipos de texto
TEXT_PATTERNS = {
    "timestamp": [
        r'\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4}',  # Datas
        r'\d{1,2}:\d{2}(:\d{2})?(\s*[AP]M)?',  # Horas
        r'\d{4}[/\-\.]\d{2}[/\-\.]\d{2}',  # Datas ISO
    ],
    "watermark": [
        r'©\s*\d{4}',                    # Copyright
        r'@\w+',                          # Handle de rede social
        r'www\.\w+',                      # URLs
        r'https?://\w+',                  # URLs completas
        r'\.(com|org|net|br|io)',         # Domínios
    ],
    "subtitle": [
        r'^[A-ZÀ-Ú].*[.!?]$',            # Frase capitalizada
        r'^\[.*\]$',                      # Texto em colchetes
        r'^-\s+\w+',                      # Diálogo
    ],
}

# Palavras-chave de overlays suspeitos
SUSPICIOUS_KEYWORDS = {
    "subscribe", "like", "follow", "share",  # CTA de redes sociais
    "inscreva", "curta", "siga", "compartilhe",  # CTA em português
    "download", "baixe", "clique", "acesse",  # CTA de download
    "promo", "oferta", "desconto", "grátis",  # Promoções
    "live", "ao vivo", "streaming",  # Indicadores de live
    "rec", "recording", "gravando",  # Indicadores de gravação
}

from .config import should_use_gpu


class OverlayDetector:
    """
    Detector de overlays, texto e elementos visuais sobrepostos.
    Usa OCR para identificar texto e padrões visuais não-naturais.
    """
    
    def __init__(
        self,
        use_easyocr: bool = True,
        min_text_confidence: float = 0.5,
        enable_gpu: bool = None
    ):
        """
        Args:
            use_easyocr: Se True, usa EasyOCR (padrão, 100% Python). False tenta pytesseract.
            min_text_confidence: Confiança mínima para detecção de texto
            enable_gpu: Se True, habilita GPU para EasyOCR. None usa config global.
        """
        self.min_confidence = min_text_confidence
        self.overlay_counter = 0
        self.use_easyocr = use_easyocr
        self.enable_gpu = enable_gpu if enable_gpu is not None else should_use_gpu()
        
        # Cache de regiões para evitar processamento redundante
        self.region_cache: Dict[str, List[OverlayDetection]] = {}
        self.last_frame_hash: Optional[int] = None
        
        # Histórico para detecção de overlays persistentes
        self.overlay_history: Dict[str, int] = {}  # region_key -> count
        self.history_threshold = 10  # Frames para considerar overlay persistente
        
        self._init_ocr(self.enable_gpu)
    
    def _init_ocr(self, enable_gpu: bool):
        """Inicializa engine de OCR."""
        self.ocr_loaded = False
        self.reader = None
        
        if self.use_easyocr:
            try:
                import easyocr
                self.reader = easyocr.Reader(['pt', 'en'], gpu=enable_gpu, verbose=False)
                self.ocr_loaded = True
                gpu_status = "GPU" if enable_gpu else "CPU"
                print(f"[INFO] EasyOCR carregado (PT+EN, {gpu_status})")
            except ImportError:
                print("[AVISO] EasyOCR não disponível. Tentando pytesseract...")
                self.use_easyocr = False
        
        if not self.use_easyocr:
            try:
                import pytesseract
                # Testa se tesseract está instalado
                pytesseract.get_tesseract_version()
                self.ocr_loaded = True
                print("[INFO] Pytesseract carregado")
            except Exception as e:
                print(f"[AVISO] OCR não disponível: {e}")
                self.ocr_loaded = False
    
    def detect(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        detect_in_regions: bool = True
    ) -> List[OverlayDetection]:
        """
        Detecta overlays e texto no frame.
        
        Args:
            frame: Imagem BGR do OpenCV
            frame_number: Número do frame atual
            detect_in_regions: Se True, detecta apenas em regiões típicas de overlay
            
        Returns:
            Lista de detecções de overlay
        """
        if not self.ocr_loaded:
            return []
        
        detections = []
        h, w = frame.shape[:2]
        
        if detect_in_regions:
            # Detecta em regiões típicas de overlays
            regions = self._get_overlay_regions(w, h)
            for region_name, (x, y, rw, rh) in regions.items():
                region = frame[y:y+rh, x:x+rw]
                region_detections = self._detect_text_in_region(
                    region, x, y, region_name, frame_number
                )
                detections.extend(region_detections)
        else:
            # Detecta no frame inteiro (mais lento)
            detections = self._detect_text_in_region(
                frame, 0, 0, "full_frame", frame_number
            )
        
        return detections
    
    def _get_overlay_regions(
        self, 
        width: int, 
        height: int
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Define regiões típicas onde overlays aparecem.
        
        Returns:
            Dict com nome da região e coordenadas (x, y, w, h)
        """
        margin_h = int(height * 0.15)  # 15% margem vertical
        margin_w = int(width * 0.20)   # 20% margem horizontal
        
        return {
            # Cantos superiores (watermarks, logos)
            "top_left": (0, 0, margin_w, margin_h),
            "top_right": (width - margin_w, 0, margin_w, margin_h),
            "top_center": (margin_w, 0, width - 2*margin_w, margin_h),
            
            # Cantos inferiores (timestamps, legendas)
            "bottom_left": (0, height - margin_h, margin_w, margin_h),
            "bottom_right": (width - margin_w, height - margin_h, margin_w, margin_h),
            "bottom_center": (margin_w, height - margin_h, width - 2*margin_w, margin_h),
        }
    
    def _detect_text_in_region(
        self,
        region: np.ndarray,
        offset_x: int,
        offset_y: int,
        region_name: str,
        frame_number: int
    ) -> List[OverlayDetection]:
        """Detecta texto em uma região específica."""
        detections = []
        
        if region.size == 0:
            return detections
        
        # Pré-processamento para melhor OCR
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Aumenta contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        if self.use_easyocr:
            text_results = self._ocr_easyocr(enhanced)
        else:
            text_results = self._ocr_pytesseract(enhanced)
        
        for text, confidence, bbox in text_results:
            if confidence < self.min_confidence or len(text.strip()) < 2:
                continue
            
            # Ajusta bbox para coordenadas globais
            global_bbox = (
                bbox[0] + offset_x,
                bbox[1] + offset_y,
                bbox[2],
                bbox[3]
            )
            
            # Classifica tipo de overlay
            overlay_type = self._classify_text(text)
            
            # Verifica se é anômalo
            is_anomalous, reason = self._check_anomaly(
                text, overlay_type, region_name, frame_number
            )
            
            self.overlay_counter += 1
            detections.append(OverlayDetection(
                overlay_id=self.overlay_counter,
                overlay_type=overlay_type,
                text=text,
                confidence=confidence,
                bbox=global_bbox,
                position_zone=region_name,
                is_anomalous=is_anomalous,
                anomaly_reason=reason
            ))
            
            # Atualiza histórico
            region_key = f"{region_name}_{text[:10]}"
            self.overlay_history[region_key] = self.overlay_history.get(region_key, 0) + 1
        
        return detections
    
    def _ocr_easyocr(
        self, 
        image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Executa OCR com EasyOCR."""
        results = []
        
        try:
            ocr_results = self.reader.readtext(image)
            
            for bbox_points, text, confidence in ocr_results:
                # Converte pontos para bbox (x, y, w, h)
                pts = np.array(bbox_points, dtype=np.int32)
                x = int(min(pts[:, 0]))
                y = int(min(pts[:, 1]))
                w = int(max(pts[:, 0]) - x)
                h = int(max(pts[:, 1]) - y)
                
                results.append((text, confidence, (x, y, w, h)))
        except Exception as e:
            print(f"[ERRO] EasyOCR: {e}")
        
        return results
    
    def _ocr_pytesseract(
        self, 
        image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Executa OCR com pytesseract."""
        results = []
        
        try:
            import pytesseract
            
            # Obtém dados detalhados
            data = pytesseract.image_to_data(
                image, 
                lang='por+eng',
                output_type=pytesseract.Output.DICT
            )
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if conf > 0 and text:
                    x, y, w, h = (
                        data['left'][i],
                        data['top'][i],
                        data['width'][i],
                        data['height'][i]
                    )
                    results.append((text, conf / 100.0, (x, y, w, h)))
        except Exception as e:
            print(f"[ERRO] Pytesseract: {e}")
        
        return results
    
    def _classify_text(self, text: str) -> OverlayType:
        """Classifica o tipo de texto/overlay."""
        text_lower = text.lower()
        
        # Verifica padrões de timestamp
        for pattern in TEXT_PATTERNS["timestamp"]:
            if re.search(pattern, text):
                return OverlayType.TIMESTAMP
        
        # Verifica padrões de watermark
        for pattern in TEXT_PATTERNS["watermark"]:
            if re.search(pattern, text, re.IGNORECASE):
                return OverlayType.WATERMARK
        
        # Verifica palavras-chave suspeitas
        for keyword in SUSPICIOUS_KEYWORDS:
            if keyword in text_lower:
                return OverlayType.BANNER
        
        # Verifica padrões de legenda
        for pattern in TEXT_PATTERNS["subtitle"]:
            if re.search(pattern, text):
                return OverlayType.SUBTITLE
        
        # Default
        return OverlayType.TEXT
    
    def _check_anomaly(
        self,
        text: str,
        overlay_type: OverlayType,
        region_name: str,
        frame_number: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifica se o overlay é anômalo.
        
        Returns:
            Tuple (is_anomalous, reason)
        """
        text_preview = text[:30] + "..." if len(text) > 30 else text
        
        # Watermarks em regiões típicas são suspeitos
        if overlay_type == OverlayType.WATERMARK:
            return True, f"Watermark detectado: '{text_preview}'"
        
        # Banners promocionais
        if overlay_type == OverlayType.BANNER:
            return True, f"Banner/CTA detectado: '{text_preview}'"
        
        # Timestamps que mudam podem indicar gravação de tela
        if overlay_type == OverlayType.TIMESTAMP:
            return True, f"Timestamp sobreposto: '{text_preview}'"
        
        # Logos corporativos
        if overlay_type == OverlayType.LOGO:
            return True, f"Logo detectado: '{text_preview}'"
        
        # Elementos de UI
        if overlay_type == OverlayType.UI_ELEMENT:
            return True, f"Elemento UI detectado: '{text_preview}'"
        
        # Legendas em posições típicas
        if overlay_type == OverlayType.SUBTITLE:
            if region_name in ["bottom_center", "bottom_left", "bottom_right"]:
                return True, f"Legenda detectada: '{text_preview}'"
        
        # Texto genérico em cantos (típico de watermarks/overlays)
        if region_name in ["top_left", "top_right", "bottom_right"]:
            return True, f"Texto sobreposto ({region_name}): '{text_preview}'"
        
        # Overlays persistentes (aparecem em muitos frames)
        region_key = f"{region_name}_{text[:10]}"
        if self.overlay_history.get(region_key, 0) > self.history_threshold:
            return True, f"Overlay persistente: '{text_preview}'"
        
        return False, None
    
    def detect_visual_overlay(self, frame: np.ndarray) -> List[Dict]:
        """
        Detecta overlays visuais sem texto (logos, watermarks gráficos).
        Usa análise de bordas e padrões repetitivos.
        """
        overlays = []
        h, w = frame.shape[:2]
        
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detecta bordas
        edges = cv2.Canny(gray, 50, 150)
        
        # Analisa cantos para padrões de logo
        corners = [
            ("top_left", gray[0:h//6, 0:w//5]),
            ("top_right", gray[0:h//6, w-w//5:w]),
            ("bottom_right", gray[h-h//6:h, w-w//5:w]),
        ]
        
        for corner_name, corner_region in corners:
            # Detecta se há conteúdo significativo (possível logo)
            if corner_region.size > 0:
                std_dev = np.std(corner_region)
                mean_val = np.mean(corner_region)
                
                # Alto contraste em região de canto = possível logo/watermark
                if std_dev > 40 and mean_val < 200 and mean_val > 50:
                    overlays.append({
                        "type": "visual_overlay",
                        "position": corner_name,
                        "contrast": float(std_dev),
                        "is_anomalous": True,
                        "reason": f"Possível logo/watermark visual em {corner_name}"
                    })
        
        return overlays
    
    def get_overlay_summary(self) -> Dict[str, int]:
        """Retorna resumo dos overlays detectados por tipo."""
        summary = {t.value: 0 for t in OverlayType}
        
        for region_key, count in self.overlay_history.items():
            # Extrai tipo do texto (simplificado)
            summary["text"] += count
        
        return summary
    
    def reset(self):
        """Reseta estado do detector."""
        self.overlay_counter = 0
        self.overlay_history.clear()
        self.region_cache.clear()
        self.last_frame_hash = None
