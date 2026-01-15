# CitrusDetector.py
# Autor: Pedro Juan Torres González -- z32togop@uco.es
# Fecha: 11 de julio de 2025

"""
Desarrollado en el marco del Proyecto Europeo CitriData
https://www.uco.es/citridata/
https://www.linkedin.com/company/citridata/


Basado en el desarrollo previo de PUTVision--DeepNess
https://github.com/PUTvision/qgis-plugin-deepness?tab=readme-ov-file
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepNessModelProcessor:
    """
    Clase base genérica para procesar modelos ONNX usando componentes de DeepNess.
    
    Esta clase sirve como PLANTILLA para crear procesadores específicos
    para diferentes tipos de modelos (segmentación, detección, etc.) [Modular y extensible].
    """
    def __init__(self, model_path: str, model_type: str = "segmentation"):
        """
        Inicializa el procesador de modelos.
        
        Args:
            model_path (str): Ruta al archivo .onnx del modelo
            model_type (str): Tipo de modelo ('segmentation', 'detection', 'classification')
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        
        # Verificar que el modelo existe
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        
        # Inicializar sesión ONNX
        self._initialize_session()
        
        logger.info(f"Procesador inicializado para modelo: {self.model_path.name}")
    
    def _initialize_session(self):
        """
        Inicializa la sesión ONNX Runtime y extrae metadatos del modelo.
        """
        try:
            # Crear sesión ONNX
            self.session = ort.InferenceSession(str(self.model_path))
            
            # Extraer información de entrada del modelo
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape
            
            # Extraer información de salida del modelo
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"Modelo cargado exitosamente:")
            logger.info(f"  - Input: {self.input_name}, Shape: {self.input_shape}")
            logger.info(f"  - Outputs: {self.output_names}")
            
        except Exception as e:
            logger.error(f"Error al inicializar sesión ONNX: {e}")
            raise   
    
    def run_inference(self, preprocessed_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Ejecuta inferencia en el modelo ONNX.
        
        Args:
            preprocessed_image (np.ndarray): Imagen preprocesada
            
        Returns:
            Dict[str, np.ndarray]: Diccionario con resultados de inferencia
        """
        if self.session is None:
            raise RuntimeError("Sesión ONNX no inicializada")
        
        try:
            # Asegurar que el input sea float32
            if preprocessed_image.dtype != np.float32:
                logger.warning(f"Convirtiendo input de {preprocessed_image.dtype} a float32")
                preprocessed_image = preprocessed_image.astype(np.float32)
            
            logger.info(f"Ejecutando inferencia con input tipo: {preprocessed_image.dtype}")
            
            # Ejecutar inferencia
            input_dict = {self.input_name: preprocessed_image}
            outputs = self.session.run(self.output_names, input_dict)
            
            # Crear diccionario de resultados
            results = {}
            for name, output in zip(self.output_names, outputs):
                results[name] = output
                logger.info(f"Output {name}: shape {output.shape}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error en inferencia: {e}")
            raise


class CitrusDetector_vf(DeepNessModelProcessor):
    """
    Detector especializado para copas de árboles/cítricos usando YOLOv9.
    
    - Conversión de coordenadas precisa
    - NMS robusto
    - Visualización integrada
    - Manejo correcto de transformaciones de imagen
    
    Características técnicas:
    - Input: [1, 3, 640, 640] RGB normalizado [0-1]
    - Output: [1, 5, 8400] coordenadas en píxeles del input (640x640)
    - Single-class: Solo detecta árboles/cítricos
    - NMS y filtrado por confianza incluidos
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Inicializa el detector de cítricos corregido.
        
        Args:
            model_path (str): Ruta al modelo YOLOv9 ONNX
            confidence_threshold (float): Umbral mínimo de confianza [0-1]
            nms_threshold (float): Umbral IoU para Non-Maximum Suppression [0-1]
        """
        # Llamar al constructor padre
        super().__init__(model_path, model_type="detection")
        
        # Configuración específica del modelo
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (640, 640)  # Tamaño fijo YOLOv9
        
        # Variables para conversión de coordenadas
        self.scale = 1.0
        self.x_offset = 0
        self.y_offset = 0
        self.original_shape = None
        
        print(f"🌳 CitrusDetector_vf inicializado:")
        print(f"   📁 Modelo: {model_path}")
        print(f"   🎯 Confianza mínima: {confidence_threshold}")
        print(f"   🔄 NMS threshold: {nms_threshold}")
        print(f"   📐 Tamaño entrada: {self.input_size}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento específico para YOLOv9 con transformaciones correctas.
        
        Implementa:
        - Redimensionamiento manteniendo aspect ratio
        - Padding para centrar la imagen
        - Normalización [0-1]
        - Conversión RGB y formato CHW
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"No se pudo cargar la imagen: {image}")
        
        # Guardar dimensiones originales para conversión posterior
        self.original_shape = image.shape[:2]  # (height, width)
        
        # Convertir BGR a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ========================================
        # REDIMENSIONAMIENTO CON ASPECT RATIO
        # ========================================
        h, w = rgb_image.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Redimensionar imagen manteniendo proporciones
        resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # ========================================
        # PADDING PARA CENTRAR
        # ========================================
        # Crear imagen con padding (fondo gris: 114)
        padded = np.full((self.input_size[1], self.input_size[0], 3), 114, dtype=np.uint8)
        
        # Calcular offsets para centrar
        y_offset = (self.input_size[1] - new_h) // 2
        x_offset = (self.input_size[0] - new_w) // 2
        
        # Colocar imagen redimensionada en el centro
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # Guardar información para conversión de coordenadas
        self.scale = scale
        self.x_offset = x_offset
        self.y_offset = y_offset
        
        # ========================================
        # NORMALIZACIÓN Y FORMATO TENSOR
        # ========================================
        # Normalizar píxeles [0-255] -> [0-1]
        normalized = padded.astype(np.float32) / 255.0
        
        # Convertir HWC -> CHW (ONNX format)
        tensor = np.transpose(normalized, (2, 0, 1))
        
        # Añadir dimensión batch [C,H,W] -> [1,C,H,W]
        batched = np.expand_dims(tensor, axis=0)
        
        # Verificar formato final
        assert batched.shape == (1, 3, 640, 640), f"Shape incorrecto: {batched.shape}"
        assert batched.dtype == np.float32, f"Tipo incorrecto: {batched.dtype}"
        
        logger.info(f"Preprocesamiento: {image.shape} -> {batched.shape}")
        return batched
    
    def postprocess_results(self, results: Dict[str, np.ndarray], 
                          original_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Postprocesamiento CORREGIDO específico para formato YOLOv9.
        
        ✅ CORREGIDO: Maneja correctamente coordenadas en píxeles del input (640x640)
        
        Implementa:
        1. Extracción correcta de salida transpuesta
        2. Filtrado por confianza
        3. Conversión centro+tamaño -> esquinas
        4. Non-Maximum Suppression robusto
        5. Conversión precisa a coordenadas de imagen original
        """
        # ========================================
        # 1. EXTRAER Y VERIFICAR SALIDA
        # ========================================
        output_names = list(results.keys())
        predictions = results[output_names[0]]  # Shape: [1, 5, 8400]
        
        logger.info(f"Output original shape: {predictions.shape}")
        
        # ⚠️ CRÍTICO: Transponer [1,5,8400] -> [1,8400,5]
        predictions = np.transpose(predictions, (0, 2, 1))  # [1, 8400, 5]
        predictions = predictions[0]  # Remover batch: [8400, 5]
        
        logger.info(f"Output transpuesto: {predictions.shape}")
        
        # ========================================
        # 2. EXTRAER COMPONENTES
        # ========================================
        # Cada detección: [x_center, y_center, width, height, confidence]
        # ✅ CORREGIDO: Coordenadas están en píxeles del input (640x640)
        x_center = predictions[:, 0]  # Píxeles en espacio 640x640
        y_center = predictions[:, 1]
        width = predictions[:, 2]
        height = predictions[:, 3]
        confidence = predictions[:, 4]
        
        # ========================================
        # 3. FILTRAR POR CONFIANZA
        # ========================================
        valid_mask = confidence >= self.confidence_threshold
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            logger.warning(f"No hay detecciones con confianza >= {self.confidence_threshold}")
            return []
        
        logger.info(f"Detecciones válidas: {valid_count} / 8400")
        
        # Aplicar filtro
        x_center = x_center[valid_mask]
        y_center = y_center[valid_mask]
        width = width[valid_mask]
        height = height[valid_mask]
        confidence = confidence[valid_mask]
        
        # ========================================
        # 4. CONVERTIR A FORMATO ESQUINAS (en espacio 640x640)
        # ========================================
        x1_640 = x_center - width / 2
        y1_640 = y_center - height / 2
        x2_640 = x_center + width / 2
        y2_640 = y_center + height / 2
        
        # ========================================
        # 5. APLICAR NON-MAXIMUM SUPPRESSION
        # ========================================
        boxes_for_nms = np.column_stack([x1_640, y1_640, x2_640, y2_640, confidence])
        keep_indices = self._apply_nms(boxes_for_nms, self.nms_threshold)
        
        if len(keep_indices) == 0:
            logger.warning("No quedan detecciones después de NMS")
            return []
        
        logger.info(f"Detecciones después de NMS: {len(keep_indices)}")
        
        # ========================================
        # 6. CONVERTIR A COORDENADAS DE IMAGEN ORIGINAL
        # ========================================
        img_height, img_width = original_image.shape[:2]
        final_detections = []
        
        for idx in keep_indices:
            # Coordenadas en espacio 640x640
            x1_640_val = x1_640[idx]
            y1_640_val = y1_640[idx]
            x2_640_val = x2_640[idx]
            y2_640_val = y2_640[idx]
            conf = confidence[idx]
            
            # ✅ CORREGIDO: Transformación inversa correcta
            x1_orig = (x1_640_val - self.x_offset) / self.scale
            y1_orig = (y1_640_val - self.y_offset) / self.scale
            x2_orig = (x2_640_val - self.x_offset) / self.scale
            y2_orig = (y2_640_val - self.y_offset) / self.scale
            
            # Clamp a límites válidos
            x1_orig = max(0, min(x1_orig, img_width))
            y1_orig = max(0, min(y1_orig, img_height))
            x2_orig = max(0, min(x2_orig, img_width))
            y2_orig = max(0, min(y2_orig, img_height))
            
            # Verificar que el bbox tiene área válida
            if x2_orig > x1_orig and y2_orig > y1_orig:
                detection = {
                    'bbox': [float(x1_orig), float(y1_orig), float(x2_orig), float(y2_orig)],
                    'confidence': float(conf),
                    'class_id': 0,  # Single class: árboles/cítricos
                    'class_name': 'citrus_tree',
                    'center': [float((x1_orig + x2_orig) / 2), float((y1_orig + y2_orig) / 2)],
                    'area': float((x2_orig - x1_orig) * (y2_orig - y1_orig))
                }
                final_detections.append(detection)
        
        logger.info(f"✅ Detecciones finales: {len(final_detections)} copas detectadas")
        return final_detections
    
    def _apply_nms(self, boxes: np.ndarray, nms_threshold: float) -> List[int]:
        """
        Implementa Non-Maximum Suppression usando IoU.
        
        Args:
            boxes: Array [N, 5] con [x1, y1, x2, y2, confidence]
            nms_threshold: Umbral IoU para eliminar duplicados
            
        Returns:
            List[int]: Índices de las detecciones a mantener
        """
        if len(boxes) == 0:
            return []
        
        # Extraer coordenadas y confidencias
        x1, y1, x2, y2, scores = boxes.T
        
        # Calcular áreas de todos los bounding boxes
        areas = (x2 - x1) * (y2 - y1)
        
        # Ordenar por confianza (descendente)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            # Tomar la detección con mayor confianza
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calcular IoU con el resto
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            # Calcular intersección
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            # Calcular IoU
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-8)
            
            # Mantener solo detecciones con IoU menor al threshold
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                           confidence_threshold: float = 0.5, 
                           title: str = "Detección de Cítricos",
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> int:
        """
        Visualiza las detecciones con bounding boxes y etiquetas.
        
        Args:
            image: Imagen original en formato RGB
            detections: Lista de detecciones del postprocesamiento
            confidence_threshold: Umbral mínimo para mostrar detecciones
            title: Título de la visualización
            save_path: Ruta opcional para guardar la imagen
            figsize: Tamaño de la figura
            
        Returns:
            int: Número de detecciones válidas visualizadas
        """
        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.title(title, fontsize=16, fontweight='bold')
        
        valid_count = 0
        for det in detections:
            if det['confidence'] >= confidence_threshold:
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                
                # Dibujar rectángulo
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='lime', facecolor='none', alpha=0.8)
                plt.gca().add_patch(rect)
                
                # Etiqueta con confianza
                plt.text(x1, y1-5, f'{det["confidence"]:.2f}', 
                        color='lime', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                valid_count += 1
        
        plt.axis('off')
        
        # Estadísticas en la esquina
        if len(detections) > 0:
            confidences = [d['confidence'] for d in detections if d['confidence'] >= confidence_threshold]
            stats_text = f'Detecciones: {valid_count}/{len(detections)}\n'
            if confidences:
                stats_text += f'Conf. promedio: {np.mean(confidences):.2f}\n'
                stats_text += f'Conf. máxima: {max(confidences):.2f}'
        else:
            stats_text = 'No hay detecciones'
            
        plt.text(0.02, 0.98, stats_text, 
                transform=plt.gca().transAxes, color='white', fontsize=12, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualización guardada en: {save_path}")
        
        plt.show()
        return valid_count
    
    def detect_and_visualize(self, image_path: str, 
                           confidence_threshold: Optional[float] = None,
                           save_path: Optional[str] = None) -> Tuple[List[Dict], int]:
        """
        Función completa: carga imagen, detecta y visualiza en un solo paso.
        
        Args:
            image_path: Ruta a la imagen
            confidence_threshold: Umbral de confianza (usa el default si es None)
            save_path: Ruta para guardar visualización
            
        Returns:
            Tuple[List[Dict], int]: (detecciones, número_válidas_visualizadas)
        """
        # Usar umbral de la instancia si no se especifica otro
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar
        preprocessed = self.preprocess_image(image)
        raw_results = self.run_inference(preprocessed)
        detections = self.postprocess_results(raw_results, image)
        
        # Visualizar
        valid_count = self.visualize_detections(
            image_rgb, detections, 
            confidence_threshold=conf_thresh,
            title=f"🌳 Detecciones de Cítricos - {os.path.basename(image_path)}",
            save_path=save_path
        )
        
        return detections, valid_count
    
    def get_statistics(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas de las detecciones.
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Dict con estadísticas
        """
        if not detections:
            return {'total': 0, 'message': 'No hay detecciones'}
        
        confidences = [d['confidence'] for d in detections]
        areas = [d['area'] for d in detections]
        
        stats = {
            'total': len(detections),
            'confidence': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(min(confidences)),
                'max': float(max(confidences)),
                'median': float(np.median(confidences))
            },
            'area': {
                'mean': float(np.mean(areas)),
                'std': float(np.std(areas)),
                'min': float(min(areas)),
                'max': float(max(areas)),
                'median': float(np.median(areas))
            },
            'by_confidence': {
                'high_conf_0.8': len([d for d in detections if d['confidence'] >= 0.8]),
                'medium_conf_0.5': len([d for d in detections if 0.5 <= d['confidence'] < 0.8]),
                'low_conf_0.3': len([d for d in detections if 0.3 <= d['confidence'] < 0.5])
            }
        }
        
        return stats


#######################################################################################################
#  EJEMPLO DE USO DEL DETECTOR
#########################

# def main():
#     """
#     Función de ejemplo para demostrar el uso de CitrusDetector_vf
#     """
#     print("🌳 EJEMPLO DE USO DE CitrusDetector_vf")
#     print("=" * 50)
    
#     # Rutas de ejemplo (ajustar según tu configuración)
#     model_path = "models/yolov9_trees.onnx"
#     image_path = "data/palma.png"
    
#     # Verificar que los archivos existen
#     if not os.path.exists(model_path):
#         print(f"❌ Modelo no encontrado: {model_path}")
#         return
    
#     if not os.path.exists(image_path):
#         print(f"❌ Imagen no encontrada: {image_path}")
#         return
    
#     try:
#         # Crear detector
#         detector = CitrusDetector_vf(
#             model_path=model_path,
#             confidence_threshold=0.3,
#             nms_threshold=0.4
#         )
        
#         # Detectar y visualizar
#         detections, valid_count = detector.detect_and_visualize(
#             image_path=image_path,
#             confidence_threshold=0.5,
#             save_path="citrus_detection_result.png"
#         )
        
#         # Mostrar estadísticas
#         stats = detector.get_statistics(detections)
#         print(f"\n📊 ESTADÍSTICAS:")
#         print(f"   Total: {stats['total']}")
#         print(f"   Visualizadas: {valid_count}")
#         if stats['total'] > 0:
#             print(f"   Confianza promedio: {stats['confidence']['mean']:.3f}")
#             print(f"   Área promedio: {stats['area']['mean']:.1f} píxeles²")
        
#         print(f"\n✅ ¡Detección completada exitosamente!")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")


# if __name__ == "__main__":
#     main()
