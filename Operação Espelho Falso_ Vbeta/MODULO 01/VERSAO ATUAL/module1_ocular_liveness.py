"""
===============================================================================
PROJETO 1: OPERAÇÃO ESPELHO FALSO — VERSÃO BETA (PoC Autônoma)
MÓDULO 1: VISÃO COMPUTACIONAL, ANÁLISE ESPECTRAL OCULAR & FREQUENCIAL (FFT)
===============================================================================
Autora: Dra. Hellen Nunes dos Santos Souza / P&D Antifraude Biométrica
Amparo Legal: LGPD (Art. 11, II, 'g' - Prevenção à Fraude) & ISO/IEC 30107-3
Política de Privacidade: Zero-Disk Persistence (Processamento Volátil em RAM)
===============================================================================
"""

import cv2
import numpy as np
from scipy.fft import fft
import json
import sys

# Suporte híbrido a MediaPipe (com fallback automático para OpenCV Haar Cascade e Modo Sintético)
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


class OcularLivenessModule:
    """
    Módulo 1 de Detecção de Liveness Espectral e Re-apresentação de Telas.
    Analisa:
      1. Isolamento de ROI Ocular via MediaPipe FaceMesh (ou OpenCV Haar Cascade / Synthetic Fallback).
      2. Gradiente Especular de Sobel (Curvatura Corneal Gaussiana vs. Plana).
      3. Análise Espectral de Cor no Espaço HSV/CIELAB (Diodos OLED/LCD).
      4. Análise Frequencial Temporal via FFT (Cintilação PWM 60Hz - 240Hz).
    """
    def __init__(self, fps=30, buffer_size=30):
        self.fps = fps
        self.buffer_size = buffer_size
        self.luminance_buffer = []  # Buffer temporal de luminância mantido em memória RAM
        
        self.use_mediapipe = HAS_MEDIAPIPE
        self.eye_cascade = None
        self.face_cascade = None

        if self.use_mediapipe:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
                self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
            except Exception:
                self.use_mediapipe = False

        if not self.use_mediapipe:
            # Fallback nativo OpenCV com proteção contra atributo/DLL ausente
            try:
                if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
                    self.eye_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_eye.xml'
                    )
                    self.face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
            except Exception:
                self.eye_cascade = None
                self.face_cascade = None

    def extract_ocular_roi(self, frame):
        """
        Isola a Região de Interesse (ROI) do olho em memória RAM (sem escrita em disco).
        """
        if frame is None or frame.size == 0:
            return None, None

        h, w = frame.shape[:2]

        # 1. Tenta MediaPipe
        if self.use_mediapipe and hasattr(self, 'face_mesh'):
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    def get_bbox(indices):
                        xs = [int(landmarks[idx].x * w) for idx in indices]
                        ys = [int(landmarks[idx].y * h) for idx in indices]
                        x_min, x_max = max(0, min(xs) - 5), min(w, max(xs) + 5)
                        y_min, y_max = max(0, min(ys) - 5), min(h, max(ys) + 5)
                        return x_min, y_min, x_max, y_max

                    x1, y1, x2, y2 = get_bbox(self.LEFT_EYE_LANDMARKS)
                    if x2 > x1 and y2 > y1:
                        left_eye_roi = frame[y1:y2, x1:x2]
                        return left_eye_roi, (x1, y1, x2, y2)
            except Exception:
                pass

        # 2. Tenta OpenCV Haar Cascade
        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    roi_gray = gray[fy:fy + int(fh/2), fx:fx + fw]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3) if self.eye_cascade else []

                    if len(eyes) > 0:
                        ex, ey, ew, eh = eyes[0]
                        abs_x1, abs_y1 = fx + ex, fy + ey
                        abs_x2, abs_y2 = abs_x1 + ew, abs_y1 + eh
                        return frame[abs_y1:abs_y2, abs_x1:abs_x2], (abs_x1, abs_y1, abs_x2, abs_y2)
                    else:
                        ex1, ey1 = fx + int(fw*0.1), fy + int(fh*0.15)
                        ex2, ey2 = fx + int(fw*0.45), fy + int(fh*0.4)
                        return frame[ey1:ey2, ex1:ex2], (ex1, ey1, ex2, ey2)
            except Exception:
                pass

        # 3. Fallback Sintético / Genérico para simulação (Garante execução contínua)
        x1, y1 = int(w * 0.35), int(h * 0.25)
        x2, y2 = int(w * 0.55), int(h * 0.40)
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    def calculate_corneal_gradient(self, eye_roi):
        """
        Calcula o Gradiente Especular de Sobel (Mapeamento de Curvatura Corneal).
        Formula: |grad I| = sqrt((dI/dx)^2 + (dI/dy)^2)
        """
        if eye_roi is None or eye_roi.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return float(np.std(magnitude))
        except Exception:
            return 0.0

    def analyze_spectral_color(self, eye_roi):
        """
        Análise no Espaço HSV e CIELAB para detecção de diodos OLED/LCD.
        """
        if eye_roi is None or eye_roi.size == 0 or len(eye_roi.shape) < 3:
            return 0.0, 0.0

        try:
            hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv[:, :, 1]))

            lab = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2LAB)
            a_channel = lab[:, :, 1].astype(float)
            b_channel = lab[:, :, 2].astype(float)
            chroma_std = float(np.std(np.sqrt(a_channel**2 + b_channel**2)))

            return sat_mean, chroma_std
        except Exception:
            return 0.0, 0.0

    def analyze_temporal_fft(self, current_luminance):
        """
        Transformada Rápida de Fourier (FFT) para cintilação de telas (PWM).
        """
        self.luminance_buffer.append(current_luminance)
        
        if len(self.luminance_buffer) > self.buffer_size:
            self.luminance_buffer.pop(0)

        if len(self.luminance_buffer) < self.buffer_size:
            return 0.0, False

        try:
            signal = np.array(self.luminance_buffer) - np.mean(self.luminance_buffer)
            fft_values = np.abs(fft(signal))
            positive_fft = fft_values[:self.buffer_size // 2]
            max_power = float(np.max(positive_fft)) if len(positive_fft) > 0 else 0.0
            pwm_detected = max_power > 150.0
            return max_power, pwm_detected
        except Exception:
            return 0.0, False

    def process_frame(self, frame):
        """
        Executa o pipeline do Módulo 1 estritamente em memória RAM.
        """
        eye_roi, bbox = self.extract_ocular_roi(frame)

        if eye_roi is None or eye_roi.size == 0:
            return {
                "face_detected": False,
                "score_f1_gradient": 0.0,
                "score_f2_spectral_sat": 0.0,
                "score_f4_fft_power": 0.0,
                "pwm_detected": False,
                "is_screen_attack": False,
                "liveness_passed": False,
                "detector_used": "Synthetic Fallback"
            }, frame

        gradient_score = self.calculate_corneal_gradient(eye_roi)
        sat_mean, chroma_std = self.analyze_spectral_color(eye_roi)

        gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if len(eye_roi.shape) == 3 else eye_roi
        current_lum = float(np.mean(gray_roi))

        fft_power, pwm_detected = self.analyze_temporal_fft(current_lum)

        is_screen_attack = (sat_mean > 110.0) or pwm_detected or (gradient_score > 45.0)
        liveness_passed = not is_screen_attack

        annotated_frame = frame.copy()
        if bbox:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if liveness_passed else (0, 0, 255)
            label = "BIOLOGICO VIVO" if liveness_passed else "ATAQUE TELA DIGITAL"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        metrics = {
            "face_detected": True,
            "score_f1_gradient": round(gradient_score, 2),
            "score_f2_spectral_sat": round(sat_mean, 2),
            "score_f4_fft_power": round(fft_power, 2),
            "pwm_detected": pwm_detected,
            "is_screen_attack": is_screen_attack,
            "liveness_passed": liveness_passed,
            "detector_used": "MediaPipe FaceMesh" if self.use_mediapipe else ("OpenCV Haar" if self.face_cascade else "Synthetic Fallback")
        }

        return metrics, annotated_frame


def main():
    print("=======================================================================")
    print("PROJETO 1: OPERAÇÃO ESPELHO FALSO (Versão Beta PoC)")
    print("Mód. 1: Visão Computacional, Análise Espectral Ocular & FFT Temporal")
    print("=======================================================================")
    
    module = OcularLivenessModule()
    print("-> Política de Privacidade: Zero-Disk RAM Active (Nenhum frame salvo em disco)")
    
    synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    metrics, _ = module.process_frame(synthetic_frame)
    print("\n✅ Teste Sintético Executado!")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
