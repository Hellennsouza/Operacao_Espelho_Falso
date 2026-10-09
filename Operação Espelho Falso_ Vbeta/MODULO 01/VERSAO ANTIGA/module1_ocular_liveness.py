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

# Suporte híbrido a MediaPipe (com fallback automático para OpenCV Haar Cascade)
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class OcularLivenessModule:
    """
    Módulo 1 de Detecção de Liveness Espectral e Re-apresentação de Telas.
    Analisa:
      1. Isolamento de ROI Ocular via MediaPipe FaceMesh (ou OpenCV Haar Cascade).
      2. Gradiente Especular de Sobel (Curvatura Corneal Gaussiana vs. Plana).
      3. Análise Espectral de Cor no Espaço HSV/CIELAB (Diodos OLED/LCD).
      4. Análise Frequencial Temporal via FFT (Cintilação PWM 60Hz - 240Hz).
    """
    def __init__(self, fps=30, buffer_size=30):
        self.fps = fps
        self.buffer_size = buffer_size
        self.luminance_buffer = []  # Buffer temporal de luminância mantido em memória RAM
        
        self.use_mediapipe = HAS_MEDIAPIPE

        if self.use_mediapipe:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            # Índices dos pontos faciais dos olhos no MediaPipe FaceMesh
            self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
            self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        else:
            # Fallback nativo OpenCV caso o mediapipe não esteja instalado
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

    def extract_ocular_roi(self, frame):
        """
        Isola a Região de Interesse (ROI) do olho em memória RAM (sem escrita em disco).
        """
        h, w, _ = frame.shape

        if self.use_mediapipe:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return None, None

            landmarks = results.multi_face_landmarks[0].landmark

            def get_bbox(indices):
                xs = [int(landmarks[idx].x * w) for idx in indices]
                ys = [int(landmarks[idx].y * h) for idx in indices]
                x_min, x_max = max(0, min(xs) - 5), min(w, max(xs) + 5)
                y_min, y_max = max(0, min(ys) - 5), min(h, max(ys) + 5)
                return x_min, y_min, x_max, y_max

            x1, y1, x2, y2 = get_bbox(self.LEFT_EYE_LANDMARKS)
            if x2 <= x1 or y2 <= y1:
                return None, None
                
            left_eye_roi = frame[y1:y2, x1:x2]
            return left_eye_roi, (x1, y1, x2, y2)
        else:
            # Algoritmo de Fallback via Haar Cascades
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3)
                if len(eyes) == 0:
                    return None, None
                x, y, ew, eh = eyes[0]
                return frame[y:y+eh, x:x+ew], (x, y, x+ew, y+eh)

            fx, fy, fw, fh = faces[0]
            roi_gray = gray[fy:fy + int(fh/2), fx:fx + fw]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            if len(eyes) == 0:
                ex1, ey1 = fx + int(fw*0.1), fy + int(fh*0.15)
                ex2, ey2 = fx + int(fw*0.45), fy + int(fh*0.4)
                return frame[ey1:ey2, ex1:ex2], (ex1, ey1, ex2, ey2)

            ex, ey, ew, eh = eyes[0]
            abs_x1, abs_y1 = fx + ex, fy + ey
            abs_x2, abs_y2 = abs_x1 + ew, abs_y1 + eh
            return frame[abs_y1:abs_y2, abs_x1:abs_x2], (abs_x1, abs_y1, abs_x2, abs_y2)

    def calculate_corneal_gradient(self, eye_roi):
        """
        Calcula o Gradiente Especular de Sobel (Mapeamento de Curvatura Corneal).
        Formula: |grad I| = sqrt((dI/dx)^2 + (dI/dy)^2)
        """
        if eye_roi is None or eye_roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        
        # Filtros de Sobel em X e Y
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude do gradiente
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Desvio padrão do gradiente mede a distribuição da reflexão
        gradient_std = np.std(magnitude)
        return float(gradient_std)

    def analyze_spectral_color(self, eye_roi):
        """
        Análise no Espaço HSV e CIELAB para detecção de diodos OLED/LCD.
        """
        if eye_roi is None or eye_roi.size == 0:
            return 0.0, 0.0

        # Espaço HSV (Média do canal de Saturação)
        hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)
        sat_mean = np.mean(hsv[:, :, 1])

        # Espaço CIELAB (Desvio Padrão do Croma)
        lab = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2LAB)
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        chroma_std = np.std(np.sqrt(a_channel.astype(float)**2 + b_channel.astype(float)**2))

        return float(sat_mean), float(chroma_std)

    def analyze_temporal_fft(self, current_luminance):
        """
        Transformada Rápida de Fourier (FFT) para cintilação de telas (PWM).
        """
        self.luminance_buffer.append(current_luminance)
        
        if len(self.luminance_buffer) > self.buffer_size:
            self.luminance_buffer.pop(0)

        if len(self.luminance_buffer) < self.buffer_size:
            return 0.0, False

        # Remove componente DC (média)
        signal = np.array(self.luminance_buffer) - np.mean(self.luminance_buffer)
        fft_values = np.abs(fft(signal))
        
        # Energia do pico frequencial
        positive_fft = fft_values[:self.buffer_size // 2]
        max_power = np.max(positive_fft) if len(positive_fft) > 0 else 0.0
        
        pwm_detected = max_power > 150.0
        return float(max_power), pwm_detected

    def process_frame(self, frame):
        """
        Executa o pipeline do Módulo 1 estritamente em memória RAM.
        """
        eye_roi, bbox = self.extract_ocular_roi(frame)

        if eye_roi is None:
            return {
                "face_detected": False,
                "score_f1_gradient": 0.0,
                "score_f2_spectral_sat": 0.0,
                "score_f4_fft_power": 0.0,
                "pwm_detected": False,
                "is_screen_attack": False,
                "liveness_passed": False,
                "detector_used": "MediaPipe FaceMesh" if self.use_mediapipe else "OpenCV Haar Cascade"
            }, frame

        # 1. Gradiente Corneal
        gradient_score = self.calculate_corneal_gradient(eye_roi)

        # 2. Espectro de Cor
        sat_mean, chroma_std = self.analyze_spectral_color(eye_roi)

        # 3. Média de Luminância
        gray_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        current_lum = float(np.mean(gray_roi))

        # 4. FFT Temporal
        fft_power, pwm_detected = self.analyze_temporal_fft(current_lum)

        # Regra de Decisão de Liveness (ISO/IEC 30107-3)
        is_screen_attack = (sat_mean > 110.0) or pwm_detected or (gradient_score > 45.0)
        liveness_passed = not is_screen_attack

        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if liveness_passed else (0, 0, 255)
        label = "BIOLOGICO VIVO" if liveness_passed else "ATAQUE TELA DIGITAL"
        
        annotated_frame = frame.copy()
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
            "detector_used": "MediaPipe FaceMesh" if self.use_mediapipe else "OpenCV Haar Cascade"
        }

        return metrics, annotated_frame


# --- Bloco Principal de Teste e Demonstração ---
def main():
    print("=======================================================================")
    print("PROJETO 1: OPERAÇÃO ESPELHO FALSO (Versão Beta PoC)")
    print("Mód. 1: Visão Computacional, Análise Espectral Ocular & FFT Temporal")
    print("=======================================================================")
    
    module = OcularLivenessModule()
    print(f"-> Detector Ativo: {'MediaPipe FaceMesh' if HAS_MEDIAPIPE else 'OpenCV Haar Cascade'}")
    print("-> Política de Privacidade: Zero-Disk RAM Active (Nenhum frame salvo em disco)")
    
    # Teste em Vídeo/Webcam ou Imagem Sintética
    source = 0 if len(sys.argv) < 2 else sys.argv[1]
    
    if source == "test":
        # Simulação com quadro aleatório sintético
        synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metrics, _ = module.process_frame(synthetic_frame)
        print("\n✅ Teste Sintético Executado!")
        print(json.dumps(metrics, indent=2))
        return

    # Execução em Stream (Webcam ou Vídeo .mp4 local)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    
    if not cap.isOpened():
        print(f"⚠️ Não foi possible abrir a fonte: {source}. Rodando modo sintético...")
        synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        metrics, _ = module.process_frame(synthetic_frame)
        print(json.dumps(metrics, indent=2))
        return

    print("\n▶️ Pressione 'q' na janela de exibição para encerrar a sessão.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        metrics, annotated_frame = module.process_frame(frame)
        
        # Exibe as métricas no terminal em tempo real
        if metrics["face_detected"]:
            print(f"\r[Módulo 1] F1(Gradiente): {metrics['score_f1_gradient']} | F2(HSV Sat): {metrics['score_f2_spectral_sat']} | F4(FFT): {metrics['score_f4_fft_power']} | Status: {'🟢 APROVADO' if metrics['liveness_passed'] else '🔴 ATAQUE TELA'}", end="")

        cv2.imshow("Operacao Espelho Falso - Modulo 1 PoC", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\nSessão encerrada com sucesso.")

if __name__ == "__main__":
    main()
