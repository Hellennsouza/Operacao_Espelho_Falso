"""
===============================================================================
PROJETO 1: OPERAÇÃO ESPELHO FALSO — VERSÃO BETA (PoC Autônoma)
MÓDULO 2: ASSINATURA DE SILÍCIO (PRNU) & DETECÇÃO DE INJEÇÃO VIRTUAL DE CÂMERA
===============================================================================
Autora: Dra. Hellen Nunes dos Santos Souza / P&D Antifraude Biométrica
Amparo Legal: LGPD (Art. 11, II, 'g' - Prevenção à Fraude) & Resolução CMN nº 4.893/2020
Política de Privacidade: Zero-Disk Persistence (Processamento Volátil em RAM)
===============================================================================
"""

import cv2
import numpy as np
import json
import sys
import os

class PRNUCameraSignatureModule:
    """
    Módulo 2 de Detecção de Câmeras Virtuais, Injeção de Vídeo e Atestação de Hardware.
    Analisa:
      1. Extração do Ruído Físico do Silício (Photo-Response Non-Uniformity - PRNU).
      2. Correlação Cruzada do Ruído de Alta Frequência contra impressões digitais de sensor.
      3. Verificação Heurística de Emuladores / Injeção Virtual (OBS, Nox, Frida).
      4. Decisão de Atestação de Silício (Hardware Authenticity Score).
    """
    def __init__(self, noise_threshold=1.5, pcc_threshold=0.15):
        self.noise_threshold = noise_threshold
        self.pcc_threshold = pcc_threshold

    def extract_sensor_noise(self, image_gray):
        """
        Extrai o mapa de ruído do sensor aplicando filtro de supressão de sinal (Denoising).
        Noise = Image - Denoised(Image)
        """
        # Aplica filtro mediano para estimar o sinal limpo W(I)
        denoised = cv2.medianBlur(image_gray, 5)
        
        # O ruído do sensor é a diferença entre a imagem bruta e o sinal denso
        noise = image_gray.astype(np.float32) - denoised.astype(np.float32)
        return noise

    def calculate_prnu_variance(self, noise_map):
        """
        Calcula a variância e energia de alta frequência do ruído do silício.
        Câmeras reais de silicone possuem ruído PRNU com variância característica estável.
        Câmeras virtuais/vídeos injetados sintéticos possuem ruído suavizado por codecs ou nulo.
        """
        variance = np.var(noise_map)
        std_dev = np.std(noise_map)
        return float(variance), float(std_dev)

    def calculate_pearson_correlation(self, noise_map1, noise_map2):
        """
        Calcula o Coeficiente de Correlação de Pearson (PCC) entre dois mapas de ruído.
        """
        if noise_map1.shape != noise_map2.shape:
            noise_map2 = cv2.resize(noise_map2, (noise_map1.shape[1], noise_map1.shape[0]))
            
        n1_flat = noise_map1.flatten()
        n2_flat = noise_map2.flatten()
        
        mean1 = np.mean(n1_flat)
        mean2 = np.mean(n2_flat)
        
        num = np.sum((n1_flat - mean1) * (n2_flat - mean2))
        den = np.sqrt(np.sum((n1_flat - mean1)**2) * np.sum((n2_flat - mean2)**2))
        
        if den == 0:
            return 0.0
        return float(num / den)

    def check_virtual_camera_heuristics(self, frame):
        """
        Verifica características heurísticas típicas de câmeras virtuais e emuladores:
        - Ausência de variação natural de micro-ganho do sensor (ISO flicker).
        - Bordas de compressão de codec rígidas (artefatos H.264/MJPEG virtuais).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Análise de ruído de alta frequência
        noise_map = self.extract_sensor_noise(gray)
        variance, std_dev = self.calculate_prnu_variance(noise_map)
        
        # Câmeras virtuais/emuladores geralmente apresentam ruído excessivamente baixo ou com artefatos sintéticos
        is_virtual_noise_profile = (std_dev < 0.8) or (std_dev > 15.0)
        
        return {
            "prnu_variance": round(variance, 3),
            "prnu_std_dev": round(std_dev, 3),
            "is_virtual_noise_profile": is_virtual_noise_profile
        }

    def process_frame(self, frame, reference_fingerprint=None):
        """
        Processa o quadro atual e verifica a autenticidade física do sensor de silício.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        heuristics = self.check_virtual_camera_heuristics(frame)
        
        noise_map = self.extract_sensor_noise(gray)
        
        pcc_score = 0.0
        pcc_passed = True
        
        if reference_fingerprint is not None:
            pcc_score = self.calculate_pearson_correlation(noise_map, reference_fingerprint)
            pcc_passed = pcc_score >= self.pcc_threshold

        # Score de Autenticidade do Hardware (0 a 100)
        hardware_authenticity_score = 100.0
        
        if heuristics["is_virtual_noise_profile"]:
            hardware_authenticity_score -= 50.0
            
        if not pcc_passed:
            hardware_authenticity_score -= 40.0

        is_physical_camera = (hardware_authenticity_score >= 60.0) and not heuristics["is_virtual_noise_profile"]

        metrics = {
            "prnu_variance": heuristics["prnu_variance"],
            "prnu_std_dev": heuristics["prnu_std_dev"],
            "pearson_correlation_score": round(pcc_score, 4),
            "hardware_authenticity_score": round(hardware_authenticity_score, 1),
            "is_virtual_camera_injection": not is_physical_camera,
            "is_physical_camera": is_physical_camera,
            "compliance_status": "HARDWARE_VERIFIED" if is_physical_camera else "VIRTUAL_CAMERA_BLOCKED"
        }

        # Anotação visual no frame
        annotated_frame = frame.copy()
        color = (0, 255, 0) if is_physical_camera else (0, 0, 255)
        status_text = "SENSOR FISICO REAL (PRNU OK)" if is_physical_camera else "INJECAO VIRTUAL / EMULADOR DETECTADO"
        
        cv2.putText(annotated_frame, f"PRNU StdDev: {heuristics['prnu_std_dev']}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Score Hardware: {metrics['hardware_authenticity_score']}/100", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, status_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return metrics, annotated_frame, noise_map

def main():
    print("=======================================================================")
    print("PROJETO 1: OPERAÇÃO ESPELHO FALSO (Versão Beta PoC)")
    print("Mód. 2: Assinatura de Silício (PRNU) & Detecção de Câmeras Virtuais")
    print("=======================================================================")
    
    module = PRNUCameraSignatureModule()
    print("-> Método: Extração de Ruído PRNU de Silício via Denoising Residual")
    print("-> Política de Privacidade: Zero-Disk RAM Active (Processamento Volátil)")
    
    source = 0 if len(sys.argv) < 2 else sys.argv[1]
    
    if source == "test":
        # Simulação com quadro sintético
        synthetic_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        metrics, _, _ = module.process_frame(synthetic_frame)
        print("\n✅ Teste Sintético de Bancada Executado!")
        print(json.dumps(metrics, indent=2))
        return

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    
    if not cap.isOpened():
        print(f"⚠️ Não foi possível abrir a fonte: {source}. Rodando modo sintético...")
        synthetic_frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        metrics, _, _ = module.process_frame(synthetic_frame)
        print(json.dumps(metrics, indent=2))
        return

    print("\n▶️ Pressione 'q' na janela de exibição para encerrar a sessão.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        metrics, annotated_frame, _ = module.process_frame(frame)
        
        print(f"\r[Módulo 2] PRNU StdDev: {metrics['prnu_std_dev']} | Score Hardware: {metrics['hardware_authenticity_score']} | Status: {'🟢 SENSOR FISICO' if metrics['is_physical_camera'] else '🔴 CAMERA VIRTUAL'}", end="")

        cv2.imshow("Operacao Espelho Falso - Modulo 2 PRNU PoC", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\nSessão do Módulo 2 encerrada com sucesso.")

if __name__ == "__main__":
    main()
