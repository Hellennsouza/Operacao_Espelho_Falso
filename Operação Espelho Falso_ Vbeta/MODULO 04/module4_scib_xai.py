"""
===============================================================================
PROJETO 1: OPERAÇÃO ESPELHO FALSO — VERSÃO BETA (PoC Autônoma)
MÓDULO 4: SCORE SCIB, TRAVA CAUTELAR & EXPLICABILIDADE (XAI - SHAP/LIME)
===============================================================================
Autora: Dra. Hellen Nunes dos Santos Souza / P&D Antifraude Biométrica
Amparo Legal: LGPD (Art. 20 - Direito de Revisão de Decisões Automatizadas)
               Resolução BCB nº 142/2021 (Trava Cautelar de Crédito no PIX)
Política de Privacidade: Zero-Disk Persistence (Processamento Volátil em RAM)
===============================================================================
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys

# Importação defensiva do scikit-learn e SHAP
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class SCIBScoreAndXAIModule:
    """
    Módulo 4: Fusão de Métricas, Cálculo do Score SCIB (0-1000) e XAI.
    
    Características de Entrada (Features f1 a f7):
      f1_corneal_gradient: Gradiente de Sobel (0.0 - 100.0)
      f2_spectral_saturation: Saturação HSV (0.0 - 255.0)
      f3_fft_power: Energia Frequencial FFT de Cintilação (0.0 - 500.0)
      f4_prnu_correlation: Correlação de Pearson PRNU (0.0 - 1.0)
      f5_device_integrity: Key Attestation TEE (0 = Íntegro, 1 = Comprometido/Frida/Root)
      f6_graph_mule_risk: Risk Score de Comunidade no Grafo (0.0 - 1.0)
      f7_behavioral_score: Telemetria de Giroscópio/Teclado (0.0 - 100.0)
    """

    def __init__(self):
        self.has_sklearn = HAS_SKLEARN
        self.has_shap = HAS_SHAP
        self.model = None
        self.feature_names = [
            "f1_corneal_gradient",
            "f2_spectral_saturation",
            "f3_fft_power",
            "f4_prnu_correlation",
            "f5_device_integrity",
            "f6_graph_mule_risk",
            "f7_behavioral_score"
        ]
        self._train_baseline_model()

    def _generate_synthetic_training_data(self, n_samples=800):
        """Gera base sintética para calibração de bancada local."""
        np.random.seed(42)
        
        # 60% Legítimos (Classe 0) e 40% Fraudes/Deepfakes (Classe 1)
        y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
        
        f1 = np.where(y == 0, np.random.uniform(10, 35, n_samples), np.random.uniform(45, 90, n_samples))
        f2 = np.where(y == 0, np.random.uniform(30, 90, n_samples), np.random.uniform(115, 200, n_samples))
        f3 = np.where(y == 0, np.random.uniform(0, 80, n_samples), np.random.uniform(160, 450, n_samples))
        f4 = np.where(y == 0, np.random.uniform(0.75, 0.99, n_samples), np.random.uniform(0.0, 0.35, n_samples))
        f5 = np.where(y == 0, np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02]), np.random.choice([0, 1], size=n_samples, p=[0.15, 0.85]))
        f6 = np.where(y == 0, np.random.uniform(0.0, 0.25, n_samples), np.random.uniform(0.65, 0.98, n_samples))
        f7 = np.where(y == 0, np.random.uniform(70, 100, n_samples), np.random.uniform(10, 50, n_samples))

        X = np.column_stack([f1, f2, f3, f4, f5, f6, f7])
        return X, y

    def _train_baseline_model(self):
        """Treina um classificador RandomForest calibrado para o Score SCIB."""
        if not self.has_sklearn:
            return

        X, y = self._generate_synthetic_training_data()
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=42
        )
        self.model.fit(X_train, y_train)

    def calculate_scib_heuristic(self, features_dict):
        """Mecanismo heurístico de fallback caso o sklearn não esteja carregado."""
        f1 = features_dict.get("f1_corneal_gradient", 20.0)
        f2 = features_dict.get("f2_spectral_saturation", 50.0)
        f3 = features_dict.get("f3_fft_power", 30.0)
        f4 = features_dict.get("f4_prnu_correlation", 0.9)
        f5 = features_dict.get("f5_device_integrity", 0)
        f6 = features_dict.get("f6_graph_mule_risk", 0.1)
        f7 = features_dict.get("f7_behavioral_score", 90.0)

        risk_f1 = min(1.0, max(0.0, (f1 - 30) / 40))
        risk_f2 = min(1.0, max(0.0, (f2 - 100) / 80))
        risk_f3 = min(1.0, max(0.0, (f3 - 100) / 200))
        risk_f4 = min(1.0, max(0.0, (0.7 - f4) / 0.7))
        risk_f5 = float(f5)
        risk_f6 = f6
        risk_f7 = min(1.0, max(0.0, (60 - f7) / 60))

        # Ponderação Multimodal da v8.0
        weighted_risk = (
            0.20 * risk_f1 +
            0.15 * risk_f2 +
            0.20 * risk_f3 +
            0.20 * risk_f4 +
            0.10 * risk_f5 +
            0.10 * risk_f6 +
            0.05 * risk_f7
        )

        scib_score = int(weighted_risk * 1000)
        return min(1000, max(0, scib_score))

    def _generate_heuristic_contributions(self, features_dict):
        """Gera pesos relativos dos atributos quando SHAP não puder ser chamado."""
        f1 = features_dict.get("f1_corneal_gradient", 20.0)
        f2 = features_dict.get("f2_spectral_saturation", 50.0)
        f3 = features_dict.get("f3_fft_power", 30.0)
        f4 = features_dict.get("f4_prnu_correlation", 0.9)
        f5 = features_dict.get("f5_device_integrity", 0)
        f6 = features_dict.get("f6_graph_mule_risk", 0.1)
        f7 = features_dict.get("f7_behavioral_score", 90.0)

        return {
            "f1_corneal_gradient": float(np.round(min(1.0, max(0.0, (f1 - 30) / 40)) * 0.20, 4)),
            "f2_spectral_saturation": float(np.round(min(1.0, max(0.0, (f2 - 100) / 80)) * 0.15, 4)),
            "f3_fft_power": float(np.round(min(1.0, max(0.0, (f3 - 100) / 200)) * 0.20, 4)),
            "f4_prnu_correlation": float(np.round(min(1.0, max(0.0, (0.7 - f4) / 0.7)) * 0.20, 4)),
            "f5_device_integrity": float(np.round(float(f5) * 0.10, 4)),
            "f6_graph_mule_risk": float(np.round(f6 * 0.10, 4)),
            "f7_behavioral_score": float(np.round(min(1.0, max(0.0, (60 - f7) / 60)) * 0.05, 4))
        }

    def compute_scib_score(self, features_dict):
        """Calcula a probabilidade de fraude e escala para o Score SCIB (0 a 1000)."""
        if self.model is not None:
            input_vector = np.array([[features_dict.get(col, 0.0) for col in self.feature_names]])
            prob_fraud = self.model.predict_proba(input_vector)[0][1]
            scib_score = int(prob_fraud * 1000)
        else:
            scib_score = self.calculate_scib_heuristic(features_dict)

        return scib_score

    def generate_xai_explanation(self, features_dict, scib_score):
        """Gera a explicação de IA (SHAP/LIME) em conformidade com o Art. 20 da LGPD."""
        contributions = {}
        
        if self.model is not None and self.has_shap:
            try:
                input_vector = np.array([[features_dict.get(col, 0.0) for col in self.feature_names]])
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(input_vector)
                
                # Trata SHAP para classificação binária
                vals = shap_values[1][0] if isinstance(shap_values, list) and len(shap_values) > 1 else (shap_values[0] if isinstance(shap_values, list) else shap_values[0])
                
                for col, val in zip(self.feature_names, vals):
                    contributions[col] = float(np.round(val, 4))
            except Exception:
                contributions = self._generate_heuristic_contributions(features_dict)
        else:
            contributions = self._generate_heuristic_contributions(features_dict)

        # Seleciona os 3 principais fatores de risco
        sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_risk_factors = sorted_factors[:3]

        # Tradução dos fatores para linguagem natural pericial
        factor_descriptions = {
            "f1_corneal_gradient": "Superfície óptica reflexiva incompatível com curvatura corneal humana (Ataque por Tela Digital).",
            "f2_spectral_saturation": "Pico não-natural de saturação de cor característico de diodos OLED/LCD.",
            "f3_fft_power": "Frequência de cintilação temporal detectada (Modulação PWM de Tela de Display).",
            "f4_prnu_correlation": "Ausência de ruído físico de silício do sensor de câmera (Injeção Virtual por Software).",
            "f5_device_integrity": "Dispositivo móvel com ambiente adulterado (Detecção de Root/Jailbreak/Frida).",
            "f6_graph_mule_risk": "Vínculo direto com cluster de contas mulas e identidades sintéticas no Identity Graph.",
            "f7_behavioral_score": "Padrão de interação física e micro-tremores divergente do comportamento humano."
        }

        natural_language_reasons = []
        for factor_name, score_val in top_risk_factors:
            if score_val > 0.01 or scib_score >= 750:
                desc = factor_descriptions.get(factor_name, "Anomalia no parâmetro de verificação.")
                natural_language_reasons.append(f"• {desc} (Peso de impacto: {score_val:+.3f})")

        if not natural_language_reasons and scib_score < 750:
            natural_language_reasons.append("• Todos os parâmetros biométricos e de hardware atenderam aos padrões de autenticidade.")

        return {
            "shap_feature_contributions": contributions,
            "top_risk_drivers": [f[0] for f in top_risk_factors if f[1] > 0.0],
            "lgpd_art20_audit_text": "\n".join(natural_language_reasons)
        }

    def process_transaction(self, features_dict, transaction_id="TX_SIMULATION_001"):
        """
        Pipeline final do Módulo 4: Score SCIB, Trava Cautelar e Laudo XAI.
        """
        scib_score = self.compute_scib_score(features_dict)
        
        # Limiar de decisão regulatório (Resolução BCB nº 142/2021)
        LOCK_THRESHOLD = 750
        is_locked = scib_score >= LOCK_THRESHOLD
        
        action = "LOCK_ACCOUNTS_TRAVA_CAUTELAR" if is_locked else "APPROVE_TRANSACTION"
        
        xai_report = self.generate_xai_explanation(features_dict, scib_score)

        laudo_pericial_lgpd = (
            f"=======================================================================\n"
            f"LAUDO PERICIAL DE EXPLICABILIDADE ALGORÍTMICA (XAI) — LGPD ART. 20\n"
            f"PROJETO 1: OPERAÇÃO ESPELHO FALSO (v8.0)\n"
            f"=======================================================================\n"
            f"ID Transação: {transaction_id}\n"
            f"Data/Hora UTC: {datetime.now(timezone.utc).isoformat()}\n"
            f"Score SCIB Calculado: {scib_score} / 1000 pontos\n"
            f"Decisão Automatizada: {'🔴 REJEITADO / RETIDO' if is_locked else '🟢 APROVADO'}\n"
            f"Ação no Core Bancário: {action}\n"
            f"Respaldo Legal: Resolução BCB nº 142/2021 & Art. 11, II, 'g' da LGPD\n"
            f"-----------------------------------------------------------------------\n"
            f"JUSTIFICATIVA TÉCNICA MOTIVADORA (AUDITORIA DE DECISÃO IA):\n"
            f"{xai_report['lgpd_art20_audit_text']}\n"
            f"======================================================================="
        )

        return {
            "transaction_id": transaction_id,
            "scib_score": scib_score,
            "lock_threshold": LOCK_THRESHOLD,
            "trava_cautelar_triggered": is_locked,
            "action": action,
            "xai_explanation": xai_report,
            "laudo_pericial_lgpd_text": laudo_pericial_lgpd,
            "compliance_status": "PROCESSED_ZERO_DISK_RAM"
        }


# --- Bloco de Execução e Demonstração ---
def main():
    print("=======================================================================")
    print("PROJETO 1: OPERAÇÃO ESPELHO FALSO (Versão Beta PoC)")
    print("Mód. 4: Score SCIB, Trava Cautelar BCB 142/21 & Explicabilidade LGPD Art. 20")
    print("=======================================================================")

    module = SCIBScoreAndXAIModule()

    # Cenário A: Simulação de Transação Legítima
    legitimate_sample = {
        "f1_corneal_gradient": 18.2,     # Curvatura biológica viva
        "f2_spectral_saturation": 45.0,    # Saturação natural de pele
        "f3_fft_power": 12.0,              # Sem cintilação PWM
        "f4_prnu_correlation": 0.92,       # Sensor físico autêntico
        "f5_device_integrity": 0,          # TEE íntegro
        "f6_graph_mule_risk": 0.05,        # Sem vínculo com mulas
        "f7_behavioral_score": 95.0        # Micro-tremores biológicos
    }

    # Cenário B: Simulação de Ataque por Injeção Virtual + Deepfake em Tela OLED 4K
    fraud_attack_sample = {
        "f1_corneal_gradient": 72.5,     # Reflexo plano de tela
        "f2_spectral_saturation": 145.0,   # Pico de saturação OLED
        "f3_fft_power": 210.0,             # Cintilação PWM de tela detectada
        "f4_prnu_correlation": 0.12,       # Falha no PRNU (Câmera Virtual / OBS)
        "f5_device_integrity": 1,          # Frida / Jailbreak detectado
        "f6_graph_mule_risk": 0.88,        # Pertence a cluster de contas mulas
        "f7_behavioral_score": 25.0        # Telemetria sintética estática
    }

    print("\n--- 1. PROCESSANDO SIMULAÇÃO DE TRANSAÇÃO LEGÍTIMA ---")
    result_legit = module.process_transaction(legitimate_sample, "TX_LEGIT_8891")
    print(result_legit["laudo_pericial_lgpd_text"])

    print("\n--- 2. PROCESSANDO SIMULAÇÃO DE ATAQUE POR DEEPFAKE / INJEÇÃO ---")
    result_fraud = module.process_transaction(fraud_attack_sample, "TX_FRAUD_4402")
    print(result_fraud["laudo_pericial_lgpd_text"])

    print("\n✅ Saída Estruturada do Laudo XAI JSON (Ataque):")
    print(json.dumps(result_fraud, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
