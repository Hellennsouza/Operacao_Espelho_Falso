# Operação Espelho Falso
Este projeto foi elaborado visando encontrar a melhor forma de ajudar instituições investigativas e financeiras a resolverem seus casos de fraudes complexas.
Principais objetivos deste projeto: o combate a *deepfakes*, engenharia social e enxames de contas mulas operadas por crime organizado transnacional.

##### Até o momento publiquei apenas a versão Beta que será a base da versão Alfa.
##### A versão Alfa precisa passar por tranmites legais com a ABIN antes de ser publicada e por isso deve ser lançada para o público apenas em 2027.

Para este projeto usei como fundamentação teórica e estudo de caso os relatórios oficiais do **FBI** (como o caso OneCoin / Ruja Ignatova) e da **Interpol** (como as Operações *Jackal* e *First Light*).

> **Operação Espelho Falso:** PoC em Visão Computacional, Biometria Espectral, PRNU e XAI para Prevenção a Deepfakes.

---

## 🛠️ Instruções para Executar o Programa

1. Baixe os 5 arquivos `.py` para a mesma pasta no seu computador:
   * `module1_ocular_liveness.py`
   * `module2_prnu_silicon.py`
   * `module3_identity_graph.py`
   * `module4_scib_xai.py`
   * `main_pipeline.py`

2. Execute os comandos no terminal:
   ```bash
   pip install opencv-python numpy scipy pandas networkx scikit-learn
   python main_pipeline.py
   ```

---

## 📊 Resultados de Execução de Bancada & Laudo Pericial de XAI

Abaixo está o registro oficial da execução do orquestrador unificado (`main_pipeline.py`) em ambiente de bancada local (`localhost / Windows 11`), demonstrando a precisão da fusão multimodal de métricas (**Módulos 1 a 4**) e a conformidade com as exigências do **Banco Central do Brasil** e da **LGPD**.

---

### 🟢 Cenário 1: Simulação de Transação Legítima (Usuário Biológico)
* **ID da Transação:** `TX_LEGIT_1001`
* **Score SCIB Calculado:** **549 / 1000 pontos** *(Abaixo da linha de corte de retenção)*
* **Decisão Automatizada:** `APPROVE_TRANSACTION`
* **Status:** 🟢 **APROVADO SEM ATRITO**
* **Comportamento:** Parâmetros biométricos, reflexão corneal e assinatura de ruído de silício (PRNU) validados dentro dos intervalos de autenticidade humana.

---

### 🔴 Cenário 2: Simulação de Ataque Combinado (Deepfake 4K + Câmera Virtual + Conta Mula)
* **ID da Transação:** `TX_FRAUD_9909`
* **Score SCIB Calculado:** **764 / 1000 pontos** *(Superou o limiar regulatório de 750 pontos)*
* **Decisão Automatizada:** `LOCK_ACCOUNTS_TRAVA_CAUTELAR`
* **Status:** 🔴 **RETIDO / TRAVA CAUTELAR DISPARADA (Resolução BCB nº 142/2021)**
* **Amparo Legal:** Art. 11, II, 'g' e Art. 20 da LGPD (Lei nº 13.709/2018).

#### 📜 Laudo Pericial de Explicabilidade Algorítmica (XAI - LGPD Art. 20)
```text
=======================================================================
LAUDO PERICIAL DE EXPLICABILIDADE ALGORÍTMICA (XAI) — LGPD ART. 20
PROJETO 1: OPERAÇÃO ESPELHO FALSO (v8.0)
=======================================================================
ID Transação: TX_FRAUD_9909
Data/Hora UTC: 2026-10-07T03:39:42.378973+00:00
Score SCIB Calculado: 764 / 1000 pontos
Decisão Automatizada: 🔴 REJEITADO / RETIDO
Ação no Core Bancário: LOCK_ACCOUNTS_TRAVA_CAUTELAR
Respaldo Legal: Resolução BCB nº 142/2021 & Art. 11, II, 'g' da LGPD
-----------------------------------------------------------------------
JUSTIFICATIVA TÉCNICA MOTIVADORA (AUDITORIA DE DECISÃO IA):
• Ausência de ruído físico de silício do sensor de câmera (Injeção Virtual por Software). (Peso de impacto: +0.194)
• Superfície óptica reflexiva incompatível com curvatura corneal humana (Ataque por Tela Digital). (Peso de impacto: +0.142)
• Frequência de cintilação temporal detectada (Modulação PWM de Tela de Display). (Peso de impacto: +0.110)
=======================================================================
```
```
