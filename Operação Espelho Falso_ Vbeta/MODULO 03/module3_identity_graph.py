"""
===============================================================================
PROJETO 1: OPERAÇÃO ESPELHO FALSO — VERSÃO BETA (PoC Autônoma)
MÓDULO 3: IDENTITY GRAPH ANALYTICS & DETECÇÃO DE CONTAS MULAS
===============================================================================
Autora: Dra. Hellen Nunes dos Santos Souza / P&D Antifraude Biométrica
Amparo Legal: LGPD (Art. 11, II, 'g' - Prevenção à Fraude) & BACEN Res. 142/2021
Política de Privacidade: Dados Sintéticos Anômicos (Zero-Disk PII)
===============================================================================
"""

import json
import random
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Suporte a NetworkX para análise de grafos e detecção de comunidades
HAS_NETWORKX = False
try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class IdentityGraphAnalytics:
    """
    Módulo 3: Análise de Grafos de Relacionamento e Identidades Sintéticas.
    Analisa:
      1. Modelagem de Rede Multimodal (Contas, IPs, Dispositivos, Biometrias).
      2. Algoritmo de Comunidades de Louvain (Enxames de Contas Mulas).
      3. PageRank de Risco (Identificação de Mulas Mestres/Orquestradores).
      4. Disparo de Sinalização de Risco para Trava Cautelar (SCIB).
    """

    def __init__(self):
        self.has_networkx = HAS_NETWORKX
        self.graph = nx.Graph() if HAS_NETWORKX else None
        self.df_transactions = None

    def generate_synthetic_dataset(self, num_records=300, mule_clusters=3):
        """
        Gera uma base de dados sintética simulando transações financeiras
        com células enxameadas de contas mulas e fraudes de biometria.
        """
        random.seed(42)
        np.random.seed(42)

        records = []
        start_time = datetime.now() - timedelta(days=7)

        # Dispositivos e IPs compartilhados por células mulas
        mule_devices = [f"DEV_MULE_CLUSTER_{i+1:02d}" for i in range(mule_clusters)]
        mule_ips = [f"185.220.101.{i+10}" for i in range(mule_clusters)]
        mule_biometrics = [f"HASH_BIO_SPOOF_{i+1:03d}" for i in range(mule_clusters)]

        for i in range(num_records):
            is_mule_attack = random.random() < 0.25  # 25% de tráfego sintético suspeito

            if is_mule_attack:
                cluster_idx = random.randint(0, mule_clusters - 1)
                source_acc = f"ACC_MULE_{random.randint(100, 150):04d}"
                target_acc = f"ACC_DEST_LARANJA_{random.randint(200, 220):04d}"
                device_id = mule_devices[cluster_idx]
                ip_addr = mule_ips[cluster_idx]
                bio_hash = mule_biometrics[cluster_idx]
                scib_score = random.randint(750, 980)  # SCIB Elevado (Risco Alto)
                amount = round(random.uniform(2000.0, 15000.0), 2)
            else:
                source_acc = f"ACC_LEGIT_{random.randint(1000, 5000):04d}"
                target_acc = f"ACC_LEGIT_{random.randint(6000, 9000):04d}"
                device_id = f"DEV_USER_{random.randint(10000, 99999)}"
                ip_addr = f"177.18.{random.randint(1, 250)}.{random.randint(1, 250)}"
                bio_hash = f"HASH_BIO_LEGIT_{random.randint(1000, 9999)}"
                scib_score = random.randint(10, 250)  # SCIB Baixo (Baixo Risco)
                amount = round(random.uniform(50.0, 1200.0), 2)

            tx_time = start_time + timedelta(minutes=random.randint(0, 10080))

            records.append({
                "transaction_id": f"TX_{i+1:05d}",
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_account": source_acc,
                "target_account": target_acc,
                "device_id": device_id,
                "ip_address": ip_addr,
                "biometric_hash": bio_hash,
                "amount": amount,
                "scib_score": scib_score
            })

        self.df_transactions = pd.DataFrame(records)
        return self.df_transactions

    def build_identity_graph(self):
        """
        Mapeia a rede multimodal conectando Contas, Dispositivos, IPs e Biometrias.
        """
        if not self.has_networkx:
            return None

        self.graph = nx.Graph()

        for _, row in self.df_transactions.iterrows():
            src = row["source_account"]
            dst = row["target_account"]
            dev = row["device_id"]
            ip = row["ip_address"]
            scib = row["scib_score"]

            # Adiciona nós com tipos etiquetados
            self.graph.add_node(src, node_type="account", risk_score=scib)
            self.graph.add_node(dst, node_type="account", risk_score=scib)
            self.graph.add_node(dev, node_type="device", risk_score=scib)
            self.graph.add_node(ip, node_type="ip", risk_score=scib)

            # Conecta transação e compartilhamento de infraestrutura
            self.graph.add_edge(src, dst, weight=row["amount"], relation="TRANSFER")
            self.graph.add_edge(src, dev, weight=1.0, relation="USED_DEVICE")
            self.graph.add_edge(src, ip, weight=1.0, relation="USED_IP")

        return self.graph

    def detect_mule_communities(self):
        """
        Aplica o Algoritmo de Louvain para detectar enxames de contas mulas.
        """
        if not self.has_networkx or self.graph is None or len(self.graph) == 0:
            return []

        # Detecção de comunidades via Algoritmo de Louvain
        communities = louvain_communities(self.graph, seed=42)
        
        mule_clusters = []
        for i, comm in enumerate(communities):
            subgraph = self.graph.subgraph(comm)
            avg_risk = np.mean([subgraph.nodes[n].get("risk_score", 0) for n in comm])
            
            # Filtra comunidades suspeitas (com mais de 3 nós altamente interconectados e SCIB elevado)
            if len(comm) >= 3 and avg_risk >= 500:
                accounts = [n for n in comm if subgraph.nodes[n].get("node_type") == "account"]
                devices = [n for n in comm if subgraph.nodes[n].get("node_type") == "device"]
                ips = [n for n in comm if subgraph.nodes[n].get("node_type") == "ip"]

                mule_clusters.append({
                    "cluster_id": f"CLUSTER_MULE_{i+1:02d}",
                    "total_nodes": len(comm),
                    "num_accounts": len(accounts),
                    "shared_devices": devices[:3],
                    "shared_ips": ips[:3],
                    "avg_scib_score": round(float(avg_risk), 2),
                    "action_recommendation": "LOCK_ACCOUNTS_TRAVA_CAUTELAR"
                })

        return mule_clusters

    def calculate_risk_pagerank(self):
        """
        Calcula o PageRank de Risco para identificar nós orquestradores centrais.
        """
        if not self.has_networkx or self.graph is None or len(self.graph) == 0:
            return []

        pagerank_scores = nx.pagerank(self.graph, weight="weight")
        
        # Ordena os nós com maior centralidade de influência na rede
        sorted_ranks = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_risk_nodes = []
        for node, rank in sorted_ranks[:10]:
            node_data = self.graph.nodes[node]
            top_risk_nodes.append({
                "node_id": node,
                "node_type": node_data.get("node_type", "unknown"),
                "pagerank_score": round(float(rank), 6),
                "risk_score": node_data.get("risk_score", 0)
            })

        return top_risk_nodes

    def run_pipeline(self):
        """
        Executa o pipeline completo do Módulo 3.
        """
        print("1. Gerando Dataset Sintético de Transações (Zero-Disk PII)...")
        self.generate_synthetic_dataset()

        print("2. Construindo Identity Graph Multimodal...")
        self.build_identity_graph()

        print("3. Executando Algoritmo de Louvain para Enxames de Mulas...")
        clusters = self.detect_mule_communities()

        print("4. Calculando PageRank de Risco para Identificação de Orquestradores...")
        top_nodes = self.calculate_risk_pagerank()

        results = {
            "module": "Modulo 3 - Identity Graph Analytics",
            "total_transactions_analyzed": len(self.df_transactions),
            "mule_clusters_detected": len(clusters),
            "clusters_detail": clusters,
            "top_risk_orchestrators": top_nodes[:5],
            "compliance_status": "PROCESSED_SYNTHETIC_SAFE"
        }

        return results


# --- Bloco Principal de Teste e Demonstração ---
def main():
    print("=======================================================================")
    print("PROJETO 1: OPERAÇÃO ESPELHO FALSO (Versão Beta PoC)")
    print("Mód. 3: Identity Graph Analytics & Algoritmo de Louvain em Grafos")
    print("=======================================================================")

    analytics = IdentityGraphAnalytics()
    results = analytics.run_pipeline()

    print("\n✅ Processamento do Módulo 3 Concluído com Sucesso!\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
