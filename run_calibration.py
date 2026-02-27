import os
import sys
import glob
from pathlib import Path

# Adiciona a raiz ao sys.path para importações absolutas do src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.data.models import get_engine

FEATURE_COLS = [
    "ewma5_xg_pro_home",  "ewma10_xg_pro_home",
    "ewma5_xg_con_home",  "ewma10_xg_con_home",
    "ewma5_xg_pro_away",  "ewma10_xg_pro_away",
    "ewma5_xg_con_away",  "ewma10_xg_con_away",
]

def main():
    print("=" * 60)
    print("  AUDITORIA DE CALIBRAÇÃO (Calibration Curve & Brier Score)")
    print("=" * 60)

    # 1. Carregar o modelo mais recente da diretoria artifacts
    artifact_dir = Path("artifacts")
    if not artifact_dir.exists():
        print(f"Erro: A diretoria {artifact_dir} não existe.")
        return

    # Procura ficheiros .pkl omitindo 'scaler' e 'metadata'
    model_files = [
        f for f in artifact_dir.glob("*.pkl") 
        if "scaler" not in f.name and "meta" not in f.name
    ]

    if not model_files:
        print("Erro: Nenhum modelo (.pkl) encontrado em artifacts/")
        return

    # Escolher o mais recente
    latest_model_path = max(model_files, key=os.path.getmtime)
    print(f"Carregando modelo: {latest_model_path}")
    model = joblib.load(latest_model_path)

    # Verifica se existe um scaler associado
    scaler = None
    scaler_path = artifact_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print("Scaler associado carregado com sucesso.")

    # 2. Conectar à base de dados
    db_path = "sqlite:///understat_premier_league.db"
    print(f"Conectando à base de dados: {db_path} ...")
    engine = get_engine(db_path)

    # 3. Extrair dados da última temporada viável (que tenha features) 
    # Usamos os resultados das vitórias através das equipas
    query = """
        SELECT m.home_team, m.away_team, m.home_goals, m.away_goals,
               f.ewma5_xg_pro_home, f.ewma10_xg_pro_home,
               f.ewma5_xg_con_home, f.ewma10_xg_con_home,
               f.ewma5_xg_pro_away, f.ewma10_xg_pro_away,
               f.ewma5_xg_con_away, f.ewma10_xg_con_away
        FROM matches m
        INNER JOIN match_features f ON m.id = f.match_id
        ORDER BY m.date DESC
        LIMIT 380 -- Uma temporada inteira típica
    """
    df = pd.read_sql(query, engine)
    df = df.dropna(subset=FEATURE_COLS).copy()

    if df.empty:
        print("Erro: Não há dados suficientes de feature na base de dados para 1 temporada.")
        return

    print(f"Dados extraídos: {len(df)} partidas.")

    # Criar o target ('H': Home Win, 'D': Draw, 'A': Away Win)
    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
        df["home_goals"] < df["away_goals"],
    ]
    df["target"] = np.select(conditions, ["H", "D", "A"], default="")
    # Remove qualquer anomalia
    df = df[df["target"] != ""]

    X = df[FEATURE_COLS].values
    y_true_all = df["target"].values

    if scaler:
        # Se for um modelo que precise do sklearn Pipeline que não guardou o scaler, aplicar
        # Neste caso o scaler local será aplicado:
        # Se o obj model for Pipeline de Sklearn pode não necessitar do scaler extra 
        # mas estamos a seguir a separação habitual.
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"Aviso: Não foi possível escalar os dados ({e}). O modelo pode ser um Pipiline que já o faça.")
            
    # 4. Probabilidades Previstas
    y_prob = model.predict_proba(X)
    classes = model.classes_  # Tipicamente ['A', 'D', 'H']

    # 5. Cálculos (Brier Score) e Gráficos (Calibration Curve)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Curvas de Calibração (Por Classe) - Fundo Quantitativo de Apostas", fontsize=16)

    brier_scores = {}

    for i, cls in enumerate(classes):
        # Transforma num problema binário 1 (classe atual) vs 0 (restantes)
        y_true_binary = (y_true_all == cls).astype(int)
        y_prob_cls = y_prob[:, i]

        # Brier Score = Mean Squared Error das probabilidades (0 a 1, onde 0 é calibração perfeita)
        bs = brier_score_loss(y_true_binary, y_prob_cls)
        brier_scores[cls] = bs

        # Curva de Calibração Visual
        ax = axes[i]
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob_cls, n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, "s-", color="darkblue", label=f"Classe {cls}")
        ax.plot([0, 1], [0, 1], "k:", color="gray", label="Perfeita")
        ax.legend()
        ax.set_title(f"Calibração: {cls} (Brier: {bs:.4f})")
        ax.set_ylabel("Frequência Observada (True)")
        ax.set_xlabel("Frequência Predita (Mean)")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    
    # 6. Salvar gráfico
    plot_path = "calibration_curve.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Gráfico de calibração guardado em: {plot_path}")

    # 7. Print do Brier Score
    print("\nResultados do Brier Score (Mais próximo de 0 é melhor):")
    print("-" * 40)
    for cls, bs in brier_scores.items():
        print(f"Classe {cls:2s}: {bs:.4f}")
    print("-" * 40)
    print("Auditoria concluída com sucesso!")

if __name__ == "__main__":
    main()
