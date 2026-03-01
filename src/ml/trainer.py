"""
trainer.py
----------
Pipeline de Machine Learning para predição de resultados de futebol.

Target (variável alvo):
    "H"  →  Vitória do Mandante
    "D"  →  Empate
    "A"  →  Vitória do Visitante

Features de entrada (8 colunas — saída de feature_engineering.py):
    ewma{5,10}_xg_pro_{home,away}  — Média de ataque (xG produzido)
    ewma{5,10}_xg_con_{home,away}  — Média de defesa (xG concedido)

Restrição crítica — TimeSeriesSplit:
    O modelo é SEMPRE treinado em dados do passado e validado em dados
    do futuro imediato. Nenhum dado futuro vaza para o treino.

Modelos:
    1. Logistic Regression Multinomial  (baseline linear)
    2. Random Forest Classifier          (baseline de árvores)

Métricas de avaliação (por fold + sumário final):
    - Log Loss          (penaliza confiança errada — principal métrica)
    - Brier Score       (erro quadrático nas probabilidades)
    - Accuracy          (referência comparativa)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "media_marcados_casa",
    "media_sofridos_casa",
    "media_marcados_fora",
    "media_sofridos_fora",
]

# Ordem canônica das classes — mantida consistente em todo o pipeline
CLASSES = ["A", "D", "H"]   # Away win | Draw | Home win


# ── Carregamento dos dados ────────────────────────────────────────────────────

def load_dataset(engine) -> pd.DataFrame:
    """
    Faz JOIN entre `matches` e `match_features`, cria a coluna `target`
    e retorna o DataFrame ordenado cronologicamente (obrigatório para TSS).

    Retorna apenas linhas com features completas (sem NaN), descartando
    os primeiros jogos de cada time onde o histórico ainda não existe.
    """
    query = """
        SELECT
            id,
            data as date,
            placar_casa as home_goals,
            placar_fora as away_goals,
            media_marcados_casa,
            media_sofridos_casa,
            media_marcados_fora,
            media_sofridos_fora
        FROM flashscore_matches
        WHERE placar_casa IS NOT NULL AND placar_fora IS NOT NULL
        ORDER BY date ASC
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])

    # Cria o target: "H" / "D" / "A"
    conditions = [
        df["home_goals"] > df["away_goals"],   # Vitória mandante
        df["home_goals"] == df["away_goals"],  # Empate
        df["home_goals"] < df["away_goals"],   # Vitória visitante
    ]
    df["target"] = np.select(conditions, ["H", "D", "A"], default="")

    # Remove as poucas linhas onde o target ficou indefinido (não deve acontecer)
    df = df[(df["target"] != "") & df[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)

    logger.info(
        f"Dataset carregado: {len(df)} jogos | "
        f"H={( df['target']=='H').sum()} | "
        f"D={(df['target']=='D').sum()} | "
        f"A={(df['target']=='A').sum()}"
    )
    return df


# ── Métrica auxiliar ──────────────────────────────────────────────────────────

def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, classes: list) -> float:
    """
    Brier Score para classificação multiclasse.

    Fórmula: BS = (1/N) * Σ_i Σ_k (p_ik - o_ik)²
      p_ik = probabilidade prevista para a classe k do jogo i
      o_ik = 1 se o resultado real é k, 0 caso contrário

    Intervalo: [0, 2]  —  quanto menor, melhor.
    Um modelo ingênuo (1/3 para cada classe) produz BS ≈ 0.667.
    """
    lb = LabelBinarizer()
    lb.fit(classes)
    y_bin = lb.transform(y_true)           # shape (N, 3)
    return float(np.mean(np.sum((y_prob - y_bin) ** 2, axis=1)))


# ── Definição dos modelos ─────────────────────────────────────────────────────

from sklearn.calibration import CalibratedClassifierCV
from typing import Dict

def _build_models() -> Dict[str, Pipeline]:
    """
    Retorna um dicionário com os pipelines de cada modelo.
    Todos os modelos base são envolvidos em CalibratedClassifierCV (Platt Scaling)
    para garantir que o Log Loss reportado reflita as probabilidades reais de mercado.
    """
    # Instancia os classificadores base
    lr_base = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )

    rf_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Envolve com o calibrador (cv=3 é usado para evitar erros em folds pequenos)
    calibrated_lr = CalibratedClassifierCV(estimator=lr_base, method='sigmoid', cv=3)
    calibrated_rf = CalibratedClassifierCV(estimator=rf_base, method='sigmoid', cv=3)

    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", calibrated_lr),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", calibrated_rf),
        ]),
    }


# ── Pipeline de treinamento principal ────────────────────────────────────────

def run_training_pipeline(engine, n_splits: int = 5) -> dict:
    """
    Executa a validação cruzada temporal (TimeSeriesSplit) para cada modelo.

    TimeSeriesSplit com n_splits=5 cria 5 folds onde:
      Fold 1: treino=jogos[0:N/6]       validação=jogos[N/6 : 2N/6]
      Fold 2: treino=jogos[0:2N/6]      validação=jogos[2N/6 : 3N/6]
      ...  (treino sempre cresce, validação sempre é o futuro imediato)

    NUNCA há sobreposição entre treino e validação — data leakage impossível.

    Parâmetros
    ----------
    engine : SQLAlchemy engine   conectado ao banco de dados
    n_splits : int               número de folds temporais (padrão: 5)

    Retorna
    -------
    dict com resultados por modelo: métricas por fold + sumário mean ± std
    """
    df = load_dataset(engine)

    X = df[FEATURE_COLS].values
    y = df["target"].values

    tss = TimeSeriesSplit(n_splits=n_splits)
    models = _build_models()
    all_results = {}

    logger.info(f"Iniciando TimeSeriesSplit com {n_splits} folds...")
    logger.info(f"Total de amostras: {len(X)} | Features: {len(FEATURE_COLS)}")
    logger.info("-" * 70)

    for model_name, pipeline in models.items():
        logger.info(f"\n{'═'*70}")
        logger.info(f"  Modelo: {model_name}")
        logger.info(f"{'═'*70}")

        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(X), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Treina APENAS com dados do passado (train_idx < val_idx garantido pelo TSS)
            pipeline.fit(X_train, y_train)

            # Predições probabilísticas — necessárias para Log Loss e Brier Score
            y_prob = pipeline.predict_proba(X_val)        # shape (n_val, 3)
            y_pred = pipeline.predict(X_val)

            # ── Métricas ──────────────────────────────────────────────────────
            ll   = log_loss(y_val, y_prob, labels=CLASSES)
            bs   = multiclass_brier_score(y_val, y_prob, CLASSES)
            acc  = accuracy_score(y_val, y_pred)

            fold_metrics.append({"fold": fold_idx, "log_loss": ll, "brier": bs, "accuracy": acc})

            logger.info(
                f"  Fold {fold_idx:2d} | "
                f"Treino: {len(train_idx):4d} jogos → Validação: {len(val_idx):4d} jogos | "
                f"LogLoss={ll:.4f}  Brier={bs:.4f}  Acc={acc:.3f}"
            )

        # ── Sumário estatístico ───────────────────────────────────────────────
        metrics_df = pd.DataFrame(fold_metrics)
        summary = {
            col: {"mean": metrics_df[col].mean(), "std": metrics_df[col].std()}
            for col in ("log_loss", "brier", "accuracy")
        }

        logger.info(f"\n  {'─'*60}")
        logger.info(f"  SUMÁRIO — {model_name}")
        logger.info(f"  {'─'*60}")
        logger.info(f"  Log Loss  : {summary['log_loss']['mean']:.4f} ± {summary['log_loss']['std']:.4f}")
        logger.info(f"  Brier     : {summary['brier']['mean']:.4f} ± {summary['brier']['std']:.4f}")
        logger.info(f"  Accuracy  : {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
        logger.info(f"  {'─'*60}")
        logger.info(f"  Referência ingênua (1/3 por classe):")
        logger.info(f"    Log Loss ≈ 1.0986  |  Brier ≈ 0.6667  |  Acc ≈ 0.333")

        all_results[model_name] = {"folds": fold_metrics, "summary": summary}

    # ── Comparativo final ─────────────────────────────────────────────────────
    logger.info(f"\n{'═'*70}")
    logger.info("  COMPARATIVO FINAL (média sobre todos os folds)")
    logger.info(f"{'═'*70}")
    logger.info(f"  {'Modelo':<22} {'Log Loss':>10} {'Brier':>10} {'Accuracy':>10}")
    logger.info(f"  {'-'*54}")
    for name, res in all_results.items():
        s = res["summary"]
        logger.info(
            f"  {name:<22} "
            f"{s['log_loss']['mean']:>10.4f} "
            f"{s['brier']['mean']:>10.4f} "
            f"{s['accuracy']['mean']:>10.3f}"
        )
    logger.info(f"  {'Ingênuo (baseline)':<22} {'1.0986':>10} {'0.6667':>10} {'0.333':>10}")
    logger.info(f"{'═'*70}\n")

    return all_results
