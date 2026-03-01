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
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from src.data.feature_engineering import add_ewma_features
from src.data.models import create_tables

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Gols Básicos
    "media_marcados_casa",
    "media_sofridos_casa",
    "media_marcados_fora",
    "media_sofridos_fora",
    # xG (Expected Goals)
    "media_xg_casa",
    "media_xga_casa",
    "media10_xg_casa",
    "media10_xga_casa",
    "media_xg_fora",
    "media_xga_fora",
    "media10_xg_fora",
    "media10_xga_fora",
    # Chutes no Alvo
    "media_chutes_alvo_casa",
    "media_chutes_alvo_sofrido_casa",
    "media_chutes_alvo_fora",
    "media_chutes_alvo_sofrido_fora",
    # Posse de Bola
    "media_posse_casa",
    "media_posse_sofrida_casa",
    "media_posse_fora",
    "media_posse_sofrida_fora",
]

# Ordem canônica das classes — mantida consistente em todo o pipeline
CLASSES = ["A", "D", "H"]  # Away win | Draw | Home win


# ── Carregamento dos dados ────────────────────────────────────────────────────


def load_dataset(engine) -> pd.DataFrame:
    """
    Carrega partidas finalizadas de `flashscore_matches`, aplica
    `add_ewma_features` para gerar as features avançadas (xG, chutes,
    posse) via EWMA anti-data-leakage e retorna o DataFrame ordenado
    cronologicamente (obrigatório para TimeSeriesSplit).

    Colunas avançadas ausentes no banco (home_xG, home_shots_target,
    home_possession e equivalentes) são inicializadas com pd.NA e
    preenchidas com mediana=0 pelo próprio add_ewma_features — resultado
    esperado até que o ETL seja expandido para fontes com esses dados.
    """    # Garante que match_advanced_stats existe no banco (no-op se já existir)
    create_tables(engine)
    query = """
        SELECT
            fm.id,
            fm.data              AS date,
            fm.time_casa         AS home_team,
            fm.time_fora         AS away_team,
            fm.placar_casa       AS home_goals,
            fm.placar_fora       AS away_goals,
            COALESCE(fm.media_marcados_casa, 0) AS media_marcados_casa,
            COALESCE(fm.media_sofridos_casa, 0) AS media_sofridos_casa,
            COALESCE(fm.media_marcados_fora, 0) AS media_marcados_fora,
            COALESCE(fm.media_sofridos_fora, 0) AS media_sofridos_fora,
            COALESCE(mas.home_xg,           0.0) AS home_xG,
            COALESCE(mas.away_xg,           0.0) AS away_xG,
            COALESCE(mas.home_shots_target, 0.0) AS home_shots_target,
            COALESCE(mas.away_shots_target, 0.0) AS away_shots_target,
            COALESCE(mas.home_possession,   0.0) AS home_possession,
            COALESCE(mas.away_possession,   0.0) AS away_possession
        FROM flashscore_matches fm
        LEFT JOIN match_advanced_stats mas
            ON  LOWER(fm.time_casa) = LOWER(mas.home_team)
            AND LOWER(fm.time_fora) = LOWER(mas.away_team)
            AND DATE(fm.data)       = DATE(mas.date)
        WHERE fm.placar_casa IS NOT NULL AND fm.placar_fora IS NOT NULL
        ORDER BY fm.data ASC
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])

    # Calcula features EWMA avançadas (xG, chutes no alvo, posse).
    # Colunas-fonte ausentes no banco são inseridas como pd.NA e preenchidas
    # com mediana=0 internamente — sem NaN residuais no DataFrame resultante.
    df = add_ewma_features(df)

    # Cria o target: "H" / "D" / "A"
    conditions = [
        df["home_goals"] > df["away_goals"],  # Vitória mandante
        df["home_goals"] == df["away_goals"],  # Empate
        df["home_goals"] < df["away_goals"],  # Vitória visitante
    ]
    df["target"] = np.select(conditions, ["H", "D", "A"], default="")

    # Remove as poucas linhas onde o target ficou indefinido (não deve acontecer)
    df = df[(df["target"] != "") & df[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)

    logger.info(f"Dataset carregado: {len(df)} jogos | H={(df['target'] == 'H').sum()} | D={(df['target'] == 'D').sum()} | A={(df['target'] == 'A').sum()}")
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
    y_bin = lb.transform(y_true)  # shape (N, 3)
    return float(np.mean(np.sum((y_prob - y_bin) ** 2, axis=1)))


# ── Definição dos modelos ─────────────────────────────────────────────────────

from typing import Dict

from sklearn.calibration import CalibratedClassifierCV


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
    calibrated_lr = CalibratedClassifierCV(estimator=lr_base, method="sigmoid", cv=3)
    calibrated_rf = CalibratedClassifierCV(estimator=rf_base, method="sigmoid", cv=3)

    return {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", calibrated_lr),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", calibrated_rf),
            ]
        ),
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
        logger.info(f"\n{'═' * 70}")
        logger.info(f"  Modelo: {model_name}")
        logger.info(f"{'═' * 70}")

        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(X), start=1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Treina APENAS com dados do passado (train_idx < val_idx garantido pelo TSS)
            pipeline.fit(X_train, y_train)

            # Predições probabilísticas — necessárias para Log Loss e Brier Score
            y_prob = pipeline.predict_proba(X_val)  # shape (n_val, 3)
            y_pred = pipeline.predict(X_val)

            # ── Métricas ──────────────────────────────────────────────────────
            ll = log_loss(y_val, y_prob, labels=CLASSES)
            bs = multiclass_brier_score(y_val, y_prob, CLASSES)
            acc = accuracy_score(y_val, y_pred)

            fold_metrics.append({"fold": fold_idx, "log_loss": ll, "brier": bs, "accuracy": acc})

            logger.info(f"  Fold {fold_idx:2d} | Treino: {len(train_idx):4d} jogos → Validação: {len(val_idx):4d} jogos | LogLoss={ll:.4f}  Brier={bs:.4f}  Acc={acc:.3f}")

        # ── Sumário estatístico ───────────────────────────────────────────────
        metrics_df = pd.DataFrame(fold_metrics)
        summary = {col: {"mean": metrics_df[col].mean(), "std": metrics_df[col].std()} for col in ("log_loss", "brier", "accuracy")}

        logger.info(f"\n  {'─' * 60}")
        logger.info(f"  SUMÁRIO — {model_name}")
        logger.info(f"  {'─' * 60}")
        logger.info(f"  Log Loss  : {summary['log_loss']['mean']:.4f} ± {summary['log_loss']['std']:.4f}")
        logger.info(f"  Brier     : {summary['brier']['mean']:.4f} ± {summary['brier']['std']:.4f}")
        logger.info(f"  Accuracy  : {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
        logger.info(f"  {'─' * 60}")
        logger.info("  Referência ingênua (1/3 por classe):")
        logger.info("    Log Loss ≈ 1.0986  |  Brier ≈ 0.6667  |  Acc ≈ 0.333")

        all_results[model_name] = {"folds": fold_metrics, "summary": summary}

    # ── Comparativo final ─────────────────────────────────────────────────────
    logger.info(f"\n{'═' * 70}")
    logger.info("  COMPARATIVO FINAL (média sobre todos os folds)")
    logger.info(f"{'═' * 70}")
    logger.info(f"  {'Modelo':<22} {'Log Loss':>10} {'Brier':>10} {'Accuracy':>10}")
    logger.info(f"  {'-' * 54}")
    for name, res in all_results.items():
        s = res["summary"]
        logger.info(f"  {name:<22} {s['log_loss']['mean']:>10.4f} {s['brier']['mean']:>10.4f} {s['accuracy']['mean']:>10.3f}")
    logger.info(f"  {'Ingênuo (baseline)':<22} {'1.0986':>10} {'0.6667':>10} {'0.333':>10}")
    logger.info(f"{'═' * 70}\n")

    # ── Treinamento final no dataset completo + persistência do modelo ────────
    best_model_name = min(
        all_results,
        key=lambda n: all_results[n]["summary"]["log_loss"]["mean"],
    )
    logger.info(f"Melhor modelo (menor Log Loss médio): {best_model_name}")
    logger.info("Treinando modelo final no dataset completo...")

    final_pipeline = _build_models()[best_model_name]
    final_pipeline.fit(X, y)

    Path("artifacts").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path = Path("artifacts") / f"best_model_{timestamp}.pkl"
    joblib.dump(
        {"model": final_pipeline, "feature_cols": FEATURE_COLS, "classes": CLASSES},
        pkl_path,
    )
    logger.info(f"Modelo salvo em: {pkl_path}")

    all_results["model_path"] = str(pkl_path)
    return all_results
