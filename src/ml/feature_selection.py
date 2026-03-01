"""
feature_selection.py
--------------------
Seleção automatizada de features para o pipeline de predição.

Estratégia dupla (ensemble de critérios):
  1. SelectFromModel com RandomForest  — captura importância não-linear
  2. Análise de correlação com o target — remove features redundantes

O pipeline garante que no máximo `max_features` variáveis sobrevivam,
reduzindo o risco de overfitting quando dezenas de colunas estão disponíveis.

Nota de design:
  Com 20 features disponíveis (gols, xG, chutes no alvo, posse), este
  módulo é responsável por conter a dimensionalidade e reduzir o risco
  de overfitting ao selecionar as features mais informativas.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


# ── Seleção via importância do Random Forest ──────────────────────────────────


def select_by_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_features: int = 20,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray]:
    """
    Usa um RandomForest treinado no dataset completo para ranquear features
    por importância (Gini impurity) e seleciona as top `max_features`.

    Parâmetros
    ----------
    X : np.ndarray           Matriz de features (N, F)
    y : np.ndarray           Vetor target (N,) com classes "H", "D", "A"
    feature_names : list     Nomes das colunas de features
    max_features : int       Máximo de features a manter (padrão: 15)
    random_state : int       Seed para reprodutibilidade

    Retorna
    -------
    (selected_names, importances)
        selected_names : lista dos nomes das features selecionadas
        importances    : importâncias do RF para TODAS as features (ordenação original)
    """
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importances = rf.feature_importances_

    # Ranqueia por importância decrescente
    sorted_idx = np.argsort(importances)[::-1]
    top_idx = sorted_idx[:max_features]
    top_idx_sorted = np.sort(top_idx)  # mantém ordem original das colunas

    selected = [feature_names[i] for i in top_idx_sorted]

    logger.info(f"SelectFromModel (RF): {len(feature_names)} → {len(selected)} features")
    for i, idx in enumerate(sorted_idx[:max_features]):
        logger.info(f"  #{i + 1:2d} {feature_names[idx]:<30s} importância={importances[idx]:.4f}")

    return selected, importances


# ── Remoção por correlação redundante ─────────────────────────────────────────


def remove_highly_correlated(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """
    Remove features com correlação de Pearson acima do `threshold`.
    Quando um par correlacionado é encontrado, a feature com menor
    correlação média com o restante das features é mantida.

    Parâmetros
    ----------
    X : np.ndarray           Matriz de features (N, F)
    feature_names : list     Nomes das colunas
    threshold : float        Limiar de correlação (padrão: 0.95)

    Retorna
    -------
    list[str]  — nomes das features sobreviventes
    """
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()

    # Triângulo superior (sem diagonal)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        highly_correlated = upper.index[upper[col] > threshold].tolist()
        if highly_correlated:
            # Entre o par, remove a feature com maior correlação média geral
            for corr_feature in highly_correlated:
                mean_corr_col = corr_matrix[col].mean()
                mean_corr_feat = corr_matrix[corr_feature].mean()
                drop = corr_feature if mean_corr_feat > mean_corr_col else col
                to_drop.add(drop)

    surviving = [f for f in feature_names if f not in to_drop]

    if to_drop:
        logger.info(f"Correlação (>{threshold}): removidas {len(to_drop)} features → {to_drop}")
    else:
        logger.info(f"Correlação (>{threshold}): nenhuma feature redundante encontrada")

    return surviving


# ── Pipeline completo de seleção ──────────────────────────────────────────────


def run_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_features: int = 20,
    corr_threshold: float = 0.95,
) -> tuple[list[str], dict]:
    """
    Pipeline completo de seleção de features:
      1. Remove features altamente correlacionadas (>threshold)
      2. Seleciona as top `max_features` por importância do RF

    Retorna
    -------
    (selected_features, report)
        selected_features : lista final de features
        report : dicionário com detalhes do processo
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"  FEATURE SELECTION -- {len(feature_names)} features de entrada")
    logger.info(f"{'=' * 60}")

    report = {"input_features": len(feature_names)}

    # Etapa 1: Remove correlações altas
    surviving_corr = remove_highly_correlated(X, feature_names, corr_threshold)
    report["after_correlation_filter"] = len(surviving_corr)

    # Filtra X para manter apenas as colunas sobreviventes
    surviving_idx = [feature_names.index(f) for f in surviving_corr]
    X_filtered = X[:, surviving_idx]

    # Etapa 2: Seleção por importância do RF
    selected, importances_all = select_by_model(X_filtered, y, surviving_corr, max_features=max_features)
    report["after_model_selection"] = len(selected)
    report["selected_features"] = selected

    # Mapa de importâncias para as features finais
    report["importances"] = {name: float(importances_all[surviving_corr.index(name)]) for name in selected}

    logger.info(f"\n  Resultado final: {len(feature_names)} -> {len(selected)} features")
    logger.info(f"  Features selecionadas: {selected}")
    logger.info(f"{'=' * 60}\n")

    return selected, report
