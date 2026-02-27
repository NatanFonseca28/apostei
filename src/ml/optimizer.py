"""
optimizer.py
------------
Otimização Bayesiana de hiperparâmetros via Optuna.

Fluxo:
  1. Carrega o dataset (JOIN matches + match_features)
  2. Executa Feature Selection automatizada
  3. Para cada trial do Optuna:
     a. Optuna sugere hiperparâmetros do modelo
     b. Avalia via TimeSeriesSplit (sem data leakage)
     c. Registra Log Loss médio (minimização)
  4. Salva o melhor modelo treinado no dataset completo em .pkl

Métrica alvo: Log Loss (minimização) — validação temporal.

Modelos suportados:
  - LogisticRegression (C, solver, penalty)
  - RandomForest (n_estimators, max_depth, min_samples_leaf, ...)
  - GradientBoosting (n_estimators, learning_rate, max_depth, subsample)
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    raise ImportError(
        "Optuna não encontrado. Instale com: pip install optuna"
    )

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from .feature_selection import run_feature_selection
from .trainer import CLASSES, FEATURE_COLS, load_dataset, multiclass_brier_score

logger = logging.getLogger(__name__)

# Diretório para salvar artefatos
ARTIFACTS_DIR = Path("artifacts")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRUTOR DE MODELO POR TRIAL
# ═══════════════════════════════════════════════════════════════════════════════

def _suggest_model(trial: "optuna.Trial") -> Pipeline:
    """
    Optuna sugere o tipo de modelo E os hiperparâmetros correspondentes.
    Retorna um Pipeline sklearn pronto para .fit().
    """
    model_type = trial.suggest_categorical(
        "model_type", ["LogisticRegression", "RandomForest", "GradientBoosting"]
    )

    if model_type == "LogisticRegression":
        C = trial.suggest_float("lr_C", 1e-3, 100.0, log=True)
        solver = trial.suggest_categorical("lr_solver", ["lbfgs", "saga"])

        clf = LogisticRegression(
            C=C,
            solver=solver,
            l1_ratio=0,  # equivalente a penalty='l2' (sem deprecation warning)
            max_iter=2000,
            random_state=42,
        )

    elif model_type == "RandomForest":
        n_estimators = trial.suggest_int("rf_n_estimators", 100, 800, step=50)
        max_depth = trial.suggest_int("rf_max_depth", 3, 15)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 5, 50)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 30)
        max_features = trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2", None]
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    else:  # GradientBoosting
        n_estimators = trial.suggest_int("gb_n_estimators", 100, 600, step=50)
        learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True)
        max_depth = trial.suggest_int("gb_max_depth", 2, 10)
        subsample = trial.suggest_float("gb_subsample", 0.6, 1.0)
        min_samples_leaf = trial.suggest_int("gb_min_samples_leaf", 5, 50)

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO OBJETIVO DO OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════

def _create_objective(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Closure que encapsula os dados e retorna a função objetivo.
    A métrica retornada é o Log Loss médio dos folds temporais (minimização).
    """
    tss = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: "optuna.Trial") -> float:
        pipeline = _suggest_model(trial)

        fold_losses = []
        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_val)

            ll = log_loss(y_val, y_prob, labels=CLASSES)
            fold_losses.append(ll)

            # Pruning: se o fold atual já indica resultado ruim, aborta cedo
            trial.report(ll, fold_idx)
            if trial.should_prune():
                raise TrialPruned()

        mean_ll = float(np.mean(fold_losses))
        # Registra métricas adicionais como user_attrs
        trial.set_user_attr("log_loss_std", float(np.std(fold_losses)))
        trial.set_user_attr("model_type", trial.params.get("model_type", "?"))

        return mean_ll

    return objective


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL DE OTIMIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def run_optimization(
    engine,
    n_trials: int = 100,
    n_splits: int = 5,
    max_features: int = 15,
    timeout: int | None = None,
) -> dict:
    """
    Pipeline completo: Feature Selection → Optuna → Salvar modelo.

    Parâmetros
    ----------
    engine        : SQLAlchemy engine
    n_trials      : número de trials do Optuna (padrão: 100)
    n_splits      : folds do TimeSeriesSplit (padrão: 5)
    max_features  : máximo de features após seleção (padrão: 15)
    timeout       : tempo máximo em segundos (None = sem limite)

    Retorna
    -------
    dict com chaves:
        best_params, best_log_loss, selected_features, model_path, study
    """
    logger.info("=" * 62)
    logger.info("  OTIMIZACAO BAYESIANA -- Optuna + TimeSeriesSplit")
    logger.info("=" * 62)

    # ── 1. Carregar dados ─────────────────────────────────────────────────────
    logger.info("\n[1/7] Carregando dataset...")
    df = load_dataset(engine)

    X_full = df[FEATURE_COLS].values
    y = df["target"].values

    # ── 2. Feature Selection ──────────────────────────────────────────────────
    logger.info("\n[2/7] Executando Feature Selection...")
    selected_features, selection_report = run_feature_selection(
        X_full, y, FEATURE_COLS, max_features=max_features
    )

    # Filtra X para as features selecionadas
    selected_idx = [FEATURE_COLS.index(f) for f in selected_features]
    X = X_full[:, selected_idx]

    logger.info(f"\n  Matriz final: {X.shape[0]} amostras x {X.shape[1]} features")

    # ── 3. Optuna ─────────────────────────────────────────────────────────────
    logger.info(f"\n[3/7] Iniciando Optuna: {n_trials} trials, {n_splits} folds temporais...")

    # Sampler TPE (Tree-structured Parzen Estimator) — default do Optuna
    sampler = optuna.samplers.TPESampler(
        seed=42, multivariate=True, warn_independent_sampling=False
    )
    # Pruner MedianPruner — aborta trials ruins com base na mediana
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)

    study = optuna.create_study(
        study_name="epl_xg_predictor",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = _create_objective(X, y, n_splits=n_splits)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # ── 4. Resultados ─────────────────────────────────────────────────────────
    best = study.best_trial
    logger.info(f"\n{'='*60}")
    logger.info(f"  MELHOR TRIAL: #{best.number}")
    logger.info(f"{'='*60}")
    logger.info(f"  Log Loss:  {best.value:.4f} +/- {best.user_attrs.get('log_loss_std', 0):.4f}")
    logger.info(f"  Modelo:    {best.params.get('model_type', '?')}")
    logger.info(f"  Params:    {best.params}")
    logger.info(f"{'='*60}")

    # ── 5. Treinar modelo final no dataset completo ──────────────────────────
    logger.info("\n[5/7] Treinando modelo final com os melhores hiperparametros...")

    # Recria o trial vencedor como um FrozenTrial "simulado"
    # para passar ao _suggest_model — abordagem mais limpa: recriar direto
    final_pipeline = _rebuild_best_pipeline(best.params)
    final_pipeline.fit(X, y)

    # Validação sanity-check: probabilidades no dataset completo
    y_prob_full = final_pipeline.predict_proba(X)
    final_ll = log_loss(y, y_prob_full, labels=CLASSES)
    final_bs = multiclass_brier_score(y, y_prob_full, CLASSES)
    logger.info(f"  Log Loss (treino completo): {final_ll:.4f}")
    logger.info(f"  Brier Score (treino completo): {final_bs:.4f}")

    # ── 6. Salvar artefatos ──────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 6a. Modelo .pkl
    model_filename = f"best_model_{timestamp}.pkl"
    model_path = ARTIFACTS_DIR / model_filename

    artifact = {
        "pipeline": final_pipeline,
        "selected_features": selected_features,
        "feature_names": FEATURE_COLS,
        "classes": CLASSES,
        "best_params": best.params,
        "best_log_loss_cv": best.value,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "train_samples": len(X),
        "timestamp": timestamp,
        "selection_report": selection_report,
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"\n[6/7] Modelo salvo em: {model_path}")
    logger.info(f"     Tamanho: {model_path.stat().st_size / 1024:.1f} KB")

    # 6b. Relatório do estudo em CSV
    trials_df = study.trials_dataframe()
    csv_path = ARTIFACTS_DIR / f"optuna_trials_{timestamp}.csv"
    trials_df.to_csv(csv_path, index=False)
    logger.info(f"  Trials exportados: {csv_path}")

    # ── 7. Sumário do estudo ──────────────────────────────────────────────────
    _print_study_summary(study)

    return {
        "best_params": best.params,
        "best_log_loss": best.value,
        "selected_features": selected_features,
        "model_path": str(model_path),
        "study": study,
        "artifact": artifact,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

def _rebuild_best_pipeline(params: dict) -> Pipeline:
    """Reconstrói o pipeline a partir dos parâmetros do melhor trial."""
    model_type = params["model_type"]

    if model_type == "LogisticRegression":
        clf = LogisticRegression(
            C=params["lr_C"],
            solver=params["lr_solver"],
            l1_ratio=0,  # equivalente a penalty='l2'
            max_iter=2000,
            random_state=42,
        )

    elif model_type == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            min_samples_leaf=params["rf_min_samples_leaf"],
            min_samples_split=params["rf_min_samples_split"],
            max_features=params["rf_max_features"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    elif model_type == "GradientBoosting":
        clf = GradientBoostingClassifier(
            n_estimators=params["gb_n_estimators"],
            learning_rate=params["gb_learning_rate"],
            max_depth=params["gb_max_depth"],
            subsample=params["gb_subsample"],
            min_samples_leaf=params["gb_min_samples_leaf"],
            random_state=42,
        )
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

    # Envolve o modelo base em calibração de probabilidades (Platt Scaling)
    # Isso resolve o problema de 'overconfidence' e gera EVs mais realistas.
    calibrated_clf = CalibratedClassifierCV(
        estimator=clf,
        method='sigmoid',
        cv=5  # Usa 5-fold interno para calibrar sem viés
    )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", calibrated_clf),
    ])


def _print_study_summary(study: "optuna.Study") -> None:
    """Imprime um sumário detalhado do estudo Optuna."""
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]

    logger.info(f"\n{'='*60}")
    logger.info(f"  SUMARIO DO ESTUDO OPTUNA")
    logger.info(f"{'='*60}")
    logger.info(f"  Trials totais:    {len(trials)}")
    logger.info(f"  Concluidos:       {len(completed)}")
    logger.info(f"  Podados (pruned): {len(pruned)}")
    logger.info(f"  Melhor Log Loss:  {study.best_value:.4f}")

    # Top 5 trials
    logger.info(f"\n  TOP 5 TRIALS:")
    logger.info(f"  {'#':>4} {'Log Loss':>10} {'Modelo':<22}")
    logger.info(f"  {'-'*40}")

    sorted_trials = sorted(completed, key=lambda t: t.value)
    for t in sorted_trials[:5]:
        model = t.params.get("model_type", "?")
        logger.info(f"  {t.number:>4} {t.value:>10.4f} {model:<22}")

    # Distribuição por tipo de modelo
    model_counts = {}
    model_best = {}
    for t in completed:
        mt = t.params.get("model_type", "?")
        model_counts[mt] = model_counts.get(mt, 0) + 1
        if mt not in model_best or t.value < model_best[mt]:
            model_best[mt] = t.value

    logger.info(f"\n  POR TIPO DE MODELO:")
    logger.info(f"  {'Modelo':<22} {'Trials':>8} {'Melhor LL':>12}")
    logger.info(f"  {'-'*44}")
    for mt in sorted(model_counts.keys()):
        logger.info(f"  {mt:<22} {model_counts[mt]:>8} {model_best[mt]:>12.4f}")

    logger.info(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIO — Carregar modelo salvo
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str | Path) -> dict:
    """
    Carrega um artefato .pkl salvo pelo optimizer.

    Retorna dict com chaves:
        pipeline, selected_features, classes, best_params, ...

    Uso:
        artifact = load_model("artifacts/best_model_20260226_123456.pkl")
        pipeline = artifact["pipeline"]
        features = artifact["selected_features"]
        probs = pipeline.predict_proba(X[:, feature_idx])
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")

    with open(path, "rb") as f:
        artifact = pickle.load(f)

    logger.info(f"Modelo carregado: {path.name}")
    logger.info(f"  Features: {artifact['selected_features']}")
    logger.info(f"  Log Loss CV: {artifact['best_log_loss_cv']:.4f}")
    logger.info(f"  Params: {artifact['best_params']}")

    return artifact
