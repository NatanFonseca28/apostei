"""
run_dashboard.py
----------------
Pipeline completo + Dashboard visual profissional.

Executa, em ordem:
  1. ETL  — extrai dados do Understat e persiste no banco
  2. Features — valida as EWMA calculadas
  3. ML       — treina modelos com TimeSeriesSplit e coleta métricas
  4. EV       — demonstra a calculadora de valor esperado
  5. Dashboard — exibe todos os resultados em um painel gráfico único
"""

import logging
import sys, os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from src.core.ev_calculator import calculate_ev, scan_matches
from src.ml.pregame_scanner import PregameScanner
from src.data.feature_engineering import add_ewma_features
from src.data.models import create_tables, get_engine, get_session
from src.data.persistence import DataPersister, FeaturePersister
from src.data.extractor import DataExtractor

# ── Logging configurado para console limpo ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Dashboard")

# ── Paleta de cores ───────────────────────────────────────────────────────────
CLR = {
    "bg":       "#0f1117",
    "panel":    "#1a1d27",
    "border":   "#2e3147",
    "green":    "#00c853",
    "red":      "#ff1744",
    "yellow":   "#ffd600",
    "blue":     "#2979ff",
    "purple":   "#d500f9",
    "orange":   "#ff6d00",
    "text":     "#e8eaf6",
    "subtext":  "#7c83a0",
}
FEATURE_COLS = [
    "ewma5_xg_pro_home",  "ewma10_xg_pro_home",
    "ewma5_xg_con_home",  "ewma10_xg_con_home",
    "ewma5_xg_pro_away",  "ewma10_xg_pro_away",
    "ewma5_xg_con_away",  "ewma10_xg_con_away",
]
CLASSES = ["A", "D", "H"]


# ═══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — ETL
# ═══════════════════════════════════════════════════════════════════════════════

def run_etl(engine, Session) -> int:
    logger.info("━" * 60)
    logger.info("ETAPA 1 — ETL: Extração Híbrida (soccerdata + Understat)")
    logger.info("━" * 60)

    extractor = DataExtractor()
    persister = DataPersister(Session)
    feature_persister = FeaturePersister(Session)

    df_rich = extractor.fetch_rich_dataset(2020, 2025)
    saved = persister.save_matches(df_rich)

    df = persister.load_as_dataframe(engine)
    df_feat = add_ewma_features(df)
    feature_persister.save_features(df_feat)

    logger.info(f"ETL concluído: {saved} jogos | {len(df_feat.dropna(subset=FEATURE_COLS))} com features")
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — Carrega dataset para ML
# ═══════════════════════════════════════════════════════════════════════════════

def load_ml_dataset(engine) -> pd.DataFrame:
    logger.info("━" * 60)
    logger.info("ETAPA 2 — Carregando dataset para ML")
    logger.info("━" * 60)

    query = """
        SELECT m.id, m.date, m.home_team, m.away_team,
               m.home_goals, m.away_goals,
               m.home_xG, m.away_xG,
               f.ewma5_xg_pro_home,  f.ewma10_xg_pro_home,
               f.ewma5_xg_con_home,  f.ewma10_xg_con_home,
               f.ewma5_xg_pro_away,  f.ewma10_xg_pro_away,
               f.ewma5_xg_con_away,  f.ewma10_xg_con_away
        FROM matches m
        INNER JOIN match_features f ON m.id = f.match_id
        ORDER BY m.date ASC
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])
    conditions = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
        df["home_goals"] < df["away_goals"],
    ]
    df["target"] = np.select(conditions, ["H", "D", "A"], default="")
    df = df[(df["target"] != "") & df[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)

    logger.info(f"Dataset: {len(df)} jogos | H={( df.target=='H').sum()} D={(df.target=='D').sum()} A={(df.target=='A').sum()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — Treinamento com TimeSeriesSplit
# ═══════════════════════════════════════════════════════════════════════════════

def multiclass_brier(y_true, y_prob):
    lb = LabelBinarizer()
    lb.fit(CLASSES)
    y_bin = lb.transform(y_true)
    return float(np.mean(np.sum((y_prob - y_bin) ** 2, axis=1)))


def run_training(df: pd.DataFrame, n_splits: int = 5):
    logger.info("━" * 60)
    logger.info("ETAPA 3 — Treinamento ML com TimeSeriesSplit")
    logger.info("━" * 60)

    X = df[FEATURE_COLS].values
    y = df["target"].values

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                            class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
    }

    tss = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    # Dados para calibração (último fold)
    calibration_data = {}

    for name, pipe in models.items():
        folds = []
        last_probs, last_true = None, None

        for fold_i, (train_idx, val_idx) in enumerate(tss.split(X), 1):
            pipe.fit(X[train_idx], y[train_idx])
            y_prob = pipe.predict_proba(X[val_idx])
            y_pred = pipe.predict(X[val_idx])
            y_val  = y[val_idx]

            ll  = log_loss(y_val, y_prob, labels=CLASSES)
            bs  = multiclass_brier(y_val, y_prob)
            acc = accuracy_score(y_val, y_pred)
            folds.append({"fold": fold_i, "n_train": len(train_idx),
                          "log_loss": ll, "brier": bs, "accuracy": acc})

            last_probs, last_true = y_prob, y_val
            logger.info(f"  [{name}] Fold {fold_i} | Treino {len(train_idx):4d} → Val {len(val_idx):4d} | "
                        f"LL={ll:.4f} BS={bs:.4f} Acc={acc:.3f}")

        calibration_data[name] = (last_true, last_probs)
        results[name] = pd.DataFrame(folds)

    return results, calibration_data


# ═══════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — EV Calculator
# ═══════════════════════════════════════════════════════════════════════════════

def run_ev_analysis():
    logger.info("━" * 60)
    logger.info("ETAPA 4 — Análise de Valor Esperado (EV)")
    logger.info("━" * 60)

    scanner = PregameScanner(db_path="sqlite:///understat_premier_league.db")
    value_bets = []
    mode = "AO VIVO"
    
    try:
        report = scanner.scan(
            min_ev=0.0,
            bankroll=1000.0,
            hours_window=24.0,
            odds_source="pinnacle",
        )
        value_bets = report.value_bets
    except Exception as e:
        logger.warning(f"Falha ao buscar odds ao vivo na The Odds API: {e}")

    if not value_bets:
        logger.warning("Nenhuma partida 'AO VIVO' encontrada nas próximas 24h ou falha na API.")
        logger.warning("Acionando Fallback Gracioso: 'OFFLINE' demostration.")
        mode = "OFFLINE"
        
        try:
            report = scanner.scan_offline(limit=6, min_ev=0.0)
            value_bets = report.value_bets
        except Exception as e:
            logger.error(f"Falha crítica no scan offline: {e}")
            
    logger.info(f"Modo do Dashboard: [{mode}]")
    
    ev_rows = []
    for bet in value_bets:
        ev_rows.append({
            "partida": f"{bet.home_team}\nvs {bet.away_team}",
            "resultado": bet.outcome_label,
            "model_pct": bet.model_prob * 100,
            "casa_pct":  bet.implied_prob * 100,
            "odd":       bet.odds_taken,
            "ev":        bet.ev_pct,
            "kelly":     bet.stake_pct * 100,
            "is_value":  bet.ev_pct > 0.0,
        })

    return pd.DataFrame(ev_rows), mode == "AO VIVO"


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def build_dashboard(df_raw: pd.DataFrame, ml_results: dict,
                    calibration_data: dict, df_ev: pd.DataFrame,
                    is_live_data: bool):

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "text.color":        CLR["text"],
        "axes.facecolor":    CLR["panel"],
        "axes.edgecolor":    CLR["border"],
        "axes.labelcolor":   CLR["text"],
        "axes.titlecolor":   CLR["text"],
        "xtick.color":       CLR["subtext"],
        "ytick.color":       CLR["subtext"],
        "figure.facecolor":  CLR["bg"],
        "grid.color":        CLR["border"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  CLR["panel"],
        "legend.edgecolor":  CLR["border"],
        "legend.labelcolor": CLR["text"],
    })

    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor(CLR["bg"])

    gs = gridspec.GridSpec(
        5, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.38,
        top=0.94, bottom=0.04,
        left=0.06, right=0.97,
    )

    # ── Título principal ──────────────────────────────────────────────────────
    fig.text(0.5, 0.97, "⚽  EPL xG Prediction Dashboard",
             ha="center", va="top", fontsize=22, fontweight="bold",
             color=CLR["text"])
    fig.text(0.5, 0.955, "Pipeline: ETL → Feature Engineering → ML (TimeSeriesSplit) → Expected Value",
             ha="center", va="top", fontsize=11, color=CLR["subtext"])
             
    import datetime
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.97, 0.02, f"Gerado em: {timestamp_str}",
             ha="right", va="bottom", fontsize=9, color=CLR["subtext"], style="italic")

    model_names   = list(ml_results.keys())
    model_colors  = [CLR["blue"], CLR["orange"]]
    baseline_vals = {"log_loss": 1.0986, "brier": 0.6667, "accuracy": 0.333}

    # ══════════════════════════════════════════════════════════════════════════
    # LINHA 0 — KPI Cards (3 métricas como texto estilizado)
    # ══════════════════════════════════════════════════════════════════════════

    kpi_specs = []
    for i, name in enumerate(model_names):
        df_m = ml_results[name]
        kpi_specs.append({
            "title":    name,
            "log_loss": df_m["log_loss"].mean(),
            "brier":    df_m["brier"].mean(),
            "accuracy": df_m["accuracy"].mean(),
            "color":    model_colors[i],
        })
    # Baseline ingênuo
    kpi_specs.append({
        "title": "Ingênuo (1/3 cada)",
        "log_loss": 1.0986, "brier": 0.6667, "accuracy": 0.333,
        "color": CLR["subtext"],
    })

    for col_i, kpi in enumerate(kpi_specs):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_facecolor(CLR["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(kpi["color"])
            spine.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])

        ax.text(0.5, 0.82, kpi["title"], transform=ax.transAxes,
                ha="center", fontsize=12, fontweight="bold", color=kpi["color"])

        metrics = [
            ("Log Loss",   f"{kpi['log_loss']:.4f}", CLR["red"]   if kpi["log_loss"] < 1.0986 else CLR["subtext"]),
            ("Brier Score",f"{kpi['brier']:.4f}",    CLR["green"] if kpi["brier"]    < 0.6667 else CLR["subtext"]),
            ("Accuracy",   f"{kpi['accuracy']:.1%}", CLR["blue"]  if kpi["accuracy"] > 0.333  else CLR["subtext"]),
        ]
        for j, (label, val, color) in enumerate(metrics):
            y_pos = 0.52 - j * 0.22
            ax.text(0.28, y_pos, label + ":", transform=ax.transAxes,
                    ha="right", fontsize=9, color=CLR["subtext"])
            ax.text(0.32, y_pos, val, transform=ax.transAxes,
                    ha="left", fontsize=13, fontweight="bold", color=color)

    # ══════════════════════════════════════════════════════════════════════════
    # LINHA 1a — Log Loss por fold (esq) | Brier por fold (centro)
    # ══════════════════════════════════════════════════════════════════════════

    metric_plots = [
        (gs[1, 0], "log_loss",  "Log Loss por Fold",   CLR["red"],   baseline_vals["log_loss"],  "↓ melhor"),
        (gs[1, 1], "brier",     "Brier Score por Fold", CLR["yellow"], baseline_vals["brier"],   "↓ melhor"),
        (gs[1, 2], "accuracy",  "Accuracy por Fold",   CLR["green"], baseline_vals["accuracy"],  "↑ melhor"),
    ]

    for spec_gs, metric, title, color, baseline, note in metric_plots:
        ax = fig.add_subplot(spec_gs)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(True, axis="y")

        n_folds = len(ml_results[model_names[0]])
        x = np.arange(1, n_folds + 1)
        width = 0.35

        for idx, (name, df_m) in enumerate(ml_results.items()):
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, df_m[metric], width,
                          color=model_colors[idx], alpha=0.85,
                          label=name, zorder=3)

        ax.axhline(baseline, color=CLR["subtext"], linestyle="--",
                   linewidth=1.4, label=f"Baseline: {baseline:.3f}", zorder=2)

        ax.set_xlabel("Fold", fontsize=9)
        ax.set_xticks(x)
        ax.legend(fontsize=8)
        ax.set_facecolor(CLR["panel"])
        ax.text(0.98, 0.96, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=CLR["subtext"],
                style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # LINHA 2a — EWMA de xG ao longo do tempo (mandante) — time destaque
    # ══════════════════════════════════════════════════════════════════════════

    ax_ewma = fig.add_subplot(gs[2, :2])
    ax_ewma.set_title("EWMA-5 de xG (Ataque vs Defesa) — Arsenal", fontsize=11, pad=8)
    ax_ewma.grid(True)

    query_arsenal = """
        SELECT m.date, m.home_team, m.away_team,
               f.ewma5_xg_pro_home, f.ewma5_xg_con_home,
               f.ewma5_xg_pro_away, f.ewma5_xg_con_away
        FROM matches m
        INNER JOIN match_features f ON m.id = f.match_id
        ORDER BY m.date ASC
    """
    df_raw2 = df_raw.copy()
    # Monta série temporal do Arsenal como mandante
    home_ars = df_raw2[df_raw2["home_team"] == "Arsenal"][["date", "ewma5_xg_pro_home", "ewma5_xg_con_home"]].copy()
    home_ars.columns = ["date", "xg_pro", "xg_con"]
    away_ars = df_raw2[df_raw2["away_team"] == "Arsenal"][["date", "ewma5_xg_pro_away", "ewma5_xg_con_away"]].copy()
    away_ars.columns = ["date", "xg_pro", "xg_con"]

    ars_ts = pd.concat([home_ars, away_ars]).sort_values("date").drop_duplicates("date")

    if len(ars_ts) > 5:
        ax_ewma.plot(ars_ts["date"], ars_ts["xg_pro"], color=CLR["green"],
                     linewidth=2, label="Ataque (xG produzido)", zorder=3)
        ax_ewma.fill_between(ars_ts["date"], ars_ts["xg_pro"], alpha=0.15, color=CLR["green"])
        ax_ewma.plot(ars_ts["date"], ars_ts["xg_con"], color=CLR["red"],
                     linewidth=2, label="Defesa (xG concedido)", zorder=3)
        ax_ewma.fill_between(ars_ts["date"], ars_ts["xg_con"], alpha=0.15, color=CLR["red"])
        ax_ewma.axhline(ars_ts["xg_pro"].mean(), color=CLR["green"],
                        linestyle=":", linewidth=1, alpha=0.6)
        ax_ewma.axhline(ars_ts["xg_con"].mean(), color=CLR["red"],
                        linestyle=":", linewidth=1, alpha=0.6)

    ax_ewma.set_xlabel("Data", fontsize=9)
    ax_ewma.set_ylabel("xG (EWMA-5)", fontsize=9)
    ax_ewma.legend(fontsize=9)
    ax_ewma.set_facecolor(CLR["panel"])

    # ══════════════════════════════════════════════════════════════════════════
    # LINHA 2b — Distribuição dos resultados (pie chart)
    # ══════════════════════════════════════════════════════════════════════════

    ax_pie = fig.add_subplot(gs[2, 2])
    ax_pie.set_title("Distribuição de Resultados", fontsize=11, pad=8)
    ax_pie.set_facecolor(CLR["panel"])

    counts = df_raw2["home_goals"].apply(
        lambda x: x  # dummy — vamos contar direto
    )
    h = (df_raw2["home_goals"] > df_raw2["away_goals"]).sum()
    d = (df_raw2["home_goals"] == df_raw2["away_goals"]).sum()
    a = (df_raw2["home_goals"] < df_raw2["away_goals"]).sum()
    total = h + d + a

    wedge_props = dict(width=0.55, edgecolor=CLR["bg"], linewidth=2)
    ax_pie.pie(
        [h, d, a],
        labels=[f"Mandante\n{h/total:.1%}", f"Empate\n{d/total:.1%}", f"Visitante\n{a/total:.1%}"],
        colors=[CLR["blue"], CLR["yellow"], CLR["orange"]],
        autopct=lambda p: f"{p:.0f}",
        wedgeprops=wedge_props,
        textprops={"color": CLR["text"], "fontsize": 9},
        startangle=90,
        pctdistance=0.75,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # LINHA 3a — Curva de Calibração
    # ══════════════════════════════════════════════════════════════════════════

    ax_cal = fig.add_subplot(gs[3, 0])
    ax_cal.set_title("Calibração — Vitória Mandante (H)", fontsize=11, pad=8)
    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.2,
                label="Calibração perfeita", color=CLR["subtext"])
    ax_cal.grid(True)
    ax_cal.set_facecolor(CLR["panel"])

    for idx, (name, (y_true, y_prob)) in enumerate(calibration_data.items()):
        # índice da classe H no predict_proba
        temp_pipe = list(ml_results.keys())
        clf_classes = sorted(set(y_true))  # ["A", "D", "H"]
        h_idx = clf_classes.index("H")

        prob_h = y_prob[:, h_idx]
        true_h = (y_true == "H").astype(int)

        try:
            frac_pos, mean_pred = calibration_curve(true_h, prob_h, n_bins=8, strategy="quantile")
            ax_cal.plot(mean_pred, frac_pos, "o-", color=model_colors[idx],
                        linewidth=2, markersize=5, label=name)
        except Exception:
            pass

    ax_cal.set_xlabel("Probabilidade prevista", fontsize=9)
    ax_cal.set_ylabel("Frequência real", fontsize=9)
    ax_cal.legend(fontsize=8)
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)

    # ══════════════════════════════════════════════════════════════════════════
    # LINHA 3b — Feature Importance (LogReg — coeficientes)
    # ══════════════════════════════════════════════════════════════════════════

    ax_fi = fig.add_subplot(gs[3, 1:])
    ax_fi.set_title("Importância das Features (Logistic Regression — classe H)", fontsize=11, pad=8)
    ax_fi.grid(True, axis="x")
    ax_fi.set_facecolor(CLR["panel"])

    # Re-treina LR com todos os dados para extrair coefs
    X_all = df_raw[FEATURE_COLS].values
    y_all = np.select(
        [df_raw["home_goals"] > df_raw["away_goals"],
         df_raw["home_goals"] == df_raw["away_goals"],
         df_raw["home_goals"] < df_raw["away_goals"]],
        ["H", "D", "A"], default=""
    )
    mask = y_all != ""
    X_all, y_all = X_all[mask], y_all[mask]

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_all)
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    lr.fit(X_scaled, y_all)

    h_class_idx = list(lr.classes_).index("H")
    coefs = lr.coef_[h_class_idx]
    sort_idx = np.argsort(np.abs(coefs))

    feat_labels = [f.replace("ewma", "EWMA-").replace("_xg_", " xG ").replace("_home", " 🏠").replace("_away", " ✈")
                   for f in FEATURE_COLS]
    sorted_labels = [feat_labels[i] for i in sort_idx]
    sorted_coefs  = coefs[sort_idx]
    bar_colors    = [CLR["green"] if c > 0 else CLR["red"] for c in sorted_coefs]

    bars = ax_fi.barh(sorted_labels, sorted_coefs, color=bar_colors, alpha=0.85, zorder=3)
    ax_fi.axvline(0, color=CLR["subtext"], linewidth=1)
    ax_fi.set_xlabel("Coeficiente (escala padronizada)", fontsize=9)

    for bar, val in zip(bars, sorted_coefs):
        ax_fi.text(val + (0.004 if val >= 0 else -0.004), bar.get_y() + bar.get_height() / 2,
                   f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                   fontsize=8, color=CLR["text"])

    # ══════════════════════════════════════════════════════════════════════════
    # LINHA 4 — Tabela de EV
    # ══════════════════════════════════════════════════════════════════════════

    ax_ev = fig.add_subplot(gs[4, :])
    ev_title = "📊  Análise de Valor Esperado (EV) — Mercado AO VIVO (Próximas 24h)" if is_live_data else "📊  Análise de Valor Esperado (EV) — Dados Históricos / Offline"
    ax_ev.set_title(ev_title, fontsize=12, pad=10)
    ax_ev.set_facecolor(CLR["panel"])
    ax_ev.set_xticks([]); ax_ev.set_yticks([])
    for spine in ax_ev.spines.values():
        spine.set_visible(False)

    col_labels = ["Partida", "Resultado", "Modelo %", "Casa %",  "Odd", "EV %", "Kelly %", "Sinal"]
    col_keys   = ["partida", "resultado", "model_pct", "casa_pct", "odd", "ev", "kelly", "is_value"]
    col_widths = [0.18, 0.13, 0.09, 0.09, 0.07, 0.08, 0.08, 0.07]

    # Cabeçalho
    x_cursor = 0.01
    for label, w in zip(col_labels, col_widths):
        ax_ev.text(x_cursor, 0.93, label, transform=ax_ev.transAxes,
                   ha="left", va="top", fontsize=9, fontweight="bold",
                   color=CLR["yellow"])
        x_cursor += w

    ax_ev.axhline(0.895, color=CLR["border"], linewidth=1)

    # Linhas de dados
    row_h = 0.085
    for row_i, (_, row) in enumerate(df_ev.iterrows()):
        y_pos = 0.87 - row_i * row_h
        if y_pos < 0.02:
            break
        bg_alpha = 0.08 if row["is_value"] else 0.0
        bg_color = CLR["green"] if row["is_value"] else CLR["panel"]

        rect = mpatches.FancyBboxPatch(
            (0.005, y_pos - 0.065), 0.99, row_h - 0.005,
            boxstyle="round,pad=0.005",
            facecolor=bg_color, alpha=0.12 if row["is_value"] else 0,
            transform=ax_ev.transAxes, zorder=1,
        )
        ax_ev.add_patch(rect)

        row_vals = [
            row["partida"],
            row["resultado"],
            f"{row['model_pct']:.1f}%",
            f"{row['casa_pct']:.1f}%",
            f"{row['odd']:.2f}",
            f"{row['ev']:+.1f}%",
            f"{row['kelly']:.1f}%" if row["is_value"] else "—",
            "✅ EV+" if row["is_value"] else "❌",
        ]
        text_colors = [CLR["text"]] * 5 + [
            CLR["green"] if row["ev"] > 0 else CLR["red"],
            CLR["green"] if row["is_value"] else CLR["subtext"],
            CLR["green"] if row["is_value"] else CLR["red"],
        ]

        x_cursor = 0.01
        for val, color, w in zip(row_vals, text_colors, col_widths):
            ax_ev.text(x_cursor, y_pos, str(val), transform=ax_ev.transAxes,
                       ha="left", va="top", fontsize=8.5, color=color, zorder=2)
            x_cursor += w

    # ── Salva e exibe ──────────────────────────────────────────────────────────
    output_path = "dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=CLR["bg"], edgecolor="none")
    logger.info(f"Dashboard salvo em: {output_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    engine  = get_engine("sqlite:///understat_premier_league.db")
    create_tables(engine)
    Session = lambda: get_session(engine)

    # 1. ETL
    run_etl(engine, Session)

    # 2. Dataset
    df = load_ml_dataset(engine)

    # 3. ML
    ml_results, calibration_data = run_training(df, n_splits=5)

    # 4. EV
    df_ev, is_live_data = run_ev_analysis()

    # 5. Dashboard
    logger.info("━" * 60)
    logger.info("ETAPA 5 — Gerando Dashboard")
    logger.info("━" * 60)
    build_dashboard(df, ml_results, calibration_data, df_ev, is_live_data)


if __name__ == "__main__":
    main()
