"""
feature_engineering.py
----------------------
Módulo responsável pela criação de Rolling Features baseadas em xG.

Estratégia anti-data-leakage:
  Para cada time, os valores de xG são ordenados cronologicamente e a EWMA
  é calculada ANTES de ser deslocada 1 posição com shift(1).
  Isso garante que a feature da Rodada N reflita apenas os jogos 1…N-1,
  nunca o jogo atual.

Formato esperado do DataFrame de entrada (colunas mínimas):
  - date       : datetime | O momento do jogo
  - home_team  : str      | Nome do time mandante
  - away_team  : str      | Nome do time visitante
  - home_xG    : float    | xG gerado pelo mandante
  - away_xG    : float    | xG gerado pelo visitante
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Colunas que serão geradas por este módulo
EWMA_SPANS = {"5": 5, "10": 10}

# Nome das colunas de saída (sufixo _home / _away adicionado ao juntar)
FEATURE_COLS = [
    # xG
    "ewma5_xg_pro",
    "ewma10_xg_pro",
    "ewma5_xg_con",
    "ewma10_xg_con",
    # Shots on target
    "ewma5_shots_target_pro",
    "ewma10_shots_target_pro",
    "ewma5_shots_target_con",
    "ewma10_shots_target_con",
    # Possession
    "ewma5_possession_pro",
    "ewma10_possession_pro",
    "ewma5_possession_con",
    "ewma10_possession_con",
]


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------


def _build_team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma o DataFrame de jogos (formato largo, 1 linha = 1 jogo)
    em formato longo (2 linhas por jogo — perspectiva de cada time).

    Colunas do resultado:
        match_id | date | team | xg_pro | xg_con
    """
    # Perspectiva do mandante: atacou com home_xG/home_shots/home_possession, sofreu o inverso
    home_view = df[["id", "date", "home_team", "home_xG", "away_xG", "home_shots_target", "away_shots_target", "home_possession", "away_possession"]].copy()
    home_view.columns = ["match_id", "date", "team", "xg_pro", "xg_con", "shots_target_pro", "shots_target_con", "possession_pro", "possession_con"]

    # Perspectiva do visitante: atacou com away_xG/away_shots/away_possession, sofreu o inverso
    away_view = df[["id", "date", "away_team", "away_xG", "home_xG", "away_shots_target", "home_shots_target", "away_possession", "home_possession"]].copy()
    away_view.columns = ["match_id", "date", "team", "xg_pro", "xg_con", "shots_target_pro", "shots_target_con", "possession_pro", "possession_con"]

    long_df = pd.concat([home_view, away_view], ignore_index=True)
    long_df.sort_values(["team", "date"], inplace=True)
    long_df.reset_index(drop=True, inplace=True)

    return long_df


def _compute_ewma_per_team(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula as EWMAs de xg_pro e xg_con para cada time,
    aplicando shift(1) APÓS o cálculo para garantir que a feature
    da rodada N use apenas os dados das rodadas 1…N-1.

    ┌─────────────────────────────────────────────────────────────────┐
    │  REGRA DE OURO — PREVENÇÃO DE DATA LEAKAGE                      │
    │                                                                 │
    │  ewma.shift(1)  →  o valor da linha N é a média das linhas 0…N-1│
    │  Sem o shift, a linha N incluiria o próprio jogo N no cálculo.  │
    └─────────────────────────────────────────────────────────────────┘

    Nota: usa loop explícito em vez de groupby().apply() para compatibilidade
    com pandas 3.x, que exclui a coluna de agrupamento do grupo passado à função.
    """
    chunks = []

    for _team, group in long_df.groupby("team", sort=False):
        group = group.copy()

        # Iterar sobre todas as colunas de estatísticas de base
        stat_cols = ["xg_pro", "xg_con", "shots_target_pro", "shots_target_con", "possession_pro", "possession_con"]

        for col in stat_cols:
            for label, span in EWMA_SPANS.items():
                feature_name = f"ewma{label}_{col}"
                # Calcula a ewma crua ignorando NaNs quando possível
                raw_ewma = group[col].ewm(span=span, adjust=False).mean()
                group[feature_name] = raw_ewma.shift(1)
        chunks.append(group)

    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------------------
# Função pública principal
# ---------------------------------------------------------------------------


def add_ewma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features de Média Móvel Exponencial (EWMA) ao DataFrame de jogos.

    Para cada jogo são criadas 8 novas colunas (4 para o mandante, 4 para visitante):

        ewma5_xg_pro_home   — EWMA-5 do xG produzido pelo mandante
        ewma10_xg_pro_home  — EWMA-10 do xG produzido pelo mandante
        ewma5_xg_con_home   — EWMA-5 do xG concedido pelo mandante
        ewma10_xg_con_home  — EWMA-10 do xG concedido pelo mandante

        ewma5_xg_pro_away   — EWMA-5 do xG produzido pelo visitante
        ewma10_xg_pro_away  — EWMA-10 do xG produzido pelo visitante
        ewma5_xg_con_away   — EWMA-5 do xG concedido pelo visitante
        ewma10_xg_con_away  — EWMA-10 do xG concedido pelo visitante

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com colunas obrigatórias:
        id, date, home_team, away_team, home_xG, away_xG

    Retorna
    -------
    pd.DataFrame
        O mesmo DataFrame de entrada enriquecido com as 8 novas colunas.
    """
    # Garantir que colunas avançadas existam com valor neutro (0.0) caso ausentes no banco.
    # O EWMA de uma coluna zerada produz zeros, contribuição nula ao modelo.
    advanced_cols = ["home_xG", "away_xG", "home_shots_target", "away_shots_target", "home_possession", "away_possession"]
    for col in advanced_cols:
        if col not in df.columns:
            df[col] = 0.0

    logger.info("Calculando features EWMA de métricas avançadas (anti data leakage)...")

    # Passo 1: formato longo — 1 linha por (time × jogo)
    long_df = _build_team_perspective(df)

    # Passo 2: calcula EWMAs com shift(1) agrupadas por time
    long_features = _compute_ewma_per_team(long_df)

    # Passo 3: reintegra as features ao DataFrame original
    # —— perspectiva do mandante ——
    home_features = long_features.merge(
        df[["id", "home_team"]],
        left_on=["match_id", "team"],
        right_on=["id", "home_team"],
        how="inner",
    )[["match_id"] + FEATURE_COLS].rename(columns={col: f"{col}_home" for col in FEATURE_COLS})

    # —— perspectiva do visitante ——
    away_features = long_features.merge(
        df[["id", "away_team"]],
        left_on=["match_id", "team"],
        right_on=["id", "away_team"],
        how="inner",
    )[["match_id"] + FEATURE_COLS].rename(columns={col: f"{col}_away" for col in FEATURE_COLS})

    # Passo 4: une tudo ao DataFrame original pela chave do jogo
    result = (
        df.merge(home_features, left_on="id", right_on="match_id", how="left").drop(columns=["match_id"]).merge(away_features, left_on="id", right_on="match_id", how="left").drop(columns=["match_id"])
    )

    # Passo 5: Renomear para o padrão final solicitado pelo usuário e tratar Nulos (NaN)
    rename_map = {
        # xG Casa
        "ewma5_xg_pro_home": "media_xg_casa",
        "ewma5_xg_con_home": "media_xga_casa",
        "ewma10_xg_pro_home": "media10_xg_casa",
        "ewma10_xg_con_home": "media10_xga_casa",
        # xG Fora
        "ewma5_xg_pro_away": "media_xg_fora",
        "ewma5_xg_con_away": "media_xga_fora",
        "ewma10_xg_pro_away": "media10_xg_fora",
        "ewma10_xg_con_away": "media10_xga_fora",
        # Chutes Alvo
        "ewma5_shots_target_pro_home": "media_chutes_alvo_casa",
        "ewma5_shots_target_con_home": "media_chutes_alvo_sofrido_casa",
        "ewma5_shots_target_pro_away": "media_chutes_alvo_fora",
        "ewma5_shots_target_con_away": "media_chutes_alvo_sofrido_fora",
        # Posse
        "ewma5_possession_pro_home": "media_posse_casa",
        "ewma5_possession_con_home": "media_posse_sofrida_casa",
        "ewma5_possession_pro_away": "media_posse_fora",
        "ewma5_possession_con_away": "media_posse_sofrida_fora",
    }

    result = result.rename(columns=rename_map)

    # Tratamento de Nulos
    for col in rename_map.values():
        if col in result.columns:
            # Preenche com a média atual da liga nessa coluna, ou 0 caso tudo seja NaN
            mean_val = result[col].median()
            if pd.isna(mean_val):
                mean_val = 0.0
            result[col] = result[col].fillna(mean_val)

    n_novas_cols = len(FEATURE_COLS) * 2
    logger.info(f"EWMA concluída. {n_novas_cols} colunas adicionadas ao DataFrame (preenchidas com média).")
    return result
