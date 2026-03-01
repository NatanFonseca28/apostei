"""
fbref_collector.py
------------------
Coleta estatísticas avançadas de partidas (xG, chutes no alvo, posse de bola)
do FBref via a biblioteca soccerdata e persiste em `match_advanced_stats`.

Fonte: https://fbref.com — scraping livre via cloudscraper (sem API key).
Rate limit do FBref: 6 segundos entre requisições (respeitado automaticamente
pelo soccerdata). Os dados são cacheados em ~/soccerdata/data/FBref/, então
execuções subsequentes são quase instantâneas.

Ligas suportadas (configuradas via ~/soccerdata/config/league_dict.json):
  - BRA-Serie A  (Brasileirão — requer entrada no league_dict.json)
  - ENG-Premier League, ESP-La Liga, ITA-Serie A, GER-Bundesliga, FRA-Ligue 1

Uso rápido:
    from src.data.fbref_collector import collect_advanced_stats
    df = collect_advanced_stats(league="BRA-Serie A", seasons=[2024, 2025])
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapa de normalização de nomes: FBref → Flashscore
# Adicione entradas quando o nome do time diferir entre as duas fontes.
# ---------------------------------------------------------------------------
TEAM_NAME_MAP: dict[str, str] = {
    # Brasileirão — casos conhecidos
    "Athletico Paranaense": "Athletico-PR",
    "Atlético Paranaense": "Athletico-PR",
    "Atlético Mineiro": "Atlético-MG",
    "Atletico Paranaense": "Athletico-PR",
    "Atletico Mineiro": "Atlético-MG",
    "Atletico-MG": "Atlético-MG",
    "Red Bull Bragantino": "RB Bragantino",
    "America Mineiro": "América-MG",
    "América Mineiro": "América-MG",
    "Cuiabá": "Cuiabá",
    "Goiás EC": "Goiás",
    "Sport Recife": "Sport",
    "Ceará SC": "Ceará",
    "Santos FC": "Santos",
    "Grêmio": "Grêmio",
}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _normalize_team(name: str) -> str:
    """Aplica o mapa de normalização FBref → Flashscore."""
    return TEAM_NAME_MAP.get(name, name)


def _find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> Optional[Any]:
    """
    Localiza o nome da coluna no DataFrame a partir de uma lista de candidatos.
    Suporta MultiIndex (estrutura de tabela aninhada do FBref) e Index simples.

    Retorna o nome da coluna encontrada (pode ser tupla em MultiIndex) ou None.
    """
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        for col_tuple in cols:
            # Verifica tanto o nível 0 quanto o nível 1 do MultiIndex
            for part in col_tuple:
                if isinstance(part, str) and part in candidates:
                    return col_tuple
    else:
        for candidate in candidates:
            if candidate in cols:
                return candidate
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Converte para float; preenche NaN/invalidos com 0.0."""
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------------
# Coleta principal
# ---------------------------------------------------------------------------


def collect_advanced_stats(
    league: str = "BRA-Serie A",
    seasons: Optional[Union[int, list[int]]] = None,
    no_cache: bool = False,
) -> pd.DataFrame:
    """
    Coleta xG, chutes no alvo e posse de bola do FBref via soccerdata.

    Parâmetros
    ----------
    league : str
        ID canônico da liga no soccerdata. Exemplos:
        - "BRA-Serie A"          ← Brasileirão (requer league_dict.json)
        - "ENG-Premier League"
        - "ESP-La Liga"
    seasons : int ou list[int], opcional
        Temporada(s) a coletar. Ex: 2024, [2023, 2024].
        Se None, soccerdata usa a última temporada disponível.
    no_cache : bool
        Se True, ignora o cache local e faz novo scraping.

    Retorna
    -------
    pd.DataFrame com colunas:
        home_team, away_team, date,
        home_xg, away_xg,
        home_shots_target, away_shots_target,
        home_possession, away_possession,
        league, season

    Observação
    ----------
    Partidas sem dados (futuras ou sem match report) retornam 0.0 em todas
    as métricas avançadas. O Flashscore-matcher usará COALESCE para isso.
    """
    try:
        import soccerdata as sd  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("soccerdata não está instalado. Execute: pip install soccerdata") from e

    if seasons is None:
        seasons_arg = None
    elif isinstance(seasons, int):
        seasons_arg = [seasons]
    else:
        seasons_arg = list(seasons)

    logger.info(f"[FBref] Iniciando coleta — liga: {league} | temporadas: {seasons_arg}")
    fbref = sd.FBref(leagues=[league], seasons=seasons_arg, no_cache=no_cache)

    # ── 1. xG via read_schedule ──────────────────────────────────────────────
    logger.info("[FBref] Buscando schedule (xG)...")
    schedule = fbref.read_schedule().reset_index()

    # Normaliza nomes, garante colunas mínimas
    for col in ("home_xg", "away_xg"):
        if col not in schedule.columns:
            schedule[col] = 0.0

    result = schedule[["league", "season", "game", "home_team", "away_team", "date", "home_xg", "away_xg"]].copy()
    result["home_xg"] = _safe_numeric(result["home_xg"])
    result["away_xg"] = _safe_numeric(result["away_xg"])
    result["date"] = pd.to_datetime(result["date"])

    # ── 2. Posse + Chutes no Alvo via read_team_match_stats('schedule') ──────
    logger.info("[FBref] Buscando match logs (posse, chutes)...")
    try:
        team_stats = fbref.read_team_match_stats(stat_type="schedule").reset_index()

        poss_col = _find_column(team_stats, ["Poss", "poss", "possession", "Possession"])
        sot_col = _find_column(team_stats, ["SoT", "sot", "shots_on_target", "Sot"])

        if poss_col is not None and sot_col is not None:
            logger.info(f"[FBref] Colunas encontradas — Posse: {poss_col!r}  SoT: {sot_col!r}")
            # home = linhas com venue == 'Home'; away = linhas com venue == 'Away'
            for col in [poss_col, sot_col, "venue", "game"]:
                if col not in team_stats.columns:
                    raise KeyError(f"Coluna '{col}' ausente em read_team_match_stats. Colunas disponíveis: {list(team_stats.columns)}")

            home_rows = team_stats[team_stats["venue"] == "Home"][["game", poss_col, sot_col]].rename(columns={poss_col: "home_possession", sot_col: "home_shots_target"})
            away_rows = team_stats[team_stats["venue"] == "Away"][["game", poss_col, sot_col]].rename(columns={poss_col: "away_possession", sot_col: "away_shots_target"})

            result = result.merge(home_rows, on="game", how="left").merge(away_rows, on="game", how="left")
        else:
            logger.warning(f"[FBref] Colunas Poss/SoT não encontradas em read_team_match_stats. Colunas disponíveis: {[str(c) for c in team_stats.columns[:30]]}. Usando 0.0 como fallback.")
            result["home_possession"] = 0.0
            result["away_possession"] = 0.0
            result["home_shots_target"] = 0.0
            result["away_shots_target"] = 0.0

    except Exception as exc:
        logger.error(f"[FBref] Falha ao buscar match logs de equipe: {exc}\nPosse e chutes no alvo serão preenchidos com 0.0.")
        result["home_possession"] = 0.0
        result["away_possession"] = 0.0
        result["home_shots_target"] = 0.0
        result["away_shots_target"] = 0.0

    # ── 3. Normalização final ────────────────────────────────────────────────
    for col in ("home_possession", "away_possession", "home_shots_target", "away_shots_target"):
        if col not in result.columns:
            result[col] = 0.0
        result[col] = _safe_numeric(result[col])

    result["home_team"] = result["home_team"].apply(_normalize_team)
    result["away_team"] = result["away_team"].apply(_normalize_team)

    # Remove matches sem data (partidas futuras sem schedule completo)
    result = result[result["date"].notna()].copy()
    result["date"] = result["date"].dt.normalize()  # apenas data, sem hora

    output_cols = [
        "home_team",
        "away_team",
        "date",
        "league",
        "season",
        "home_xg",
        "away_xg",
        "home_shots_target",
        "away_shots_target",
        "home_possession",
        "away_possession",
    ]
    result = result[output_cols].reset_index(drop=True)

    logger.info(
        f"[FBref] Coleta concluída: {len(result)} partidas | "
        f"xG>0: {(result['home_xg'] > 0).sum()} | "
        f"Posse>0: {(result['home_possession'] > 0).sum()} | "
        f"SoT>0: {(result['home_shots_target'] > 0).sum()}"
    )
    return result
