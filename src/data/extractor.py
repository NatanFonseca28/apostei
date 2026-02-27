"""
extractor.py
------------
Modulo de extracao de dados hibrido (Apenas soccerdata):

  1. soccerdata / MatchHistory (football-data.co.uk)
     Fornece: resultados, chutes, cartoes, odds reais de multiplas casas
     (Bet365, Pinnacle, Betfair, media do mercado, etc.).

  2. soccerdata / FBref
     Fornece: xG (Expected Goals) por partida.

O pipeline une ambas as fontes por (data, home_team, away_team), produzindo
um dataset rico com xG real da FBref + odds de mercado MatchHistory.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import soccerdata as sd

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _season_code(year: int) -> str:
    """Converte ano de inicio em codigo football-data.co.uk (ex: 2024 -> '2425')."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


# Colunas que queremos do football-data.co.uk
MATCH_HIST_COLS = {
    # Resultado
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Date": "date",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    # Stats
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    # Odds Bet365
    "B365H": "odds_home_b365",
    "B365D": "odds_draw_b365",
    "B365A": "odds_away_b365",
    # Odds Pinnacle (sharp)
    "PSH": "odds_home_pin",
    "PSD": "odds_draw_pin",
    "PSA": "odds_away_pin",
    # Odds media do mercado
    "AvgH": "odds_home_avg",
    "AvgD": "odds_draw_avg",
    "AvgA": "odds_away_avg",
    # Odds max do mercado
    "MaxH": "odds_home_max",
    "MaxD": "odds_draw_max",
    "MaxA": "odds_away_max",
}

# Normalizacao de nomes de times entre Understat <-> football-data.co.uk
_TEAM_NAME_MAP = {
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham": "Nottingham Forest",
    "Sheffield United": "Sheffield United",
    "Spurs": "Tottenham",
    "Wolves": "Wolverhampton Wanderers",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton",
    "Leicester": "Leicester",
    "Leeds": "Leeds",
}


# ============================================================================
# MATCH HISTORY READER (football-data.co.uk)
# ============================================================================

class MatchHistoryReader:
    """
    Le dados de football-data.co.uk com compatibilidade pandas 3.x.

    Resolve o bug do soccerdata 1.5.1 onde:
      - CSV tem BOM UTF-8 (coluna 'Div' aparece como BOM + 'Div')
      - _translate_league quebra com KeyError: 'league' no pandas 3.0

    Usa cache local no diretorio do soccerdata.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

        if cache_dir is None:
            self.cache_dir = Path.home() / "soccerdata" / "data" / "MatchHistory"
        else:
            self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def read_season(self, season: int) -> pd.DataFrame:
        """
        Le uma temporada do CSV do football-data.co.uk.

        Args:
            season: Ano de inicio (ex: 2024 para 2024/25).

        Returns:
            DataFrame com resultados, stats de jogo e odds.
        """
        code = _season_code(season)
        url = FOOTBALL_DATA_URL.format(code, "E0")  # E0 = Premier League
        filepath = self.cache_dir / f"E0_{code}.csv"

        if filepath.exists():
            logger.info(f"MatchHistory {season}/{season+1}: usando cache {filepath.name}")
            raw = filepath.read_bytes()
        else:
            logger.info(f"MatchHistory {season}/{season+1}: baixando de football-data.co.uk")
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            raw = resp.content
            filepath.write_bytes(raw)

        # utf-8-sig remove BOM automaticamente (fix para pandas 3.x)
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")

        available = {k: v for k, v in MATCH_HIST_COLS.items() if k in df.columns}
        df = df[list(available.keys())].rename(columns=available)

        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        df["season"] = season
        df["league"] = "EPL"
        df = df.dropna(subset=["home_team", "away_team"])

        logger.info(f"  -> {len(df)} partidas com {len(available)} colunas")
        return df

    def read_seasons(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Le multiplas temporadas e concatena."""
        frames = []
        for year in range(start_year, end_year + 1):
            try:
                df = self.read_season(year)
                frames.append(df)
            except Exception as e:
                logger.warning(f"MatchHistory {year}/{year+1} falhou: {e}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


# ============================================================================
# FBREF READER (xG) via Soccerdata
# ============================================================================

class FBrefReader:
    """
    Extrai xG de partidas da EPL via package soccerdata.FBref.
    Retorna dataframe limpo com a carga xG pronta para o merge.
    """

    def __init__(self):
        pass

    def fetch_seasons_df(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Busca múltiplas temporadas do FBref e extrai id, data, times e xG."""
        seasons_str = [_season_code(y) for y in range(start_year, end_year + 1)]
        
        logger.info(f"FBref: Inicializando crawler para as temporadas {seasons_str}...")
        try:
            fb = sd.FBref(leagues="ENG-Premier League", seasons=seasons_str)
            df = fb.read_schedule()
        except Exception as e:
            logger.error(f"Pesquisa FBref falhou: {e}")
            return pd.DataFrame()

        df = df.reset_index()

        # Filtrando jogos que já aconteceram (possuem score legível)
        if "score" in df.columns:
            df = df.dropna(subset=["score"])
        
        # Mapeamento do FBref
        rows = []
        for _, row in df.iterrows():
            try:
                # O Soccerdata do FBref armazena as datas no formato sting/datetime
                dt = pd.to_datetime(row["date"])
                season_yr = int(row["season"][:2]) + 2000 # "2021" -> 2020

                rows.append({
                    "fbref_id": hash(f"{row['home_team']}{row['away_team']}{dt.date()}"),
                    "date": dt,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_xG": float(row["home_xg"]) if pd.notna(row["home_xg"]) else float("nan"),
                    "away_xG": float(row["away_xg"]) if pd.notna(row["away_xg"]) else float("nan"),
                    "season": season_yr
                })
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)



# ============================================================================
# NORMALIZACAO
# ============================================================================

def _normalize_team(name: str) -> str:
    """Normaliza nome do time para matching entre fontes."""
    return _TEAM_NAME_MAP.get(name, name)


# ============================================================================
# EXTRATOR HIBRIDO
# ============================================================================

class DataExtractor:
    """
    Extrator Híbrido que combina:
      - MatchHistory (soccerdata) -> resultados + odds mercado
      - FBref (soccerdata) -> métricas reais de xG
    """

    def __init__(self):
        self.match_history = MatchHistoryReader()
        self.fbref = FBrefReader()

    # -- Interface nova: dataset rico via FBref xG e MatchHistory Odds --

    def fetch_rich_dataset(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """
        Extrai e combina dados do FBref (xG) com MatchHistory (odds)
        para reconstruir o pipeline que usava Understat.
        """
        logger.info("=" * 60)
        logger.info("EXTRACAO: MatchHistory + FBref (soccerdata framework)")
        logger.info("=" * 60)

        logger.info("[1/3] Buscando MatchHistory (football-data.co.uk)...")
        df_odds = self.match_history.read_seasons(start_year, end_year)

        logger.info("[2/3] Buscando xG Data (FBref)...")
        df_xg = self.fbref.fetch_seasons_df(start_year, end_year)

        if df_odds.empty and df_xg.empty:
            logger.warning("Nenhum dado extraido!")
            return pd.DataFrame()

        # 3. Merge
        logger.info("[3/3] Unindo fontes por (data, times)...")

        if not df_odds.empty:
            df_odds["_home_norm"] = df_odds["home_team"].map(_normalize_team)
            df_odds["_away_norm"] = df_odds["away_team"].map(_normalize_team)
            df_odds["_date_key"] = df_odds["date"].dt.date

        if not df_xg.empty:
            df_xg["_home_norm"] = df_xg["home_team"].map(_normalize_team)
            df_xg["_away_norm"] = df_xg["away_team"].map(_normalize_team)
            df_xg["_date_key"] = df_xg["date"].dt.date

        if df_odds.empty:
            logger.warning("MatchHistory vazio -- usando apenas FBref")
            df_xg["id"] = df_xg["fbref_id"]
            return df_xg.drop(columns=["_home_norm", "_away_norm", "_date_key"])

        if df_xg.empty:
            logger.warning("FBref vazio -- sem xG disponivel")
            df_odds["id"] = range(1, len(df_odds) + 1)
            df_odds["home_xG"] = float("nan")
            df_odds["away_xG"] = float("nan")
            return df_odds.drop(columns=["_home_norm", "_away_norm", "_date_key"])

        # Identificar colunas unicas
        odds_cols = [c for c in df_odds.columns
                     if c not in ("home_team", "away_team", "date", "season",
                                  "league", "home_goals", "away_goals",
                                  "_home_norm", "_away_norm", "_date_key")]

        # Left join priorizando MatchHistory
        df_merged = pd.merge(
            df_odds,
            df_xg[["_date_key", "_home_norm", "_away_norm", "home_xG", "away_xG", "fbref_id"]],
            on=["_date_key", "_home_norm", "_away_norm"],
            how="left",
        )

        df_merged["id"] = df_merged["fbref_id"].fillna(0).astype(int)

        drop_cols = ["_home_norm", "_away_norm", "_date_key", "fbref_id"]
        df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

        n_total = len(df_merged)
        n_xg = df_merged["home_xG"].notna().sum()
        logger.info(
            f"Merge concluido: {n_total} partidas, "
            f"{n_xg} recuperaram xG via FBref ({n_xg/max(n_total,1)*100:.0f}%)"
        )

        return df_merged.sort_values("date").reset_index(drop=True)

