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
# CONSTANTES E MAPEAMENTO
# ============================================================================

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{}/{}.csv"

# Mapeamento de ligas suportadas entre as fontes
LEAGUE_MAPPING = {
    'soccer_epl': {
        'fbref': 'ENG-Premier League', 
        'fd_code': 'E0'
    },
    'soccer_brazil_campeonato': {
        'fbref': 'BRA-Série A', 
        'fd_code': None  # Football-Data não suporta Brasileirão
    },
    'soccer_uefa_champs_league': {
        'fbref': 'INT-Champions League', 
        'fd_code': None  # Histórico de MatchHistory varia para CL
    }
}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

def _season_code(year: int) -> str:
    """Converte ano de inicio em codigo football-data.co.uk (ex: 2024 -> '2425')."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"

# Colunas que queremos do football-data.co.uk
MATCH_HIST_COLS = {
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Date": "date",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_target",
    "AST": "away_shots_target",
    "B365H": "odds_home_b365",
    "B365D": "odds_draw_b365",
    "B365A": "odds_away_b365",
    "PSH": "odds_home_pin",
    "PSD": "odds_draw_pin",
    "PSA": "odds_away_pin",
    "AvgH": "odds_home_avg",
    "AvgD": "odds_draw_avg",
    "AvgA": "odds_away_avg",
    "MaxH": "odds_home_max",
    "MaxD": "odds_draw_max",
    "MaxA": "odds_away_max",
}

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
    "Leicester City": "Leicester",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
}

# ============================================================================
# MATCH HISTORY READER
# ============================================================================

class MatchHistoryReader:
    """Le dados de football-data.co.uk para ligas suportadas."""

    def __init__(self, cache_dir: Path | None = None):
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        if cache_dir is None:
            self.cache_dir = Path.home() / "soccerdata" / "data" / "MatchHistory"
        else:
            self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def read_season(self, season: int, fd_code: str | None) -> pd.DataFrame:
        """Le uma temporada. Se fd_code for None, retorna DF vazio."""
        if not fd_code:
            return pd.DataFrame(columns=list(MATCH_HIST_COLS.values()) + ["season", "league"])

        code = _season_code(season)
        url = FOOTBALL_DATA_URL.format(code, fd_code)
        filepath = self.cache_dir / f"{fd_code}_{code}.csv"

        try:
            if filepath.exists():
                logger.info(f"MatchHistory {fd_code} {season}: usando cache")
                raw = filepath.read_bytes()
            else:
                logger.info(f"MatchHistory {fd_code} {season}: baixando")
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                raw = resp.content
                filepath.write_bytes(raw)

            df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
            available = {k: v for k, v in MATCH_HIST_COLS.items() if k in df.columns}
            df = df[list(available.keys())].rename(columns=available)
            df["date"] = pd.to_datetime(df["date"], dayfirst=True)
            df["season"] = season
            df["league"] = fd_code
            return df.dropna(subset=["home_team", "away_team"])
        except Exception as e:
            logger.warning(f"Falha ao ler MatchHistory {fd_code} {season}: {e}")
            return pd.DataFrame()

# ============================================================================
# FBREF READER
# ============================================================================

class FBrefReader:
    """Extrai xG do FBref via soccerdata para multiplas ligas."""

    def read_leagues(self, start_year: int, end_year: int, fbref_leagues: list[str]) -> pd.DataFrame:
        """Busca temporadas para uma lista de ligas do FBref."""
        seasons_str = [_season_code(y) for y in range(start_year, end_year + 1)]
        import time, random
        time.sleep(random.uniform(1.0, 3.0)) # Atraso polido para evitar blocks
        logger.info(f"FBref: Coletando {fbref_leagues} para {seasons_str}")
        
        try:
            fb = sd.FBref(leagues=fbref_leagues, seasons=seasons_str)
            df = fb.read_schedule()
            df = df.reset_index()
        except Exception as e:
            logger.error(f"Erro no FBref: {e}")
            return pd.DataFrame()

        if "score" in df.columns:
            df = df.dropna(subset=["score"])

        rows = []
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row["date"])
                # Extrai ano da string de temporada "2425" ou similar
                # O soccerdata as vezes retorna "2024-2025" ou "2425"
                s_str = str(row["season"])
                season_yr = int(s_str[:2]) + 2000 if len(s_str) == 4 else int(s_str[:4])
                
                rows.append({
                    "fbref_id": hash(f"{row['home_team']}{row['away_team']}{dt.date()}{row['league']}"),
                    "date": dt,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_xG": float(row["home_xg"]) if pd.notna(row["home_xg"]) else float("nan"),
                    "away_xG": float(row["away_xg"]) if pd.notna(row["away_xg"]) else float("nan"),
                    "season": season_yr,
                    "league_key": row["league"]
                })
            except Exception:
                continue

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ============================================================================
# DATA EXTRACTOR PRINCIPAL
# ============================================================================

class DataExtractor:
    """Orquestra a extração multi-liga combinando xG e Odds."""

    def __init__(self):
        self.match_history = MatchHistoryReader()
        self.fbref = FBrefReader()

    def fetch_rich_dataset(
        self,
        start_year: int,
        end_year: int,
        target_leagues: list[str] = ["soccer_epl"]
    ) -> pd.DataFrame:
        """Pipeline muli-liga robusto."""
        logger.info("=" * 60)
        logger.info(f"EXTRACAO MULTI-LIGA: {target_leagues}")
        logger.info("=" * 60)

        all_leagues_data = []

        for league_id in target_leagues:
            if league_id not in LEAGUE_MAPPING:
                logger.warning(f"Liga {league_id} nao mapeada. Pulando.")
                continue
            
            mapping = LEAGUE_MAPPING[league_id]
            fb_name = mapping['fbref']
            fd_code = mapping['fd_code']

            logger.info(f"\n[LIGA] {league_id} | FBref: {fb_name} | FD: {fd_code}")

            try:
                # 1. FBref (xG)
                df_xg = self.fbref.read_leagues(start_year, end_year, [fb_name])
                
                # 2. MatchHistory (Odds)
                frames_fd = []
                if fd_code:
                    for yr in range(start_year, end_year + 1):
                        frames_fd.append(self.match_history.read_season(yr, fd_code))
                
                df_odds = pd.concat(frames_fd) if frames_fd else pd.DataFrame()

                # 3. Merge por liga
                if df_xg.empty:
                    logger.warning(f"Sem dados FBref para {league_id}")
                    continue

                if df_odds.empty:
                    logger.info(f"Sem odds (FD) para {league_id} - Mantendo apenas xG")
                    df_league = df_xg.copy()
                    # Garante que as colunas de odds existam como NaN
                    for col in MATCH_HIST_COLS.values():
                        if col not in df_league.columns:
                            df_league[col] = float("nan")
                    df_league["id"] = df_league["fbref_id"]
                else:
                    # Normalizacao para merge
                    for df in [df_odds, df_xg]:
                        df["_h"] = df["home_team"].apply(lambda x: _TEAM_NAME_MAP.get(x, x))
                        df["_a"] = df["away_team"].apply(lambda x: _TEAM_NAME_MAP.get(x, x))
                        df["_d"] = df["date"].dt.date
                    
                    df_league = pd.merge(
                        df_xg, 
                        df_odds.drop(columns=["home_team", "away_team", "date", "season", "league"]),
                        on=["_d", "_h", "_a"],
                        how="left"
                    )
                    df_league["id"] = df_league["fbref_id"]
                    df_league = df_league.drop(columns=["_h", "_a", "_d"])

                all_leagues_data.append(df_league)
                logger.info(f"Sucesso: {len(df_league)} partidas processadas para {league_id}")

            except Exception as e:
                logger.error(f"Falha critica na liga {league_id}: {e}")
                continue

        if not all_leagues_data:
            return pd.DataFrame()

        df_final = pd.concat(all_leagues_data, ignore_index=True)
        # Limpeza final
        drop_internal = ["fbref_id", "league_key"]
        df_final = df_final.drop(columns=[c for c in drop_internal if c in df_final.columns])
        
        logger.info(f"\nEXTRACAO FINALIZADA: {len(df_final)} partidas totais.")
        return df_final.sort_values("date").reset_index(drop=True)
