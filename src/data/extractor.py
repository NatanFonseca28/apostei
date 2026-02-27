"""
extractor.py
------------
Modulo de extracao de dados hibrido:

  1. soccerdata / football-data.co.uk (MatchHistory)
     Fornece: resultados, chutes, cartoes, odds reais de multiplas casas
     (Bet365, Pinnacle, Betfair, media do mercado, etc.).
     Corrige bug de compatibilidade pandas 3.x (BOM UTF-8 no CSV).

  2. Understat (API AJAX direta)
     Fornece: xG (Expected Goals) por partida -- essencial para as features
     EWMA que alimentam o modelo de predicao.

O pipeline une ambas as fontes por (data, home_team, away_team), produzindo
um dataset rico com xG + odds reais de mercado.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

UNDERSTAT_BASE_URL = "https://understat.com"
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
# UNDERSTAT READER (xG)
# ============================================================================

class UnderstatReader:
    """
    Extrai xG de partidas da EPL via API AJAX do Understat.
    Usa requests sincrono com retry + exponential backoff.
    """

    def __init__(self, retries: int = 4, backoff_factor: int = 2):
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        self.session.headers.update({
            **DEFAULT_HEADERS,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://understat.com/league/EPL",
        })

    def fetch_season(self, season: int) -> list[dict]:
        """Busca resultados de uma temporada. Retorna lista de dicts."""
        url = f"{UNDERSTAT_BASE_URL}/getLeagueData/EPL/{season}"

        for attempt in range(self.retries):
            try:
                logger.info(
                    f"Understat EPL {season}/{season+1} "
                    f"(tentativa {attempt+1}/{self.retries})"
                )
                resp = self.session.get(url, timeout=30)

                if resp.status_code == 429:
                    wait = self.backoff_factor ** (attempt + 2)
                    logger.warning(f"Rate limited (429). Aguardando {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                payload = resp.json()

                matches = [m for m in payload.get("dates", []) if m.get("isResult")]
                logger.info(f"  -> {len(matches)} partidas finalizadas")
                return matches

            except requests.RequestException as e:
                logger.warning(f"Erro na tentativa {attempt+1}: {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_factor ** attempt)
                else:
                    logger.error(
                        f"Desistindo de EPL {season} apos {self.retries} tentativas."
                    )
                    raise
        return []

    def fetch_seasons_raw(self, start_year: int, end_year: int) -> list[tuple[int, dict]]:
        """
        Busca multiplas temporadas no formato antigo (compatibilidade).
        Retorna lista de (season_year, match_dict).
        """
        results: list[tuple[int, dict]] = []
        for year in range(start_year, end_year + 1):
            try:
                matches = self.fetch_season(year)
                results.extend((year, m) for m in matches)
                time.sleep(1.5)
            except Exception as e:
                logger.error(f"Temporada {year} falhou: {e}")
        return results

    def fetch_seasons_df(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Busca multiplas temporadas e retorna DataFrame com xG."""
        rows = []
        for year in range(start_year, end_year + 1):
            try:
                matches = self.fetch_season(year)
                for m in matches:
                    rows.append({
                        "understat_id": int(m["id"]),
                        "date": datetime.strptime(m["datetime"], "%Y-%m-%d %H:%M:%S"),
                        "home_team": m["h"]["title"],
                        "away_team": m["a"]["title"],
                        "home_goals": int(m["goals"]["h"]),
                        "away_goals": int(m["goals"]["a"]),
                        "home_xG": float(m["xG"]["h"]),
                        "away_xG": float(m["xG"]["a"]),
                        "season": year,
                    })
                time.sleep(1.5)
            except Exception as e:
                logger.error(f"Understat temporada {year} falhou: {e}")
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
    Extrator hibrido que combina:
      - football-data.co.uk (MatchHistory / soccerdata) -> resultados + odds
      - Understat -> xG

    O merge e feito por (date, home_team, away_team) apos normalizacao.
    """

    def __init__(self):
        self.match_history = MatchHistoryReader()
        self.understat = UnderstatReader()

    # -- Interface compativel com pipeline antigo --

    def fetch_multiple_seasons(
        self,
        league: str,
        start_year: int,
        end_year: int,
    ) -> list[tuple[int, dict]]:
        """
        Backward compatible: retorna lista de (season, match_dict)
        no formato Understat original para DataPersister.save_matches().
        """
        return self.understat.fetch_seasons_raw(start_year, end_year)

    # -- Interface nova: dataset rico com xG + odds --

    def fetch_rich_dataset(
        self,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        """
        Extrai e combina dados de AMBAS as fontes em um unico DataFrame.

        Colunas do resultado:
          - Metadados: id, date, season, home_team, away_team
          - Resultado: home_goals, away_goals, result
          - xG (Understat): home_xG, away_xG
          - Stats: home_shots, away_shots, home_shots_target, away_shots_target
          - Odds: odds_{home,draw,away}_{b365,pin,avg,max}
        """
        logger.info("=" * 60)
        logger.info("EXTRACAO HIBRIDA: MatchHistory + Understat")
        logger.info("=" * 60)

        # 1. football-data.co.uk: odds + stats
        logger.info("[1/3] Buscando MatchHistory (football-data.co.uk)...")
        df_odds = self.match_history.read_seasons(start_year, end_year)

        # 2. Understat: xG
        logger.info("[2/3] Buscando xG do Understat...")
        df_xg = self.understat.fetch_seasons_df(start_year, end_year)

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
            logger.warning("MatchHistory vazio -- usando apenas Understat")
            df_xg["id"] = df_xg["understat_id"]
            return df_xg.drop(columns=["_home_norm", "_away_norm", "_date_key"])

        if df_xg.empty:
            logger.warning("Understat vazio -- sem xG disponivel")
            df_odds["id"] = range(1, len(df_odds) + 1)
            df_odds["home_xG"] = float("nan")
            df_odds["away_xG"] = float("nan")
            return df_odds.drop(columns=["_home_norm", "_away_norm", "_date_key"])

        # Odds columns to merge (drop duplicates from df_odds)
        odds_cols = [c for c in df_odds.columns
                     if c not in ("home_team", "away_team", "date", "season",
                                  "league", "home_goals", "away_goals",
                                  "_home_norm", "_away_norm", "_date_key")]

        df_merged = pd.merge(
            df_xg,
            df_odds[["_date_key", "_home_norm", "_away_norm"] + odds_cols],
            on=["_date_key", "_home_norm", "_away_norm"],
            how="left",
        )

        df_merged["id"] = df_merged["understat_id"]

        drop_cols = ["_home_norm", "_away_norm", "_date_key", "understat_id"]
        df_merged = df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns])

        n_total = len(df_merged)
        n_odds = df_merged["odds_home_avg"].notna().sum() if "odds_home_avg" in df_merged.columns else 0
        logger.info(
            f"Merge concluido: {n_total} partidas, "
            f"{n_odds} com odds ({n_odds/max(n_total,1)*100:.0f}%)"
        )

        return df_merged.sort_values("date").reset_index(drop=True)
