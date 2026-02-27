import logging
import math
from datetime import datetime

import pandas as pd

from .models import Match, MatchFeatures

logger = logging.getLogger(__name__)

# Colunas de features que serão persistidas na tabela match_features
_FEATURE_COLS = [
    "ewma5_xg_pro_home",
    "ewma10_xg_pro_home",
    "ewma5_xg_con_home",
    "ewma10_xg_con_home",
    "ewma5_xg_pro_away",
    "ewma10_xg_pro_away",
    "ewma5_xg_con_away",
    "ewma10_xg_con_away",
]

# Colunas de odds opcionais para persistencia
_ODDS_COLS = [
    "odds_home_b365", "odds_draw_b365", "odds_away_b365",
    "odds_home_pin",  "odds_draw_pin",  "odds_away_pin",
    "odds_home_avg",  "odds_draw_avg",  "odds_away_avg",
    "odds_home_max",  "odds_draw_max",  "odds_away_max",
]

_STATS_COLS = [
    "home_shots", "away_shots",
    "home_shots_target", "away_shots_target",
]


class DataPersister:
    def __init__(self, session_factory):
        self.Session = session_factory

    def save_matches(self, matches):
        """
        Persists match data using session.merge() — a database-agnostic upsert
        strategy that works on any SQLite version.
        If a match ID already exists the row is updated; otherwise inserted.

        Aceita dois formatos:
          1. list[tuple[int, dict]] — formato Understat (season_year, match_data)
          2. pd.DataFrame — formato rico (rich dataset do extractor hibrido)
        """
        if isinstance(matches, pd.DataFrame):
            return self._save_from_dataframe(matches)
        return self._save_from_understat(matches)

    def _save_from_understat(self, matches: list) -> int:
        """Salva no formato antigo Understat: list[(season, dict)]."""
        session = self.Session()
        processed_count = 0

        try:
            for season_year, match_data in matches:
                if not match_data.get("id"):
                    continue

                match_dt = datetime.strptime(match_data["datetime"], "%Y-%m-%d %H:%M:%S")

                match_obj = Match(
                    id=int(match_data["id"]),
                    league="EPL",
                    season=season_year,
                    date=match_dt,
                    home_team=match_data["h"]["title"],
                    away_team=match_data["a"]["title"],
                    home_goals=int(match_data["goals"]["h"]),
                    away_goals=int(match_data["goals"]["a"]),
                    home_xG=float(match_data["xG"]["h"]),
                    away_xG=float(match_data["xG"]["a"]),
                    is_result=bool(match_data["isResult"]),
                )

                session.merge(match_obj)
                processed_count += 1

                if processed_count % 100 == 0:
                    session.flush()

            session.commit()
            logger.info(f"Committed {processed_count} match records (Understat).")
            return processed_count

        except Exception as e:
            logger.error(f"Error during persistence: {e}")
            session.rollback()
            raise e
        finally:
            session.close()

    def _save_from_dataframe(self, df: pd.DataFrame) -> int:
        """Salva a partir do DataFrame rico (hibrido: xG + odds)."""
        session = self.Session()
        processed_count = 0

        try:
            for _, row in df.iterrows():
                match_obj = Match(
                    id=int(row["id"]),
                    league=row.get("league", "EPL"),
                    season=int(row["season"]),
                    date=pd.Timestamp(row["date"]).to_pydatetime(),
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    home_goals=int(row["home_goals"]),
                    away_goals=int(row["away_goals"]),
                    home_xG=_safe_float(row.get("home_xG")),
                    away_xG=_safe_float(row.get("away_xG")),
                    is_result=True,
                    # Stats
                    home_shots=_safe_float(row.get("home_shots")),
                    away_shots=_safe_float(row.get("away_shots")),
                    home_shots_target=_safe_float(row.get("home_shots_target")),
                    away_shots_target=_safe_float(row.get("away_shots_target")),
                    # Odds
                    odds_home_b365=_safe_float(row.get("odds_home_b365")),
                    odds_draw_b365=_safe_float(row.get("odds_draw_b365")),
                    odds_away_b365=_safe_float(row.get("odds_away_b365")),
                    odds_home_pin=_safe_float(row.get("odds_home_pin")),
                    odds_draw_pin=_safe_float(row.get("odds_draw_pin")),
                    odds_away_pin=_safe_float(row.get("odds_away_pin")),
                    odds_home_avg=_safe_float(row.get("odds_home_avg")),
                    odds_draw_avg=_safe_float(row.get("odds_draw_avg")),
                    odds_away_avg=_safe_float(row.get("odds_away_avg")),
                    odds_home_max=_safe_float(row.get("odds_home_max")),
                    odds_draw_max=_safe_float(row.get("odds_draw_max")),
                    odds_away_max=_safe_float(row.get("odds_away_max")),
                )

                session.merge(match_obj)
                processed_count += 1

                if processed_count % 100 == 0:
                    session.flush()

            session.commit()
            logger.info(f"Committed {processed_count} match records (rich dataset).")
            return processed_count

        except Exception as e:
            logger.error(f"Error during persistence: {e}")
            session.rollback()
            raise e
        finally:
            session.close()

    def load_as_dataframe(self, engine) -> pd.DataFrame:
        """
        Carrega todos os jogos da tabela `matches` em um DataFrame Pandas,
        ordenado por data (necessário para o cálculo correto das EWMAs).
        """
        query = "SELECT id, date, home_team, away_team, home_xG, away_xG FROM matches ORDER BY date ASC"
        df = pd.read_sql(query, engine, parse_dates=["date"])
        logger.info(f"Carregados {len(df)} jogos do banco de dados.")
        return df

    def load_rich_dataframe(self, engine) -> pd.DataFrame:
        """
        Carrega todos os jogos com todas as colunas (xG + odds + stats).
        """
        query = "SELECT * FROM matches ORDER BY date ASC"
        df = pd.read_sql(query, engine, parse_dates=["date"])
        logger.info(f"Carregados {len(df)} jogos (rico) do banco de dados.")
        return df


class FeaturePersister:
    """
    Responsável por salvar as Rolling Features na tabela `match_features`.
    Usa upsert via session.merge() — seguro para re-execuções do pipeline.
    """

    def __init__(self, session_factory):
        self.Session = session_factory

    def save_features(self, df: pd.DataFrame) -> int:
        """
        Recebe o DataFrame enriquecido (saída de add_ewma_features) e persiste
        as colunas de features na tabela match_features.

        Apenas jogos com ao menos uma feature não-nula são salvos,
        descartando os primeiros jogos de cada time (NaN por falta de histórico).
        """
        session = self.Session()
        saved_count = 0

        # Filtra linhas que tenham ao menos 1 feature calculada
        mask = df[_FEATURE_COLS].notna().any(axis=1)
        df_valid = df[mask].copy()

        logger.info(f"Persistindo features para {len(df_valid)}/{len(df)} jogos...")

        try:
            for _, row in df_valid.iterrows():
                feature_obj = MatchFeatures(
                    match_id=int(row["id"]),
                    ewma5_xg_pro_home=_safe_float(row.get("ewma5_xg_pro_home")),
                    ewma10_xg_pro_home=_safe_float(row.get("ewma10_xg_pro_home")),
                    ewma5_xg_con_home=_safe_float(row.get("ewma5_xg_con_home")),
                    ewma10_xg_con_home=_safe_float(row.get("ewma10_xg_con_home")),
                    ewma5_xg_pro_away=_safe_float(row.get("ewma5_xg_pro_away")),
                    ewma10_xg_pro_away=_safe_float(row.get("ewma10_xg_pro_away")),
                    ewma5_xg_con_away=_safe_float(row.get("ewma5_xg_con_away")),
                    ewma10_xg_con_away=_safe_float(row.get("ewma10_xg_con_away")),
                )
                session.merge(feature_obj)
                saved_count += 1

                if saved_count % 200 == 0:
                    session.flush()

            session.commit()
            logger.info(f"Features salvas/atualizadas: {saved_count} registros.")
            return saved_count

        except Exception as e:
            logger.error(f"Erro ao salvar features: {e}")
            session.rollback()
            raise
        finally:
            session.close()


def _safe_float(value):
    """Converte para float, retornando None em caso de NaN ou ausência."""
    try:
        import math
        return None if value is None or math.isnan(float(value)) else float(value)
    except (TypeError, ValueError):
        return None
