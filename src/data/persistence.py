import logging
import math

import pandas as pd

from .models import FlashscoreMatch, MatchAdvancedStats

logger = logging.getLogger(__name__)


class DataPersister:
    def __init__(self, session_factory):
        self.Session = session_factory

    def save_matches(self, df: pd.DataFrame) -> int:
        """
        Salva os dados extraídos do Flashscore no SQLite, fazendo merge pelo ID.
        """
        session = self.Session()
        processed_count = 0

        try:
            for _, row in df.iterrows():
                # Tratando nans para null do banco
                def _s(val):
                    if pd.isna(val):
                        return None
                    if isinstance(val, float) and math.isnan(val):
                        return None
                    return val

                match_dt = row["data"].to_pydatetime() if pd.notnull(row["data"]) else None

                match_obj = FlashscoreMatch(
                    id=str(row["id"]),
                    campeonato=str(row["campeonato"]),
                    status=str(row["status"]),
                    data=match_dt,
                    placar_casa=int(row["placar_casa"]) if pd.notna(row["placar_casa"]) else None,
                    placar_fora=int(row["placar_fora"]) if pd.notna(row["placar_fora"]) else None,
                    time_casa=str(row["time_casa"]),
                    jogos_casa=int(row["jogos_casa"]) if pd.notna(row["jogos_casa"]) else 0,
                    gols_marcados_casa=int(row["gols_marcados_casa"]) if pd.notna(row["gols_marcados_casa"]) else 0,
                    gols_sofridos_casa=int(row["gols_sofridos_casa"]) if pd.notna(row["gols_sofridos_casa"]) else 0,
                    media_marcados_casa=float(row["media_marcados_casa"]) if pd.notna(row["media_marcados_casa"]) else 0.0,
                    media_sofridos_casa=float(row["media_sofridos_casa"]) if pd.notna(row["media_sofridos_casa"]) else 0.0,
                    time_fora=str(row["time_fora"]),
                    jogos_fora=int(row["jogos_fora"]) if pd.notna(row["jogos_fora"]) else 0,
                    gols_marcados_fora=int(row["gols_marcados_fora"]) if pd.notna(row["gols_marcados_fora"]) else 0,
                    gols_sofridos_fora=int(row["gols_sofridos_fora"]) if pd.notna(row["gols_sofridos_fora"]) else 0,
                    media_marcados_fora=float(row["media_marcados_fora"]) if pd.notna(row["media_marcados_fora"]) else 0.0,
                    media_sofridos_fora=float(row["media_sofridos_fora"]) if pd.notna(row["media_sofridos_fora"]) else 0.0,
                    ofensividade_casa=str(row["ofensividade_casa"]),
                    defensividade_casa=str(row["defensividade_casa"]),
                    ofensividade_fora=str(row["ofensividade_fora"]),
                    defensividade_fora=str(row["defensividade_fora"]),
                    probabilidade_gol=str(row["probabilidade_gol"]),
                    melhor_chance=str(row["melhor_chance"]),
                )

                session.merge(match_obj)
                processed_count += 1

                if processed_count % 100 == 0:
                    session.flush()

            session.commit()
            logger.info(f"Committed {processed_count} Flashscore match records.")
            return processed_count

        except Exception as e:
            logger.error(f"Error during persistence: {e}")
            session.rollback()
            raise e
        finally:
            session.close()

    def load_as_dataframe(self, engine) -> pd.DataFrame:
        """
        Carrega todos os jogos do Flashscore do banco para DataFrames (ex: para ML).
        """
        query = "SELECT * FROM flashscore_matches ORDER BY data ASC"
        df = pd.read_sql(query, engine, parse_dates=["data"])
        logger.info(f"Carregados {len(df)} jogos (Flashscore) do banco de dados.")
        return df

    def save_advanced_stats(self, df: pd.DataFrame) -> int:
        """
        Persiste estatísticas avançadas coletadas pelo FBrefCollector na tabela
        `match_advanced_stats`, usando UPSERT por (home_team, away_team, date).

        Parâmetros
        ----------
        df : pd.DataFrame
            Saída de fbref_collector.collect_advanced_stats().
            Colunas esperadas: home_team, away_team, date, league, season,
            home_xg, away_xg, home_shots_target, away_shots_target,
            home_possession, away_possession.

        Retorna
        -------
        int — número de registros processados (novos + atualizados).
        """
        required_cols = {
            "home_team",
            "away_team",
            "date",
            "home_xg",
            "away_xg",
            "home_shots_target",
            "away_shots_target",
            "home_possession",
            "away_possession",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no DataFrame: {missing}")

        session = self.Session()
        processed = 0

        try:
            for _, row in df.iterrows():
                row_date = row["date"].date() if hasattr(row["date"], "date") else row["date"]

                # Tenta encontrar registro existente pela chave única
                existing = (
                    session.query(MatchAdvancedStats)
                    .filter_by(
                        home_team=str(row["home_team"]),
                        away_team=str(row["away_team"]),
                        date=row_date,
                    )
                    .first()
                )

                def _f(val, default: float = 0.0) -> float:
                    try:
                        v = float(val)
                        return v if v == v else default  # NaN check
                    except (TypeError, ValueError):
                        return default

                if existing:
                    existing.home_xg = _f(row.get("home_xg"))
                    existing.away_xg = _f(row.get("away_xg"))
                    existing.home_shots_target = _f(row.get("home_shots_target"))
                    existing.away_shots_target = _f(row.get("away_shots_target"))
                    existing.home_possession = _f(row.get("home_possession"))
                    existing.away_possession = _f(row.get("away_possession"))
                    if pd.notna(row.get("league")):
                        existing.league = str(row["league"])
                    if pd.notna(row.get("season")):
                        existing.season = str(row["season"])
                else:
                    obj = MatchAdvancedStats(
                        home_team=str(row["home_team"]),
                        away_team=str(row["away_team"]),
                        date=row_date,
                        league=str(row.get("league", "") or ""),
                        season=str(row.get("season", "") or ""),
                        home_xg=_f(row.get("home_xg")),
                        away_xg=_f(row.get("away_xg")),
                        home_shots_target=_f(row.get("home_shots_target")),
                        away_shots_target=_f(row.get("away_shots_target")),
                        home_possession=_f(row.get("home_possession")),
                        away_possession=_f(row.get("away_possession")),
                        source="fbref",
                    )
                    session.add(obj)

                processed += 1
                if processed % 100 == 0:
                    session.flush()

            session.commit()
            logger.info(f"[MatchAdvancedStats] {processed} registros persistidos (novos + atualizados).")
            return processed

        except Exception as e:
            logger.error(f"Erro ao persistir MatchAdvancedStats: {e}")
            session.rollback()
            raise
        finally:
            session.close()
