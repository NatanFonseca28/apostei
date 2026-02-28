import logging
import math
from datetime import datetime
import pandas as pd

from .models import FlashscoreMatch

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
                    melhor_chance=str(row["melhor_chance"])
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
