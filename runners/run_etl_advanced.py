"""
run_etl_advanced.py
-------------------
Coleta estatísticas avançadas (xG, chutes no alvo, posse) via Sofascore —
sem autenticação, sem limite rígido de requisições — e persiste na tabela
`match_advanced_stats` do banco SQLite.

Este runner é independente do ETL principal (run_etl.py). Deve ser executado
após o ETL do Flashscore já ter populado `flashscore_matches`.

Ligas suportadas:
    BRA-Serie A           (Brasileirão Série A)
    ENG-Premier League    (Premier League)
    ESP-La Liga           (La Liga)
    FRA-Ligue 1           (Ligue 1)
    UEFA-Champions League (Champions League)

Não é necessária nenhuma chave de API. O soccerdata gerencia o cache
das schedules em ~/soccerdata/data/Sofascore/ e as stats por partida
são cacheadas em ~/.apostei/cache/sofascore/.

Uso:
    python runners/run_etl_advanced.py
    python runners/run_etl_advanced.py --leagues "ENG-Premier League" --seasons 2024 2025
    python runners/run_etl_advanced.py --leagues "UEFA-Champions League" --seasons 2024
    python runners/run_etl_advanced.py --no-cache
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.models import create_tables, get_engine, get_session
from src.data.persistence import DataPersister
from src.data.sofascore_collector import LEAGUE_MAP, collect_advanced_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    leagues: list[str] | None = None,
    seasons: list[int] | None = None,
    db_path: str = "sqlite:///flashscore_data.db",
    no_cache: bool = False,
) -> None:
    """
    Pipeline completo: coleta Sofascore → persiste em match_advanced_stats.

    Parâmetros
    ----------
    leagues : list[str] | None
        Ligas a coletar. None = apenas BRA-Serie A.
        Opções: "BRA-Serie A", "ENG-Premier League", "ESP-La Liga",
                "FRA-Ligue 1", "UEFA-Champions League"
    seasons : list[int] ou None
        Ex: [2025, 2026]. None = duas temporadas mais recentes.
    db_path : str
        URL de conexão SQLite.
    no_cache : bool
        Se True, força nova requisição ignorando cache em disco.
    """
    if not leagues:
        leagues = ["BRA-Serie A"]

    logger.info("=" * 65)
    logger.info("  ETL AVANÇADO — Sofascore")
    logger.info("=" * 65)
    logger.info(f"  Ligas   : {leagues}")
    logger.info(f"  Seasons : {seasons or 'últimas disponíveis'}")
    logger.info(f"  Banco   : {db_path}")
    logger.info(f"  No cache: {no_cache}")
    logger.info("-" * 65)

    import pandas as pd

    # 1. Coleta — itera pelas ligas e concatena
    frames: list[pd.DataFrame] = []
    for league in leagues:
        logger.info(f"\n>>> Liga: {league}")
        df_league = collect_advanced_stats(league=league, seasons=seasons, no_cache=no_cache)
        if df_league.empty:
            logger.warning(f"  Nenhuma partida retornada para {league}.")
        else:
            frames.append(df_league)

    if not frames:
        logger.warning("Nenhuma partida retornada pela API. Encerrando.")
        return

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"\nTotal coletado: {len(df)} partidas | xG>0: {(df['home_xg'] > 0).sum()} | Posse>0: {(df['home_possession'] > 0).sum()}")

    # 2. Persiste
    engine = get_engine(db_path)
    create_tables(engine)  # cria match_advanced_stats se não existir
    Session = lambda: get_session(engine)  # noqa: E731
    persister = DataPersister(session_factory=Session)
    count = persister.save_advanced_stats(df)

    logger.info("=" * 65)
    logger.info(f"  ETL avançado concluído — {count} registros no banco.")
    logger.info("=" * 65)
    logger.info("\nPróximo passo: execute 'python runners/run_trainer.py' para retreinar com os dados reais.")


if __name__ == "__main__":
    ligas_disponiveis = ", ".join(f"'{k}'" for k in LEAGUE_MAP)
    parser = argparse.ArgumentParser(
        description="Coleta stats avançadas via Sofascore (xG, SoT, Posse) e persiste no banco.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--leagues",
        type=str,
        nargs="+",
        default=None,
        metavar="LEAGUE",
        help=(f'Liga(s) a coletar (separadas por espaço). Padrão: BRA-Serie A\nOpções: {ligas_disponiveis}\nEx: --leagues "BRA-Serie A" "ENG-Premier League"'),
    )
    parser.add_argument(
        "--all-leagues",
        action="store_true",
        help="Coleta todas as ligas do LEAGUE_MAP. Equivalente a passar todas com --leagues.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Temporada(s) a coletar. Ex: --seasons 2025 2026",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="sqlite:///flashscore_data.db",
        help="Caminho do banco SQLite.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignora cache local e força nova requisição à API.",
    )
    args = parser.parse_args()

    selected_leagues = list(LEAGUE_MAP.keys()) if args.all_leagues else args.leagues

    run(
        leagues=selected_leagues,
        seasons=args.seasons,
        db_path=args.db,
        no_cache=args.no_cache,
    )
