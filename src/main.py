import logging
import sys

from src.data.extractor import FlashscoreExtractor
from src.data.models import create_tables, get_engine, get_session
from src.data.persistence import DataPersister

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Main")

URLS = [
    "https://www.flashscore.com.br/futebol/espanha/laliga/#/dINOZk9Q/table/overall",
    "https://www.flashscore.com.br/futebol/inglaterra/premier-league/#/dINOZk9Q/table/overall",
    "https://www.flashscore.com.br/futebol/italia/serie-a/#/dINOZk9Q/table/overall",
    "https://www.flashscore.com.br/futebol/franca/ligue-1/#/WYO1P5ch/table/overall",
    "https://www.flashscore.com.br/futebol/alemanha/bundesliga/#/nZEwjkNa/table/overall",
    # "https://www.flashscore.com.br/futebol/arabia-saudita/divisao-1/#/IeULGOgm/table/overall",
    # "https://www.flashscore.com.br/futebol/espanha/laliga2/#/YwFjV9Hs/table/overall",
    # "https://www.flashscore.com.br/futebol/grecia/superliga/#/Y5PJdcC3/table/overall",
    # "https://www.flashscore.com.br/futebol/indonesia/liga-1/#/IoZQW2N8/table/overall",
    # "https://www.flashscore.com.br/futebol/israel/ligat-ha-al/#/ltpEOpgd/table/overall",
    # "https://www.flashscore.com.br/futebol/turquia/super-lig/#/jy5jZF2o/table/overall",
    "https://www.flashscore.com.br/futebol/europa/liga-dos-campeoes/#/2oN82Fw5/table/overall",
    # "https://www.flashscore.com.br/futebol/azerbaijao/primeira-liga/#/tOIX7uHc/table/overall",
    # "https://www.flashscore.com.br/futebol/asia/liga-dos-campeoes-da-afc/#/4hWTXH4p/table/overall",
    # "https://www.flashscore.com.br/futebol/egito/primeira-liga/#/GleLtG64/table/overall",
    # "https://www.flashscore.com.br/futebol/inglaterra/2-divisao/#/2DgLvevA/table/overall",
    # "https://www.flashscore.com.br/futebol/india/liga-indiana/#/MNsj6QO9/table/overall",
    # FOCO BRASIL, ESTADUAIS e SUL-AMERICA
    "https://www.flashscore.com.br/futebol/brasil/brasileirao-betano/#/6FygD4mG/table/overall",
    "https://www.flashscore.com.br/futebol/america-do-sul/copa-libertadores/",
    "https://www.flashscore.com.br/futebol/america-do-sul/copa-sul-americana/",
    # "https://www.flashscore.com.br/futebol/brasil/acreano",
    # "https://www.flashscore.com.br/futebol/brasil/alagoano",
    # "https://www.flashscore.com.br/futebol/brasil/amapaense",
    # "https://www.flashscore.com.br/futebol/brasil/amazonense",
    # "https://www.flashscore.com.br/futebol/brasil/baiano",
    # "https://www.flashscore.com.br/futebol/brasil/brasiliense",
    # "https://www.flashscore.com.br/futebol/brasil/capixabao-superbet",
    # "https://www.flashscore.com.br/futebol/brasil/carioca-superbet",
    # "https://www.flashscore.com.br/futebol/brasil/catarinense",
    # "https://www.flashscore.com.br/futebol/brasil/cearense-superbet",
    # "https://www.flashscore.com.br/futebol/brasil/gauchao-superbet",
    # "https://www.flashscore.com.br/futebol/brasil/goiano",
    # "https://www.flashscore.com.br/futebol/brasil/maranhense",
    # "https://www.flashscore.com.br/futebol/brasil/matogrossense",
    # "https://www.flashscore.com.br/futebol/brasil/mineiro",
    # "https://www.flashscore.com.br/futebol/brasil/paraense",
    # "https://www.flashscore.com.br/futebol/brasil/paraibano",
    # "https://www.flashscore.com.br/futebol/brasil/paranaense",
    # "https://www.flashscore.com.br/futebol/brasil/paulista",
    # "https://www.flashscore.com.br/futebol/brasil/pernambucano",
    # "https://www.flashscore.com.br/futebol/brasil/piauiense",
    # "https://www.flashscore.com.br/futebol/brasil/potiguar",
    # "https://www.flashscore.com.br/futebol/brasil/rondoniense",
    # "https://www.flashscore.com.br/futebol/brasil/roraimense",
    # "https://www.flashscore.com.br/futebol/brasil/sergipano-superbet",
    # "https://www.flashscore.com.br/futebol/brasil/sul-matogrossense",
    # "https://www.flashscore.com.br/futebol/brasil/tocantinense",
]


def main():
    logger.info("Starting Apo$tei Data Pipeline (Flashscore)...")

    logger.info("Initializing Database...")
    db_path = "sqlite:///flashscore_data.db"
    engine = get_engine(db_path)
    create_tables(engine)
    Session = lambda: get_session(engine)

    extractor = FlashscoreExtractor()
    persister = DataPersister(Session)

    try:
        logger.info("Step 1: Extracting data (Flashscore)...")
        df_rich = extractor.fetch_data(URLS)

        if df_rich.empty:
            logger.warning("No data extracted. Exiting.")
            return

        logger.info("Step 2: Persisting match data...")
        saved_count = persister.save_matches(df_rich)

        # TODO: ML and Predictive Analysis features mapped to Flashscore metrics
        logger.info("Step 3: Feature Engineering and ML predictions are pending adaptation to new schema.")

        # Save to raw csv as user requested in original script
        df_rich.to_csv("dados_futebol.csv", index=False, encoding="utf-8")
        logger.info("Saved raw CSV to dados_futebol.csv")

        logger.info("=" * 60)
        logger.info(f"Pipeline completed successfully! Total matches saved: {saved_count}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
