import logging
import sys

from .extractor import DataExtractor
from .feature_engineering import add_ewma_features
from .models import create_tables, get_engine, get_session
from .persistence import DataPersister, FeaturePersister

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("Main")


def main():
    logger.info("Starting Apo$tei Data Pipeline (Hybrid: soccerdata + Understat)...")

    # 1. Setup Database
    logger.info("Initializing Database...")
    db_path = "sqlite:///understat_premier_league.db"
    engine = get_engine(db_path)
    create_tables(engine)
    Session = lambda: get_session(engine)

    # 2. Components
    extractor = DataExtractor()
    persister = DataPersister(Session)
    feature_persister = FeaturePersister(Session)

    # 3. Define Scope
    START_YEAR = 2020
    END_YEAR = 2024

    logger.info(f"Target: EPL, Seasons: {START_YEAR}-{END_YEAR}")

    try:
        # ── Step 1: Extract (Hybrid) ───────────────────────────────────────
        logger.info("Step 1: Extracting data (MatchHistory + Understat)...")
        df_rich = extractor.fetch_rich_dataset(START_YEAR, END_YEAR)

        if df_rich.empty:
            logger.warning("No data extracted. Exiting.")
            return

        logger.info(f"Extracted {len(df_rich)} matches total.")

        # ── Step 2: Load (Upsert rich matches with xG + odds) ─────────────
        logger.info("Step 2: Persisting rich match data (xG + odds)...")
        saved_count = persister.save_matches(df_rich)
        logger.info(f"Saved/updated {saved_count} match records.")

        # ── Step 3: Feature Engineering ────────────────────────────────────
        logger.info("Step 3: Computing EWMA rolling features...")

        # 3a. Carrega todos os jogos do banco em ordem cronológica
        df_matches = persister.load_as_dataframe(engine)

        # 3b. Aplica EWMA com shift(1) — sem data leakage
        df_features = add_ewma_features(df_matches)

        # 3c. Persiste as features na tabela match_features
        features_saved = feature_persister.save_features(df_features)
        logger.info(f"Rolling features salvas: {features_saved} registros em match_features.")

        # ── Summary ────────────────────────────────────────────────────────
        n_odds = df_rich["odds_home_avg"].notna().sum() if "odds_home_avg" in df_rich.columns else 0
        n_xg = df_rich["home_xG"].notna().sum() if "home_xG" in df_rich.columns else 0
        logger.info("=" * 60)
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"  Total matches: {saved_count}")
        logger.info(f"  With xG data:  {n_xg}")
        logger.info(f"  With odds:     {n_odds}")
        logger.info(f"  With features: {features_saved}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
