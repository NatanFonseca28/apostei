import logging
import sys
from src.ml.optimizer import run_optimization
from src.data.models import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

if __name__ == "__main__":
    engine = get_engine("sqlite:///flashscore_data.db")
    run_optimization(engine, n_trials=50) # fast test
