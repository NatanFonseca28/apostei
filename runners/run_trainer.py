"""
run_trainer.py
--------------
Executa o pipeline de treinamento de ML de forma independente do ETL.
O banco de dados deve já ter sido populado via run_etl.py.

Uso:
    python run_trainer.py
"""

import logging
import sys

# Adiciona o diretório raiz ao path para importar src
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.models import get_engine
from src.ml.trainer import run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    engine = get_engine("sqlite:///understat_premier_league.db")
    run_training_pipeline(engine, n_splits=5)
