"""
run_optimize.py
---------------
Script de otimização Bayesiana de hiperparâmetros.

Executa:
  1. Feature Selection automatizada (SelectFromModel + correlação)
  2. Optuna com TimeSeriesSplit (sem data leakage)
  3. Salva o melhor modelo em artifacts/best_model_*.pkl
  4. Exporta histórico de trials em CSV

Pré-requisito: banco populado via run_etl.py

Uso:
    python run_optimize.py                         # 100 trials (padrão)
    python run_optimize.py --trials 200            # mais trials
    python run_optimize.py --trials 50 --timeout 300  # máx 5 min
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.models import get_engine
from src.ml.optimizer import run_optimization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Silencia logs verbosos do Optuna (mantém apenas warnings+)
logging.getLogger("optuna").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Otimização Bayesiana de hiperparâmetros com Optuna")
    parser.add_argument("--trials", type=int, default=100, help="Número de trials do Optuna (padrão: 100)")
    parser.add_argument("--splits", type=int, default=5, help="Número de folds do TimeSeriesSplit (padrão: 5)")
    parser.add_argument("--max-features", type=int, default=15, help="Máximo de features após seleção (padrão: 15)")
    parser.add_argument("--timeout", type=int, default=None, help="Tempo máximo em segundos (padrão: sem limite)")
    parser.add_argument("--db", type=str, default="sqlite:///flashscore_data.db", help="Caminho do banco SQLite")

    args = parser.parse_args()

    engine = get_engine(args.db)

    result = run_optimization(
        engine=engine,
        n_trials=args.trials,
        n_splits=args.splits,
        max_features=args.max_features,
        timeout=args.timeout,
    )

    # Resumo final no console
    print("\n" + "=" * 60)
    print("  RESULTADO FINAL")
    print("=" * 60)
    print(f"  Melhor Log Loss (CV): {result['best_log_loss']:.4f}")
    print(f"  Modelo:               {result['best_params']['model_type']}")
    print(f"  Features:             {len(result['selected_features'])}")
    print(f"  Artefato:             {result['model_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
