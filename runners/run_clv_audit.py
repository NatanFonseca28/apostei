"""
run_clv_audit.py
----------------
Script de auditoria CLV (Closing Line Value) para o sistema Apo$tei.

Modos de execucao:
    1. Backtest simulado: Simula apostas historicas usando odds do banco
    2. Auditoria de CSV: Audita apostas registradas em arquivo CSV
    3. Calculo rapido:   CLV de uma aposta individual
    4. Auditoria com modelo: Usa o modelo treinado real (sklearn)

Uso:
    python run_clv_audit.py                     # Backtest simulado completo
    python run_clv_audit.py --csv apostas.csv   # Audita CSV de apostas
    python run_clv_audit.py --quick 2.00 1.80   # Calculo rapido

Formato do CSV de apostas:
    match_id,date,home_team,away_team,outcome,odds_taken,model_prob,stake_pct
    20424,2024-09-14,Arsenal,Wolves,H,1.25,0.82,0.02
"""

import argparse
import logging
import sys, os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.clv import (
    BetRecord,
    CLVAuditor,
    CLVReport,
    ClosingSource,
    load_bets_from_csv,
    quick_clv_check,
)
from src.data.models import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CLV-Audit")


# ============================================================================
# MODOS DE EXECUCAO
# ============================================================================

def run_backtest_simulated(
    engine,
    source: ClosingSource,
    min_ev: float,
    seasons: list[int] | None = None,
):
    """
    Backtest simulado: identifica apostas historicas EV+ e mede CLV.

    Estrategia:
        - 'Odd apostada' = Bet365 (publicada cedo, proxy de abertura)
        - 'Odd de fechamento' = Pinnacle (sharp line, proxy do mercado eficiente)
        - O CLV mede se apostando na B365 antes do kickoff, o modelo
          teria capturado valor vs. o fechamento Pinnacle
    """
    auditor = CLVAuditor(engine)
    report = auditor.backtest_historical(
        source=source,
        min_ev=min_ev,
        seasons=seasons,
    )

    print(report)

    # Mostra amostra das apostas individuais
    print_sample(report, n=15)

    return report


def run_backtest_with_model(
    engine,
    source: ClosingSource,
    min_ev: float,
    seasons: list[int] | None = None,
):
    """
    Auditoria CLV usando o modelo treinado real.

    Carrega o modelo salvo (se existir) e executa predicoes sobre dados
    historicos para identificar apostas EV+ e medir CLV.
    """
    import joblib

    model_path = Path("artifacts/model.pkl")
    scaler_path = Path("artifacts/scaler.pkl")

    if not model_path.exists() or not scaler_path.exists():
        logger.warning(
            "Modelo treinado nao encontrado em artifacts/. "
            "Execute 'python run_trainer.py' primeiro, ou use o backtest simulado."
        )
        logger.info("Executando backtest simulado como fallback...")
        return run_backtest_simulated(engine, source, min_ev, seasons)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info(f"Modelo carregado: {type(model).__name__}")

    auditor = CLVAuditor(engine)
    report = auditor.audit_with_model(
        model=model,
        scaler=scaler,
        source=source,
        min_ev=min_ev,
        seasons=seasons,
    )

    print(report)
    print_sample(report, n=15)

    return report


def run_csv_audit(
    engine,
    csv_path: str,
    source: ClosingSource,
):
    """Audita apostas registradas em arquivo CSV."""
    bets = load_bets_from_csv(csv_path)

    if not bets:
        logger.warning("Nenhuma aposta encontrada no CSV.")
        return None

    auditor = CLVAuditor(engine)
    report = auditor.audit_bets(bets, source)

    print(report)
    print_sample(report, n=20)

    return report


def run_quick_check(odds_taken: float, odds_closing: float, label: str = ""):
    """Calculo rapido de CLV para uma aposta individual."""
    quick_clv_check(odds_taken, odds_closing, label)


def run_demo(engine):
    """
    Demo interativa: mostra exemplos de CLV com apostas sinteticas
    baseadas em jogos reais do banco de dados.
    """
    import pandas as pd

    logger.info("=" * 60)
    logger.info("DEMO: CLV (Closing Line Value) — Auditoria de Apostas")
    logger.info("=" * 60)

    # 1. Exemplos rapidos de CLV
    print("\n" + "─" * 60)
    print("  EXEMPLOS RÁPIDOS DE CLV")
    print("─" * 60)

    examples = [
        (2.00, 1.80, "Arsenal ML — Capturou valor cedo"),
        (2.00, 2.20, "Chelsea ML — Mercado moveu contra"),
        (3.50, 3.50, "Empate — Neutro (sem CLV)"),
        (1.50, 1.35, "Man City ML — Edge significativo"),
        (4.00, 5.00, "Underdog — Linha subiu, CLV negativo"),
    ]

    for taken, closing, label in examples:
        quick_clv_check(taken, closing, label)

    # 2. Busca jogos reais para demo
    print("\n" + "─" * 60)
    print("  DEMO COM JOGOS REAIS (últimos 50 jogos do banco)")
    print("─" * 60)

    query = """
        SELECT id, date, home_team, away_team,
               odds_home_b365, odds_draw_b365, odds_away_b365,
               odds_home_pin, odds_draw_pin, odds_away_pin,
               home_goals, away_goals, season
        FROM matches
        WHERE odds_home_b365 IS NOT NULL
          AND odds_home_pin IS NOT NULL
        ORDER BY date DESC
        LIMIT 50
    """
    df = pd.read_sql(query, engine, parse_dates=["date"])

    if df.empty:
        logger.warning("Banco vazio. Execute run_etl.py primeiro.")
        return

    # Simula: aposta em todos os mandantes usando B365, compara com Pinnacle
    bets = []
    for _, row in df.iterrows():
        bets.append(BetRecord(
            match_id=int(row["id"]),
            date=pd.Timestamp(row["date"]).to_pydatetime(),
            home_team=row["home_team"],
            away_team=row["away_team"],
            outcome="H",
            odds_taken=float(row["odds_home_b365"]),
            model_prob=1.0 / float(row["odds_home_b365"]),  # prob implicita
        ))

    auditor = CLVAuditor(engine)
    report = auditor.audit_bets(bets, ClosingSource.PINNACLE)

    print(report)

    # Mostra 10 melhores e 5 piores CLV
    if report.results:
        sorted_results = sorted(
            [r for r in report.results if r.odds_closing > 0],
            key=lambda r: r.clv,
            reverse=True,
        )

        print("\n  TOP 5 — Melhor CLV:")
        for r in sorted_results[:5]:
            print(f"    {r}")

        print("\n  BOTTOM 5 — Pior CLV:")
        for r in sorted_results[-5:]:
            print(f"    {r}")

    # 3. Comparativo entre fontes de fechamento
    print("\n" + "─" * 60)
    print("  COMPARATIVO: CLV por fonte de fechamento")
    print("─" * 60)

    for src in ClosingSource:
        rpt = auditor.audit_bets(bets, src)
        print(
            f"  {src.value:<10} | Beat Rate: {rpt.beat_rate*100:5.1f}% | "
            f"CLV Médio: {rpt.avg_clv*100:+.2f}% | "
            f"Apostas: {rpt.bets_with_closing}"
        )


# ============================================================================
# FUNCOES AUXILIARES
# ============================================================================

def print_sample(report: CLVReport, n: int = 10):
    """Imprime amostra dos resultados individuais."""
    valid = [r for r in report.results if r.odds_closing > 0]
    if not valid:
        return

    print(f"\n{'─'*75}")
    print(f"  AMOSTRA ({min(n, len(valid))}/{len(valid)} apostas)")
    print(f"{'─'*75}")

    # Ordena por CLV (melhor primeiro)
    sorted_results = sorted(valid, key=lambda r: r.clv, reverse=True)

    for r in sorted_results[:n]:
        print(r)

    if len(valid) > n:
        print(f"  ... e mais {len(valid) - n} apostas")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Auditoria CLV — Closing Line Value",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_clv_audit.py                          # Demo + backtest simulado
  python run_clv_audit.py --backtest               # Backtest simulado
  python run_clv_audit.py --backtest --min-ev 0.05 # EV minimo 5%
  python run_clv_audit.py --model                  # Usa modelo treinado
  python run_clv_audit.py --csv apostas.csv        # Audita CSV
  python run_clv_audit.py --quick 2.00 1.80        # Calculo rapido
  python run_clv_audit.py --source average          # Fechamento pela media
        """,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", default=True,
                       help="Executa demo interativa (padrao)")
    group.add_argument("--backtest", action="store_true",
                       help="Backtest simulado com dados historicos")
    group.add_argument("--model", action="store_true",
                       help="Auditoria usando modelo treinado")
    group.add_argument("--csv", type=str, metavar="FILE",
                       help="Audita apostas de um arquivo CSV")
    group.add_argument("--quick", nargs=2, type=float, metavar=("TAKEN", "CLOSING"),
                       help="Calculo rapido de CLV")

    parser.add_argument("--source", type=str, default="pinnacle",
                        choices=["pinnacle", "average", "bet365", "max"],
                        help="Fonte das odds de fechamento (default: pinnacle)")
    parser.add_argument("--min-ev", type=float, default=0.02,
                        help="EV minimo para backtest (default: 0.02 = 2%%)")
    parser.add_argument("--seasons", type=int, nargs="+",
                        help="Temporadas a incluir (ex: 2023 2024)")
    parser.add_argument("--db", type=str,
                        default="sqlite:///understat_premier_league.db",
                        help="Caminho do banco de dados")

    args = parser.parse_args()

    source = ClosingSource(args.source)

    # Calculo rapido (nao precisa de banco)
    if args.quick:
        run_quick_check(args.quick[0], args.quick[1])
        return

    # Modos que precisam do banco
    engine = get_engine(args.db)

    if args.csv:
        run_csv_audit(engine, args.csv, source)
    elif args.model:
        run_backtest_with_model(engine, source, args.min_ev, args.seasons)
    elif args.backtest:
        run_backtest_simulated(engine, source, args.min_ev, args.seasons)
    else:
        # Demo (padrao)
        run_demo(engine)
        print("\n" + "═" * 60)
        print("  Para backtest completo: python run_clv_audit.py --backtest")
        print("  Para usar modelo real:  python run_clv_audit.py --model")
        print("═" * 60)


if __name__ == "__main__":
    main()
