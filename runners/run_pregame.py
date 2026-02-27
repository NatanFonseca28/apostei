#!/usr/bin/env python
"""
run_pregame.py
--------------
Script de orquestracao pre-jogo — event-driven.

Uso:
  # Modo LIVE (The Odds API → modelo → filtro EV+ > 3%)
  python run_pregame.py --live --min-ev 3

  # Modo OFFLINE (odds do banco, sem gastar creditos da API)
  python run_pregame.py --offline --min-ev 3

  # Com bankroll e staking agressivo
  python run_pregame.py --live --bankroll 5000 --staking aggressive --min-ev 2

  # Exportar JSON
  python run_pregame.py --live --output apostas_hoje.json

  # Modelo especifico + odds de todas as casas (melhor odd)
  python run_pregame.py --live --model artifacts/best_model_20260226_190625.pkl --best-odds

Variaveis de ambiente:
  ODDS_API_KEY  —  Chave da The Odds API (obrigatoria para modo live)
                   Cadastro gratis: https://the-odds-api.com
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Carrega .env da raiz do projeto
load_dotenv(Path(__file__).resolve().parent / ".env")

# Garante que o diretorio raiz este no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.pregame_scanner import PregameScanner, ScanReport
from src.core.staking import CONSERVATIVE, MODERATE, AGGRESSIVE

# ============================================================================
# CONSTANTES
# ============================================================================

STAKING_PRESETS = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE,
}


# ============================================================================
# FORMATACAO DE SAIDA
# ============================================================================

def print_header():
    """Imprime cabecalho do scanner."""
    print()
    print("=" * 72)
    print("  APO$TEI — Pregame Scanner v1.0")
    print("  Escaneamento Pre-Jogo | Event-Driven Pipeline")
    print("=" * 72)
    print()


def print_report(report: ScanReport):
    """Imprime relatorio formatado no console."""
    print("-" * 72)
    print(f"  Timestamp:           {report.timestamp}")
    print(f"  Modelo:              {report.model_name} ({report.model_path})")
    print(f"  Features:            {', '.join(report.features_used)}")
    print(f"  EV minimo:           {report.min_ev_threshold*100:.1f}%")
    print(f"  Staking:             {report.staking_config}")
    print(f"  Banca:               ${report.bankroll:,.2f}")
    print("-" * 72)
    print()

    # Sumario
    print(f"  Eventos escaneados:     {report.events_scanned}")
    print(f"  Com features no banco:  {report.events_matched}")
    print(f"  Apostas com EV+:        {report.total_value_bets}")
    print()

    if report.api_requests_remaining is not None:
        print(f"  [API] Requests restantes: {report.api_requests_remaining}")
        print(f"  [API] Requests usadas:    {report.api_requests_used}")
        print()

    if not report.value_bets:
        print("  Nenhuma aposta de valor encontrada com os criterios atuais.")
        print()
        print("  Possibilidades:")
        print("    - Nao ha jogos nas proximas horas")
        print("    - Modelo e mercado estao alinhados (sem edge)")
        print("    - Tente reduzir --min-ev para ampliar o filtro")
        return

    # Tabela de apostas
    print("  APOSTAS DE VALOR ENCONTRADAS:")
    print("  " + "-" * 68)
    print(f"  {'#':>2s}  {'Jogo':<35s}  {'Tip':>4s}  {'Odd':>5s}  {'EV%':>6s}  {'Stake':>8s}")
    print("  " + "-" * 68)

    for i, bet in enumerate(report.value_bets, 1):
        match = f"{bet.home_team} vs {bet.away_team}"
        if len(match) > 33:
            match = match[:32] + "."
        print(
            f"  {i:2d}  {match:<35s}  {bet.outcome:>4s}  "
            f"{bet.odds_taken:5.2f}  {bet.ev_pct:5.1f}%  "
            f"${bet.stake_amount:7.2f}"
        )

    print("  " + "-" * 68)
    total_stake = sum(b.stake_amount for b in report.value_bets)
    total_exposure = total_stake / report.bankroll * 100 if report.bankroll else 0
    print(f"  {'':>2s}  {'TOTAL':>35s}  {'':>4s}  {'':>5s}  {'':>6s}  ${total_stake:7.2f}")
    print(f"  {'':>2s}  {'Exposicao da banca':>35s}  {'':>4s}  {'':>5s}  {total_exposure:5.1f}%")
    print()

    # Detalhamento por aposta
    print("  DETALHAMENTO:")
    print("  " + "-" * 68)
    for i, bet in enumerate(report.value_bets, 1):
        print(f"\n  [{i}] {bet.home_team} vs {bet.away_team}")
        print(f"      Kickoff:     {bet.commence_time}")
        print(f"      Tip:         {bet.outcome} — {bet.outcome_label}")
        print(f"      Bookmaker:   {bet.bookmaker}")
        print(f"      Odd:         {bet.odds_taken:.2f}")
        print(f"      Prob modelo: {bet.model_prob*100:.1f}% | Implicita: {bet.implied_prob*100:.1f}%")
        print(f"      Edge:        {bet.edge*100:.2f}pp")
        print(f"      EV:          {bet.ev_pct:.2f}%")
        print(f"      Kelly:       {bet.kelly_full*100:.2f}% (full) -> {bet.kelly_shrunk*100:.2f}% (shrunk)")
        print(f"      Stake:       ${bet.stake_amount:.2f} ({bet.stake_pct*100:.2f}% da banca)")

    print()


def save_json(report: ScanReport, filepath: str):
    """Salva relatorio como JSON."""
    out = Path(filepath)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report.to_json(indent=2))
    print(f"  JSON exportado: {out.resolve()}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="APO$TEI Pregame Scanner — Escaneamento pre-jogo event-driven",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_pregame.py --live --min-ev 3
  python run_pregame.py --offline --min-ev 2 --limit 100
  python run_pregame.py --live --bankroll 5000 --staking aggressive --output apostas.json
  python run_pregame.py --live --best-odds --min-ev 1
        """,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--live", action="store_true",
        help="Busca odds ao vivo da The Odds API (requer ODDS_API_KEY)",
    )
    mode.add_argument(
        "--offline", action="store_true",
        help="Usa odds do banco local (Bet365 historicas)",
    )

    # Modelo
    p.add_argument(
        "--model", type=str, default=None,
        help="Caminho para o modelo .pkl (default: auto-detect mais recente)",
    )

    # Filtros
    p.add_argument(
        "--min-ev", type=float, default=3.0,
        help="EV minimo em %% para filtrar apostas (default: 3.0)",
    )
    p.add_argument(
        "--hours", type=float, default=24.0,
        help="Janela de tempo em horas — so jogos nas proximas N horas (default: 24)",
    )
    p.add_argument(
        "--sport", type=str, default="soccer_epl",
        help="Esporte na The Odds API (default: soccer_epl)",
    )

    # Staking
    p.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Banca disponivel em $ (default: 1000)",
    )
    p.add_argument(
        "--staking", type=str, default="moderate",
        choices=["conservative", "moderate", "aggressive"],
        help="Perfil de risco (default: moderate)",
    )

    # Odds
    p.add_argument(
        "--odds-source", type=str, default="pinnacle",
        help="Bookmaker para odds de referencia (default: pinnacle)",
    )
    p.add_argument(
        "--best-odds", action="store_true",
        help="Usar a melhor odd disponivel entre todas as casas",
    )

    # Offline
    p.add_argument(
        "--limit", type=int, default=50,
        help="Numero de jogos a escanear no modo offline (default: 50)",
    )
    p.add_argument(
        "--season", type=int, default=None,
        help="Temporada especifica para modo offline (ex: 2023)",
    )

    # Saida
    p.add_argument(
        "--output", "-o", type=str, default=None,
        help="Caminho para salvar JSON com resultado",
    )
    p.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suprime saida console (util com --output)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Modo verboso (debug logging)",
    )

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print_header()

    # Inicializa scanner
    scanner = PregameScanner(
        model_path=args.model,
        odds_api_key=os.environ.get("ODDS_API_KEY", ""),
    )

    # Executa scan
    staking_config = STAKING_PRESETS[args.staking]

    if args.live:
        print(f"  Modo: LIVE (The Odds API)")
        print(f"  Esporte: {args.sport}")
        print()

        if not os.environ.get("ODDS_API_KEY"):
            print("  [AVISO] ODDS_API_KEY nao definida!")
            print("  Configure via: $env:ODDS_API_KEY = 'sua_chave'")
            print("  Cadastro gratis: https://the-odds-api.com")
            print()
            return

        report = scanner.scan(
            min_ev=args.min_ev / 100.0,
            bankroll=args.bankroll,
            staking_config=staking_config,
            odds_source=args.odds_source,
            sport=args.sport,
            hours_window=args.hours,
            use_best_odds=args.best_odds,
        )

    else:  # --offline
        print(f"  Modo: OFFLINE (odds Bet365 do banco)")
        season_str = f"Temporada {args.season}" if args.season else "Todas"
        print(f"  Temporada: {season_str} | Limite: {args.limit} jogos")
        print()

        report = scanner.scan_offline(
            min_ev=args.min_ev / 100.0,
            bankroll=args.bankroll,
            staking_config=staking_config,
            season=args.season,
            limit=args.limit,
        )

    # Saida
    if not args.quiet:
        print_report(report)

    if args.output:
        save_json(report, args.output)

    # Retorna JSON estruturado para integracao com automacoes
    return report.to_dict()


if __name__ == "__main__":
    result = main()
