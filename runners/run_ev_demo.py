"""
run_ev_demo.py
--------------
Demonstração da calculadora de Valor Esperado (EV).

Simula a saída do modelo de ML aplicada a odds reais de bookmakers.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.ev_calculator import calculate_ev, scan_matches

# ─────────────────────────────────────────────────────────────────────────────
# Exemplo 1 — Partida isolada com EV+ identificado
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*75)
print("  EXEMPLO 1 — Partida isolada")
print("="*75)

report = calculate_ev(
    home_team="Arsenal",
    away_team="Chelsea",
    probs={
        "H": 0.55,   # modelo diz: 55% chance de Arsenal ganhar
        "D": 0.22,   #             22% empate
        "A": 0.23,   #             23% Chelsea ganhar
    },
    odds={
        "H": 2.10,   # casa paga 2.10 × para Arsenal
        "D": 3.40,   # casa paga 3.40 × para Empate
        "A": 3.80,   # casa paga 3.80 × para Chelsea
    },
    min_ev_threshold=0.0,   # sinaliza qualquer EV positivo
)
print(report)

# ─────────────────────────────────────────────────────────────────────────────
# Exemplo 2 — Filtro de EV mínimo de 5% (reduz falsos positivos)
# ─────────────────────────────────────────────────────────────────────────────

print("="*75)
print("  EXEMPLO 2 — Filtro EV > 5%  (min_ev_threshold=0.05)")
print("="*75)

report2 = calculate_ev(
    home_team="Man City",
    away_team="Everton",
    probs={"H": 0.72, "D": 0.17, "A": 0.11},
    odds={"H": 1.45, "D": 4.50, "A": 7.00},
    min_ev_threshold=0.05,   # ignora EV entre 0% e 5%
)
print(report2)

# ─────────────────────────────────────────────────────────────────────────────
# Exemplo 3 — scan_matches: múltiplas partidas da rodada
# ─────────────────────────────────────────────────────────────────────────────

print("="*75)
print("  EXEMPLO 3 — scan_matches: rodada completa")
print("="*75)

rodada = [
    {
        "home_team": "Liverpool",
        "away_team": "Man United",
        "probs": {"H": 0.58, "D": 0.22, "A": 0.20},
        "odds":  {"H": 1.90, "D": 3.60, "A": 4.20},
    },
    {
        "home_team": "Tottenham",
        "away_team": "Newcastle",
        "probs": {"H": 0.45, "D": 0.27, "A": 0.28},
        "odds":  {"H": 2.30, "D": 3.20, "A": 3.10},
    },
    {
        "home_team": "Aston Villa",
        "away_team": "Brighton",
        "probs": {"H": 0.40, "D": 0.28, "A": 0.32},
        "odds":  {"H": 2.60, "D": 3.10, "A": 2.80},
    },
    {
        "home_team": "Wolves",
        "away_team": "Brentford",
        # Modelo e casa concordam — sem value
        "probs": {"H": 0.35, "D": 0.30, "A": 0.35},
        "odds":  {"H": 2.85, "D": 3.30, "A": 2.85},
    },
]

value_reports = scan_matches(rodada, min_ev_threshold=0.02)

if value_reports:
    print(f"\n  🎯 {len(value_reports)} partida(s) com apostas EV+ (threshold > 2%):\n")
    for r in value_reports:
        print(r)
else:
    print("\n  Nenhuma aposta com valor encontrada nesta rodada.\n")
