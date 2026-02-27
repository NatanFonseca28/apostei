"""
run_staking_demo.py
-------------------
Demonstracao do modulo de Gestao de Risco Quantitativo (staking.py).

Exemplos praticos de como usar o Kelly Fracionario com protecoes.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.staking import (
    fractional_kelly,
    plan_round,
    StakingConfig,
    CONSERVATIVE,
    MODERATE,
    AGGRESSIVE,
)


BANKROLL = 1_000.00  # Banca inicial em dolares


def exemplo_1_aposta_individual():
    """Aposta individual com diferentes niveis de agressividade."""
    print("=" * 75)
    print("  EXEMPLO 1: Aposta Individual -- Arsenal vs Chelsea")
    print("  Modelo da 55% de chance para vitoria do Arsenal, odd 2.10")
    print("=" * 75)

    for label, cfg in [("CONSERVATIVE", CONSERVATIVE), ("MODERATE", MODERATE), ("AGGRESSIVE", AGGRESSIVE)]:
        rec = fractional_kelly(
            model_prob=0.55,
            odd=2.10,
            bankroll=BANKROLL,
            config=cfg,
            home_team="Arsenal",
            away_team="Chelsea",
            outcome="H",
        )
        print(f"\n  [{label}] Kelly x{cfg.kelly_fraction}")
        print(f"    Kelly completo:     {rec.kelly_full*100:.2f}%")
        print(f"    Kelly c/ shrinkage: {rec.kelly_shrunk*100:.2f}%")
        print(f"    Stake final:        {rec.stake_pct*100:.2f}% = ${rec.stake_amount:.2f}")
        print(f"    Teto aplicado?      {'Sim' if rec.is_capped else 'Nao'}")
        print(f"    Lucro potencial:    ${rec.potential_profit:.2f}")
        print(f"    EV:                 {rec.ev_pct:+.1f}%")

    print()


def exemplo_2_teto_em_acao():
    """Mostra o teto de seguranca limitando uma aposta de alto EV."""
    print("=" * 75)
    print("  EXEMPLO 2: Teto de Seguranca em Acao")
    print("  Modelo da 70% mas odd apenas 1.80 (edge enorme)")
    print("  Sem teto, Kelly sugeriria apostar ~45% da banca!")
    print("=" * 75)

    rec_sem_teto = fractional_kelly(
        model_prob=0.70,
        odd=1.80,
        bankroll=BANKROLL,
        config=StakingConfig(kelly_fraction=1.0, max_stake_pct=0.50),  # Kelly completo, teto alto
        home_team="Man City",
        away_team="Burnley",
        outcome="H",
    )

    rec_com_teto = fractional_kelly(
        model_prob=0.70,
        odd=1.80,
        bankroll=BANKROLL,
        config=CONSERVATIVE,
        home_team="Man City",
        away_team="Burnley",
        outcome="H",
    )

    print(f"\n  SEM protecoes (Full Kelly):")
    print(f"    Kelly: {rec_sem_teto.kelly_full*100:.2f}% -> Stake: ${rec_sem_teto.stake_amount:.2f}")

    print(f"\n  COM protecoes (Eighth Kelly + teto 2%):")
    print(f"    Kelly: {rec_com_teto.kelly_full*100:.2f}% -> {rec_com_teto.stake_pct*100:.2f}%")
    print(f"    Stake: ${rec_com_teto.stake_amount:.2f}")
    print(f"    Teto aplicado? {'Sim' if rec_com_teto.is_capped else 'Nao'}")

    print()


def exemplo_3_aposta_sem_valor():
    """Mostra uma aposta que e corretamente rejeitada."""
    print("=" * 75)
    print("  EXEMPLO 3: Aposta Rejeitada (EV insuficiente)")
    print("  Modelo da 30% para empate, odd 3.20")
    print("  EV = -4%, abaixo do minimo -> SKIP")
    print("=" * 75)

    rec = fractional_kelly(
        model_prob=0.30,
        odd=3.20,
        bankroll=BANKROLL,
        config=MODERATE,
        home_team="Liverpool",
        away_team="Tottenham",
        outcome="D",
    )

    print(f"\n  Resultado: {'APOSTAR' if rec.is_actionable else 'NAO APOSTAR'}")
    print(f"  EV: {rec.ev_pct:+.1f}%")
    print(f"  Kelly completo: {rec.kelly_full*100:.2f}%")
    print(f"  Stake: ${rec.stake_amount:.2f}")

    print()


def exemplo_4_plano_rodada():
    """Plano completo de staking para uma rodada com 3 jogos."""
    print("=" * 75)
    print("  EXEMPLO 4: Plano de Staking para Rodada Completa")
    print("  3 partidas, configuracao MODERATE")
    print("=" * 75)

    matches = [
        {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "probs": {"H": 0.55, "D": 0.25, "A": 0.20},
            "odds":  {"H": 2.10, "D": 3.40, "A": 3.80},
        },
        {
            "home_team": "Man City",
            "away_team": "Newcastle",
            "probs": {"H": 0.65, "D": 0.20, "A": 0.15},
            "odds":  {"H": 1.65, "D": 4.00, "A": 5.50},
        },
        {
            "home_team": "Liverpool",
            "away_team": "Tottenham",
            "probs": {"H": 0.50, "D": 0.25, "A": 0.25},
            "odds":  {"H": 1.90, "D": 3.60, "A": 4.20},
        },
    ]

    plan = plan_round(matches, bankroll=BANKROLL, config=MODERATE)
    print(plan)


if __name__ == "__main__":
    print("\n")
    print("  APO$TEI -- Modulo de Gestao de Risco Quantitativo")
    print("  Kelly Fracionario com Shrinkage + Teto de Seguranca")
    print("\n")

    exemplo_1_aposta_individual()
    exemplo_2_teto_em_acao()
    exemplo_3_aposta_sem_valor()
    exemplo_4_plano_rodada()

    print("\n  Demo concluida.\n")
