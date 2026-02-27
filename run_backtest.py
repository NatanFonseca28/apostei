import os
import sys

# Adiciona a raiz ao sys.path para importações absolutas do src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from sqlalchemy import text
from src.ml.pregame_scanner import PregameScanner
from src.core.staking import MODERATE

def main():
    print("=" * 70)
    print("  SIMULADOR DE BACKTEST DIRECIONAL E VARIÂNCIA")
    print("  Estratégia: Kelly Fracionário Dinâmico (Config: MODERATE)")
    print("=" * 70)

    # 1. Inicializar Motor de Scan
    scanner = PregameScanner(db_path="sqlite:///understat_premier_league.db")
    
    # 2. Encontrar a última temporada completa
    with scanner.engine.connect() as conn:
        res = conn.execute(text("""
            SELECT MAX(season) FROM matches 
            WHERE home_goals IS NOT NULL 
              AND odds_home_b365 IS NOT NULL
        """)).fetchone()
        latest_season = res[0]

    print(f"Temporada em análise: {latest_season}")

    # 3. Obter resultados reais dos jogos da temporada
    with scanner.engine.connect() as conn:
        df_matches = pd.read_sql(text(f"""
            SELECT id, home_goals, away_goals 
            FROM matches 
            WHERE season = {latest_season}
        """), conn)
        
    outcomes_dict = {}
    for _, row in df_matches.iterrows():
        mid = row['id']
        hg = row['home_goals']
        ag = row['away_goals']
        
        if pd.isna(hg) or pd.isna(ag):
            continue
            
        if hg > ag:
            real_outcome = "H"
        elif hg == ag:
            real_outcome = "D"
        else:
            real_outcome = "A"
            
        outcomes_dict[mid] = real_outcome

    # 4. Executar Scan Offline para identificar oportunidades
    initial_bankroll = 10000.0
    print(f"\nA pesquisar oportunidades com EV+ mínimo de {MODERATE.min_ev*100:.1f}%...")
    
    report = scanner.scan_offline(
        min_ev=MODERATE.min_ev,
        bankroll=initial_bankroll,  # Usado apenas por padrão, o valor é recalculado iterativamente
        staking_config=MODERATE,
        season=latest_season,
        limit=1000
    )

    bets = report.value_bets
    if not bets:
        print("Nenhuma aposta de valor encontrada nesta temporada!")
        return

    # Ordenação estrita por data do evento para compounding real
    bets.sort(key=lambda x: x.commence_time)

    # 5. Motor Direcional de Bankroll e Backtesting
    bankroll = initial_bankroll
    history = []
    peak_bankroll = bankroll
    max_drawdown = 0.0

    print(f"\nIniciando simulação com banca virtual: ${initial_bankroll:,.2f}")
    
    for bet in bets:
        real_outcome = outcomes_dict.get(bet.match_id)
        if not real_outcome:
            continue
            
        # Actualizar valor apostado (Stake) *dinamicamente*
        # porque bet.stake_pct mantém a % ideal de Kelly do saldo ativod
        stake_amount = bankroll * bet.stake_pct
        
        # Resolução da aposta direcional
        if bet.outcome == real_outcome:
            pnl = stake_amount * (bet.odds_taken - 1.0)
            status = '✅ WON'
        else:
            pnl = -stake_amount
            status = '❌ LOST'
            
        # Atualização rigorosa de banca
        bankroll += pnl
        
        # Métrica de queda (Drawdown)
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        
        drawdown = (peak_bankroll - bankroll) / peak_bankroll
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            
        history.append({
            "Data": bet.commence_time,
            "Mandante": bet.home_team,
            "Visitante": bet.away_team,
            "Mercado": bet.outcome_label,
            "Odd": bet.odds_taken,
            "EV.Pct(%)": bet.ev_pct,
            "Stake(%)": bet.stake_pct * 100,
            "Apostado($)": stake_amount,
            "Status": status,
            "PnL($)": pnl,
            "Banca($)": bankroll
        })

    # 6. Extracção de Métricas para o Relatório Executivo
    df_history = pd.DataFrame(history)
    
    total_bets = len(df_history)
    won_bets = len(df_history[df_history['Status'] == '✅ WON'])
    win_rate = won_bets / total_bets if total_bets > 0 else 0
    total_staked = df_history['Apostado($)'].sum()
    total_pnl = bankroll - initial_bankroll
    yield_pct = (total_pnl / total_staked) * 100 if total_staked > 0 else 0
    
    # Opcional: Salvar em CSV para análise local
    df_history.to_csv("artifacts/backtest_history.csv", index=False)
    
    print("\n" + "=" * 70)
    print("  RELATÓRIO EXECUTIVO - RESULTADOS DO BACKTEST")
    print("=" * 70)
    print(f"Banca Inicial:     ${initial_bankroll:,.2f}")
    print(f"Banca Final:       ${bankroll:,.2f}")
    print(f"Pico de Banca:     ${peak_bankroll:,.2f}")
    print(f"PnL Total:         ${total_pnl:+,.2f}")
    print(f"Volume Apostado:   ${total_staked:,.2f}")
    print("-" * 70)
    print(f"Qtd. de Apostas:   {total_bets}")
    print(f"Win Rate:          {win_rate * 100:.1f}%")
    print(f"Yield (ROI):       {yield_pct:+.2f}%")
    print(f"Max Drawdown:      {max_drawdown * 100:.1f}%")
    print("=" * 70)
    print("Histórico detalhado exportado em: artifacts/backtest_history.csv\n")

if __name__ == "__main__":
    main()
