from src.ml.pregame_scanner import PregameScanner
import logging
logging.basicConfig(level=logging.INFO)

scanner = PregameScanner(
    db_path="sqlite:///flashscore_data.db"
)

report = scanner.scan(
    min_ev=0.0, 
    bankroll=1000.0,
    leagues=['soccer_brazil_campeonato'],
    hours_window=336.0, 
    odds_source="pinnacle"
)

print(f"BETS FOUND: {len(report.value_bets)}")
for b in report.value_bets:
    print(f"{b.home_team} vs {b.away_team} -> {b.outcome_label} (EV: {b.ev_pct}%)")
