"""
run_clv_tracker.py
------------------
Rastreia o Closing Line Value (CLV) comparando as odds de valor (capturadas antecipadamente)
com as odds exatas do fechamento (Pinnacle) no momento ou logo apos o inicio do jogo.
"""

import logging
import os
import sys
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

# Ajuste do path para rodar na raiz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.models import CLVTracking, create_tables, get_engine, get_session
from src.ml.pregame_scanner import ODDS_API_BASE, ODDS_API_SPORT, PregameScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")


def capture_early_value(db_path: str = "sqlite:///flashscore_data.db"):
    """
    Busca apostas de valor nas proximas 24 horas usando o PregameScanner.
    Registra novas oportunidades na tabela clv_tracking.
    """
    logger.info("Iniciando captura de early value...")
    scanner = PregameScanner(db_path=db_path)

    try:
        report = scanner.scan(min_ev=0.0, bankroll=1000.0, hours_window=24.0, odds_source="pinnacle")
    except Exception as e:
        logger.error(f"Falha no scan: {e}")
        return

    engine = get_engine(db_path)
    create_tables(engine)
    session = get_session(engine)

    novos_registros = 0
    now = datetime.utcnow()

    for bet in report.value_bets:
        # Verifica se já existe o mesmo bet analisado
        exists = session.query(CLVTracking).filter_by(event_id=bet.event_id, outcome=bet.outcome).first()

        if not exists:
            clv_entry = CLVTracking(
                match_id=bet.match_id,
                event_id=bet.event_id,
                home_team=bet.home_team,
                away_team=bet.away_team,
                commence_time=bet.commence_time,
                timestamp_captura=now,
                outcome=bet.outcome,
                odd_capturada=bet.odds_taken,
                prob_modelo=bet.model_prob,
                bookmaker_usado=bet.bookmaker,
            )
            session.add(clv_entry)
            novos_registros += 1

    session.commit()
    logger.info(f"Captura concluida. {novos_registros} novas bets adicionadas ao rastreio CLV.")
    session.close()


def verify_closing_lines(db_path: str = "sqlite:///flashscore_data.db"):
    """
    Consulta jogos que começaram nas últimas 2 horas.
    Coleta a odd final da Pinnacle via The Odds API e calcula o CLV.
    """
    if not API_KEY:
        logger.error("ODDS_API_KEY não definida no arquivo .env")
        return

    logger.info("Iniciando verificação de closing lines...")
    engine = get_engine(db_path)
    session = get_session(engine)

    now = datetime.utcnow()
    two_hours_ago = now - timedelta(hours=2)
    two_hours_from_now = now + timedelta(hours=2)

    # Identifica bets pendentes cujo jogo deve ter começado (ou prestes a começar)
    # Comparação literal de strings ISO-8601 (A-Z chars funcionam bem)
    iso_now = now.isoformat() + "Z"
    iso_two_hours_ago = two_hours_ago.isoformat() + "Z"

    pending_bets = session.query(CLVTracking).filter(CLVTracking.pinnacle_closing_odd.is_(None), CLVTracking.commence_time >= iso_two_hours_ago, CLVTracking.commence_time <= iso_now).all()

    if not pending_bets:
        logger.info("Sem partidas pendentes iniciadas nas últimas 2 horas para verificar CLV.")
        session.close()
        return

    logger.info(f"Encontradas {len(pending_bets)} bets pendentes de closing line. Consultando API...")

    url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/odds"
    params = {"apiKey": API_KEY, "regions": "eu", "markets": "h2h", "bookmakers": "pinnacle"}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        logger.error(f"Falha ao consultar The Odds API: {e}")
        session.close()
        return

    # Mapeia eventos
    event_odds = {}
    for ev in events:
        ev_id = ev["id"]
        bookmakers = ev.get("bookmakers", [])
        pin_bkm = next((b for b in bookmakers if b["key"] == "pinnacle"), None)
        if pin_bkm:
            h2h_market = next((m for m in pin_bkm.get("markets", []) if m["key"] == "h2h"), None)
            if h2h_market:
                odds_dict = {}
                for obj in h2h_market.get("outcomes", []):
                    name = obj["name"]
                    # Compatibilizar output (nome) p/ 'H', 'D', 'A' baseados no ev["home_team"] etc
                    home = ev["home_team"]
                    away = ev["away_team"]
                    if name == home:
                        odds_dict["H"] = obj["price"]
                    elif name == away:
                        odds_dict["A"] = obj["price"]
                    elif name.lower() == "draw":
                        odds_dict["D"] = obj["price"]

                event_odds[ev_id] = odds_dict

    # Atualiza DB
    atualizados = 0
    for bet in pending_bets:
        if bet.event_id in event_odds:
            closing_odd = event_odds[bet.event_id].get(bet.outcome)
            if closing_odd:
                bet.pinnacle_closing_odd = closing_odd
                bet.clv_positivo = bet.odd_capturada > closing_odd
                atualizados += 1

    session.commit()
    logger.info(f"Verificação concluida. {atualizados} closing lines atualizadas.")
    session.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apo$tei CLV Tracker")
    parser.add_argument("--capture", action="store_true", help="Captura value bets antecipados")
    parser.add_argument("--verify", action="store_true", help="Verifica as closing lines de jogos iniciados")

    args = parser.parse_args()

    # Se nao passar flag nenhuma, roda os dois por padrão no cron
    if not args.capture and not args.verify:
        capture_early_value()
        verify_closing_lines()
    else:
        if args.capture:
            capture_early_value()
        if args.verify:
            verify_closing_lines()
