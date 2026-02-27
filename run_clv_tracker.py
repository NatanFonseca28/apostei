import os
import sys
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Adicionar a raiz ao sys.path para importações do src funcionarem
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import requests
from sqlalchemy.orm import sessionmaker

from src.data.models import CLVTracking, get_engine, create_tables
from src.ml.pregame_scanner import PregameScanner
from src.core.staking import MODERATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "soccer_epl"

def capture_early_value():
    """ Procura apostas EV+ até 24h para o futuro e regista-as na BD """
    logger.info("Iniciando captura de apostas de valor (Early Value)...")
    try:
        scanner = PregameScanner(db_path="sqlite:///understat_premier_league.db")
        # Criar a nova tabela (se não existir)
        create_tables(scanner.engine)
        
        Session = sessionmaker(bind=scanner.engine)
        session = Session()

        # Usar o modo de verificação nas próximas 24h
        report = scanner.scan(
            min_ev=0.02, # Limite suave de 2% para acompanhamento rigoroso do CLV
            bankroll=10000.0,
            staking_config=MODERATE,
            odds_source="pinnacle", # Referência dos sharps para tracker de mercados
            hours_window=24.0,
            use_best_odds=False
        )

        value_bets = report.value_bets
        if not value_bets:
            logger.info("Nenhuma aposta EV+ inicial encontrada na janela de tempo atual.")
            session.close()
            return

        now = datetime.now(timezone.utc)
        
        for bet in value_bets:
            # Verifica para não duplicar capturas
            existing = session.query(CLVTracking).filter_by(
                event_id=bet.event_id,
                outcome=bet.outcome
            ).first()

            if not existing:
                logger.info(f"Registando rastreio CLV: {bet.home_team} vs {bet.away_team} [{bet.outcome}] @ {bet.odds_taken}")
                tracker = CLVTracking(
                    match_id=bet.match_id,
                    event_id=bet.event_id,
                    home_team=bet.home_team,
                    away_team=bet.away_team,
                    commence_time=bet.commence_time,
                    timestamp_captura=now,
                    outcome=bet.outcome,
                    odd_capturada=bet.odds_taken,
                    prob_modelo=bet.model_prob, # Registamos para auditar as quedas de rentabilidade a posteriori
                    bookmaker_usado=bet.bookmaker
                )
                session.add(tracker)
            else:
                logger.debug(f"Aposta {bet.home_team} vs {bet.away_team} [{bet.outcome}] já rastreada. Ignorando...")
                
        session.commit()
    except Exception as e:
        logger.error(f"Erro em capture_early_value: {e}")
    finally:
        if 'session' in locals():
            session.close()

def verify_closing_lines():
    """ 
    Recupera os valores apostados nas apostas cujos kickoff terms passaram/estão 
    a passar, guardando Pinnacle Closing e determinando sucesso direcional 
    """
    logger.info("Iniciando verificação de Closing Lines (Linhas de Fecho)...")
    engine = get_engine("sqlite:///understat_premier_league.db")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        now = datetime.now(timezone.utc)
        two_hours_ago = now - timedelta(hours=2)
        
        # Filtra instâncias onde a Pinnacle Closing Line ainda não foi obtida
        pending_trackers = session.query(CLVTracking).filter(
            CLVTracking.pinnacle_closing_odd.is_(None)
        ).all()
        
        if not pending_trackers:
            logger.info("Nenhum rastreio pendente à espera de consolidação CLV.")
            return
            
        pending_to_check = []
        for t in pending_trackers:
            try:
                ct = datetime.fromisoformat(t.commence_time.replace("Z", "+00:00"))
                # Limita a chamadas estritas perto do match time (existem matches a ocorrer num delta de -2h -> hoje)
                if two_hours_ago <= ct <= now:
                    pending_to_check.append(t)
            except ValueError:
                pass
                
        if not pending_to_check:
            logger.info("Existem registos CLV na base, mas não ocorreram no delta recente de 2h/kick-off.")
            return
            
        api_key = os.environ.get("ODDS_API_KEY", "")
        if not api_key:
            logger.error("ODDS_API_KEY falhou/não foi encontrada. Configura .env obrigatoriamente para tracker de closing.")
            return

        params = {
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h",
            "bookmakers": "pinnacle", # Pinnacle para bater The Origin Lines
            "oddsFormat": "decimal",
        }
        
        url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/odds"
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        
        events = resp.json()
        
        # Cria um mapeamento O(1) de ID => Mercado H2H para os dicts da the-odds
        event_odds_map = {}
        for ev in events:
            eid = ev.get("id")
            for bk in ev.get("bookmakers", []):
                if bk["key"] == "pinnacle":
                    for market in bk.get("markets", []):
                        if market["key"] == "h2h":
                            home_team = ev.get("home_team")
                            away_team = ev.get("away_team")
                            odds_dict = {}
                            for o in market["outcomes"]:
                                if o["name"] == home_team:
                                    odds_dict["H"] = o["price"]
                                elif o["name"] == away_team:
                                    odds_dict["A"] = o["price"]
                                elif o["name"] == "Draw":
                                    odds_dict["D"] = o["price"]
                            event_odds_map[eid] = odds_dict
                            break

        for t in pending_to_check:
            odds_dict = event_odds_map.get(t.event_id)
            if odds_dict and t.outcome in odds_dict:
                closing_odd = odds_dict[t.outcome]
                t.pinnacle_closing_odd = closing_odd
                
                # Bateu a Pinnacle Closing Value: True quando Early > Closing
                t.clv_positivo = t.odd_capturada > closing_odd
                
                status_clr = "🟢" if t.clv_positivo else "🔴"
                logger.info(f"{status_clr} Pinnacle Closing validada para {t.home_team} vs {t.away_team} [{t.outcome}]: "
                            f"Early: {t.odd_capturada:.2f} | Closing: {closing_odd:.2f} | CLV+: {t.clv_positivo}")
            else:
                logger.warning(f"O Odds Event {t.event_id} não estava mais listado ou a Pinnacle não ofereceu linhas (Event Canceled/In Play longo).")

        session.commit()
    except requests.exceptions.Timeout:
        logger.error("The Odds API: Rate Limit Timeout. O tracker será executado novamente na cron_job seguinte.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"The Odds API: Erro HTTP inesperado (Status {e})")
    except Exception as e:
        logger.error(f"Exceção genérica CLV Tracking ({type(e).__name__}): {e}")
    finally:
        session.close()

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(" CRON JOB: CLV TRACKER (Tracking the Sharp Money)")
    print(f"{'='*70}\n")
    
    # 1º Passo: Early Capture
    capture_early_value()
    
    # 2º Passo: Late Validation
    verify_closing_lines()
    
    print("\nExecução do Pipeline CLV finalizada com sucesso.")
