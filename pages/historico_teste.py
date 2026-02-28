import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
from src.ml.pregame_scanner import PregameScanner
from src.data.models import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="APO$TEI - Backtest", page_icon="📈", layout="wide")

st.title("📈 APO$TEI Backtest (Teste de Efetividade)")
st.markdown("Avalie a taxa de acerto do modelo ML comparando previsões com os placares REAIS de jogos recentes arquivados.")

# Simulando uma busca de todos os jogos nos últimos 7 dias que já têm placar final
@st.cache_data(ttl=300)
def fetch_historic_data(days=7):
    # Simulando a data de hoje para bater com a semana anterior, ajustamos timezone
    query = f"""
        SELECT 
            id, data as date, time_casa as home_team, time_fora as away_team,
            placar_casa as home_goals, placar_fora as away_goals, campeonato,
            media_marcados_casa, media_sofridos_casa, media_marcados_fora, media_sofridos_fora
        FROM flashscore_matches
        WHERE placar_casa IS NOT NULL 
          AND placar_fora IS NOT NULL
          AND data >= datetime('now', '-{days} days')
        ORDER BY data DESC
    """
    engine = get_engine("sqlite:///flashscore_data.db")
    df = pd.read_sql(query, engine, parse_dates=["date"])
    return df

days_back = st.slider("Analisar jogos de até", min_value=1, max_value=30, value=7, help="Dias retroativos para buscar jogos já finalizados no banco.")

if st.button("Executar Simulação de Efetividade", type="primary"):
    with st.spinner("⏳ Rodando inferência histórica nos jogos concluidos..."):
        df = fetch_historic_data(days_back)
        
        if df.empty:
            st.warning(f"Não encontramos jogos com placar finalizado nos últimos {days_back} dias no seu banco de dados.")
        else:
            scanner = PregameScanner(db_path="sqlite:///flashscore_data.db")
            
            # Reusa scan_offline
            report = scanner.scan_offline(
                min_ev=-99.0, # Aceita tudo
                bankroll=100.0,
                limit=100
            )
            
            # Filtra o report pelos `match_id` que fetch_historic_data encontrou
            historic_ids = df['id'].tolist()
            
            results_data = []
            hits = 0
            total_evaluated = 0
            
            for bet in report.value_bets:
                if bet.match_id in historic_ids:
                    # Encontra a linha real
                    real_match = df[df['id'] == bet.match_id].iloc[0]
                    hg = int(real_match['home_goals'])
                    ag = int(real_match['away_goals'])
                    
                    real_outcome = "H" if hg > ag else ("A" if ag > hg else "D")
                    # Para simplificar na contabilidade de backtest, consideramos o outcome de MAIOR probabilidade (O que o app indicaria como favorito)
                    # Encontrar a melhor probabilidade do dicionário do modelo
                    best_pred = max(bet.model_probs, key=bet.model_probs.get)
                    
                    # Evita duplicar porque o array da scan_offline traz H,D,A para cada match
                    # Usaremos este hack -> Se outcome que estamos olhando == melhor_pred
                    if bet.outcome == best_pred:
                        total_evaluated += 1
                        acertou = (best_pred == real_outcome)
                        if acertou:
                            hits += 1
                            
                        # Traducao
                        mapper = {"H": "Mandante", "A": "Visitante", "D": "Empate"}
                        
                        results_data.append({
                            "Data": real_match['date'],
                            "Confronto": f"{bet.home_team} {hg} x {ag} {bet.away_team}",
                            "Previsão ML": mapper[best_pred],
                            "Resultado Real": mapper[real_outcome],
                            "Probabilidade": f"{bet.model_probs[best_pred]*100:.1f}%",
                            "Status": "✅ Acertou" if acertou else "❌ Errou"
                        })
            
            if total_evaluated > 0:
                accuracy = hits / total_evaluated * 100
                st.success(f"Teste concluído! O modelo avaliou {total_evaluated} partidas recentes.")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Partidas Validadas", total_evaluated)
                c2.metric("Acertos Exatos (1X2)", hits)
                c3.metric("Taxa de Acerto (Win Rate)", f"{accuracy:.1f}%")
                
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            else:
                st.info("Nenhuma previsão compatível gerada para os jogos finalizados encontrados.")
