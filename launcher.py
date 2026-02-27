import streamlit as st
import pandas as pd
import logging
from src.ml.pregame_scanner import PregameScanner

# Configuração de logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="APO$TEI - Launcher",
    page_icon="🎯",
    layout="wide"
)

# --- BARRA LATERAL (SIDEBAR) CONTROLES ---
st.sidebar.header("⚙️ Parâmetros de Execução")

# 1. Chave da API
api_key = st.sidebar.text_input(
    "The Odds API Key", 
    type="password", 
    help="Insira sua chave da API da The Odds API."
)

# 2. Banca Disponível
bankroll = st.sidebar.number_input(
    "Banca Atual ($)", 
    min_value=10.0, 
    value=1000.0, 
    step=50.0,
    help="Valor para cálculo do Staking (Kelly)."
)

# 3. Slider de EV Mínimo
min_ev_pct = st.sidebar.slider(
    "EV Mínimo (%)", 
    min_value=0.0, 
    max_value=10.0, 
    value=3.0, 
    step=0.5,
    help="Filtra apenas apostas com Valor Esperado acima desta porcentagem."
)

st.sidebar.divider()
st.sidebar.info("🎯 APO$TEI utiliza Closing Lines da Pinnacle como benchmark de eficiência.")

# --- CONTEÚDO PRINCIPAL ---
st.title("🎯 APO$TEI Launcher")
st.markdown("""
Painel de Controle do Pipeline Quantitativo. 
Configure os parâmetros na barra lateral e execute a análise de *Expected Value* em tempo real.
""")

st.divider()

# Botão principal
if st.button("🚀 Iniciar Análise Real (Live Market)", type="primary", use_container_width=True):
    
    if not api_key:
        st.error("❌ Erro: Insira a chave da API (The Odds API Key) na barra lateral antes de iniciar.")
        st.stop()
    
    scanner = PregameScanner(
        db_path="sqlite:///understat_premier_league.db",
        odds_api_key=api_key
    )
    
    try:
        with st.spinner('🚀 Buscando odds e rodando inferência real...'):
            report = scanner.scan(
                min_ev=min_ev_pct / 100.0, 
                bankroll=bankroll,
                hours_window=24.0,
                odds_source="pinnacle"
            )
        
        if not report.value_bets:
            st.warning(f"⚠️ Nenhuma oportunidade de Valor Esperado (EV > {min_ev_pct}%) encontrada para as próximas 24 horas.")
        else:
            st.success(f"✅ Análise concluída! Encontramos {len(report.value_bets)} oportunidades de valor.")
            
            # ── 1. MÉTRICAS DE RESUMO ──────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            
            total_bets = len(report.value_bets)
            total_exposure = sum(b.stake_amount for b in report.value_bets)
            max_ev = max(b.ev_pct for b in report.value_bets)
            
            m1.metric("Total de Apostas EV+", f"{total_bets}")
            m2.metric("Exposição Total", f"R$ {total_exposure:.2f}")
            m3.metric("Maior EV Encontrado", f"{max_ev:+.1f}%")
            
            # ── 2. PREPARAÇÃO DO DATAFRAME ─────────────────────────────────────
            data_rows = []
            for bet in report.value_bets:
                data_rows.append({
                    "Partida": f"{bet.home_team} vs {bet.away_team}",
                    "Resultado": bet.outcome_label,
                    "Modelo %": bet.model_prob * 100,
                    "Casa %": bet.implied_prob * 100,
                    "Odd": bet.odds_taken,
                    "EV": bet.ev_pct,
                    "Stake Sugerida": bet.stake_amount
                })
            
            df = pd.DataFrame(data_rows)
            
            # ── 3. ESTILIZAÇÃO VISUAL ──────────────────────────────────────────
            def style_ev(val):
                return 'color: #00c853; font-weight: bold'

            styled_df = df.style.format({
                "Modelo %": "{:.1f}%",
                "Casa %": "{:.1f}%",
                "Odd": "{:.2f}",
                "EV": "{:+.1f}%",
                "Stake Sugerida": "R$ {:.2f}"
            }).applymap(style_ev, subset=['EV'])

            # Exibição
            st.subheader("📊 Oportunidades de Valor Identificadas")
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            st.info(f"💡 Foram analisados {report.events_scanned} eventos. Créditos da API restantes: {report.api_requests_remaining}")

    except Exception as e:
        st.error(f"❌ Erro crítico: {str(e)}")
        logger.exception("Erro no Launcher")
