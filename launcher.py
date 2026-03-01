import streamlit as st
import pandas as pd
import logging
import os
from dotenv import load_dotenv

# Configuração de logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

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

# 2. Seleção de Ligas (Multiselect)
league_options = {
    'Premier League': 'soccer_epl',
    'Brasileirão Série A': 'soccer_brazil_campeonato',
    'Champions League': 'soccer_uefa_champs_league',
    'La Liga (Espanha)': 'soccer_spain_la_liga',
    'Serie A (Itália)': 'soccer_italy_serie_a',
    'Ligue 1 (França)': 'soccer_france_ligue_one',
    'Copa Libertadores': 'soccer_conmebol_copa_libertadores',
    'Copa Sul-Americana': 'soccer_conmebol_copa_sudamericana'
}

selected_league_names = st.sidebar.multiselect(
    "Ligas Alvo",
    options=list(league_options.keys()),
    default=['Brasileirão Série A', 'Premier League'],
    help="Selecione as ligas para escanear oportunidades."
)

# Mapeia nomes para chaves da API
selected_leagues = [league_options[name] for name in selected_league_names]

# 3. Banca Disponível
bankroll = st.sidebar.number_input(
    "Banca Atual ($)", 
    min_value=10.0, 
    value=1000.0, 
    step=50.0,
    help="Valor para cálculo do Staking (Kelly)."
)

# 4. Slider de EV Mínimo
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

tab1, tab2 = st.tabs(["📊 Terminal de Operações", "⚙️ Configurações do Motor"])

with tab1:
    # Botão principal
    if st.button("🚀 Iniciar Análise Real (Live Market)", type="primary", use_container_width=True):
        
        # Validações
        if not api_key:
            api_key = ODDS_API_KEY
            if not api_key:
                st.error("❌ Erro: Chave da API (The Odds API) ausente na barra lateral ou no .env.")
                st.stop()
        
        if not selected_leagues:
            selected_leagues = list(league_options.values())
            st.toast("ℹ️ Nenhuma liga selecionada. Escaneando todas as habilitadas por padrão.")
        
        # IMPORT ATRASADO (Late Import): Evita que o Streamlit demore a carregar a UI inicial
        with st.spinner('⚙️ Carregando motor de IA e Scanner...'):
            from src.data.models import get_engine, create_tables
            from src.ml.pregame_scanner import PregameScanner
            
            # Garante que as novas tabelas de Cache IA existam no SQLite
            db_engine = get_engine("sqlite:///flashscore_data.db")
            create_tables(db_engine)
        
        scanner = PregameScanner(
            db_path="sqlite:///flashscore_data.db",
            odds_api_key=api_key
        )
        
        try:
            with st.spinner(f'🚀 Buscando odds para {len(selected_leagues)} ligas e rodando inferência...'):
                
                # Componentes Visuais (Reativos)
                progress_bar = st.progress(0)
                status_text = st.empty()
                table_placeholder = st.empty()
                
                def update_ui(current_bets, current_idx, total_events):
                    # 1. Update Progress
                    progress_bar.progress(current_idx / max(1, total_events))
                    
                    # 2. Update Status Text
                    status_text.text(f"Analisando evento {current_idx} de {total_events}... Encontradas {len(current_bets)} apostas de valor.")
                    
                    # 3. Stream da Tabela Parcial
                    if current_bets:
                        data_rows = []
                        for bet in current_bets:
                            stats = "N/A"
                            if bet.home_features and bet.away_features:
                                hgf = bet.home_features.get('media_marcados_casa', '-')
                                hgs = bet.home_features.get('media_sofridos_casa', '-')
                                agf = bet.away_features.get('media_marcados_fora', '-')
                                ags = bet.away_features.get('media_sofridos_fora', '-')
                                stats = f"C(G:{hgf}/S:{hgs}) | V(G:{agf}/S:{ags})"
                            
                            data_rows.append({
                                "Partida": f"{bet.home_team} vs {bet.away_team}",
                                "Data": bet.commence_time,
                                "Aposta": bet.outcome_label,
                                "Oportunidade (%)": bet.model_prob * 100,
                                "Casa (%)": bet.implied_prob * 100,
                                "Odd (Valor)": bet.odds_taken,
                                "EV Extra (%)": bet.ev_pct,
                                "$$ Stake (R$)": bet.stake_amount,
                                "Base de Dados (Média de Gols)": stats,
                                "Palpite da IA (Gemini)": bet.ai_insight
                            })
                        
                        df = pd.DataFrame(data_rows)
                        df = df.sort_values(by="EV Extra (%)", ascending=False)
                        
                        table_placeholder.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Oportunidade (%)": st.column_config.ProgressColumn(
                                    "Modelo %",
                                    help="Probabilidade real calculada",
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=100,
                                ),
                                "Casa (%)": st.column_config.ProgressColumn(
                                    "Casa %",
                                    help="Probabilidade Implícita na Odd",
                                    format="%.1f%%",
                                    min_value=0,
                                    max_value=100,
                                ),
                                "Odd (Valor)": st.column_config.NumberColumn("Odd", format="%.2f"),
                                "EV Extra (%)": st.column_config.NumberColumn("EV", format="🔥 %.1f%%"),
                                "$$ Stake (R$)": st.column_config.NumberColumn("$$ Stake", format="R$ %.2f")
                            }
                        )

                # Predição para os próximos 14 dias (336h) acionando Callback Reativo
                report = scanner.scan(
                    min_ev=min_ev_pct / 100.0, 
                    bankroll=bankroll,
                    leagues=selected_leagues,
                    hours_window=336.0, 
                    odds_source="pinnacle",
                    progress_callback=update_ui
                )
            
            # Limpa carregamento assíncrono final
            progress_bar.empty()
            status_text.empty()
            
            if not report.value_bets:
                st.warning(f"⚠️ Nenhuma oportunidade de Valor Esperado (EV > {min_ev_pct}%) encontrada para os próximos 14 dias.")
            else:
                st.toast(f"✅ Análise concluída! Encontramos {len(report.value_bets)} oportunidades de valor.", icon="✅")
                
                # ── 1. MÉTRICAS FINAIS ─────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                total_bets = len(report.value_bets)
                total_exposure = sum(b.stake_amount for b in report.value_bets)
                max_ev = max(b.ev_pct for b in report.value_bets)
                
                m1.metric("Total de Apostas EV+", f"{total_bets}")
                m2.metric("Exposição Total", f"R$ {total_exposure:.2f}")
                m3.metric("Maior EV Encontrado", f"{max_ev:+.1f}%")
                
                st.toast(f"💡 Foram analisados {report.events_scanned} eventos em {len(selected_leagues)} ligas. Créditos da API restantes: {report.api_requests_remaining}", icon="💡")

        except Exception as e:
            st.error(f"❌ Erro crítico: {str(e)}")
            logger.exception("Erro no Launcher")

with tab2:
    st.subheader("⚙️ Configurações do Motor Analítico")
    st.info("Aqui entrará a camada de tuning do optuna (.pkl) e hiperparâmetros (Em breve)")
    st.info("Aqui entrará o módulo de Backtesting Histórico (Em breve)")
    
    st.markdown("---")
    st.write("Ajustes algoritimícos e validação de confiança fora de ambiente de Live Market operam centralizados nesta tela separada, sem poluir seu Terminal Operacional diário.")
