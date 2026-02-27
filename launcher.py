import streamlit as st
import time
import pandas as pd

# Configuração da página
st.set_page_config(
    page_title="APO$TEI - Launcher",
    page_icon="🎯",
    layout="centered"
)

# Estilização básica e título
st.title("🎯 APO$TEI Launcher")
st.markdown("""
Painel de Controle do Pipeline Quantitativo. 
Execute a esteira completa de extração, modelagem e cálculo de *Expected Value* em tempo real.
""")

st.divider()

# Botão principal
if st.button("🚀 Iniciar Análise Just-in-Time", type="primary", use_container_width=True):
    
    # Container de status (mostra as engrenagens ao usuário)
    with st.status("🛠️ Iniciando a esteira de dados...", expanded=True) as status:
        
        # Sprint 1: Raspagem
        st.write("⏳ **Sprint 1:** Raspagem de odds (The Odds API) e métricas (Understat)...")
        time.sleep(2.5) # Simulação
        st.write("✅ *Dados extraídos com sucesso!*")
        
        # Sprint 2: Feature Engineering
        st.write("⚙️ **Sprint 2:** Feature Engineering (Calculando EWMA, transformações temporais)...")
        time.sleep(2.0)
        st.write("✅ *Features geradas e validadas!*")
        
        # Sprint 3 e 4: Inferência
        st.write("🧠 **Sprint 3 e 4:** Inferência (Random Forest / Logistic Regression) e cálculo de EV...")
        time.sleep(3.0)
        st.write("✅ *Probabilidades inferidas e valor esperado calculado!*")
        
        # Sprint 10: Disparo de E-mail
        st.write("📧 **Sprint 10:** Registrando logs e preparando disparo de notificação por e-mail...")
        time.sleep(1.5)
        st.write("✅ *Notificações na fila de envio!*")
        
        # Finaliza o status
        status.update(label="Análise Concluída! Todas as fases finalizadas.", state="complete", expanded=False)
        
    # Feedback final
    st.success("Pipeline executado com êxito! Encontramos oportunidades de valor (EV+).")
    
    # Exibição do DataFrame com apostas EV+
    st.subheader("📊 Oportunidades de Valor Esperado Positivo")
    
    mock_data = [
        {"Partida": "Arsenal vs Chelsea",     "Mercado": "Match Winner", "Tip": "Home", "Prob. Modelo": "55.0%", "Odd Casa": 2.15, "EV": "+18.2%", "Stake Sugerida": "$32.50"},
        {"Partida": "Liverpool vs Man Utd",   "Mercado": "Match Winner", "Tip": "Home", "Prob. Modelo": "68.5%", "Odd Casa": 1.55, "EV": "+6.1%", "Stake Sugerida": "$55.00"},
        {"Partida": "Aston Villa vs Brighton","Mercado": "Match Winner", "Tip": "Away", "Prob. Modelo": "38.0%", "Odd Casa": 3.10, "EV": "+17.8%", "Stake Sugerida": "$15.00"},
    ]
    
    df_bets = pd.DataFrame(mock_data)
    
    # Renderiza o DataFrame formatado nativo do Streamlit
    st.dataframe(
        df_bets,
        use_container_width=True,
        hide_index=True
    )
    
    st.info("💡 As tips acima já foram encaminhadas para o e-mail configurado juntamente com a recomendação de *Staking*.")
