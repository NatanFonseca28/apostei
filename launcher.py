"""
launcher.py
-----------
Interface principal do APO$TEI — painel de controle quantitativo.
"""

import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="APO$TEI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Profissional ────────────────────────────────────────────────────────

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; }
section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0e2a1a 100%);
    border: 1px solid #00ff8844;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ""; position: absolute; top:0;left:0;right:0;bottom:0;
    background: radial-gradient(ellipse at 80% 50%, #00ff2215 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(90deg,#00ff88,#00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.hero-sub { color: #8b949e; font-size: 0.95rem; margin: 0; }
.hero-badge {
    display: inline-block; margin-top: 0.7rem;
    background: #00ff8820; color: #00ff88;
    border: 1px solid #00ff8855; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; padding: 0.2rem 0.7rem;
    letter-spacing: 0.5px; text-transform: uppercase;
}

/* ── Metric card ── */
.metric-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 1.1rem 1.3rem; text-align: center; transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00ff8855; }
.metric-value {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg,#00ff88,#00d4ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-label { color: #8b949e; font-size: 0.78rem; font-weight: 500; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Bet list item ── */
.bet-list-item {
    display: flex; align-items: flex-start; gap: 0.6rem;
    padding: 0.7rem 0.9rem; background: #0d1117; border-radius: 8px;
    margin-bottom: 0.4rem; border: 1px solid #21262d;
}
.bet-list-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 0.05rem; }
.bet-list-text { color: #c9d1d9; font-size: 0.88rem; line-height: 1.5; }
.bet-list-label { font-weight: 600; color: #e6edf3; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #161b22; border-radius: 10px;
    padding: 4px; border: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 8px;
    color: #8b949e !important; font-size: 0.85rem; font-weight: 500; padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] { background: #21262d !important; color: #e6edf3 !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#00a854,#00d4ff) !important;
    color: #0d1117 !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    font-size: 0.95rem !important; padding: 0.7rem 2rem !important;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #21262d; margin: 1rem 0; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg,#00ff88,#00d4ff) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161b22 !important; border-radius: 10px !important;
    color: #c9d1d9 !important;
}
.streamlit-expanderContent {
    background: #161b22 !important; border: 1px solid #30363d;
    border-top: none; border-radius: 0 0 10px 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 1.5rem 0;">
            <div style="font-size:2rem;margin-bottom:0.3rem;">🎯</div>
            <div style="font-size:1.3rem;font-weight:700;
                background:linear-gradient(90deg,#00ff88,#00d4ff);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">APO$TEI</div>
            <div style="font-size:0.72rem;color:#8b949e;letter-spacing:1px;text-transform:uppercase;">
                Motor Quantitativo v2.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Autenticação ──
    st.markdown("#### 🔑 Autenticação")
    api_key = st.text_input(
        "The Odds API Key",
        type="password",
        placeholder="Deixe em branco para usar .env",
        help="Obtenha em https://the-odds-api.com — 500 req/mês grátis",
    )
    gemini_key_override = st.text_input(
        "Gemini API Key (opcional)",
        type="password",
        placeholder="Deixe em branco para usar .env",
        help="Necessária para Laudos IA — https://aistudio.google.com",
    )

    st.divider()

    # ── Ligas ──
    st.markdown("#### 🏆 Ligas Alvo")
    LEAGUE_OPTIONS = {
        "🇧🇷  Brasileirão Série A": "soccer_brazil_campeonato",
        "🏴󠁧󠁢󠁥󠁮󠁧󠁿  Premier League": "soccer_epl",
        "🏆  Champions League": "soccer_uefa_champs_league",
        "🇪🇸  La Liga": "soccer_spain_la_liga",
        "🇮🇹  Serie A (Itália)": "soccer_italy_serie_a",
        "🇫🇷  Ligue 1": "soccer_france_ligue_one",
        "🇩🇪  Bundesliga": "soccer_germany_bundesliga",
        "🌎  Copa Libertadores": "soccer_conmebol_copa_libertadores",
    }

    selected_league_names = st.multiselect(
        "Selecione as ligas",
        options=list(LEAGUE_OPTIONS.keys()),
        default=["🇧🇷  Brasileirão Série A", "🏴󠁧󠁢󠁥󠁮󠁧󠁿  Premier League"],
        label_visibility="collapsed",
    )
    selected_leagues = [LEAGUE_OPTIONS[n] for n in selected_league_names]

    st.divider()

    # ── Gestão de Banca ──
    st.markdown("#### 💰 Gestão de Banca")
    bankroll = st.number_input(
        "Banca Disponível (R$)",
        min_value=10.0,
        value=1000.0,
        step=100.0,
        format="%.2f",
    )

    st.divider()

    # ── Filtros ──
    st.markdown("#### ⚡ Filtros de Qualidade")

    col_ev_label, col_ev_check = st.columns([3, 2])
    with col_ev_label:
        st.markdown("<small style='color:#8b949e'>Filtro de EV Mínimo</small>", unsafe_allow_html=True)
    with col_ev_check:
        disable_ev_filter = st.checkbox(
            "Ignorar",
            value=False,
            help="Exibe todas as apostas sem filtrar por EV mínimo.",
        )

    if disable_ev_filter:
        min_ev_pct = 0.0
        st.caption("🟡 Filtro EV **desativado** — todas as apostas são exibidas.")
    else:
        min_ev_pct = st.slider(
            "EV Mínimo (%)",
            min_value=0.0,
            max_value=15.0,
            value=3.0,
            step=0.5,
            help="Filtra apostas com Valor Esperado acima desta porcentagem.",
        )

    hours_window = st.slider(
        "Janela de tempo (horas)",
        min_value=1,
        max_value=336,
        value=72,
        step=1,
        help="Analisa jogos nas próximas N horas.",
    )

    st.divider()
    st.markdown(
        """
        <div style="color:#8b949e;font-size:0.72rem;text-align:center;padding:0.5rem 0;">
            📌 Benchmark: <b style="color:#58a6ff">Pinnacle</b> (Sharp Line)<br>
            Metodologia: Kelly Fracionário + EWMA<br><br>
            <i>Apostas envolvem risco financeiro.<br>Aposte com responsabilidade.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Hero ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <p class="hero-title">APO$TEI</p>
        <p class="hero-sub">Motor Quantitativo de Expected Value — análise pré-jogo em tempo real com ML + xG</p>
        <span class="hero-badge">⚡ Sistema em Produção</span>
        <span class="hero-badge" style="margin-left:0.5rem;">📊 ML + Sofascore xG</span>
        <span class="hero-badge" style="margin-left:0.5rem;">🤖 IA Gemini</span>
        <span class="hero-badge" style="margin-left:0.5rem;">🔷 Kelly Fracionário</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab_ops, tab_apostas, tab_config = st.tabs(["📡  Terminal de Operações", "🎰  Lista de Apostas", "⚙️  Configurações do Motor"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TERMINAL DE OPERAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ops:
    c_btn, c_info = st.columns([2, 3], gap="medium")
    with c_btn:
        run_analysis = st.button(
            "🚀  Iniciar Análise (Live Market)",
            type="primary",
            use_container_width=True,
        )
    with c_info:
        if disable_ev_filter:
            st.info("🟡 Filtro EV desativado — todas as apostas serão exibidas.", icon="ℹ️")
        else:
            st.info(
                f"🎯 EV mínimo: **{min_ev_pct}%**  ·  Janela: **{hours_window}h**  ·  Banca: **R$ {bankroll:,.2f}**  ·  Ligas: **{len(selected_leagues)}**",
                icon="📊",
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if run_analysis:
        effective_api_key = api_key or ODDS_API_KEY
        if not effective_api_key:
            st.error(
                "❌ **API Key ausente.** Configure `ODDS_API_KEY` no `.env` ou insira na sidebar.",
                icon="🔐",
            )
            st.stop()

        if not selected_leagues:
            selected_leagues = list(LEAGUE_OPTIONS.values())
            st.toast("ℹ️ Nenhuma liga selecionada. Usando todas.", icon="ℹ️")

        if gemini_key_override:
            os.environ["GEMINI_API_KEY"] = gemini_key_override

        with st.spinner("⚙️ Inicializando motor…"):
            from src.data.models import create_tables, get_engine
            from src.ml.pregame_scanner import PregameScanner

            db_engine = get_engine("sqlite:///flashscore_data.db")
            create_tables(db_engine)

        scanner = PregameScanner(
            db_path="sqlite:///flashscore_data.db",
            odds_api_key=effective_api_key,
        )

        try:
            prog_bar = st.progress(0, text="Inicializando…")
            status_ph = st.empty()
            kpi_ph = st.empty()
            table_ph = st.empty()

            def _render_kpis(bets):
                ev_bets = [b for b in bets if b.ev_pct >= min_ev_pct]
                if not ev_bets:
                    return
                total_stake = sum(b.stake_amount for b in ev_bets)
                max_ev = max(b.ev_pct for b in ev_bets)
                avg_ev = sum(b.ev_pct for b in ev_bets) / len(ev_bets)
                with kpi_ph.container():
                    c1, c2, c3, c4 = st.columns(4)
                    for col, val, label in [
                        (c1, str(len(ev_bets)), "Apostas EV+"),
                        (c2, f"R$ {total_stake:,.0f}", "Exposição Total"),
                        (c3, f"{max_ev:+.1f}%", "Maior EV"),
                        (c4, f"{avg_ev:.1f}%", "EV Médio"),
                    ]:
                        col.markdown(
                            f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>',
                            unsafe_allow_html=True,
                        )

            def update_ui(current_bets, current_idx, total_events):
                pct = int(current_idx / max(1, total_events) * 100)
                prog_bar.progress(pct, text=f"Evento {current_idx}/{total_events}…")
                ev_count = len([b for b in current_bets if b.ev_pct >= min_ev_pct])
                status_ph.caption(f"🔍 {current_idx}/{total_events} analisados — **{ev_count}** oportunidades de valor")
                _render_kpis(current_bets)
                _render_table(current_bets, table_ph)

            def _render_table(bets, placeholder):
                display = bets if disable_ev_filter else [b for b in bets if b.ev_pct >= min_ev_pct]
                if not display:
                    return
                rows = []
                for b in display:
                    xg_h = b.home_features.get("media_xg_casa", 0) or 0
                    xg_a = b.away_features.get("media_xg_fora", 0) or 0
                    poss = b.home_features.get("media_posse_casa", 0) or 0
                    shots = b.home_features.get("media_chutes_alvo_casa", 0) or 0
                    rows.append(
                        {
                            "Partida": f"{b.home_team} × {b.away_team}",
                            "Horário": b.commence_time[:16].replace("T", " "),
                            "Tip": b.outcome_label,
                            "Modelo %": round(b.model_prob * 100, 1),
                            "Casa %": round(b.implied_prob * 100, 1),
                            "Odd": b.odds_taken,
                            "EV %": b.ev_pct,
                            "Stake R$": b.stake_amount,
                            "xG Casa": round(xg_h, 2) if xg_h else "–",
                            "xG Fora": round(xg_a, 2) if xg_a else "–",
                            "Posse %": round(poss, 1) if poss else "–",
                            "Chutes Alvo": round(shots, 1) if shots else "–",
                            "Casa": b.bookmaker.upper(),
                        }
                    )
                df_show = pd.DataFrame(rows).sort_values("EV %", ascending=False)
                placeholder.dataframe(
                    df_show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Modelo %": st.column_config.ProgressColumn("Modelo", format="%.1f%%", min_value=0, max_value=100),
                        "Casa %": st.column_config.ProgressColumn("Implícito", format="%.1f%%", min_value=0, max_value=100),
                        "Odd": st.column_config.NumberColumn("Odd", format="%.2f"),
                        "EV %": st.column_config.NumberColumn("EV", format="🔥 %.1f%%"),
                        "Stake R$": st.column_config.NumberColumn("Stake", format="R$ %.2f"),
                    },
                )

            with st.spinner(f"🚀 Escaneando {len(selected_leagues)} liga(s) — janela {hours_window}h…"):
                report = scanner.scan(
                    min_ev=min_ev_pct / 100.0,
                    bankroll=bankroll,
                    leagues=selected_leagues,
                    hours_window=float(hours_window),
                    odds_source="pinnacle",
                    progress_callback=update_ui,
                )

            prog_bar.empty()
            status_ph.empty()

            if not report.value_bets:
                st.warning(
                    f"⚠️ Nenhuma oportunidade encontrada (EV > {min_ev_pct}%) nos próximos {hours_window}h.",
                    icon="📭",
                )
            else:
                display_bets = report.value_bets if disable_ev_filter else [b for b in report.value_bets if b.ev_pct >= min_ev_pct]
                _render_kpis(report.value_bets)
                _render_table(report.value_bets, table_ph)

                st.caption(f"📡 {report.events_scanned} eventos escaneados · {report.events_matched} com dados no banco · Créditos API restantes: {report.api_requests_remaining}")

                st.session_state["last_report"] = report
                st.session_state["last_display_bets"] = display_bets
                st.session_state["last_bankroll"] = bankroll

                st.success(
                    f"✅ Análise concluída! **{len(display_bets)}** oportunidade(s). Vá para **🎰 Lista de Apostas** para ver as recomendações detalhadas.",
                    icon="✅",
                )

        except Exception as e:
            st.error(f"❌ Erro crítico no pipeline: {str(e)}", icon="🚨")
            logger.exception("Erro no Launcher")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LISTA DE APOSTAS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_apostas:
    if "last_report" not in st.session_state:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 2rem;color:#8b949e;">
                <div style="font-size:3rem;margin-bottom:1rem;">📡</div>
                <div style="font-size:1.1rem;font-weight:500;color:#c9d1d9;">Nenhuma análise executada ainda</div>
                <div style="font-size:0.88rem;margin-top:0.5rem;">
                    Execute uma análise no <b>Terminal de Operações</b> para ver as apostas recomendadas aqui.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        report = st.session_state["last_report"]
        display_bets = st.session_state.get("last_display_bets", report.value_bets)
        bankroll_disp = st.session_state.get("last_bankroll", 1000.0)

        # Agrupa por partida
        matches: dict[str, list] = {}
        for b in display_bets:
            key = f"{b.home_team} × {b.away_team}"
            matches.setdefault(key, []).append(b)

        if not matches:
            st.warning("Nenhuma aposta de valor encontrada para exibir.")
        else:
            st.markdown(
                f"""
                <div style="margin-bottom:1.5rem;">
                    <span style="font-size:1.4rem;font-weight:700;color:#e6edf3;">🎰 Lista de Apostas Recomendadas</span><br>
                    <span style="color:#8b949e;font-size:0.88rem;">
                        {len(matches)} partida(s) · {len(display_bets)} tip(s) · 
                        {report.timestamp[:16].replace("T", " ")}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for match_key, bets in matches.items():
                b0 = bets[0]
                hf = b0.home_features
                af = b0.away_features
                probs = b0.model_probs

                best_bet = max(bets, key=lambda x: x.ev_pct)
                ev_color = "#00ff88" if best_bet.ev_pct >= 3 else ("#e3b341" if best_bet.ev_pct >= 0 else "#ff6b6b")

                with st.expander(
                    f"⚽  {match_key}  ·  {b0.commence_time[:16].replace('T', ' ')}  ·  Melhor EV: {best_bet.ev_pct:+.1f}%",
                    expanded=True,
                ):
                    col_left, col_right = st.columns([3, 2], gap="large")

                    with col_left:
                        # Probabilidades
                        st.markdown("**📊 Probabilidades do Modelo vs Mercado**")
                        outcome_labels_pt = {
                            "H": f"Vitória {b0.home_team}",
                            "D": "Empate",
                            "A": f"Vitória {b0.away_team}",
                        }
                        outcome_icons = {"H": "🏠", "D": "🤝", "A": "✈️"}

                        for code, lbl in outcome_labels_pt.items():
                            model_p = probs.get(code, 0) * 100
                            relevant = next((b for b in bets if b.outcome == code), None)
                            implied_p = (relevant.implied_prob * 100) if relevant else None
                            is_best = code == best_bet.outcome
                            ca, cb, cc = st.columns([3, 2, 2])
                            badge = "⭐ " if is_best else ""
                            ca.markdown(f"**{badge}{outcome_icons[code]} {lbl}**")
                            cb.markdown(f"🤖 `{model_p:.1f}%`")
                            if implied_p and relevant:
                                cc.metric(
                                    f"@{relevant.odds_taken}",
                                    f"{implied_p:.1f}%",
                                    f"{model_p - implied_p:+.1f}pp · EV {relevant.ev_pct:+.1f}%",
                                    delta_color="normal" if model_p >= implied_p else "inverse",
                                    label_visibility="visible",
                                )

                        st.markdown('<hr class="divider">', unsafe_allow_html=True)

                        # Lista de apostas sugeridas
                        st.markdown("**🎰 O que apostar neste jogo:**")
                        suggestions = _generate_bet_suggestions(b0, bets, hf, af, probs)
                        for s in suggestions:
                            conf_html = ""
                            if s.get("confidence"):
                                c_color = {
                                    "ALTA": "#00ff88",
                                    "MÉDIA": "#e3b341",
                                    "BAIXA": "#8b949e",
                                }.get(s["confidence"], "#8b949e")
                                conf_html = f' <span style="color:{c_color};font-weight:700;font-size:0.75rem;padding:0.1rem 0.4rem;background:{c_color}20;border-radius:4px;">{s["confidence"]}</span>'
                            st.markdown(
                                f"""
                                <div class="bet-list-item">
                                    <div class="bet-list-icon">{s["icon"]}</div>
                                    <div class="bet-list-text">
                                        <span class="bet-list-label">{s["title"]}</span>{conf_html}<br>
                                        {s["desc"]}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Laudo IA
                        if b0.ai_insight and not b0.ai_insight.startswith("Sem valor"):
                            st.markdown('<hr class="divider">', unsafe_allow_html=True)
                            st.markdown("**🤖 Laudo da IA (Gemini)**")
                            st.markdown(b0.ai_insight)

                    with col_right:
                        # Stats avançadas
                        st.markdown("**📈 Estatísticas Avançadas**")
                        xg_h = hf.get("media_xg_casa", 0) or 0
                        xg_a = af.get("media_xg_fora", 0) or 0
                        xga_h = hf.get("media_xga_casa", 0) or 0
                        xga_a = af.get("media_xga_fora", 0) or 0
                        shots_h = hf.get("media_chutes_alvo_casa", 0) or 0
                        shots_a = af.get("media_chutes_alvo_fora", 0) or 0
                        poss_h = hf.get("media_posse_casa", 0) or 0
                        poss_a = af.get("media_posse_fora", 0) or 0
                        gols_h = hf.get("media_marcados_casa", 0) or 0
                        gols_a = af.get("media_marcados_fora", 0) or 0
                        gc_h = hf.get("media_sofridos_casa", 0) or 0
                        gc_a = af.get("media_sofridos_fora", 0) or 0
                        has_adv = xg_h > 0 or xg_a > 0

                        stats_rows = [
                            ("⚽ Gols Marcados", f"{gols_h:.2f}", f"{gols_a:.2f}"),
                            ("🛡️ Gols Sofridos", f"{gc_h:.2f}", f"{gc_a:.2f}"),
                        ]
                        if has_adv:
                            stats_rows += [
                                ("📐 xG Produzido", f"{xg_h:.2f}", f"{xg_a:.2f}"),
                                ("📐 xG Concedido", f"{xga_h:.2f}", f"{xga_a:.2f}"),
                                ("🎯 Chutes no Alvo", f"{shots_h:.1f}", f"{shots_a:.1f}"),
                                ("🔵 Posse Bola", f"{poss_h:.1f}%", f"{poss_a:.1f}%"),
                            ]

                        st.dataframe(
                            pd.DataFrame(
                                stats_rows,
                                columns=["Stat", b0.home_team[:18], b0.away_team[:18]],
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                        if not has_adv:
                            st.caption("ℹ️ xG/Posse não disponíveis — rode o ETL Sofascore")

                        st.markdown('<hr class="divider">', unsafe_allow_html=True)

                        # Stakes
                        st.markdown("**💰 Kelly Fracionário**")
                        for b in bets:
                            if b.stake_amount > 0 or disable_ev_filter:
                                ev_c = "#00ff88" if b.ev_pct >= 3 else ("#e3b341" if b.ev_pct >= 0 else "#ff6b6b")
                                st.markdown(
                                    f"""
                                    <div style="display:flex;justify-content:space-between;align-items:center;
                                        padding:0.4rem 0.7rem;background:#0d1117;border-radius:8px;
                                        margin-bottom:0.3rem;border:1px solid #21262d;">
                                        <span style="color:#c9d1d9;font-size:0.85rem;"><b>{b.outcome_label}</b></span>
                                        <span>
                                            <span style="color:{ev_c};font-weight:700;font-size:0.82rem;">
                                                EV {b.ev_pct:+.1f}%
                                            </span>
                                            &nbsp;·&nbsp;
                                            <span style="color:#58a6ff;font-weight:600;font-size:0.85rem;">
                                                R$ {b.stake_amount:.2f}
                                            </span>
                                        </span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                st.markdown("")  # Espaçamento


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONFIGURAÇÕES DO MOTOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_config:
    st.markdown("### ⚙️ Configurações do Motor Analítico")

    st.markdown(
        """
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
            <div style="color:#e6edf3;font-weight:600;margin-bottom:0.5rem;">📦 Modelo ML</div>
            <div style="color:#8b949e;font-size:0.88rem;">
                Modelo carregado automaticamente de <code>artifacts/best_model_*.pkl</code>.<br>
                Para retreinar: <code>python runners/run_optimize.py</code>
            </div>
        </div>
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
            <div style="color:#e6edf3;font-weight:600;margin-bottom:0.5rem;">📊 Atualização de Dados</div>
            <div style="color:#8b949e;font-size:0.88rem;">
                • <code>python runners/run_etl.py</code> → Flashscore (Selenium, form/gols)<br>
                • <code>python runners/run_etl_advanced.py --all-leagues --seasons 2025 2026</code> → Sofascore (xG/Posse/Chutes)
            </div>
        </div>
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.5rem;">
            <div style="color:#e6edf3;font-weight:600;margin-bottom:0.5rem;">📈 Backtesting</div>
            <div style="color:#8b949e;font-size:0.88rem;">
                <code>python runners/run_backtest.py</code> — resultados em <code>artifacts/backtest_report.txt</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🗄️ Status do Banco de Dados")
    try:
        import sqlite3 as _sqlite3

        _conn = _sqlite3.connect("flashscore_data.db")
        _cur = _conn.cursor()
        _tables = ["flashscore_matches", "match_advanced_stats", "ai_predictions_cache"]
        _counts = {}
        for _t in _tables:
            try:
                _cur.execute(f"SELECT COUNT(*) FROM {_t}")
                _counts[_t] = _cur.fetchone()[0]
            except Exception:
                _counts[_t] = "–"
        _conn.close()

        _cols = st.columns(3)
        for _col, (_tbl, _cnt) in zip(_cols, _counts.items()):
            _col.markdown(
                f'<div class="metric-card"><div class="metric-value">{_cnt:,}</div><div class="metric-label">{_tbl.replace("_", " ").title()}</div></div>',
                unsafe_allow_html=True,
            )
    except Exception as _e:
        st.caption(f"ℹ️ Banco não disponível: {_e}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — GERADOR DE LISTA DE APOSTAS SUGERIDAS
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_bet_suggestions(b0, bets, hf, af, probs) -> list[dict]:
    """
    Gera lista amigável de apostas com base em todas as stats disponíveis:
    gols, xG, chutes no alvo, posse, probabilidades do modelo e EV.
    """
    suggestions = []

    gols_h = hf.get("media_marcados_casa", 0) or 0
    gols_a = af.get("media_marcados_fora", 0) or 0
    gc_h = hf.get("media_sofridos_casa", 0) or 0
    gc_a = af.get("media_sofridos_fora", 0) or 0
    xg_h = hf.get("media_xg_casa", 0) or 0
    xg_a = af.get("media_xg_fora", 0) or 0
    xga_h = hf.get("media_xga_casa", 0) or 0
    xga_a = af.get("media_xga_fora", 0) or 0
    shots_h = hf.get("media_chutes_alvo_casa", 0) or 0
    shots_a = af.get("media_chutes_alvo_fora", 0) or 0
    poss_h = hf.get("media_posse_casa", 0) or 0
    poss_a = af.get("media_posse_fora", 0) or 0
    has_adv = xg_h > 0 or xg_a > 0

    home_prob = probs.get("H", 0)
    draw_prob = probs.get("D", 0)
    away_prob = probs.get("A", 0)

    outcome_labels_pt = {
        "H": f"Vitória {b0.home_team}",
        "D": "Empate",
        "A": f"Vitória {b0.away_team}",
    }
    icons_1x2 = {"H": "🏠", "D": "🤝", "A": "✈️"}

    # ── 1. Melhor tip 1X2 com EV+ ─────────────────────────────────────────
    best_1x2 = max(bets, key=lambda b: b.ev_pct)
    if best_1x2.ev_pct >= 0:
        conf = "ALTA" if best_1x2.ev_pct >= 5 else ("MÉDIA" if best_1x2.ev_pct >= 2 else "BAIXA")
        suggestions.append(
            {
                "icon": icons_1x2.get(best_1x2.outcome, "⚽"),
                "title": outcome_labels_pt[best_1x2.outcome],
                "desc": (f"Modelo: **{best_1x2.model_prob * 100:.1f}%** · Casa @{best_1x2.odds_taken} ({best_1x2.implied_prob * 100:.1f}% implícito) · EV **{best_1x2.ev_pct:+.1f}%**"),
                "confidence": conf,
            }
        )

    # ── 2. Over/Under 2.5 ─────────────────────────────────────────────────
    indicator = (xg_h + xg_a) if has_adv else (gols_h + gols_a)
    xg_tag = "xG combinado" if has_adv else "Gols/jogo combinado"
    if indicator > 0:
        if indicator >= 2.8:
            suggestions.append(
                {
                    "icon": "📈",
                    "title": "Over 2.5 Gols",
                    "desc": (f"{xg_tag} de **{indicator:.2f}** — histórico aponta jogo aberto." + (f" Gols/jogo: {gols_h + gols_a:.2f}." if has_adv else "")),
                    "confidence": "ALTA" if indicator >= 3.2 else "MÉDIA",
                }
            )
        elif indicator <= 2.0:
            suggestions.append(
                {
                    "icon": "📉",
                    "title": "Under 2.5 Gols",
                    "desc": f"{xg_tag} de **{indicator:.2f}** — tendência de jogo fechado.",
                    "confidence": "MÉDIA",
                }
            )

    # ── 3. Ambas Marcam (BTTS) ─────────────────────────────────────────────
    if gols_h >= 1.2 and gols_a >= 1.0 and gc_h >= 1.0 and gc_a >= 1.2:
        suggestions.append(
            {
                "icon": "💥",
                "title": "Ambas Marcam — Sim",
                "desc": (f"{b0.home_team} marca **{gols_h:.2f}** e sofre **{gc_h:.2f}**/jogo em casa. {b0.away_team} marca **{gols_a:.2f}** e sofre **{gc_a:.2f}**/jogo fora."),
                "confidence": "MÉDIA",
            }
        )

    # ── 4. Posse dominante ────────────────────────────────────────────────
    if poss_h > 0 and poss_a > 0:
        dominant = b0.home_team if poss_h >= poss_a else b0.away_team
        dominant_poss = max(poss_h, poss_a)
        suggestions.append(
            {
                "icon": "🔵",
                "title": f"Maior Posse: {dominant}",
                "desc": (f"{b0.home_team} **{poss_h:.1f}%** posse em casa · {b0.away_team} **{poss_a:.1f}%** fora."),
                "confidence": "MÉDIA" if dominant_poss >= 55 else "BAIXA",
            }
        )

    # ── 5. xG dominante ───────────────────────────────────────────────────
    if has_adv and xg_h > 0 and xg_a > 0 and abs(xg_h - xg_a) >= 0.3:
        xg_dom = b0.home_team if xg_h >= xg_a else b0.away_team
        diff = abs(xg_h - xg_a)
        suggestions.append(
            {
                "icon": "📐",
                "title": f"Vantagem de xG: {xg_dom}",
                "desc": (f"xG produzido: {b0.home_team} **{xg_h:.2f}** × {b0.away_team} **{xg_a:.2f}** (+{diff:.2f} xG de diferença)."),
                "confidence": "ALTA" if diff >= 0.6 else "MÉDIA",
            }
        )

    # ── 6. Chutes no alvo ─────────────────────────────────────────────────
    if has_adv and shots_h > 0 and shots_a > 0 and abs(shots_h - shots_a) >= 1.5:
        sh_dom = b0.home_team if shots_h >= shots_a else b0.away_team
        suggestions.append(
            {
                "icon": "🎯",
                "title": f"Mais Chutes no Alvo: {sh_dom}",
                "desc": (f"{b0.home_team} **{shots_h:.1f}** chutes/jogo no alvo (casa) · {b0.away_team} **{shots_a:.1f}** (fora)."),
                "confidence": "BAIXA",
            }
        )

    # ── 7. Empate com valor ───────────────────────────────────────────────
    d_bet = next((b for b in bets if b.outcome == "D"), None)
    if draw_prob >= 0.32 and d_bet and d_bet.ev_pct >= 2:
        suggestions.append(
            {
                "icon": "🤝",
                "title": "Empate com Valor",
                "desc": (f"Modelo aponta **{draw_prob * 100:.1f}%** de empate — odd @{d_bet.odds_taken} tem EV **{d_bet.ev_pct:+.1f}%**."),
                "confidence": "MÉDIA",
            }
        )

    # ── 8. Clean sheet ────────────────────────────────────────────────────
    for team, gc in [(b0.home_team, gc_h), (b0.away_team, gc_a)]:
        if gc > 0 and gc <= 0.75:
            suggestions.append(
                {
                    "icon": "🛡️",
                    "title": f"Clean Sheet: {team}",
                    "desc": f"{team} sofre apenas **{gc:.2f}** gols/jogo — considere mercado de Clean Sheet.",
                    "confidence": "BAIXA",
                }
            )

    if not suggestions:
        suggestions.append(
            {
                "icon": "⚠️",
                "title": "Dados insuficientes",
                "desc": "Rode o ETL completo (Flashscore + Sofascore) para gerar recomendações detalhadas.",
                "confidence": "",
            }
        )

    return suggestions
