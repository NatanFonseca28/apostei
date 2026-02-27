"""
pregame_scanner.py
------------------
Motor de escaneamento pre-jogo para arquitetura event-driven.

Executa poucas horas antes dos jogos da rodada:
  1. Carrega features EWMA do banco local (ja atualizadas pelo ETL)
  2. Busca odds em tempo real da The Odds API (prox. ao kickoff = closing lines)
  3. Roda inferencia com o modelo otimizado (.pkl do Optuna)
  4. Cruza probabilidades do modelo com odds ao vivo
  5. Filtra apostas com EV+ superior ao limiar (padrao: 3%)
  6. Calcula staking via Kelly Fracionario
  7. Retorna dict/JSON com apostas acionaveis

Dependencias externas:
  - The Odds API (https://the-odds-api.com) — 500 req/mes gratis
  - Modelo .pkl em artifacts/ (gerado por run_optimize.py)
  - Banco SQLite atualizado (gerado por run_etl.py)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.data.models import get_engine
from .trainer import CLASSES, FEATURE_COLS
from src.core.staking import StakingConfig, CONSERVATIVE, MODERATE, fractional_kelly

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "soccer_epl"

OUTCOME_LABELS = {
    "H": "Vitoria Mandante",
    "D": "Empate",
    "A": "Vitoria Visitante",
}

# Mapa de normalizacao The Odds API -> nomes do Understat (nosso banco)
_API_TEAM_MAP = {
    "Arsenal":                     "Arsenal",
    "Aston Villa":                 "Aston Villa",
    "AFC Bournemouth":             "Bournemouth",
    "Brentford":                   "Brentford",
    "Brighton and Hove Albion":    "Brighton",
    "Burnley":                     "Burnley",
    "Chelsea":                     "Chelsea",
    "Crystal Palace":              "Crystal Palace",
    "Everton":                     "Everton",
    "Fulham":                      "Fulham",
    "Ipswich Town":                "Ipswich",
    "Leeds United":                "Leeds",
    "Leicester City":              "Leicester",
    "Liverpool":                   "Liverpool",
    "Luton Town":                  "Luton",
    "Manchester City":             "Manchester City",
    "Manchester United":           "Manchester United",
    "Newcastle United":            "Newcastle United",
    "Norwich City":                "Norwich",
    "Nottingham Forest":           "Nottingham Forest",
    "Sheffield United":            "Sheffield United",
    "Southampton":                 "Southampton",
    "Tottenham Hotspur":           "Tottenham",
    "Watford":                     "Watford",
    "West Bromwich Albion":        "West Bromwich Albion",
    "West Ham United":             "West Ham",
    "Wolverhampton Wanderers":     "Wolverhampton Wanderers",
}


# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

@dataclass
class LiveOdds:
    """Odds ao vivo de um evento da The Odds API."""
    event_id: str
    home_team: str          # Nome normalizado (nosso banco)
    away_team: str          # Nome normalizado (nosso banco)
    commence_time: str      # ISO 8601
    bookmakers: dict[str, dict[str, float]] = field(default_factory=dict)
    # bookmakers = {"pinnacle": {"H": 1.80, "D": 3.50, "A": 4.20}, "bet365": {...}}

    @property
    def pinnacle(self) -> dict[str, float]:
        return self.bookmakers.get("pinnacle", {})

    @property
    def best_odds(self) -> dict[str, float]:
        """Melhor odd disponivel para cada outcome (sharp shopping)."""
        best = {"H": 0.0, "D": 0.0, "A": 0.0}
        for bk_odds in self.bookmakers.values():
            for outcome in ("H", "D", "A"):
                if bk_odds.get(outcome, 0) > best[outcome]:
                    best[outcome] = bk_odds[outcome]
        return best


@dataclass
class ScanResult:
    """Resultado do escaneamento para uma aposta individual."""
    # Identificacao
    match_id: int | None
    event_id: str
    home_team: str
    away_team: str
    commence_time: str
    outcome: str
    outcome_label: str

    # Modelo
    model_prob: float
    model_probs: dict[str, float]

    # Odds
    odds_taken: float
    bookmaker: str
    implied_prob: float

    # Metricas
    edge: float                 # model_prob - implied_prob
    ev: float                   # (model_prob * odds) - 1
    ev_pct: float               # EV em %
    kelly_full: float           # Kelly completo
    kelly_shrunk: float         # Kelly com shrinkage
    stake_pct: float            # % da banca recomendado
    stake_amount: float         # Valor em $ (se bankroll fornecido)

    # Features usadas
    features_available: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanReport:
    """Relatorio completo do escaneamento pre-jogo."""
    timestamp: str
    sport: str
    model_name: str
    model_path: str
    features_used: list[str]
    min_ev_threshold: float
    bankroll: float
    staking_config: str

    # Resultados
    events_scanned: int
    events_matched: int         # Com features no banco
    total_value_bets: int
    value_bets: list[ScanResult]

    # API info
    api_requests_remaining: int | None = None
    api_requests_used: int | None = None

    def to_dict(self) -> dict:
        d = {
            "metadata": {
                "timestamp": self.timestamp,
                "sport": self.sport,
                "model": self.model_name,
                "model_path": self.model_path,
                "features": self.features_used,
                "min_ev_pct": round(self.min_ev_threshold * 100, 1),
                "bankroll": self.bankroll,
                "staking": self.staking_config,
            },
            "summary": {
                "events_scanned": self.events_scanned,
                "events_with_features": self.events_matched,
                "value_bets_found": self.total_value_bets,
            },
            "value_bets": [b.to_dict() for b in self.value_bets],
            "api": {
                "requests_remaining": self.api_requests_remaining,
                "requests_used": self.api_requests_used,
            },
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)


# ============================================================================
# PREGAME SCANNER
# ============================================================================

class PregameScanner:
    """
    Motor de escaneamento pre-jogo.

    Fluxo:
      1. Carrega modelo otimizado (.pkl)
      2. Busca features EWMA dos times no banco local
      3. Busca odds ao vivo da The Odds API
      4. Match: vincula eventos da API com dados do banco (por nome de time)
      5. Inferencia: roda predict_proba do modelo
      6. Filtro EV: retorna apenas apostas acima do limiar
      7. Staking: calcula tamanho recomendado via Kelly
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        db_path: str = "sqlite:///understat_premier_league.db",
        odds_api_key: str | None = None,
    ):
        self.engine = get_engine(db_path)
        self.odds_api_key = odds_api_key or os.environ.get("ODDS_API_KEY", "")

        # Carrega modelo
        self.artifact = self._load_model(model_path)
        self.pipeline = self.artifact["pipeline"]
        self.selected_features = self.artifact["selected_features"]
        self.classes = self.artifact.get("classes", CLASSES)
        self.model_path = str(model_path) if model_path else "auto"

        logger.info(
            f"PregameScanner inicializado | "
            f"Modelo: {self.artifact.get('best_params', {}).get('model_type', '?')} | "
            f"Features: {self.selected_features} | "
            f"API Key: {'configurada' if self.odds_api_key else 'AUSENTE'}"
        )

    # ─── Carga do modelo ────────────────────────────────────────────────────

    @staticmethod
    def _load_model(model_path: str | Path | None) -> dict:
        """Carrega o modelo .pkl mais recente de artifacts/."""
        if model_path is None:
            # Auto-detect: pega o .pkl mais recente
            artifacts_dir = Path("artifacts")
            if not artifacts_dir.exists():
                raise FileNotFoundError(
                    "Diretorio artifacts/ nao encontrado. "
                    "Execute 'python run_optimize.py' para gerar um modelo."
                )
            pkls = sorted(artifacts_dir.glob("best_model_*.pkl"))
            if not pkls:
                raise FileNotFoundError(
                    "Nenhum modelo .pkl encontrado em artifacts/. "
                    "Execute 'python run_optimize.py' primeiro."
                )
            model_path = pkls[-1]  # Mais recente
            logger.info(f"Auto-detectado modelo: {model_path}")

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo nao encontrado: {path}")

        with open(path, "rb") as f:
            artifact = pickle.load(f)

        model_type = artifact.get("best_params", {}).get("model_type", "?")
        ll = artifact.get("best_log_loss_cv", 0)
        logger.info(f"Modelo carregado: {path.name} ({model_type}, LogLoss CV: {ll:.4f})")

        return artifact

    # ─── Features do banco local ────────────────────────────────────────────

    def get_team_features(self) -> pd.DataFrame:
        """
        Retorna as features EWMA mais recentes para cada time no banco.

        Para cada time, pega a ultima partida (como mandante E visitante)
        e extrai as features EWMA, que representam o 'estado atual' do time.
        """
        query = """
            SELECT
                m.id, m.date, m.home_team, m.away_team,
                f.ewma5_xg_pro_home,  f.ewma10_xg_pro_home,
                f.ewma5_xg_con_home,  f.ewma10_xg_con_home,
                f.ewma5_xg_pro_away,  f.ewma10_xg_pro_away,
                f.ewma5_xg_con_away,  f.ewma10_xg_con_away
            FROM matches m
            INNER JOIN match_features f ON m.id = f.match_id
            ORDER BY m.date DESC
        """
        df = pd.read_sql(query, self.engine, parse_dates=["date"])
        logger.info(f"Carregadas {len(df)} partidas com features do banco")
        return df

    def build_feature_vector(
        self,
        home_team: str,
        away_team: str,
        all_matches: pd.DataFrame,
    ) -> np.ndarray | None:
        """
        Constroi o vetor de features para um confronto home vs away.

        Busca:
          - Ultimo jogo do home_team como mandante -> ewma_*_home
          - Ultimo jogo do away_team como visitante -> ewma_*_away

        Retorna array com as features selecionadas pelo optimizer,
        ou None se algum time nao tem historico.
        """
        # Features do mandante: ultimo jogo COMO MANDANTE
        home_matches = all_matches[all_matches["home_team"] == home_team]
        if home_matches.empty:
            logger.debug(f"Sem historico de mandante: {home_team}")
            return None
        home_row = home_matches.iloc[0]  # Mais recente (ja ordenado DESC)

        # Features do visitante: ultimo jogo COMO VISITANTE
        away_matches = all_matches[all_matches["away_team"] == away_team]
        if away_matches.empty:
            logger.debug(f"Sem historico de visitante: {away_team}")
            return None
        away_row = away_matches.iloc[0]

        # Monta vetor completo (8 features)
        full_vector = {
            "ewma5_xg_pro_home":  home_row["ewma5_xg_pro_home"],
            "ewma10_xg_pro_home": home_row["ewma10_xg_pro_home"],
            "ewma5_xg_con_home":  home_row["ewma5_xg_con_home"],
            "ewma10_xg_con_home": home_row["ewma10_xg_con_home"],
            "ewma5_xg_pro_away":  away_row["ewma5_xg_pro_away"],
            "ewma10_xg_pro_away": away_row["ewma10_xg_pro_away"],
            "ewma5_xg_con_away":  away_row["ewma5_xg_con_away"],
            "ewma10_xg_con_away": away_row["ewma10_xg_con_away"],
        }

        # Verifica NaNs
        if any(pd.isna(v) for v in full_vector.values()):
            logger.debug(f"Features incompletas para {home_team} vs {away_team}")
            return None

        # Seleciona apenas as features usadas pelo modelo
        selected = [full_vector[f] for f in self.selected_features]
        return np.array(selected).reshape(1, -1)

    # ─── Odds ao vivo da The Odds API ───────────────────────────────────────

    def fetch_live_odds(
        self,
        sport: str = ODDS_API_SPORT,
        bookmakers: list[str] | None = None,
    ) -> list[LiveOdds]:
        """
        Busca odds ao vivo de todas as casas para os jogos do dia.

        Parametros
        ----------
        sport : str
            Chave do esporte na API (default: soccer_epl).
        bookmakers : list[str], opcional
            Casas especificas. Se None, busca todas disponiveis.

        Retorna lista de LiveOdds com odds por bookmaker.
        """
        if not self.odds_api_key:
            logger.error(
                "ODDS_API_KEY nao configurada! "
                "Defina via: $env:ODDS_API_KEY = 'sua_chave' "
                "ou passe odds_api_key= no construtor. "
                "Cadastro gratis: https://the-odds-api.com"
            )
            return []

        params = {
            "apiKey": self.odds_api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        url = f"{ODDS_API_BASE}/sports/{sport}/odds"

        try:
            logger.info(f"Buscando odds ao vivo: {url}")
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()

            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            logger.info(f"The Odds API | Restantes: {remaining} | Usadas: {used}")
            self._api_remaining = int(remaining) if remaining != "?" else None
            self._api_used = int(used) if used != "?" else None

            events = resp.json()
            logger.info(f"Eventos retornados: {len(events)}")

            results = []
            for event in events:
                home_raw = event.get("home_team", "")
                away_raw = event.get("away_team", "")
                home = _API_TEAM_MAP.get(home_raw, home_raw)
                away = _API_TEAM_MAP.get(away_raw, away_raw)

                bk_odds = {}
                for bk in event.get("bookmakers", []):
                    bk_key = bk["key"]
                    for market in bk.get("markets", []):
                        if market["key"] == "h2h":
                            odds_map = {}
                            for o in market["outcomes"]:
                                if o["name"] == home_raw:
                                    odds_map["H"] = o["price"]
                                elif o["name"] == away_raw:
                                    odds_map["A"] = o["price"]
                                elif o["name"] == "Draw":
                                    odds_map["D"] = o["price"]
                            if len(odds_map) == 3:
                                bk_odds[bk_key] = odds_map

                results.append(LiveOdds(
                    event_id=event.get("id", ""),
                    home_team=home,
                    away_team=away,
                    commence_time=event.get("commence_time", ""),
                    bookmakers=bk_odds,
                ))

            return results

        except requests.RequestException as e:
            logger.error(f"Erro ao buscar odds: {e}")
            return []

    # ─── Escaneamento completo ──────────────────────────────────────────────

    def scan(
        self,
        min_ev: float = 0.03,
        bankroll: float = 1000.0,
        staking_config: StakingConfig | None = None,
        odds_source: str = "pinnacle",
        leagues: list[str] | None = None,
        bookmakers_to_fetch: list[str] | None = None,
        hours_window: float = 24.0,
        use_best_odds: bool = False,
    ) -> ScanReport:
        """
        Executa o escaneamento completo pre-jogo para múltiplas ligas.

        Fluxo:
          1. Busca odds ao vivo da API para cada liga na lista
          2. Carrega features do banco
          3. Para cada evento com features:
             a. Constroi vetor de features
             b. Roda inferencia (predict_proba)
             c. Cruza probs com odds de cada outcome
             d. Filtra os com EV >= min_ev
          4. Calcula staking Kelly
          5. Retorna ScanReport (exportavel como JSON)

        Parametros
        ----------
        min_ev : float
            EV minimo para filtrar apostas (0.03 = 3%).
        bankroll : float
            Banca disponivel para calcular stakes.
        staking_config : StakingConfig
            Configuracao de risco (default: MODERATE).
        odds_source : str
            Bookmaker preferido: "pinnacle" (sharp), "bet365", "best" (melhor odd).
        leagues : list[str], opcional
            Lista de ligas na API (default: ['soccer_epl']).
        bookmakers_to_fetch : list[str], opcional
            Casas a buscar. Se None, busca todas.
        hours_window : float
            Janela de tempo: so considera jogos nas proximas N horas.
        use_best_odds : bool
            Se True, usa a melhor odd entre todas as casas.

        Retorna
        -------
        ScanReport com apostas filtradas por EV+ >= min_ev.
        """
        config = staking_config or MODERATE
        self._api_remaining = None
        self._api_used = None
        
        target_leagues = leagues or [ODDS_API_SPORT]

        logger.info("=" * 60)
        logger.info("PREGAME SCANNER — Escaneamento Pre-Jogo (Múltiplas Ligas)")
        logger.info("=" * 60)
        logger.info(f"  EV minimo: {min_ev*100:.1f}% | Banca: ${bankroll:,.2f}")
        logger.info(f"  Ligas: {target_leagues} | Janela: {hours_window}h")

        # ── 1. Odds ao vivo ──────────────────────────────────────────────────
        logger.info("\n[1/4] Buscando odds ao vivo...")
        all_live_events = []
        
        for league in target_leagues:
            try:
                league_events = self.fetch_live_odds(league, bookmakers_to_fetch)
                if league_events:
                    all_live_events.extend(league_events)
            except Exception as e:
                logger.error(f"Erro ao buscar odds para a liga {league}: {e}")
                continue

        if not all_live_events:
            logger.warning("Nenhum evento encontrado nas ligas selecionadas. Verifique a API key.")
            return self._empty_report(min_ev, bankroll, config)

        # Filtra por janela de tempo
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_window)

        filtered_events = []
        for ev in all_live_events:
            try:
                ct = datetime.fromisoformat(ev.commence_time.replace("Z", "+00:00"))
                ct_naive = ct.replace(tzinfo=None)
                if now <= ct_naive <= cutoff:
                    filtered_events.append(ev)
            except (ValueError, TypeError):
                filtered_events.append(ev)  # Se nao parsear, inclui

        logger.info(f"  Eventos consolidados na janela de {hours_window}h: {len(filtered_events)}/{len(all_live_events)}")

        # ── 2. Features do banco ─────────────────────────────────────────────
        logger.info("\n[2/4] Carregando features do banco local...")
        all_matches = self.get_team_features()

        # ── 3. Inferencia ────────────────────────────────────────────────────
        logger.info("\n[3/4] Rodando inferencia do modelo...")
        value_bets: list[ScanResult] = []
        events_matched = 0

        for event in filtered_events:
            # Constroi features
            features = self.build_feature_vector(
                event.home_team, event.away_team, all_matches
            )
            if features is None:
                logger.debug(
                    f"  SKIP {event.home_team} vs {event.away_team} — sem features"
                )
                continue

            events_matched += 1

            # Inferencia
            probs_array = self.pipeline.predict_proba(features)[0]  # shape (3,)
            prob_map = {cls: float(probs_array[i]) for i, cls in enumerate(self.classes)}

            # Seleciona odds
            if use_best_odds:
                odds = event.best_odds
                bookmaker_label = "best"
            elif odds_source in event.bookmakers:
                odds = event.bookmakers[odds_source]
                bookmaker_label = odds_source
            else:
                # Fallback: primeira casa disponivel
                if event.bookmakers:
                    bookmaker_label = next(iter(event.bookmakers))
                    odds = event.bookmakers[bookmaker_label]
                else:
                    logger.debug(f"  SKIP {event.home_team} vs {event.away_team} — sem odds")
                    continue

            if not all(odds.get(o, 0) > 1.0 for o in ("H", "D", "A")):
                continue

            # Tenta buscar match_id do banco (para CLV posterior)
            match_id = self._find_match_id(
                event.home_team, event.away_team, event.commence_time
            )

            # ── 4. Calculo EV por outcome ────────────────────────────────────
            for outcome in ("H", "D", "A"):
                prob = prob_map[outcome]
                odd = odds[outcome]
                implied = 1.0 / odd
                edge = prob - implied
                ev = (prob * odd) - 1.0

                if ev < min_ev:
                    continue

                # Kelly
                b = odd - 1.0
                kelly_full = max(0.0, (b * prob - (1 - prob)) / b)
                kelly_shrunk = kelly_full * config.kelly_fraction
                stake_pct = min(kelly_shrunk, config.max_stake_pct)

                if stake_pct < config.min_stake_pct:
                    stake_pct = 0.0

                stake_amount = round(stake_pct * bankroll, 2)

                value_bets.append(ScanResult(
                    match_id=match_id,
                    event_id=event.event_id,
                    home_team=event.home_team,
                    away_team=event.away_team,
                    commence_time=event.commence_time,
                    outcome=outcome,
                    outcome_label=OUTCOME_LABELS.get(outcome, outcome),
                    model_prob=round(prob, 4),
                    model_probs={k: round(v, 4) for k, v in prob_map.items()},
                    odds_taken=odd,
                    bookmaker=bookmaker_label,
                    implied_prob=round(implied, 4),
                    edge=round(edge, 4),
                    ev=round(ev, 4),
                    ev_pct=round(ev * 100, 2),
                    kelly_full=round(kelly_full, 4),
                    kelly_shrunk=round(kelly_shrunk, 4),
                    stake_pct=round(stake_pct, 4),
                    stake_amount=stake_amount,
                    features_available=True,
                ))

        # Ordena por EV descendente
        value_bets.sort(key=lambda x: x.ev, reverse=True)

        logger.info(f"\n[4/4] Resultados:")
        logger.info(f"  Eventos escaneados:     {len(filtered_events)}")
        logger.info(f"  Com features no banco:  {events_matched}")
        logger.info(f"  Apostas EV+ >= {min_ev*100:.1f}%:  {len(value_bets)}")

        if value_bets:
            total_stake = sum(b.stake_amount for b in value_bets)
            logger.info(f"  Exposicao total:        ${total_stake:.2f} "
                        f"({total_stake/bankroll*100:.1f}% da banca)")

        model_type = self.artifact.get("best_params", {}).get("model_type", "unknown")

        return ScanReport(
            timestamp=datetime.now().isoformat(),
            sport=sport,
            model_name=model_type,
            model_path=self.model_path,
            features_used=self.selected_features,
            min_ev_threshold=min_ev,
            bankroll=bankroll,
            staking_config=f"Kelly x{config.kelly_fraction} | Teto {config.max_stake_pct*100:.1f}%",
            events_scanned=len(filtered_events),
            events_matched=events_matched,
            total_value_bets=len(value_bets),
            value_bets=value_bets,
            api_requests_remaining=getattr(self, "_api_remaining", None),
            api_requests_used=getattr(self, "_api_used", None),
        )

    # ─── Scan offline (usando odds do banco, para teste/backtest) ────────────

    def scan_offline(
        self,
        min_ev: float = 0.03,
        bankroll: float = 1000.0,
        staking_config: StakingConfig | None = None,
        season: int | None = None,
        limit: int = 50,
    ) -> ScanReport:
        """
        Escaneamento offline usando odds ja persistidas no banco.

        Util para:
          - Testar sem gastar creditos da API
          - Backtest rapido
          - Validacao do pipeline completo

        Usa odds Bet365 do banco como proxy de 'odds ao vivo'.
        """
        config = staking_config or MODERATE

        logger.info("PREGAME SCANNER — Modo offline (odds do banco)")
        logger.info(f"  EV minimo: {min_ev*100:.1f}% | Banca: ${bankroll:,.2f}")

        # Carrega jogos com features e odds
        where_season = f"AND m.season = {season}" if season else ""
        query = f"""
            SELECT
                m.id, m.date, m.home_team, m.away_team, m.season,
                m.home_goals, m.away_goals,
                m.odds_home_b365, m.odds_draw_b365, m.odds_away_b365,
                f.ewma5_xg_pro_home,  f.ewma10_xg_pro_home,
                f.ewma5_xg_con_home,  f.ewma10_xg_con_home,
                f.ewma5_xg_pro_away,  f.ewma10_xg_pro_away,
                f.ewma5_xg_con_away,  f.ewma10_xg_con_away
            FROM matches m
            INNER JOIN match_features f ON m.id = f.match_id
            WHERE m.odds_home_b365 IS NOT NULL
              {where_season}
            ORDER BY m.date DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, self.engine, parse_dates=["date"])
        logger.info(f"  Jogos carregados: {len(df)}")

        if df.empty:
            return self._empty_report(min_ev, bankroll, config)

        value_bets: list[ScanResult] = []
        events_matched = 0

        for _, row in df.iterrows():
            # Monta features selecionadas
            full_feats = {f: row[f] for f in FEATURE_COLS}
            if any(pd.isna(v) for v in full_feats.values()):
                continue

            selected = np.array([full_feats[f] for f in self.selected_features]).reshape(1, -1)
            events_matched += 1

            # Inferencia
            probs_array = self.pipeline.predict_proba(selected)[0]
            prob_map = {cls: float(probs_array[i]) for i, cls in enumerate(self.classes)}

            odds = {
                "H": float(row["odds_home_b365"]),
                "D": float(row["odds_draw_b365"]),
                "A": float(row["odds_away_b365"]),
            }

            if not all(o > 1.0 for o in odds.values()):
                continue

            for outcome in ("H", "D", "A"):
                prob = prob_map[outcome]
                odd = odds[outcome]
                implied = 1.0 / odd
                edge = prob - implied
                ev = (prob * odd) - 1.0

                if ev < min_ev:
                    continue

                b = odd - 1.0
                kelly_full = max(0.0, (b * prob - (1 - prob)) / b)
                kelly_shrunk = kelly_full * config.kelly_fraction
                stake_pct = min(kelly_shrunk, config.max_stake_pct)
                if stake_pct < config.min_stake_pct:
                    stake_pct = 0.0

                value_bets.append(ScanResult(
                    match_id=int(row["id"]),
                    event_id="",
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    commence_time=str(row["date"]),
                    outcome=outcome,
                    outcome_label=OUTCOME_LABELS.get(outcome, outcome),
                    model_prob=round(prob, 4),
                    model_probs={k: round(v, 4) for k, v in prob_map.items()},
                    odds_taken=odd,
                    bookmaker="bet365_db",
                    implied_prob=round(implied, 4),
                    edge=round(edge, 4),
                    ev=round(ev, 4),
                    ev_pct=round(ev * 100, 2),
                    kelly_full=round(kelly_full, 4),
                    kelly_shrunk=round(kelly_shrunk, 4),
                    stake_pct=round(stake_pct, 4),
                    stake_amount=round(stake_pct * bankroll, 2),
                    features_available=True,
                ))

        value_bets.sort(key=lambda x: x.ev, reverse=True)

        model_type = self.artifact.get("best_params", {}).get("model_type", "unknown")

        return ScanReport(
            timestamp=datetime.now().isoformat(),
            sport=ODDS_API_SPORT,
            model_name=model_type,
            model_path=self.model_path,
            features_used=self.selected_features,
            min_ev_threshold=min_ev,
            bankroll=bankroll,
            staking_config=f"Kelly x{config.kelly_fraction} | Teto {config.max_stake_pct*100:.1f}%",
            events_scanned=len(df),
            events_matched=events_matched,
            total_value_bets=len(value_bets),
            value_bets=value_bets,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _find_match_id(
        self, home_team: str, away_team: str, commence_time: str
    ) -> int | None:
        """Tenta encontrar o match_id no banco para um evento futuro."""
        try:
            ct = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            date_str = ct.strftime("%Y-%m-%d")
            query = """
                SELECT id FROM matches
                WHERE home_team = :home AND away_team = :away
                  AND date(date) = :dt
                LIMIT 1
            """
            df = pd.read_sql(
                query, self.engine,
                params={"home": home_team, "away": away_team, "dt": date_str},
            )
            return int(df.iloc[0, 0]) if not df.empty else None
        except Exception:
            return None

    def _empty_report(self, min_ev, bankroll, config) -> ScanReport:
        model_type = self.artifact.get("best_params", {}).get("model_type", "unknown")
        return ScanReport(
            timestamp=datetime.now().isoformat(),
            sport=ODDS_API_SPORT,
            model_name=model_type,
            model_path=self.model_path,
            features_used=self.selected_features,
            min_ev_threshold=min_ev,
            bankroll=bankroll,
            staking_config=f"Kelly x{config.kelly_fraction} | Teto {config.max_stake_pct*100:.1f}%",
            events_scanned=0,
            events_matched=0,
            total_value_bets=0,
            value_bets=[],
        )
