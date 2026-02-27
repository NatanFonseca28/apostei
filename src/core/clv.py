"""
clv.py
------
Sistema de Auditoria de CLV (Closing Line Value) para apostas esportivas.

CLV e o padrao-ouro para avaliar a qualidade de um modelo de apostas.
Mede se voce esta consistentemente capturando odds com valor ANTES que
o mercado as corrija para baixo no fechamento.

Conceito:
    CLV = (Odd Apostada / Odd de Fechamento) - 1

    Se apostamos no Mandante a @2.00 e o mercado fechou a @1.80:
        CLV = (2.00 / 1.80) - 1 = +11.1%  → Batemos a linha de fechamento.

    Isso significa que conseguimos um preco MELHOR que o mercado eficiente
    no momento do apito — evidencia forte de edge real, independente do
    resultado em campo.

Por que CLV importa:
    - Resultados de curto prazo sao dominados por variancia (sorte/azar)
    - CLV positivo consistente e evidencia ESTATISTICA de edge real
    - Casas profissionais (Pinnacle) limitam apostadores com CLV+ sustentado
    - Um modelo com CLV medio > 0% ao longo de centenas de apostas
      e quase certamente lucrativo no longo prazo

Fontes de odds de fechamento:
    1. Banco local (football-data.co.uk) — odds Pinnacle/Avg/Max ja persistidas
       como odds de fechamento (capturadas proximo ao kickoff)
    2. The Odds API (api.the-odds-api.com) — para jogos futuros/tempo real
       Plano gratuito: 500 requisicoes/mes

Uso tipico:
    # Backtest historico (usa banco local)
    auditor = CLVAuditor(engine)
    report = auditor.backtest_historical(bets_df, closing_source="pinnacle")
    print(report)

    # Jogo futuro (usa API)
    closing = auditor.fetch_closing_from_api(sport="soccer_epl", match_id="xxx")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd
import requests

from .models import get_engine, get_session

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS E CONSTANTES
# ============================================================================

class ClosingSource(str, Enum):
    """Fonte das odds de fechamento para calculo do CLV."""
    PINNACLE = "pinnacle"   # Sharp line — referencia do mercado
    AVERAGE  = "average"    # Media do mercado
    BET365   = "bet365"     # Bet365 (popular, boa liquidez)
    MAX      = "max"        # Melhor odd disponivel no mercado

    @property
    def column_map(self) -> dict[str, str]:
        """Mapeamento outcome -> coluna no banco para esta fonte."""
        _maps = {
            "pinnacle": {"H": "odds_home_pin",  "D": "odds_draw_pin",  "A": "odds_away_pin"},
            "average":  {"H": "odds_home_avg",  "D": "odds_draw_avg",  "A": "odds_away_avg"},
            "bet365":   {"H": "odds_home_b365", "D": "odds_draw_b365", "A": "odds_away_b365"},
            "max":      {"H": "odds_home_max",  "D": "odds_draw_max",  "A": "odds_away_max"},
        }
        return _maps[self.value]


# Outcomes legíveis
OUTCOME_LABELS = {
    "H": "Vitória Mandante",
    "D": "Empate",
    "A": "Vitória Visitante",
}


# The Odds API
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT = "soccer_epl"


# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

@dataclass
class BetRecord:
    """
    Registro de uma aposta sugerida pelo modelo.

    Atributos
    ---------
    match_id : int
        ID da partida (Understat ID, presente na tabela matches).
    date : datetime
        Data/hora da partida.
    home_team : str
        Time mandante.
    away_team : str
        Time visitante.
    outcome : str
        Lado apostado: "H" (home), "D" (draw), "A" (away).
    odds_taken : float
        Odd decimal no momento em que a aposta foi feita/sugerida.
    model_prob : float
        Probabilidade estimada pelo modelo para o outcome.
    stake_pct : float
        Percentual da banca apostado (0.01 = 1%).
    """
    match_id: int
    date: datetime
    home_team: str
    away_team: str
    outcome: str          # "H", "D", "A"
    odds_taken: float     # Odd no momento da aposta
    model_prob: float     # Prob do modelo
    stake_pct: float = 0.0  # % da banca apostado

    def __post_init__(self):
        if self.outcome not in ("H", "D", "A"):
            raise ValueError(f"outcome deve ser 'H', 'D' ou 'A'. Recebido: {self.outcome}")
        if self.odds_taken <= 1.0:
            raise ValueError(f"odds_taken deve ser > 1.0. Recebido: {self.odds_taken}")


@dataclass
class CLVResult:
    """
    Resultado do calculo de CLV para uma aposta individual.

    Atributos
    ---------
    bet : BetRecord
        A aposta original.
    odds_closing : float
        Odd de fechamento (no kickoff), obtida do banco ou API.
    clv : float
        Closing Line Value = (odds_taken / odds_closing) - 1.
        Positivo = bateu o mercado. Negativo = mercado era melhor.
    clv_pct : float
        CLV em percentual.
    beat_closing : bool
        True se CLV > 0 (capturou valor antes do fechamento).
    actual_result : str | None
        Resultado real do jogo ("H", "D", "A") se disponivel.
    bet_won : bool | None
        Se a aposta foi vencedora (outcome == actual_result).
    closing_source : str
        Fonte da odd de fechamento utilizada.
    implied_prob_taken : float
        Probabilidade implicita da odd tomada (1/odds_taken).
    implied_prob_closing : float
        Probabilidade implicita da odd de fechamento (1/odds_closing).
    """
    bet: BetRecord
    odds_closing: float
    clv: float
    clv_pct: float
    beat_closing: bool
    actual_result: Optional[str] = None
    bet_won: Optional[bool] = None
    closing_source: str = ""
    implied_prob_taken: float = 0.0
    implied_prob_closing: float = 0.0

    def __str__(self) -> str:
        tag = "✅ CLV+" if self.beat_closing else "❌ CLV-"
        won_tag = ""
        if self.bet_won is not None:
            won_tag = " | WON ✓" if self.bet_won else " | LOST ✗"

        return (
            f"  {tag} {self.bet.home_team} vs {self.bet.away_team} | "
            f"{OUTCOME_LABELS.get(self.bet.outcome, self.bet.outcome)} | "
            f"Apostada: @{self.bet.odds_taken:.2f} → Fechou: @{self.odds_closing:.2f} | "
            f"CLV: {self.clv_pct:+.2f}%{won_tag}"
        )


@dataclass
class CLVReport:
    """
    Relatorio completo de auditoria CLV sobre um conjunto de apostas.

    Metricas chave:
        - beat_rate: % das apostas que bateram a linha de fechamento
        - avg_clv: CLV medio (esperado > 0 para modelo com edge)
        - median_clv: CLV mediano (menos sensivel a outliers)
        - clv_by_outcome: CLV medio segmentado por H/D/A
        - correlation: correlacao entre CLV e resultado (ganho/perda)
    """
    results: list[CLVResult]
    closing_source: str
    generated_at: datetime = field(default_factory=datetime.now)

    # --- Metricas calculadas ---
    total_bets: int = 0
    bets_with_closing: int = 0
    beat_count: int = 0
    beat_rate: float = 0.0
    avg_clv: float = 0.0
    median_clv: float = 0.0
    std_clv: float = 0.0
    max_clv: float = 0.0
    min_clv: float = 0.0
    avg_clv_positive: float = 0.0
    avg_clv_negative: float = 0.0

    # Segmentado por outcome
    clv_by_outcome: dict[str, dict] = field(default_factory=dict)

    # Win rate (se resultados disponiveis)
    total_settled: int = 0
    wins: int = 0
    win_rate: float = 0.0

    # Correlacao CLV vs resultado
    clv_winners_avg: float = 0.0
    clv_losers_avg: float = 0.0

    def __post_init__(self):
        self._compute_metrics()

    def _compute_metrics(self):
        """Calcula todas as metricas a partir dos resultados individuais."""
        if not self.results:
            return

        self.total_bets = len(self.results)

        # Filtra apenas apostas com odds de fechamento validas
        valid = [r for r in self.results if r.odds_closing > 0]
        self.bets_with_closing = len(valid)

        if not valid:
            return

        clvs = [r.clv for r in valid]

        # Taxa de batida da linha de fechamento
        self.beat_count = sum(1 for r in valid if r.beat_closing)
        self.beat_rate = self.beat_count / len(valid)

        # Estatisticas descritivas do CLV
        self.avg_clv = sum(clvs) / len(clvs)
        self.std_clv = (sum((c - self.avg_clv) ** 2 for c in clvs) / len(clvs)) ** 0.5

        sorted_clvs = sorted(clvs)
        n = len(sorted_clvs)
        self.median_clv = (
            sorted_clvs[n // 2]
            if n % 2 == 1
            else (sorted_clvs[n // 2 - 1] + sorted_clvs[n // 2]) / 2
        )
        self.max_clv = max(clvs)
        self.min_clv = min(clvs)

        pos = [c for c in clvs if c > 0]
        neg = [c for c in clvs if c < 0]
        self.avg_clv_positive = sum(pos) / len(pos) if pos else 0.0
        self.avg_clv_negative = sum(neg) / len(neg) if neg else 0.0

        # CLV por outcome
        for outcome in ("H", "D", "A"):
            subset = [r for r in valid if r.bet.outcome == outcome]
            if subset:
                subset_clvs = [r.clv for r in subset]
                beat = sum(1 for r in subset if r.beat_closing)
                self.clv_by_outcome[outcome] = {
                    "count": len(subset),
                    "avg_clv": sum(subset_clvs) / len(subset_clvs),
                    "beat_rate": beat / len(subset),
                }

        # Win rate e correlacao CLV vs resultado
        settled = [r for r in valid if r.bet_won is not None]
        self.total_settled = len(settled)
        if settled:
            self.wins = sum(1 for r in settled if r.bet_won)
            self.win_rate = self.wins / len(settled)

            winners = [r.clv for r in settled if r.bet_won]
            losers = [r.clv for r in settled if not r.bet_won]
            self.clv_winners_avg = sum(winners) / len(winners) if winners else 0.0
            self.clv_losers_avg = sum(losers) / len(losers) if losers else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Exporta resultados individuais como DataFrame."""
        rows = []
        for r in self.results:
            rows.append({
                "match_id": r.bet.match_id,
                "date": r.bet.date,
                "home_team": r.bet.home_team,
                "away_team": r.bet.away_team,
                "outcome": r.bet.outcome,
                "odds_taken": r.bet.odds_taken,
                "odds_closing": r.odds_closing,
                "clv": r.clv,
                "clv_pct": r.clv_pct,
                "beat_closing": r.beat_closing,
                "model_prob": r.bet.model_prob,
                "actual_result": r.actual_result,
                "bet_won": r.bet_won,
                "closing_source": r.closing_source,
            })
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        return self._format_report()

    def _format_report(self) -> str:
        """Gera relatorio formatado em texto."""
        lines = [
            "",
            "╔" + "═" * 73 + "╗",
            "║" + "  RELATÓRIO DE AUDITORIA CLV (Closing Line Value)".center(73) + "║",
            "╠" + "═" * 73 + "╣",
            f"║  Gerado em: {self.generated_at:%Y-%m-%d %H:%M:%S}".ljust(74) + "║",
            f"║  Fonte de fechamento: {self.closing_source.upper()}".ljust(74) + "║",
            "╠" + "═" * 73 + "╣",
        ]

        # Resumo geral
        lines.append("║" + "  RESUMO GERAL".ljust(73) + "║")
        lines.append("╟" + "─" * 73 + "╢")
        lines.append(f"║  Total de apostas analisadas:  {self.total_bets:>6}".ljust(74) + "║")
        lines.append(f"║  Apostas com odd de fechamento: {self.bets_with_closing:>5}".ljust(74) + "║")
        lines.append("║" + " " * 73 + "║")

        # Metrica principal: Beat Rate
        beat_bar = self._progress_bar(self.beat_rate, 30)
        lines.append(f"║  🎯 BEAT RATE: {self.beat_rate*100:.1f}%  {beat_bar}  ({self.beat_count}/{self.bets_with_closing})".ljust(74) + "║")
        lines.append("║" + " " * 73 + "║")

        # Interpretacao do beat rate
        if self.beat_rate >= 0.55:
            interp = "  ✅ EXCELENTE — Modelo captura valor consistente antes do fechamento"
        elif self.beat_rate >= 0.50:
            interp = "  ⚠️  MARGINAL — Modelo bate a linha, mas com margem estreita"
        else:
            interp = "  ❌ ABAIXO — Modelo nao esta batendo a linha de fechamento"
        lines.append(f"║{interp}".ljust(74) + "║")

        lines.append("╟" + "─" * 73 + "╢")

        # Estatisticas CLV
        lines.append("║" + "  ESTATÍSTICAS CLV".ljust(73) + "║")
        lines.append("╟" + "─" * 73 + "╢")
        lines.append(f"║  CLV Médio:    {self.avg_clv*100:+.2f}%".ljust(74) + "║")
        lines.append(f"║  CLV Mediano:  {self.median_clv*100:+.2f}%".ljust(74) + "║")
        lines.append(f"║  Desvio Padrão: {self.std_clv*100:.2f}%".ljust(74) + "║")
        lines.append(f"║  CLV Máximo:   {self.max_clv*100:+.2f}%".ljust(74) + "║")
        lines.append(f"║  CLV Mínimo:   {self.min_clv*100:+.2f}%".ljust(74) + "║")
        lines.append("║" + " " * 73 + "║")
        lines.append(f"║  Média (quando CLV+): {self.avg_clv_positive*100:+.2f}%".ljust(74) + "║")
        lines.append(f"║  Média (quando CLV-): {self.avg_clv_negative*100:+.2f}%".ljust(74) + "║")

        # CLV por outcome
        if self.clv_by_outcome:
            lines.append("╟" + "─" * 73 + "╢")
            lines.append("║" + "  CLV POR TIPO DE APOSTA".ljust(73) + "║")
            lines.append("╟" + "─" * 73 + "╢")
            lines.append(f"║  {'Tipo':<22} {'N':>5} {'CLV Médio':>12} {'Beat Rate':>12}".ljust(74) + "║")
            lines.append("║  " + "─" * 51 + " " * 20 + "║")

            for outcome in ("H", "D", "A"):
                if outcome in self.clv_by_outcome:
                    data = self.clv_by_outcome[outcome]
                    label = OUTCOME_LABELS.get(outcome, outcome)
                    lines.append(
                        f"║  {label:<22} {data['count']:>5} "
                        f"{data['avg_clv']*100:>+10.2f}% "
                        f"{data['beat_rate']*100:>10.1f}%".ljust(74) + "║"
                    )

        # Correlacao CLV vs Resultado (se disponivel)
        if self.total_settled > 0:
            lines.append("╟" + "─" * 73 + "╢")
            lines.append("║" + "  RESULTADOS EM CAMPO".ljust(73) + "║")
            lines.append("╟" + "─" * 73 + "╢")
            lines.append(f"║  Apostas liquidadas: {self.total_settled}".ljust(74) + "║")
            lines.append(f"║  Acertos: {self.wins} ({self.win_rate*100:.1f}%)".ljust(74) + "║")
            lines.append("║" + " " * 73 + "║")
            lines.append(f"║  CLV médio (apostas GANHAS):   {self.clv_winners_avg*100:+.2f}%".ljust(74) + "║")
            lines.append(f"║  CLV médio (apostas PERDIDAS): {self.clv_losers_avg*100:+.2f}%".ljust(74) + "║")

            if self.clv_winners_avg > self.clv_losers_avg:
                lines.append("║  → Apostas ganhas tinham CLV superior — sinal de edge real".ljust(74) + "║")

        lines.append("╚" + "═" * 73 + "╝")
        return "\n".join(lines)

    @staticmethod
    def _progress_bar(ratio: float, width: int = 30) -> str:
        """Gera barra de progresso visual."""
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)


# ============================================================================
# AUDITOR CLV
# ============================================================================

class CLVAuditor:
    """
    Motor de auditoria CLV.

    Busca odds de fechamento de duas fontes:
      1. Banco de dados local (football-data.co.uk, ja persistidas)
      2. The Odds API (para jogos futuros / tempo real)

    Calcula CLV individual e agregado, gerando relatorio completo.
    """

    def __init__(self, engine=None, odds_api_key: str | None = None):
        """
        Parametros
        ----------
        engine : SQLAlchemy engine, opcional
            Conexao com o banco de dados. Se None, usa o padrao.
        odds_api_key : str, opcional
            Chave da The Odds API. Se None, tenta ler de ODDS_API_KEY env var.
        """
        self.engine = engine or get_engine()
        self.odds_api_key = odds_api_key or os.environ.get("ODDS_API_KEY", "")

    # ─── CLV: cálculo puro ─────────────────────────────────────────────────

    @staticmethod
    def calculate_clv(odds_taken: float, odds_closing: float) -> float:
        """
        Calcula o Closing Line Value.

        CLV = (odds_taken / odds_closing) - 1

        Exemplos:
            Apostou @2.00, fechou @1.80 → CLV = +11.1%  (bateu o mercado)
            Apostou @2.00, fechou @2.20 → CLV = -9.1%   (mercado era melhor)
            Apostou @2.00, fechou @2.00 → CLV = 0%       (neutro)

        Interpretação no mundo das probabilidades implícitas:
            Odds @2.00 = 50% implícito.  Odds @1.80 = 55.6% implícito.
            Se o mercado eficiente diz 55.6% mas pagamos por 50%,
            temos 5.6 pontos percentuais de vantagem.
        """
        if odds_closing <= 0 or odds_taken <= 0:
            return 0.0
        return (odds_taken / odds_closing) - 1.0

    # ─── Fonte 1: banco de dados local ─────────────────────────────────────

    def get_closing_odds_from_db(
        self,
        match_id: int,
        outcome: str,
        source: ClosingSource = ClosingSource.PINNACLE,
    ) -> float | None:
        """
        Busca a odd de fechamento do banco de dados local.

        As odds de football-data.co.uk sao capturadas proximo ao kickoff,
        servindo como proxy confiavel para odds de fechamento.

        Parametros
        ----------
        match_id : int
            ID da partida na tabela matches.
        outcome : str
            "H", "D" ou "A".
        source : ClosingSource
            Qual bookmaker usar como referencia de fechamento.

        Retorna
        -------
        float | None
            Odd de fechamento ou None se nao encontrada.
        """
        col = source.column_map.get(outcome)
        if not col:
            logger.warning(f"Outcome inválido: {outcome}")
            return None

        query = f"SELECT {col} FROM matches WHERE id = :mid"
        df = pd.read_sql(query, self.engine, params={"mid": match_id})

        if df.empty or pd.isna(df.iloc[0, 0]):
            return None

        return float(df.iloc[0, 0])

    def get_match_result(self, match_id: int) -> str | None:
        """Retorna o resultado real do jogo ("H", "D", "A") ou None."""
        query = "SELECT home_goals, away_goals FROM matches WHERE id = :mid"
        df = pd.read_sql(query, self.engine, params={"mid": match_id})

        if df.empty or pd.isna(df.iloc[0, 0]):
            return None

        hg = int(df.iloc[0]["home_goals"])
        ag = int(df.iloc[0]["away_goals"])

        if hg > ag:
            return "H"
        elif hg < ag:
            return "A"
        else:
            return "D"

    # ─── Fonte 2: The Odds API ─────────────────────────────────────────────

    def fetch_closing_from_api(
        self,
        sport: str = ODDS_API_SPORT,
        event_id: str | None = None,
        bookmaker: str = "pinnacle",
    ) -> dict[str, float]:
        """
        Busca odds atuais/de fechamento da The Odds API.

        Requer ODDS_API_KEY configurada (env var ou parametro do construtor).
        Plano gratuito: 500 requisicoes/mes.

        Parametros
        ----------
        sport : str
            Chave do esporte (ex: "soccer_epl").
        event_id : str, opcional
            ID do evento especifico. Se None, retorna todos os eventos.
        bookmaker : str
            Bookmaker preferido para odds de fechamento.

        Retorna
        -------
        dict[str, float]
            Odds por outcome: {"H": 1.80, "D": 3.50, "A": 4.20}
        """
        if not self.odds_api_key:
            logger.error(
                "ODDS_API_KEY nao configurada. "
                "Defina via variavel de ambiente ou parametro do construtor. "
                "Cadastre-se gratis em: https://the-odds-api.com"
            )
            return {}

        params = {
            "apiKey": self.odds_api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": bookmaker,
        }

        if event_id:
            url = f"{ODDS_API_BASE}/sports/{sport}/events/{event_id}/odds"
        else:
            url = f"{ODDS_API_BASE}/sports/{sport}/odds"

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # Requests restantes
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info(f"The Odds API — Requisicoes restantes: {remaining}")

            if event_id and isinstance(data, dict):
                return self._parse_api_odds(data, bookmaker)
            elif isinstance(data, list):
                # Retorna dict de event_id -> odds
                all_odds = {}
                for event in data:
                    eid = event.get("id", "")
                    odds = self._parse_api_odds(event, bookmaker)
                    if odds:
                        all_odds[eid] = {
                            "home_team": event.get("home_team", ""),
                            "away_team": event.get("away_team", ""),
                            "commence_time": event.get("commence_time", ""),
                            "odds": odds,
                        }
                return all_odds

        except requests.RequestException as e:
            logger.error(f"Erro ao acessar The Odds API: {e}")
            return {}

    @staticmethod
    def _parse_api_odds(event: dict, bookmaker: str) -> dict[str, float]:
        """Extrai odds H/D/A do JSON da The Odds API."""
        for bk in event.get("bookmakers", []):
            if bk["key"] == bookmaker:
                for market in bk.get("markets", []):
                    if market["key"] == "h2h":
                        outcomes = market["outcomes"]
                        odds_map = {}
                        home = event.get("home_team", "")
                        away = event.get("away_team", "")

                        for o in outcomes:
                            if o["name"] == home:
                                odds_map["H"] = o["price"]
                            elif o["name"] == away:
                                odds_map["A"] = o["price"]
                            elif o["name"] == "Draw":
                                odds_map["D"] = o["price"]

                        return odds_map
        return {}

    # ─── Auditoria individual ──────────────────────────────────────────────

    def audit_bet(
        self,
        bet: BetRecord,
        source: ClosingSource = ClosingSource.PINNACLE,
    ) -> CLVResult:
        """
        Audita uma aposta individual: busca odd de fechamento e calcula CLV.

        Parametros
        ----------
        bet : BetRecord
            Aposta a ser auditada.
        source : ClosingSource
            Fonte das odds de fechamento.

        Retorna
        -------
        CLVResult com CLV calculado e metadados.
        """
        # Busca odd de fechamento
        closing = self.get_closing_odds_from_db(bet.match_id, bet.outcome, source)

        if closing is None:
            logger.warning(
                f"Odd de fechamento nao encontrada: match_id={bet.match_id}, "
                f"outcome={bet.outcome}, source={source.value}"
            )
            closing = 0.0

        # Calcula CLV
        clv = self.calculate_clv(bet.odds_taken, closing) if closing > 0 else 0.0

        # Resultado real
        actual = self.get_match_result(bet.match_id)
        bet_won = (actual == bet.outcome) if actual else None

        return CLVResult(
            bet=bet,
            odds_closing=closing,
            clv=clv,
            clv_pct=clv * 100.0,
            beat_closing=clv > 0,
            actual_result=actual,
            bet_won=bet_won,
            closing_source=source.value,
            implied_prob_taken=1.0 / bet.odds_taken if bet.odds_taken > 0 else 0.0,
            implied_prob_closing=1.0 / closing if closing > 0 else 0.0,
        )

    # ─── Auditoria em lote ─────────────────────────────────────────────────

    def audit_bets(
        self,
        bets: list[BetRecord],
        source: ClosingSource = ClosingSource.PINNACLE,
    ) -> CLVReport:
        """
        Audita uma lista de apostas e gera relatorio completo.

        Parametros
        ----------
        bets : list[BetRecord]
            Lista de apostas feitas pelo modelo.
        source : ClosingSource
            Fonte das odds de fechamento.

        Retorna
        -------
        CLVReport com metricas agregadas e resultados individuais.
        """
        logger.info(f"Auditando {len(bets)} apostas (fechamento: {source.value})...")

        results = []
        for bet in bets:
            result = self.audit_bet(bet, source)
            results.append(result)

        report = CLVReport(results=results, closing_source=source.value)

        logger.info(
            f"Auditoria concluida: {report.beat_count}/{report.bets_with_closing} "
            f"bateram a linha ({report.beat_rate*100:.1f}%), "
            f"CLV medio: {report.avg_clv*100:+.2f}%"
        )

        return report

    # ─── Backtest historico usando banco de dados ──────────────────────────

    def backtest_historical(
        self,
        bets_df: pd.DataFrame | None = None,
        source: ClosingSource = ClosingSource.PINNACLE,
        min_ev: float = 0.02,
        seasons: list[int] | None = None,
    ) -> CLVReport:
        """
        Executa backtest historico CLV usando dados do banco.

        Simula: "Se o modelo tivesse apostado na odd Bet365 (abertura relativa)
        para cada jogo com EV+, como estaria o CLV contra o fechamento Pinnacle?"

        Logica de simulacao:
            - Odds Bet365 sao usadas como proxy de "odd no momento da aposta"
              (Bet365 publica odds cedo, Pinnacle e a referencia sharp)
            - Se Bet365 e a odd tomada e Pinnacle e o fechamento, o CLV mede
              quanto de valor o modelo capturaria apostando cedo

        Parametros
        ----------
        bets_df : pd.DataFrame, opcional
            DataFrame com apostas pre-registradas. Se None, simula
            automaticamente usando odds do banco (Bet365 como proxy de
            abertura, Pinnacle como fechamento).
        source : ClosingSource
            Fonte para odds de fechamento.
        min_ev : float
            EV minimo para considerar a aposta na simulacao.
        seasons : list[int], opcional
            Temporadas a incluir. Se None, usa todas.

        Retorna
        -------
        CLVReport com analise historica completa.
        """
        if bets_df is not None:
            return self._backtest_from_dataframe(bets_df, source)

        return self._backtest_simulated(source, min_ev, seasons)

    def _backtest_from_dataframe(
        self,
        df: pd.DataFrame,
        source: ClosingSource,
    ) -> CLVReport:
        """Backtest usando DataFrame de apostas pre-registradas."""
        bets = []
        for _, row in df.iterrows():
            bets.append(BetRecord(
                match_id=int(row["match_id"]),
                date=pd.Timestamp(row["date"]).to_pydatetime() if "date" in row else datetime.now(),
                home_team=row.get("home_team", ""),
                away_team=row.get("away_team", ""),
                outcome=row["outcome"],
                odds_taken=float(row["odds_taken"]),
                model_prob=float(row.get("model_prob", 0.0)),
                stake_pct=float(row.get("stake_pct", 0.0)),
            ))
        return self.audit_bets(bets, source)

    def _backtest_simulated(
        self,
        source: ClosingSource,
        min_ev: float,
        seasons: list[int] | None,
    ) -> CLVReport:
        """
        Backtest simulado: usa Bet365 como proxy de 'odd no momento da aposta'
        e compara contra a fonte de fechamento (Pinnacle/Avg/Max).

        A simulacao identifica jogos onde o modelo (probs implicitas do xG)
        teria encontrado EV+ na Bet365, e mede o CLV contra o fechamento.
        """
        logger.info("Executando backtest historico simulado...")
        logger.info(f"  Proxy de abertura: Bet365  |  Fechamento: {source.value}")
        logger.info(f"  EV minimo: {min_ev*100:.1f}%")

        # Carrega dados completos
        query = """
            SELECT m.id, m.date, m.home_team, m.away_team,
                   m.home_goals, m.away_goals, m.season,
                   m.odds_home_b365, m.odds_draw_b365, m.odds_away_b365,
                   f.ewma5_xg_pro_home, f.ewma10_xg_pro_home,
                   f.ewma5_xg_con_home, f.ewma10_xg_con_home,
                   f.ewma5_xg_pro_away, f.ewma10_xg_pro_away,
                   f.ewma5_xg_con_away, f.ewma10_xg_con_away
            FROM matches m
            JOIN match_features f ON m.id = f.match_id
            WHERE m.odds_home_b365 IS NOT NULL
        """
        if seasons:
            placeholders = ", ".join(str(s) for s in seasons)
            query += f" AND m.season IN ({placeholders})"

        query += " ORDER BY m.date"

        df = pd.read_sql(query, self.engine, parse_dates=["date"])
        logger.info(f"  Jogos com features + odds: {len(df)}")

        if df.empty:
            return CLVReport(results=[], closing_source=source.value)

        # Simula modelo: probabilidades implicitas das odds Pinnacle
        # (representam o mercado "verdadeiro" sem vig, ou usamos xG-based)
        # Para a simulacao, usamos probabilidades implicitas da MEDIA do mercado
        # removendo a margem (Shin method simplificado)
        bets: list[BetRecord] = []

        for _, row in df.iterrows():
            odds_b365 = {
                "H": row.get("odds_home_b365", 0),
                "D": row.get("odds_draw_b365", 0),
                "A": row.get("odds_away_b365", 0),
            }

            # Verifica se tem todas as odds B365
            if any(o is None or o <= 1.0 for o in odds_b365.values()):
                continue

            # Usa probabilidades implicitas das odds B365 normalizadas como 'modelo'
            # Na pratica real, o modelo EWMA treinado daria essas probs.
            # Aqui simulamos: se o modelo tivesse achado EV+ contra B365,
            # quanto o CLV seria contra Pinnacle?
            # Para isso, adicionamos edge artificial: usamos as odds B365
            # como proxy de "aposta feita" e conferimos contra fechamento.

            # Identifica o outcome favorito (menor odd = maior prob implicita)
            for outcome in ("H", "D", "A"):
                odd_taken = odds_b365[outcome]

                if odd_taken is None or odd_taken <= 1.0:
                    continue

                # Prob implicita da B365 (removendo margem pró-rata)
                margin = sum(1.0 / odds_b365[o] for o in ("H", "D", "A")) - 1.0
                raw_prob = 1.0 / odd_taken
                model_prob = raw_prob / (1.0 + margin)

                ev = (model_prob * odd_taken) - 1.0

                if ev >= min_ev:
                    bets.append(BetRecord(
                        match_id=int(row["id"]),
                        date=pd.Timestamp(row["date"]).to_pydatetime(),
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                        outcome=outcome,
                        odds_taken=odd_taken,
                        model_prob=model_prob,
                    ))

        logger.info(f"  Apostas simuladas com EV >= {min_ev*100:.1f}%: {len(bets)}")

        return self.audit_bets(bets, source)

    # ─── Auditoria com modelo treinado (prod) ─────────────────────────────

    def audit_with_model(
        self,
        model,
        scaler,
        source: ClosingSource = ClosingSource.PINNACLE,
        odds_source: str = "b365",
        min_ev: float = 0.02,
        seasons: list[int] | None = None,
    ) -> CLVReport:
        """
        Auditoria CLV usando o modelo treinado real.

        Usa as probabilidades do modelo (predict_proba) para identificar
        apostas EV+ contra as odds Bet365, e mede CLV contra o fechamento.

        Parametros
        ----------
        model : sklearn estimator
            Modelo treinado com predict_proba (ex: LogisticRegression).
        scaler : sklearn transformer
            Scaler treinado para normalizar features.
        source : ClosingSource
            Fonte para odds de fechamento (default: Pinnacle).
        odds_source : str
            Bookmaker para 'odds no momento da aposta' ("b365", "avg", "max").
        min_ev : float
            EV minimo para filtrar apostas.
        seasons : list[int], opcional
            Temporadas a incluir.

        Retorna
        -------
        CLVReport com analise real do modelo.
        """
        import numpy as np

        logger.info("Auditoria CLV com modelo treinado...")

        # Mapeamento de colunas de odds por fonte
        odds_cols_map = {
            "b365": ("odds_home_b365", "odds_draw_b365", "odds_away_b365"),
            "avg":  ("odds_home_avg",  "odds_draw_avg",  "odds_away_avg"),
            "max":  ("odds_home_max",  "odds_draw_max",  "odds_away_max"),
            "pin":  ("odds_home_pin",  "odds_draw_pin",  "odds_away_pin"),
        }
        oh, od, oa = odds_cols_map.get(odds_source, odds_cols_map["b365"])

        # Carrega dados com features
        query = f"""
            SELECT m.id, m.date, m.home_team, m.away_team,
                   m.home_goals, m.away_goals, m.season,
                   m.{oh} as odds_h, m.{od} as odds_d, m.{oa} as odds_a,
                   f.ewma5_xg_pro_home, f.ewma10_xg_pro_home,
                   f.ewma5_xg_con_home, f.ewma10_xg_con_home,
                   f.ewma5_xg_pro_away, f.ewma10_xg_pro_away,
                   f.ewma5_xg_con_away, f.ewma10_xg_con_away
            FROM matches m
            JOIN match_features f ON m.id = f.match_id
            WHERE m.{oh} IS NOT NULL AND m.{od} IS NOT NULL AND m.{oa} IS NOT NULL
        """
        if seasons:
            placeholders = ", ".join(str(s) for s in seasons)
            query += f" AND m.season IN ({placeholders})"

        query += " ORDER BY m.date"

        df = pd.read_sql(query, self.engine, parse_dates=["date"])
        logger.info(f"  Jogos disponiveis: {len(df)}")

        if df.empty:
            return CLVReport(results=[], closing_source=source.value)

        # Features para o modelo
        feature_cols = [
            "ewma5_xg_pro_home", "ewma10_xg_pro_home",
            "ewma5_xg_con_home", "ewma10_xg_con_home",
            "ewma5_xg_pro_away", "ewma10_xg_pro_away",
            "ewma5_xg_con_away", "ewma10_xg_con_away",
        ]
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)

        # Predicoes do modelo
        probs = model.predict_proba(X_scaled)  # shape: (n, 3) — H, D, A
        classes = list(model.classes_)  # ex: [0, 1, 2] com 0=A, 1=D, 2=H

        # Mapeia classes para outcomes
        # Assumindo: 0='A', 1='D', 2='H' (ordem alfabetica de sklearn)
        class_to_outcome = {c: o for c, o in zip(classes, ["A", "D", "H"])}

        bets: list[BetRecord] = []

        for i, (_, row) in enumerate(df.iterrows()):
            odds = {"H": row["odds_h"], "D": row["odds_d"], "A": row["odds_a"]}

            for cls_idx, cls in enumerate(classes):
                outcome = class_to_outcome[cls]
                prob = probs[i, cls_idx]
                odd = odds[outcome]

                if odd is None or odd <= 1.0 or np.isnan(odd):
                    continue

                ev = (prob * odd) - 1.0

                if ev >= min_ev:
                    bets.append(BetRecord(
                        match_id=int(row["id"]),
                        date=pd.Timestamp(row["date"]).to_pydatetime(),
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                        outcome=outcome,
                        odds_taken=float(odd),
                        model_prob=float(prob),
                    ))

        logger.info(f"  Apostas EV+ identificadas: {len(bets)}")

        return self.audit_bets(bets, source)


# ============================================================================
# FUNCOES AUXILIARES
# ============================================================================

def load_bets_from_csv(filepath: str) -> list[BetRecord]:
    """
    Carrega apostas de um arquivo CSV.

    Formato esperado:
        match_id,date,home_team,away_team,outcome,odds_taken,model_prob,stake_pct

    Exemplo:
        20424,2024-09-14,Arsenal,Wolves,H,1.25,0.82,0.02
        20430,2024-09-21,Man City,Arsenal,A,3.80,0.28,0.01
    """
    df = pd.read_csv(filepath, parse_dates=["date"])

    bets = []
    for _, row in df.iterrows():
        bets.append(BetRecord(
            match_id=int(row["match_id"]),
            date=pd.Timestamp(row["date"]).to_pydatetime(),
            home_team=row["home_team"],
            away_team=row["away_team"],
            outcome=row["outcome"],
            odds_taken=float(row["odds_taken"]),
            model_prob=float(row.get("model_prob", 0.0)),
            stake_pct=float(row.get("stake_pct", 0.0)),
        ))

    logger.info(f"Carregadas {len(bets)} apostas de {filepath}")
    return bets


def quick_clv_check(
    odds_taken: float,
    odds_closing: float,
    label: str = "",
) -> None:
    """
    Calculo rapido de CLV na linha de comando.

    Exemplo:
        >>> quick_clv_check(2.00, 1.80, "Arsenal ML")
        Arsenal ML: Apostou @2.00, Fechou @1.80 → CLV: +11.11%  ✅ BATEU O MERCADO
    """
    clv = CLVAuditor.calculate_clv(odds_taken, odds_closing)
    tag = "✅ BATEU O MERCADO" if clv > 0 else "❌ LINHA VENCEU"
    prefix = f"{label}: " if label else ""
    print(f"{prefix}Apostou @{odds_taken:.2f}, Fechou @{odds_closing:.2f} → CLV: {clv*100:+.2f}%  {tag}")
