"""
ev_calculator.py
----------------
Calculadora de Valor Esperado (Expected Value) para apostas esportivas.

Conceito central — EV de uma aposta:
    EV = (Probabilidade do Modelo × Odd Decimal da Casa) - 1

Interpretação:
    EV > 0  →  Aposta com VALOR POSITIVO (EV+): a casa subprecificou o evento.
               A longo prazo, essa aposta é Lucrativa.
    EV = 0  →  Ponto de equilíbrio (break-even).
    EV < 0  →  Aposta sem valor: a margem da casa consome o lucro esperado.
               A longo prazo, essa aposta é Deficitária.

Exemplo rápido:
    Modelo diz: Vitória Mandante = 55%
    Casa oferece: 2.10 para Vitória Mandante
    EV = (0.55 × 2.10) - 1 = 1.155 - 1 = +0.155  →  EV+ de 15.5%

Estrutura de entrada:
    probs: {"H": 0.50, "D": 0.25, "A": 0.25}   (soma deve ser ≈ 1.0)
    odds:  {"H": 2.10, "D": 3.40, "A": 3.80}   (odds decimais da casa)

Estrutura de saída (por resultado):
    {
        "outcome"   : str    — "H" / "D" / "A"
        "label"     : str    — Nome legível ("Vitória Mandante" etc.)
        "model_prob": float  — Probabilidade do modelo (0–1)
        "odd"       : float  — Odd decimal oferecida pela casa
        "implied_prob": float — Probabilidade implícita da casa (1 / odd)
        "edge"      : float  — Vantagem do modelo vs. casa (model_prob - implied_prob)
        "ev"        : float  — Valor Esperado (EV = model_prob × odd - 1)
        "ev_pct"    : str    — EV formatado como "+15.5%" ou "-8.3%"
        "is_value"  : bool   — True se EV > 0
        "kelly_fraction": float — Fração de Kelly para dimensionamento do stake
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Mapeamento de rótulos legíveis ────────────────────────────────────────────

OUTCOME_LABELS: dict[str, str] = {
    "H": "Vitória Mandante",
    "D": "Empate",
    "A": "Vitória Visitante",
}

# ── Estrutura de resultado ────────────────────────────────────────────────────

@dataclass
class BetAnalysis:
    """Análise completa de uma única aposta (um resultado)."""

    outcome: str          # "H", "D" ou "A"
    label: str            # Nome legível
    model_prob: float     # Probabilidade do modelo [0, 1]
    odd: float            # Odd decimal da casa (ex: 2.10)
    implied_prob: float   # Probabilidade implícita da casa = 1 / odd
    edge: float           # Vantagem: model_prob - implied_prob
    ev: float             # Valor Esperado = (model_prob × odd) - 1
    ev_pct: str           # EV formatado: "+15.5%" ou "-8.3%"
    is_value: bool        # True se EV > 0
    kelly_fraction: float # Fração de Kelly (0 se EV ≤ 0)

    def __str__(self) -> str:
        valor_tag = "✅ EV+" if self.is_value else "❌ SEM VALOR"
        return (
            f"[{valor_tag}] {self.label:<22} | "
            f"Modelo: {self.model_prob*100:5.1f}% | "
            f"Casa: {self.implied_prob*100:5.1f}% | "
            f"Odd: {self.odd:5.2f} | "
            f"Edge: {self.edge*100:+.1f}% | "
            f"EV: {self.ev_pct} | "
            f"Kelly: {self.kelly_fraction*100:.1f}%"
        )


@dataclass
class MatchEVReport:
    """Relatório completo de EV para uma partida."""

    home_team: str
    away_team: str
    analyses: list[BetAnalysis]
    value_bets: list[BetAnalysis] = field(default_factory=list)
    bookmaker_margin: float = 0.0   # Margem total da casa (vig)

    def __post_init__(self):
        self.value_bets = [a for a in self.analyses if a.is_value]

    def __str__(self) -> str:
        lines = [
            f"\n{'═'*75}",
            f"  {self.home_team}  vs  {self.away_team}",
            f"  Margem da casa (vig): {self.bookmaker_margin*100:.2f}%",
            f"{'─'*75}",
        ]
        for analysis in self.analyses:
            lines.append(f"  {analysis}")

        lines.append(f"{'─'*75}")
        if self.value_bets:
            lines.append(f"  🎯 {len(self.value_bets)} aposta(s) com VALOR POSITIVO encontrada(s):")
            for vb in self.value_bets:
                lines.append(f"     → {vb.label}: EV {vb.ev_pct}  |  Kelly: {vb.kelly_fraction*100:.1f}%")
        else:
            lines.append("  ⚠️  Nenhuma aposta com valor positivo. Fique de fora desta partida.")
        lines.append(f"{'═'*75}\n")
        return "\n".join(lines)


# ── Funções de cálculo ────────────────────────────────────────────────────────

def _kelly_fraction(model_prob: float, odd: float) -> float:
    """
    Critério de Kelly para dimensionamento ótimo do stake (versão básica).

    Fórmula: f* = (b×p - q) / b
      b = odd - 1  (lucro líquido por unidade apostada)
      p = probabilidade do modelo
      q = 1 - p    (probabilidade de perda)

    Retorna 0.0 quando EV ≤ 0 (não apostar).

    .. note::
        Para dimensionamento com shrinkage, teto de segurança e controle
        de exposição, use ``src.staking.fractional_kelly()``.
    """
    b = odd - 1.0
    p = model_prob
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


def calculate_ev(
    probs: dict[str, float],
    odds: dict[str, float],
    home_team: str = "Mandante",
    away_team: str = "Visitante",
    min_ev_threshold: float = 0.05,
    max_ev_threshold: float = 0.30,
    min_odd: Optional[float] = 1.20,
    max_odd: Optional[float] = 10.0,
) -> MatchEVReport:
    """
    Calcula o Valor Esperado (EV) para cada resultado de uma partida.

    Parâmetros
    ----------
    probs : dict[str, float]
        Probabilidades do modelo para cada resultado.
        Chaves: "H" (mandante), "D" (empate), "A" (visitante).
        Exemplo: {"H": 0.50, "D": 0.25, "A": 0.25}

    odds : dict[str, float]
        Odds decimais oferecidas pela casa de apostas.
        Exemplo: {"H": 2.10, "D": 3.40, "A": 3.80}

    home_team : str
        Nome do time mandante (para exibição).

    away_team : str
        Nome do time visitante (para exibição).

    min_ev_threshold : float
        EV mínimo para considerar uma aposta com valor.
        Padrão 0.0 (qualquer EV positivo). Use 0.05 para filtrar apenas
        apostas com EV > 5% (recomendado para reduzir falsos positivos).

    min_odd / max_odd : float, opcional
        Filtros de odd mínima/máxima. Apostas fora do intervalo não são
        sinalizadas como valor mesmo se EV > 0.

    Retorna
    -------
    MatchEVReport com análise completa e lista de apostas EV+.
    """
    # ── Validações ────────────────────────────────────────────────────────────
    outcomes = set(OUTCOME_LABELS.keys())
    missing_probs = outcomes - set(probs.keys())
    missing_odds  = outcomes - set(odds.keys())

    if missing_probs:
        raise ValueError(f"Probabilidades ausentes para: {missing_probs}")
    if missing_odds:
        raise ValueError(f"Odds ausentes para: {missing_odds}")

    prob_sum = sum(probs.values())
    if not (0.97 <= prob_sum <= 1.03):
        raise ValueError(
            f"As probabilidades devem somar ≈ 1.0. Soma atual: {prob_sum:.4f}"
        )

    for outcome, odd in odds.items():
        if odd <= 1.0:
            raise ValueError(
                f"Odd inválida para '{outcome}': {odd}. Odds decimais devem ser > 1.0."
            )

    # ── Margem da casa (vig) ──────────────────────────────────────────────────
    # Soma das probabilidades implícitas: quanto > 1.0, maior a margem
    implied_sum = sum(1.0 / odd for odd in odds.values())
    bookmaker_margin = implied_sum - 1.0

    # ── Cálculo de EV por resultado ───────────────────────────────────────────
    analyses: list[BetAnalysis] = []

    for outcome in ("H", "D", "A"):
        p     = probs[outcome]
        odd   = odds[outcome]
        impl  = 1.0 / odd
        edge  = p - impl
        ev    = (p * odd) - 1.0

        # Aplica filtros de odd
        in_odd_range = True
        if min_odd is not None and odd < min_odd:
            in_odd_range = False
        if max_odd is not None and odd > max_odd:
            in_odd_range = False

        is_value = (min_ev_threshold <= ev <= max_ev_threshold) and in_odd_range
        kelly    = _kelly_fraction(p, odd) if is_value else 0.0

        sign = "+" if ev >= 0 else ""
        ev_pct = f"{sign}{ev*100:.1f}%"

        analyses.append(BetAnalysis(
            outcome=outcome,
            label=OUTCOME_LABELS[outcome],
            model_prob=p,
            odd=odd,
            implied_prob=impl,
            edge=edge,
            ev=ev,
            ev_pct=ev_pct,
            is_value=is_value,
            kelly_fraction=kelly,
        ))

    report = MatchEVReport(
        home_team=home_team,
        away_team=away_team,
        analyses=analyses,
        bookmaker_margin=bookmaker_margin,
    )

    n_value = len(report.value_bets)
    logger.info(
        f"{home_team} vs {away_team} | "
        f"Margem casa: {bookmaker_margin*100:.2f}% | "
        f"EV+ encontradas: {n_value}"
    )
    return report


def scan_matches(
    matches: list[dict],
    min_ev_threshold: float = 0.05,
    max_ev_threshold: float = 0.30,
    min_odd: Optional[float] = 1.20,
    max_odd: Optional[float] = 10.0,
) -> list[MatchEVReport]:
    """
    Analisa múltiplas partidas de uma vez e retorna apenas as que
    contêm ao menos uma aposta com valor positivo (EV+).

    Parâmetros
    ----------
    matches : list[dict]
        Lista de dicionários, cada um com:
        {
            "home_team": str,
            "away_team": str,
            "probs": {"H": float, "D": float, "A": float},
            "odds":  {"H": float, "D": float, "A": float},
        }

    min_ev_threshold : float
        EV mínimo para sinalizar. Use 0.05 para filtrar ruído.

    Retorna
    -------
    Lista de MatchEVReport que possuem ao menos 1 aposta EV+,
    ordenada pelo maior EV encontrado (descendente).
    """
    value_reports: list[MatchEVReport] = []

    for m in matches:
        report = calculate_ev(
            probs=m["probs"],
            odds=m["odds"],
            home_team=m.get("home_team", "Mandante"),
            away_team=m.get("away_team", "Visitante"),
            min_ev_threshold=min_ev_threshold,
            max_ev_threshold=max_ev_threshold,
            min_odd=min_odd,
            max_odd=max_odd,
        )
        if report.value_bets:
            value_reports.append(report)

    # Ordena pelo maior EV individual encontrado na partida
    value_reports.sort(
        key=lambda r: max(b.ev for b in r.value_bets),
        reverse=True,
    )

    logger.info(
        f"scan_matches: {len(matches)} partidas analisadas | "
        f"{len(value_reports)} com apotas EV+"
    )
    return value_reports
