"""
staking.py
----------
Modulo de Gestao de Risco Quantitativo para dimensionamento de apostas.

Implementa o Criterio de Kelly Fracionario com protecoes para ligas de
alta variancia (futebol), onde a incerteza do modelo e significativa.

Fundamento matematico:
    Kelly Completo:  f* = (b*p - q) / b
    onde:
      b = odd - 1      (lucro liquido por unidade apostada)
      p = prob modelo   (probabilidade estimada do evento)
      q = 1 - p         (probabilidade de perda)

    Kelly Fracionario: stake = f* x shrinkage_factor
    Reduz a variancia da banca em troca de crescimento mais lento
    mas significativamente mais seguro.

Protecoes implementadas:
    1. Shrinkage ajustavel (padrao: Eighth-Kelly = 0.125)
       - Full Kelly e agressivo demais para modelos imperfeitos
       - Quarter-Kelly (0.25) e popular mas ainda arriscado em futebol
       - Eighth-Kelly (0.125) e conservador, ideal para alta variancia
    2. Teto de seguranca duro (padrao: 3% da banca)
       - Nenhuma aposta individual ultrapassa esse limite
       - Protege contra erros de calibracao do modelo
    3. Piso minimo de aposta (padrao: 0.5% da banca)
       - Evita apostas tao pequenas que nao compensam o esforco
    4. Filtro de EV minimo (padrao: 2%)
       - Ignora apostas com edge minusculo (ruido do modelo)
    5. Limite de exposicao total por rodada (padrao: 15% da banca)
       - Diversificacao forcada: nao concentra risco em uma rodada

Niveis de agressividade pre-definidos:
    CONSERVATIVE  : Eighth-Kelly (0.125), teto 2%
    MODERATE      : Quarter-Kelly (0.25), teto 3%
    AGGRESSIVE    : Half-Kelly (0.50), teto 5%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURACOES PRE-DEFINIDAS
# ============================================================================

@dataclass
class StakingConfig:
    """Parametros de configuracao do staking."""

    # Kelly shrinkage: fracao do Kelly completo a ser utilizada
    # 1.0 = Full Kelly, 0.5 = Half, 0.25 = Quarter, 0.125 = Eighth
    kelly_fraction: float = 0.125

    # Teto duro: maximo percentual da banca por aposta individual
    max_stake_pct: float = 0.03  # 3%

    # Piso: minimo percentual da banca (abaixo disso, nao aposta)
    min_stake_pct: float = 0.005  # 0.5%

    # EV minimo para considerar a aposta
    min_ev: float = 0.02  # 2%

    # Limite de exposicao total por rodada
    max_exposure_pct: float = 0.15  # 15%

    def __post_init__(self):
        assert 0 < self.kelly_fraction <= 1.0, \
            f"kelly_fraction deve estar em (0, 1]. Recebido: {self.kelly_fraction}"
        assert 0 < self.max_stake_pct <= 1.0, \
            f"max_stake_pct deve estar em (0, 1.0]. Recebido: {self.max_stake_pct}"
        assert self.min_stake_pct < self.max_stake_pct, \
            "min_stake_pct deve ser menor que max_stake_pct"
        if self.max_stake_pct > 0.10:
            logger.warning(
                f"max_stake_pct={self.max_stake_pct:.0%} e muito alto! "
                f"Recomendado <= 5%% para futebol."
            )


# Presets
CONSERVATIVE = StakingConfig(kelly_fraction=0.125, max_stake_pct=0.02)
MODERATE     = StakingConfig(kelly_fraction=0.25,  max_stake_pct=0.03)
AGGRESSIVE   = StakingConfig(kelly_fraction=0.50,  max_stake_pct=0.05)


# ============================================================================
# RESULTADO DO DIMENSIONAMENTO
# ============================================================================

@dataclass
class StakeRecommendation:
    """Resultado do calculo de dimensionamento para uma aposta."""

    # Identificacao
    home_team: str
    away_team: str
    outcome: str        # "H", "D" ou "A"
    outcome_label: str  # "Vitoria Mandante", etc.

    # Inputs
    model_prob: float   # Probabilidade do modelo
    odd: float          # Odd decimal da casa
    bankroll: float     # Banca atual

    # Calculos intermediarios
    kelly_full: float         # Kelly completo (f*)
    kelly_shrunk: float       # Kelly apos shrinkage
    shrinkage_factor: float   # Multiplicador usado

    # Resultado final
    stake_pct: float      # Percentual da banca a apostar (apos caps)
    stake_amount: float   # Valor monetario da aposta
    ev: float             # Valor Esperado
    ev_pct: float         # EV em percentual
    potential_profit: float  # Lucro potencial se acertar
    potential_loss: float    # Perda se errar (= stake_amount)

    # Flags
    is_capped: bool       # True se o teto duro limitou o stake
    is_actionable: bool   # True se vale a pena executar

    def __str__(self) -> str:
        if not self.is_actionable:
            return (
                f"  SKIP  {self.home_team} vs {self.away_team} | "
                f"{self.outcome_label} | EV={self.ev_pct:+.1f}% | "
                f"Kelly muito baixo ou EV insuficiente"
            )

        cap_tag = " [CAPPED]" if self.is_capped else ""
        return (
            f"  BET   {self.home_team} vs {self.away_team} | "
            f"{self.outcome_label} @{self.odd:.2f} | "
            f"EV={self.ev_pct:+.1f}% | "
            f"Kelly={self.kelly_full*100:.2f}% -> {self.stake_pct*100:.2f}%{cap_tag} | "
            f"Stake=${self.stake_amount:.2f} | "
            f"Lucro potencial=${self.potential_profit:.2f}"
        )


@dataclass
class RoundStakingPlan:
    """Plano de staking para uma rodada completa."""

    bankroll: float
    config: StakingConfig
    recommendations: list[StakeRecommendation] = field(default_factory=list)

    @property
    def actionable_bets(self) -> list[StakeRecommendation]:
        return [r for r in self.recommendations if r.is_actionable]

    @property
    def total_exposure(self) -> float:
        return sum(r.stake_amount for r in self.actionable_bets)

    @property
    def total_exposure_pct(self) -> float:
        return self.total_exposure / self.bankroll if self.bankroll > 0 else 0.0

    def __str__(self) -> str:
        lines = [
            "",
            "=" * 75,
            f"  PLANO DE STAKING -- Banca: ${self.bankroll:,.2f}",
            f"  Config: Kelly x{self.config.kelly_fraction} | "
            f"Teto: {self.config.max_stake_pct*100:.1f}% | "
            f"EV min: {self.config.min_ev*100:.1f}%",
            "=" * 75,
        ]

        for rec in self.recommendations:
            lines.append(str(rec))

        lines.append("-" * 75)

        n_bets = len(self.actionable_bets)
        lines.append(
            f"  Apostas ativas: {n_bets}/{len(self.recommendations)} | "
            f"Exposicao total: ${self.total_exposure:.2f} "
            f"({self.total_exposure_pct*100:.2f}% da banca)"
        )

        if self.total_exposure_pct > self.config.max_exposure_pct:
            lines.append(
                f"  !! ALERTA: Exposicao ({self.total_exposure_pct*100:.1f}%) "
                f"excede limite ({self.config.max_exposure_pct*100:.1f}%)"
            )

        lines.append("=" * 75)
        return "\n".join(lines)


# ============================================================================
# FUNCAO PRINCIPAL: KELLY FRACIONARIO COM PROTECOES
# ============================================================================

OUTCOME_LABELS = {
    "H": "Vitoria Mandante",
    "D": "Empate",
    "A": "Vitoria Visitante",
}


def fractional_kelly(
    model_prob: float,
    odd: float,
    bankroll: float,
    config: StakingConfig | None = None,
    home_team: str = "Mandante",
    away_team: str = "Visitante",
    outcome: str = "H",
) -> StakeRecommendation:
    """
    Calcula o tamanho otimo da aposta usando Kelly Fracionario com protecoes.

    Algoritmo:
    ----------
    1. Calcula o Kelly completo:
           f* = (b*p - q) / b
       onde b = odd - 1, p = model_prob, q = 1 - p

    2. Aplica shrinkage (Kelly fracionario):
           f_shrunk = f* x kelly_fraction

       Niveis tipicos:
         Full Kelly   (1.0)  : crescimento otimo mas variancia extrema
         Half Kelly   (0.5)  : reduz variancia ~50%, perde ~25% do crescimento
         Quarter Kelly(0.25) : popular em esportes
         Eighth Kelly (0.125): conservador, ideal para futebol
                               onde calibracao do modelo e imperfeita

    3. Aplica teto de seguranca duro:
           stake_pct = min(f_shrunk, max_stake_pct)

       Independente da edge ou EV, NUNCA aposta mais que max_stake_pct
       da banca em uma unica aposta. Protege contra:
         - Erros de calibracao do modelo
         - Odds manipuladas ou erroneas
         - Eventos de cauda (black swans)

    4. Verifica piso minimo:
           Se stake_pct < min_stake_pct: nao aposta (custo/beneficio ruim)

    5. Verifica EV minimo:
           Se EV < min_ev: nao aposta (edge muito fino, possivelmente ruido)

    Parametros
    ----------
    model_prob : float
        Probabilidade estimada pelo modelo para o evento [0, 1].

    odd : float
        Odd decimal oferecida pela casa de apostas (ex: 2.10).

    bankroll : float
        Valor total da banca disponivel.

    config : StakingConfig, opcional
        Configuracao de risco. Se None, usa MODERATE (Quarter-Kelly, 3%).

    home_team, away_team : str
        Nomes dos times (para exibicao).

    outcome : str
        Resultado apostado: "H", "D" ou "A".

    Retorna
    -------
    StakeRecommendation
        Objeto com stake calculado, flags e metadados.

    Exemplos
    --------
    >>> from src.core.staking import fractional_kelly, CONSERVATIVE
    >>> rec = fractional_kelly(
    ...     model_prob=0.55,
    ...     odd=2.10,
    ...     bankroll=1000.0,
    ...     config=CONSERVATIVE,
    ...     home_team="Arsenal",
    ...     away_team="Chelsea",
    ...     outcome="H",
    ... )
    >>> rec.stake_amount   # valor em $
    >>> rec.stake_pct      # percentual da banca
    >>> rec.is_actionable  # se deve executar
    """
    if config is None:
        config = MODERATE

    # Validacoes basicas
    if not (0.0 < model_prob < 1.0):
        raise ValueError(f"model_prob deve estar em (0, 1). Recebido: {model_prob}")
    if odd <= 1.0:
        raise ValueError(f"Odd deve ser > 1.0. Recebido: {odd}")
    if bankroll <= 0:
        raise ValueError(f"Bankroll deve ser > 0. Recebido: {bankroll}")

    # -- Passo 1: Kelly completo --
    b = odd - 1.0          # lucro liquido por unidade
    p = model_prob         # prob de acerto
    q = 1.0 - p            # prob de erro
    kelly_full = (b * p - q) / b

    # -- Passo 2: Shrinkage --
    kelly_shrunk = kelly_full * config.kelly_fraction

    # -- EV --
    ev = (p * odd) - 1.0
    ev_pct = ev * 100.0

    # -- Passo 3 & 4 & 5: Caps e filtros --
    is_capped = False
    is_actionable = True

    if kelly_full <= 0 or ev < config.min_ev:
        # EV negativo ou edge insuficiente -> nao aposta
        stake_pct = 0.0
        is_actionable = False
    else:
        stake_pct = kelly_shrunk

        # Teto duro
        if stake_pct > config.max_stake_pct:
            stake_pct = config.max_stake_pct
            is_capped = True

        # Piso minimo
        if stake_pct < config.min_stake_pct:
            stake_pct = 0.0
            is_actionable = False

    stake_amount = round(stake_pct * bankroll, 2)
    potential_profit = round(stake_amount * (odd - 1.0), 2) if is_actionable else 0.0
    potential_loss = stake_amount if is_actionable else 0.0

    rec = StakeRecommendation(
        home_team=home_team,
        away_team=away_team,
        outcome=outcome,
        outcome_label=OUTCOME_LABELS.get(outcome, outcome),
        model_prob=model_prob,
        odd=odd,
        bankroll=bankroll,
        kelly_full=max(0.0, kelly_full),
        kelly_shrunk=max(0.0, kelly_shrunk),
        shrinkage_factor=config.kelly_fraction,
        stake_pct=stake_pct,
        stake_amount=stake_amount,
        ev=ev,
        ev_pct=ev_pct,
        potential_profit=potential_profit,
        potential_loss=potential_loss,
        is_capped=is_capped,
        is_actionable=is_actionable,
    )

    action = "BET" if is_actionable else "SKIP"
    logger.info(
        f"[{action}] {home_team} vs {away_team} | "
        f"{OUTCOME_LABELS.get(outcome, outcome)} @{odd:.2f} | "
        f"p={model_prob:.3f} | EV={ev_pct:+.1f}% | "
        f"Kelly={kelly_full*100:.2f}% -> {stake_pct*100:.2f}%"
    )

    return rec


# ============================================================================
# PLANO DE STAKING PARA UMA RODADA
# ============================================================================

def plan_round(
    matches: list[dict],
    bankroll: float,
    config: StakingConfig | None = None,
) -> RoundStakingPlan:
    """
    Gera um plano de staking completo para uma rodada.

    Para cada partida, calcula o stake de cada resultado com EV+
    e verifica se a exposicao total respeita o limite da configuracao.
    Se exceder, reduz proporcionalmente todos os stakes.

    Parametros
    ----------
    matches : list[dict]
        Lista de partidas. Cada dict deve conter:
        {
            "home_team": str,
            "away_team": str,
            "probs": {"H": float, "D": float, "A": float},
            "odds":  {"H": float, "D": float, "A": float},
        }

    bankroll : float
        Valor total da banca.

    config : StakingConfig, opcional
        Configuracao. Padrao: MODERATE.

    Retorna
    -------
    RoundStakingPlan com todas as recomendacoes e metadados.
    """
    if config is None:
        config = MODERATE

    all_recs: list[StakeRecommendation] = []

    for match in matches:
        home = match.get("home_team", "Mandante")
        away = match.get("away_team", "Visitante")
        probs = match["probs"]
        odds = match["odds"]

        for outcome in ("H", "D", "A"):
            rec = fractional_kelly(
                model_prob=probs[outcome],
                odd=odds[outcome],
                bankroll=bankroll,
                config=config,
                home_team=home,
                away_team=away,
                outcome=outcome,
            )
            all_recs.append(rec)

    plan = RoundStakingPlan(
        bankroll=bankroll,
        config=config,
        recommendations=all_recs,
    )

    # -- Controle de exposicao total --
    if plan.total_exposure_pct > config.max_exposure_pct:
        _apply_exposure_cap(plan)

    return plan


def _apply_exposure_cap(plan: RoundStakingPlan) -> None:
    """
    Reduz proporcionalmente todos os stakes ativos para respeitar
    o limite de exposicao total da rodada.
    """
    target = plan.config.max_exposure_pct * plan.bankroll
    current = plan.total_exposure

    if current <= 0:
        return

    ratio = target / current
    logger.info(
        f"Exposicao ({current/plan.bankroll*100:.1f}%) excede limite "
        f"({plan.config.max_exposure_pct*100:.1f}%). "
        f"Reduzindo stakes por fator {ratio:.3f}"
    )

    for rec in plan.actionable_bets:
        rec.stake_amount = round(rec.stake_amount * ratio, 2)
        rec.stake_pct = rec.stake_amount / plan.bankroll
        rec.potential_profit = round(rec.stake_amount * (rec.odd - 1.0), 2)
        rec.potential_loss = rec.stake_amount
