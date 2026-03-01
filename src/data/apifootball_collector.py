"""
apifootball_collector.py
------------------------
Coleta estatísticas avançadas de partidas (xG, chutes no alvo, posse de bola)
via API-Football (api-sports.io / RapidAPI) e retorna DataFrame padronizado
compatível com `match_advanced_stats`.

Ligas suportadas:
    BRA-Serie A          →  league_id=71
    ENG-Premier League   →  league_id=39
    ESP-La Liga          →  league_id=140
    FRA-Ligue 1          →  league_id=61
    UEFA-Champions League→  league_id=2

Autenticação (variável de ambiente — defina no .env ou no shell):
    APIFOOTBALL_KEY     →  chave direta em api-sports.io (header x-apisports-key)
    RAPIDAPI_KEY        →  chave RapidAPI (header x-rapidapi-key); alternativa

Rate limits (plano gratuito):
    100 req/dia | 10 req/min  →  coletor respeita automaticamente.
    Para um campeonato completo (380 jogos), a coleta pode levar ~4 dias.
    O cache em disco garante que partidas já coletadas não sejam re-buscadas.

Cache em disco:
    ~/.apostei/cache/apifootball/
        fixtures_{league_id}_{season}.json        ← lista de partidas
        stats_{fixture_id}.json                   ← stats por partida

Uso:
    from src.data.apifootball_collector import collect_advanced_stats

    df = collect_advanced_stats("BRA-Serie A", seasons=[2024, 2025])
    df = collect_advanced_stats("ENG-Premier League", seasons=[2024])
    df = collect_advanced_stats("UEFA-Champions League", seasons=[2024])
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração das ligas
# ---------------------------------------------------------------------------

LEAGUE_MAP: dict[str, int] = {
    "BRA-Serie A": 71,
    "ENG-Premier League": 39,
    "ESP-La Liga": 140,
    "FRA-Ligue 1": 61,
    "UEFA-Champions League": 2,
}

# Ligas com temporada dividida em dois anos civis (ex: 2024/25).
# Para essas ligas, a API-Football usa o ano de INÍCIO como season ID.
# Ou seja: usuário passa "2025" → API season = 2024  (temporada 2024/25)
#           usuário passa "2026" → API season = 2025  (temporada 2025/26)
# Brasileirão NÃO entra aqui — usa ano calendário (2025 = 2025).
SPLIT_SEASON_LEAGUES: set[str] = {
    "ENG-Premier League",
    "ESP-La Liga",
    "FRA-Ligue 1",
    "UEFA-Champions League",
}

# Status da API-Football que representam jogo encerrado:
#   FT  = Full Time | AET = After Extra Time | PEN = Penalty Shootout
#   AWD = Awarded (WO) | WO = Walkover
FINISHED_STATUSES = "FT-AET-PEN-AWD-WO"


def _user_season_to_api(league: str, user_season: int) -> int:
    """
    Converte o ano que o usuário pensa como 'temporada' para o ID
    de season usado pela API-Football.

    Exemplos:
        BRA-Serie A      2025 → 2025  (temporada calendário)
        ENG-Premier League 2025 → 2024  (temporada 2024/25)
        ENG-Premier League 2026 → 2025  (temporada 2025/26)
    """
    if league in SPLIT_SEASON_LEAGUES:
        return user_season - 1
    return user_season


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".apostei" / "cache" / "apifootball"


def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.json"


def _load_cache(name: str):
    p = _cache_path(name)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(name: str, data) -> None:
    with open(_cache_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Cliente HTTP
# ---------------------------------------------------------------------------


def _get_headers() -> dict[str, str]:
    """Retorna headers de autenticação. Prioriza api-sports.io, fallback RapidAPI."""
    key = os.getenv("APIFOOTBALL_KEY")
    if key:
        return {"x-apisports-key": key}

    rapid_key = os.getenv("RAPIDAPI_KEY")
    if rapid_key:
        return {
            "x-rapidapi-key": rapid_key,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
        }

    raise EnvironmentError("Nenhuma chave de API encontrada.\nDefina APIFOOTBALL_KEY (api-sports.io) ou RAPIDAPI_KEY (RapidAPI) no arquivo .env ou como variável de ambiente.")


def _base_url() -> str:
    if os.getenv("RAPIDAPI_KEY") and not os.getenv("APIFOOTBALL_KEY"):
        return "https://api-football-v1.p.rapidapi.com/v3"
    return "https://v3.football.api-sports.io"


def _get(endpoint: str, params: dict, no_cache: bool = False, cache_key: str | None = None):
    """GET com cache em disco e rate limiting."""
    if cache_key and not no_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    headers = _get_headers()
    url = f"{_base_url()}/{endpoint}"

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError:
        if resp.status_code == 429:
            logger.warning("Rate limit atingido. Aguardando 61 segundos...")
            time.sleep(61)
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        else:
            raise

    # Respeita 10 req/min do plano gratuito
    time.sleep(6.5)

    if cache_key:
        _save_cache(cache_key, data)

    return data


# ---------------------------------------------------------------------------
# Coleta de fixtures
# ---------------------------------------------------------------------------


def _fetch_fixtures(league_id: int, season: int, no_cache: bool = False) -> list[dict]:
    """Retorna lista de fixtures encerrados para a liga/temporada."""
    cache_key = f"fixtures_{league_id}_{season}"
    data = _get(
        "fixtures",
        params={"league": league_id, "season": season, "status": FINISHED_STATUSES},
        no_cache=no_cache,
        cache_key=cache_key,
    )
    return data.get("response", [])


def _fetch_stats(fixture_id: int, no_cache: bool = False) -> list[dict]:
    """Retorna estatísticas de uma partida (chutes, posse, xG)."""
    cache_key = f"stats_{fixture_id}"
    data = _get(
        "fixtures/statistics",
        params={"fixture": fixture_id},
        no_cache=no_cache,
        cache_key=cache_key,
    )
    return data.get("response", [])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_stat(stats_list: list[dict], team_side: str, stat_name: str) -> float:
    """
    Extrai valor numérico de uma estatística de um time.
    team_side: 'home' ou 'away' (baseado na posição na lista)
    """
    idx = 0 if team_side == "home" else 1
    if idx >= len(stats_list):
        return 0.0
    for item in stats_list[idx].get("statistics", []):
        if item["type"].lower() == stat_name.lower():
            val = item["value"]
            if val is None:
                return 0.0
            # Possessão vem como "45%"
            if isinstance(val, str) and val.endswith("%"):
                return float(val.replace("%", "").strip())
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------


def collect_advanced_stats(
    league: str,
    seasons: list[int] | None = None,
    no_cache: bool = False,
) -> pd.DataFrame:
    """
    Coleta xG, chutes no alvo e posse de bola via API-Football.

    Parâmetros
    ----------
    league : str
        Nome da liga conforme LEAGUE_MAP.
        Ex: "BRA-Serie A", "ENG-Premier League", "UEFA-Champions League"
    seasons : list[int] | None
        Ex: [2024, 2025]. None = apenas temporada corrente estimada.
    no_cache : bool
        Se True, ignora cache e força nova requisição.

    Retorna
    -------
    pd.DataFrame com colunas:
        home_team, away_team, date, league, season,
        home_xg, away_xg, home_shots_target, away_shots_target,
        home_possession, away_possession, source
    """
    if league not in LEAGUE_MAP:
        known = ", ".join(f'"{k}"' for k in LEAGUE_MAP)
        raise ValueError(f"Liga '{league}' não reconhecida. Opções: {known}")

    league_id = LEAGUE_MAP[league]

    if seasons is None:
        from datetime import date

        year = date.today().year
        seasons = [year - 1, year]

    is_split = league in SPLIT_SEASON_LEAGUES
    if is_split:
        logger.info(f"[API-Football] Liga com temporada dividida ({league}). Mapeando anos: {seasons} → {[s - 1 for s in seasons]} (ano de início da temporada na API)")
    logger.info(f"[API-Football] Iniciando coleta — liga: {league} (id={league_id}) | temporadas (usuário): {seasons}")

    # Verifica autenticação antes de fazer qualquer requisição
    try:
        _get_headers()
    except EnvironmentError as e:
        logger.error(str(e))
        raise

    rows: list[dict] = []

    for user_season in seasons:
        api_season = _user_season_to_api(league, user_season)
        season_label = f"{api_season}/{str(api_season + 1)[-2:]}" if is_split else str(user_season)
        logger.info(f"  Buscando fixtures encerrados — {league} {season_label} (API season={api_season})...")
        fixtures = _fetch_fixtures(league_id, api_season, no_cache=no_cache)
        logger.info(f"  {len(fixtures)} fixtures encontrados.")

        if not fixtures:
            if is_split:
                logger.warning(f"  0 jogos encontrados para API season={api_season}. Verifique se a temporada {season_label} já foi disputada.")
            continue

        # Verifica quantos já têm stats em cache
        cached_count = sum(1 for fix in fixtures if _load_cache(f"stats_{fix['fixture']['id']}") is not None)
        need_fetch = len(fixtures) - cached_count
        logger.info(f"  Stats em cache: {cached_count} | A buscar: {need_fetch} (≈{need_fetch * 6.5 / 60:.0f} min)")

        for i, fix in enumerate(fixtures, 1):
            fix_id = fix["fixture"]["id"]
            date_str = fix["fixture"]["date"][:10]  # "2024-06-15T..."
            home_name = fix["teams"]["home"]["name"]
            away_name = fix["teams"]["away"]["name"]

            stats = _fetch_stats(fix_id, no_cache=no_cache)

            if i % 50 == 0 or i == len(fixtures):
                logger.info(f"  Progresso: {i}/{len(fixtures)} fixtures processados")

            row = {
                "home_team": home_name,
                "away_team": away_name,
                "date": date_str,
                "league": league,
                "season": user_season,
                "home_xg": _parse_stat(stats, "home", "expected_goals"),
                "away_xg": _parse_stat(stats, "away", "expected_goals"),
                "home_shots_target": _parse_stat(stats, "home", "shots on goal"),
                "away_shots_target": _parse_stat(stats, "away", "shots on goal"),
                "home_possession": _parse_stat(stats, "home", "ball possession"),
                "away_possession": _parse_stat(stats, "away", "ball possession"),
                "source": "api-football",
            }
            rows.append(row)

    if not rows:
        logger.warning("Nenhuma partida retornada.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    xg_count = (df["home_xg"] > 0).sum()
    sot_count = (df["home_shots_target"] > 0).sum()
    poss_count = (df["home_possession"] > 0).sum()
    logger.info(f"[API-Football] Coleta concluída — {len(df)} partidas | xG>0: {xg_count} | SoT>0: {sot_count} | Posse>0: {poss_count}")
    if xg_count == 0:
        logger.info("  xG retornou 0 — provável plano gratuito (xG requer plano Pro/Ultra). Chutes e posse seguem disponíveis.")

    return df
