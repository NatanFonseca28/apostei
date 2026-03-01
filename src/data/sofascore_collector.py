"""
sofascore_collector.py
----------------------
Coleta estatísticas avançadas (xG, chutes no alvo, posse de bola) via API
pública do Sofascore — sem autenticação, sem dependência do soccerdata.

Estratégia:
    1. GET /unique-tournament/{tid}/seasons          → descobre season_id
    2. GET /unique-tournament/{tid}/season/{sid}/rounds  → lista de rodadas
    3. GET /unique-tournament/{tid}/season/{sid}/events/round/{r} → partidas
    4. GET /event/{game_id}/statistics               → xG, SoT, posse

Ligas suportadas:
    BRA-Serie A           → Brasileirão Série A  (tournament_id 325)
    ENG-Premier League    → Premier League        (tournament_id  17)
    ESP-La Liga           → La Liga               (tournament_id   8)
    FRA-Ligue 1           → Ligue 1               (tournament_id  34)
    GER-Bundesliga        → Bundesliga            (tournament_id  35)
    ITA-Serie A           → Serie A italiana      (tournament_id  23)
    UEFA-Champions League → Champions League      (tournament_id   7)

Cache em disco:
    ~/.apostei/cache/sofascore/
        seasons_{tid}.json                ← lista de temporadas
        rounds_{tid}_{sid}.json           ← rodadas de uma temporada
        round_{tid}_{sid}_{r}.json        ← eventos de uma rodada
        stats_{game_id}.json              ← stats por partida

Dependência extra (instalada automaticamente pelo pip):
    curl-cffi    → impersona TLS fingerprint do Chrome — contorna Cloudflare

Semântica de seasons:
    Brasileirão (single-year)  : seasons=[2025] → temporada "2025"
    Premier League (split-year): seasons=[2025] → temporada "24/25" (encerra em 2025)
                                  seasons=[2026] → temporada "25/26" (encerra em 2026)

Uso:
    from src.data.sofascore_collector import collect_advanced_stats

    df = collect_advanced_stats("BRA-Serie A", seasons=[2024, 2025])
    df = collect_advanced_stats("ENG-Premier League", seasons=[2025])
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapa de ligas: nome → tournament_id do Sofascore
# ---------------------------------------------------------------------------

LEAGUE_MAP: dict[str, int] = {
    "BRA-Serie A": 325,
    "ENG-Premier League": 17,
    "ESP-La Liga": 8,
    "FRA-Ligue 1": 34,
    "GER-Bundesliga": 35,
    "ITA-Serie A": 23,
    "UEFA-Champions League": 7,
    "CONMEBOL Libertadores": 384,
}

# Ligas com temporada dividida entre dois anos civis (ex: 2024/2025).
# Para essas ligas, "user_year=2025" mapeia para a season "2024/2025".
_SPLIT_SEASON_TID: set[int] = {17, 8, 34, 35, 23, 7}

# ---------------------------------------------------------------------------
# Normalização de nomes de times Sofascore → Flashscore
# ---------------------------------------------------------------------------

TEAM_NAME_MAP: dict[str, str] = {
    # Brasileirão
    "Athletico Paranaense": "Athletico-PR",
    "Paranaense": "Athletico-PR",
    "Atlético Mineiro": "Atletico MG",
    "Atlético-GO": "Atletico GO",
    "Bragantino": "RB Bragantino",
    "Red Bull Bragantino": "RB Bragantino",
    "Cuiabá": "Cuiaba",
    "Cuiabá Esporte Clube": "Cuiaba",
    "América Mineiro": "America MG",
    "América-MG": "America MG",
    "Grêmio": "Gremio",
    "São Paulo": "Sao Paulo",
    "Vasco da Gama": "Vasco",
    "Ceará": "Ceara",
    "Goiás": "Goias",
    # Premier League
    "Manchester United": "Manchester Utd",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "Nottingham Forest": "Nott'm Forest",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    # La Liga
    "Atlético Madrid": "Atletico Madrid",
    "Atletico de Madrid": "Atletico Madrid",
    # Bundesliga
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt",
    # Ligue 1
    "Paris Saint-Germain": "Paris SG",
    "Saint-Étienne": "Saint-Etienne",
    # Champions League extras
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "Internazionale": "Inter",
}

# ---------------------------------------------------------------------------
# HTTP — sessão com headers de browser real
# ---------------------------------------------------------------------------

_API = "https://api.sofascore.com/api/v1/"

# Headers adicionais (o curl_cffi já gerencia TLS fingerprint e sec-* automaticamente)
_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,pt-BR;q=0.8",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "Cache-Control": "no-cache",
}

# Perfil de impersonação do Chrome para curl_cffi
_IMPERSONATE = "chrome124"

# Delay entre requests (segundos)
REQUEST_DELAY = 1.2


def _make_session() -> cffi_requests.Session:
    s = cffi_requests.Session(impersonate=_IMPERSONATE)
    s.headers.update(_HEADERS)
    return s


# ---------------------------------------------------------------------------
# Cache em disco
# ---------------------------------------------------------------------------

_CACHE_DIR = Path.home() / ".apostei" / "cache" / "sofascore"


def _ensure_cache() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_json_cache(name: str) -> dict | list | None:
    p = _CACHE_DIR / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            p.unlink(missing_ok=True)
    return None


def _save_json_cache(name: str, data: dict | list) -> None:
    _ensure_cache()
    (_CACHE_DIR / name).write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Chamadas à API
# ---------------------------------------------------------------------------


def _get(session: cffi_requests.Session, url: str, retries: int = 3) -> dict | None:
    """GET com retry e tratamento de erros."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                logger.debug(f"    404: {url}")
                return None
            if resp.status_code == 403:
                logger.warning(f"    403 Forbidden: {url} (tentativa {attempt}/{retries})")
                time.sleep(2 * attempt)
                continue
            logger.warning(f"    HTTP {resp.status_code}: {url}")
            return None
        except cffi_requests.RequestException as exc:
            logger.warning(f"    Erro de rede ({attempt}/{retries}): {exc}")
            if attempt < retries:
                time.sleep(2)
    return None


def _fetch_seasons(session: cffi_requests.Session, tid: int, no_cache: bool) -> list[dict]:
    """Retorna lista de temporadas de um torneio."""
    cache_name = f"seasons_{tid}.json"
    if not no_cache:
        cached = _load_json_cache(cache_name)
        if cached is not None:
            return cached  # type: ignore[return-value]

    data = _get(session, f"{_API}unique-tournament/{tid}/seasons")
    if not data:
        return []
    seasons = data.get("seasons", [])
    _save_json_cache(cache_name, seasons)
    return seasons


def _fetch_rounds(session: cffi_requests.Session, tid: int, sid: int, no_cache: bool) -> list[dict]:
    """Retorna lista de rodadas de uma temporada."""
    cache_name = f"rounds_{tid}_{sid}.json"
    if not no_cache:
        cached = _load_json_cache(cache_name)
        if cached is not None:
            return cached  # type: ignore[return-value]

    data = _get(session, f"{_API}unique-tournament/{tid}/season/{sid}/rounds")
    if not data:
        return []
    rounds = data.get("rounds", [])
    _save_json_cache(cache_name, rounds)
    return rounds


def _fetch_round_events(session: cffi_requests.Session, tid: int, sid: int, round_num: int, no_cache: bool) -> list[dict]:
    """Retorna eventos (partidas) de uma rodada."""
    cache_name = f"round_{tid}_{sid}_{round_num}.json"
    if not no_cache:
        cached = _load_json_cache(cache_name)
        if cached is not None:
            return cached  # type: ignore[return-value]

    data = _get(session, f"{_API}unique-tournament/{tid}/season/{sid}/events/round/{round_num}")
    if not data:
        return []
    events = data.get("events", [])
    # Cacheia apenas se houver partidas finalizadas (evita cachear rodadas futuras)
    if any(e.get("status", {}).get("code") == 100 for e in events):
        _save_json_cache(cache_name, events)
    return events


def _fetch_stats(session: cffi_requests.Session, game_id: int, no_cache: bool) -> dict:
    """Busca xG, SoT e posse de um jogo."""
    cache_name = f"stats_{game_id}.json"
    if not no_cache:
        cached = _load_json_cache(cache_name)
        if cached is not None:
            return cached  # type: ignore[return-value]

    data = _get(session, f"{_API}event/{game_id}/statistics")
    result = _parse_stats(data) if data else _empty_stats()
    if any(v > 0 for v in result.values()):
        _save_json_cache(cache_name, result)
    return result


def _parse_stats(data: dict) -> dict:
    """Extrai xG, SoT e posse do JSON de estatísticas do Sofascore."""
    result = _empty_stats()
    statistics = data.get("statistics", [])
    period_data = next(
        (s for s in statistics if s.get("period") == "ALL"),
        statistics[0] if statistics else None,
    )
    if period_data is None:
        return result

    for group in period_data.get("groups", []):
        for item in group.get("statisticsItems", []):
            name_lower = item.get("name", "").lower()

            if "expected goals" in name_lower or name_lower == "xg":
                result["home_xg"] = _safe_float(item.get("homeValueFloat", item.get("home", 0)))
                result["away_xg"] = _safe_float(item.get("awayValueFloat", item.get("away", 0)))
            elif "shots on target" in name_lower:
                result["home_shots_target"] = _safe_float(item.get("home", 0))
                result["away_shots_target"] = _safe_float(item.get("away", 0))
            elif "ball possession" in name_lower:
                result["home_possession"] = _safe_float(str(item.get("homeValueFloat", item.get("home", 0))).replace("%", ""))
                result["away_possession"] = _safe_float(str(item.get("awayValueFloat", item.get("away", 0))).replace("%", ""))

    return result


def _empty_stats() -> dict:
    return {
        "home_xg": 0.0,
        "away_xg": 0.0,
        "home_shots_target": 0.0,
        "away_shots_target": 0.0,
        "home_possession": 0.0,
        "away_possession": 0.0,
    }


def _safe_float(val) -> float:
    try:
        return float(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Lógica de matching de temporadas
# ---------------------------------------------------------------------------


def _season_matches(year_str: str, user_year: int, tid: int) -> bool:
    """Verifica se a string de season do Sofascore corresponde ao ano do usuário.

    Ligas com split season (Premier League, La Liga etc.):
        Sofascore usa formato curto: "24/25", "25/26".
        user_year=2025 → procura season que termina em "/25"
        user_year=2026 → procura season que termina em "/26"
    Ligas com season em único ano calendário (Brasileirão):
        user_year=2025 → procura "2025"
    """
    if tid in _SPLIT_SEASON_TID:
        # "24/25" → termina com "/25" → user_year=2025 (season que encerra em 2025)
        suffix = f"/{str(user_year)[-2:]}"
        return year_str.rstrip().endswith(suffix)
    return year_str.strip() == str(user_year)


# ---------------------------------------------------------------------------
# Função pública principal
# ---------------------------------------------------------------------------


def collect_advanced_stats(
    league: str,
    seasons: list[int] | None = None,
    no_cache: bool = False,
) -> pd.DataFrame:
    """Coleta estatísticas avançadas de uma liga via Sofascore (sem soccerdata).

    Parâmetros
    ----------
    league : str
        Ex: "BRA-Serie A", "ENG-Premier League".
    seasons : list[int] | None
        Ex: [2024, 2025]. None = todas as temporadas disponíveis.
    no_cache : bool
        Se True, ignora cache em disco.

    Retorna
    -------
    pd.DataFrame
        Colunas: home_team, away_team, date, league, season,
                 home_xg, away_xg, home_shots_target, away_shots_target,
                 home_possession, away_possession, source
    """
    if league not in LEAGUE_MAP:
        raise ValueError(f"Liga '{league}' não suportada. Opções: {list(LEAGUE_MAP)}")

    tid = LEAGUE_MAP[league]
    session = _make_session()
    _ensure_cache()

    # 1. Descobre as temporadas disponíveis
    logger.info(f"  Buscando temporadas para {league} (tid={tid})...")
    all_seasons = _fetch_seasons(session, tid, no_cache=no_cache)
    if not all_seasons:
        logger.error(f"  Não foi possível obter temporadas para {league} (tid={tid})")
        return pd.DataFrame()

    time.sleep(REQUEST_DELAY)

    # Filtra pelo(s) ano(s) solicitados
    if seasons:
        target_seasons = [s for s in all_seasons if any(_season_matches(s.get("year", ""), y, tid) for y in seasons)]
    else:
        target_seasons = all_seasons

    if not target_seasons:
        available = [s.get("year") for s in all_seasons[:10]]
        logger.warning(f"  Nenhuma temporada encontrada para {league} com seasons={seasons}. Disponíveis (últimas 10): {available}")
        return pd.DataFrame()

    logger.info(f"  Temporadas selecionadas: {[s['year'] for s in target_seasons]}")

    rows: list[dict] = []

    for season_info in target_seasons:
        sid = season_info["id"]
        year_label = season_info.get("year", str(sid))

        logger.info(f"  Processando temporada: {year_label} (season_id={sid})")

        # 2. Rodadas
        rounds = _fetch_rounds(session, tid, sid, no_cache=no_cache)
        if not rounds:
            logger.warning(f"    Sem rodadas para {year_label}")
            continue
        time.sleep(REQUEST_DELAY)

        round_nums = [r["round"] for r in rounds if "round" in r]
        logger.info(f"    {len(round_nums)} rodadas encontradas")

        finished_count = 0
        for rn in round_nums:
            events = _fetch_round_events(session, tid, sid, rn, no_cache=no_cache)
            time.sleep(REQUEST_DELAY)

            for event in events:
                status_code = event.get("status", {}).get("code", -1)
                if status_code != 100:  # 100 = jogo finalizado
                    continue

                game_id = event["id"]
                home_score = int(event["homeScore"]["current"])
                away_score = int(event["awayScore"]["current"])
                ts = event.get("startTimestamp", 0)
                match_date = datetime.fromtimestamp(ts, tz=timezone.utc).date()

                home_raw = event["homeTeam"]["name"]
                away_raw = event["awayTeam"]["name"]
                home_team = TEAM_NAME_MAP.get(home_raw, home_raw)
                away_team = TEAM_NAME_MAP.get(away_raw, away_raw)

                # 3. Stats da partida
                stats = _fetch_stats(session, game_id, no_cache=no_cache)
                time.sleep(REQUEST_DELAY)

                rows.append(
                    {
                        "home_team": home_team,
                        "away_team": away_team,
                        "date": match_date,
                        "league": league,
                        "season": year_label,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_xg": stats["home_xg"],
                        "away_xg": stats["away_xg"],
                        "home_shots_target": stats["home_shots_target"],
                        "away_shots_target": stats["away_shots_target"],
                        "home_possession": stats["home_possession"],
                        "away_possession": stats["away_possession"],
                        "source": "sofascore",
                    }
                )
                finished_count += 1

        logger.info(f"    {finished_count} partidas finalizadas — temporada {year_label}")

    if not rows:
        logger.warning(f"  Nenhuma partida retornada para {league}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    xg_ok = (df["home_xg"] > 0).sum()
    sot_ok = (df["home_shots_target"] > 0).sum()
    pos_ok = (df["home_possession"] > 0).sum()
    logger.info(f"  Total: {len(df)} partidas | xG>0: {xg_ok} | SoT>0: {sot_ok} | Posse>0: {pos_ok}")
    return df
