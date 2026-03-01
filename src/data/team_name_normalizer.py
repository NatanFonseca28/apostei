"""
team_name_normalizer.py
-----------------------
Normaliza nomes de times provenientes de fontes diferentes (Flashscore e
Sofascore) para uma forma canônica comum, permitindo o JOIN correto entre
`flashscore_matches` e `match_advanced_stats`.

Problemas resolvidos:
  - Flashscore usa nomes abreviados em PT/ES ("Ath. Bilbao", "Atl. Madrid")
    enquanto Sofascore usa nomes completos em EN ("Athletic Club", "Atletico Madrid")
  - Flashscore adiciona códigos de país na Copa Libertadores: "Botafogo (Bra)"
  - Flashscore adiciona sufixos como "2", "Avança na competição"
  - Prefixos oficiais diferem: "Olympique de Marseille" vs "Marseille",
    "RC Lens" vs "Lens", "Stade Brestois" vs "Brest", etc.

Uso:
    from src.data.team_name_normalizer import canonical

    df['_home'] = df['time_casa'].apply(canonical)
    df2['_home'] = df2['home_team'].apply(canonical)
    merged = df.merge(df2, on=['_home', '_away', 'date'], how='left')
"""

import re

# ── Sufixos a remover (Flashscore adiciona texto contextual ao nome do time) ──

_SUFFIX_PATTERNS = re.compile(
    r"\s*(Avança na competição|Eliminado|Rebaixado|Promovido|Despromovido)\s*$",
    re.IGNORECASE,
)

# ── Prefixos descartáveis quando aparecem ANTES do nome real do clube ──
# (apenas quando não fazem parte da identidade do clube)
_PREFIX_PATTERNS = re.compile(
    r"^(Olympique de|Olympique|Stade|RC |AS |SC |FK |1\.\s+FC |1\.\s+FSV |1\.\s+)\s*",
    re.IGNORECASE,
)

# ── Mapeamento alias → nome canônico (tudo em lowercase) ──────────────────────
# Ordem: variante Flashscore (PT/ES/abrev.) → variante canônica (geralmente Sofascore EN)
# Inclui também variantes Sofascore que não mapeiam diretamente após a limpeza genérica.

_ALIAS: dict[str, str] = {
    # ── La Liga ──────────────────────────────────────────────────────────────
    "ath. bilbao": "athletic club",
    "ath. club": "athletic club",
    "athletic bilbao": "athletic club",
    "atl. madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "bétis": "real betis",
    "betis": "real betis",
    "celta de vigo": "celta vigo",
    "girona fc": "girona",
    "girona": "girona",
    "deportivo alavés": "deportivo alaves",
    "alavés": "deportivo alaves",
    "alaves": "deportivo alaves",
    "levante ud": "levante",
    "real oviedo": "oviedo",
    "osasuna": "osasuna",
    "ca osasuna": "osasuna",
    "rayo vallecano": "rayo vallecano",
    "real sociedad": "real sociedad",
    "real valladolid": "valladolid",
    "valladolid": "valladolid",
    "elche cf": "elche",
    "getafe cf": "getafe",
    "villarreal cf": "villarreal",
    "atletico madrid": "atletico madrid",
    # ── Premier League ───────────────────────────────────────────────────────
    "brighton": "brighton",
    "brighton & hove albion": "brighton",
    "nottingham": "nottingham forest",
    "nott'm forest": "nottingham forest",
    "wolverhampton": "wolves",
    "wolves": "wolves",
    "leeds": "leeds",
    "leeds united": "leeds",
    "manchester utd": "manchester utd",
    "manchester united": "manchester utd",
    "man utd": "manchester utd",
    "man united": "manchester utd",
    "manchester city": "manchester city",
    "man city": "manchester city",
    "newcastle utd": "newcastle",
    "newcastle united": "newcastle",
    "newcastle": "newcastle",
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
    "west ham united": "west ham",
    "west ham": "west ham",
    "crystal palace": "crystal palace",
    "aston villa": "aston villa",
    # ── Ligue 1 ──────────────────────────────────────────────────────────────
    "psg": "paris sg",
    "paris sg": "paris sg",
    "paris saint-germain": "paris sg",
    "monaco": "monaco",
    "as monaco": "monaco",
    "marseille": "marseille",
    "olympique de marseille": "marseille",
    "lyon": "lyon",
    "olympique lyonnais": "lyon",
    "lens": "lens",
    "rc lens": "lens",
    "strasbourg": "strasbourg",
    "rc strasbourg": "strasbourg",
    "rc strasbourg alsace": "strasbourg",
    "brest": "brest",
    "stade brestois": "brest",
    "stade brestois 29": "brest",
    "rennes": "rennes",
    "stade rennais": "rennes",
    "stade rennais fc": "rennes",
    "paris fc": "paris fc",
    "nice": "nice",
    "ogc nice": "nice",
    "nantes": "nantes",
    "fc nantes": "nantes",
    "toulouse": "toulouse",
    "toulouse fc": "toulouse",
    "auxerre": "auxerre",
    "aaj auxerre": "auxerre",
    "metz": "metz",
    "fc metz": "metz",
    "le havre": "le havre",
    "le havre ac": "le havre",
    "angers": "angers",
    "angers sco": "angers",
    "lorient": "lorient",
    "fc lorient": "lorient",
    "lille": "lille",
    "losc lille": "lille",
    # ── Serie A (Itália) ─────────────────────────────────────────────────────
    "ac milan": "milan",
    "milan": "milan",
    "as roma": "roma",
    "roma": "roma",
    "verona": "verona",
    "hellas verona": "verona",
    "inter": "inter",
    "fc internazionale milano": "inter",
    "internazionale": "inter",
    "juventus fc": "juventus",
    "juventus": "juventus",
    "fiorentina": "fiorentina",
    "acf fiorentina": "fiorentina",
    "napoli": "napoli",
    "ssc napoli": "napoli",
    "lazio": "lazio",
    "ss lazio": "lazio",
    "atalanta": "atalanta",
    "atalanta bc": "atalanta",
    "bologna": "bologna",
    "bologna fc": "bologna",
    "torino": "torino",
    "torino fc": "torino",
    "udinese": "udinese",
    "udinese calcio": "udinese",
    "genoa": "genoa",
    "genoa cfc": "genoa",
    "cagliari": "cagliari",
    "cagliari calcio": "cagliari",
    "lecce": "lecce",
    "us lecce": "lecce",
    "sassuolo": "sassuolo",
    "us sassuolo": "sassuolo",
    "parma": "parma",
    "cremonese": "cremonese",
    "us cremonese": "cremonese",
    "como": "como",
    "como 1907": "como",
    "pisa": "pisa",
    "ac pisa": "pisa",
    # ── Bundesliga ───────────────────────────────────────────────────────────
    "fc bayern münchen": "bayern munchen",
    "fc bayern munchen": "bayern munchen",
    "fc bayern munich": "bayern munchen",
    "bayern münchen": "bayern munchen",
    "bayern munchen": "bayern munchen",
    "bayer 04 leverkusen": "bayer leverkusen",
    "bayer leverkusen": "bayer leverkusen",
    "borussia m'gladbach": "gladbach",
    "borussia mönchengladbach": "gladbach",
    "borussia monchengladbach": "gladbach",
    "gladbach": "gladbach",
    "dortmund": "dortmund",
    "borussia dortmund": "dortmund",
    "bvb": "dortmund",
    "rb leipzig": "rb leipzig",
    "rasenballsport leipzig": "rb leipzig",
    "ein frankfurt": "eintracht frankfurt",
    "eintracht frankfurt": "eintracht frankfurt",
    "1. fc köln": "koln",
    "1. fc koln": "koln",
    "koln": "koln",
    "1. fc heidenheim": "heidenheim",
    "fc heidenheim": "heidenheim",
    "heidenheim": "heidenheim",
    "1. fc union berlin": "union berlin",
    "union berlin": "union berlin",
    "1. fsv mainz 05": "mainz",
    "fsv mainz 05": "mainz",
    "mainz 05": "mainz",
    "mainz": "mainz",
    "fc augsburg": "augsburg",
    "augsburg": "augsburg",
    "fc st. pauli": "st pauli",
    "st. pauli": "st pauli",
    "hamburger sv": "hamburger sv",
    "hamburger": "hamburger sv",
    "hsv": "hamburger sv",
    "tsg hoffenheim": "hoffenheim",
    "hoffenheim": "hoffenheim",
    "tsg 1899 hoffenheim": "hoffenheim",
    "vfb stuttgart": "stuttgart",
    "stuttgart": "stuttgart",
    "vfl wolfsburg": "wolfsburg",
    "wolfsburg": "wolfsburg",
    "sc freiburg": "sc freiburg",
    "freiburg": "sc freiburg",
    "sv werder bremen": "werder bremen",
    "werder bremen": "werder bremen",
    "werder": "werder bremen",
    # ── Champions League (outros clubes) ─────────────────────────────────────
    "afc ajax": "ajax",
    "ajax": "ajax",
    "club brugge kv": "brugge",
    "club brugge": "brugge",
    "brugge": "brugge",
    "bodø/glimt": "bodo glimt",
    "bodo/glimt": "bodo glimt",
    "bodo glimt": "bodo glimt",
    "fc københavn": "kobenhavn",
    "fc koebenhavn": "kobenhavn",
    "kobenhavn": "kobenhavn",
    "psv eindhoven": "psv",
    "psv": "psv",
    "royale union saint-gilloise": "union sg",
    "union saint-gilloise": "union sg",
    "sk slavia praha": "slavia prague",
    "slavia prague": "slavia prague",
    "slavia praha": "slavia prague",
    "sporting cp": "sporting cp",
    "sporting": "sporting cp",
    "pafos fc": "pafos",
    "olympiacos fc": "olympiacos",
    "olympiacos": "olympiacos",
    "qarabağ fk": "qarabag",
    "qarabag fk": "qarabag",
    "qarabag": "qarabag",
    "kairat almaty": "kairat",
    "galatasaray": "galatasaray",
    "galatasaray sk": "galatasaray",
    "benfica": "benfica",
    "sl benfica": "benfica",
    "porto": "porto",
    "fc porto": "porto",
    # ── Brasileirão ──────────────────────────────────────────────────────────
    "athletico": "athletico paranaense",
    "athletico paranaense": "athletico paranaense",
    "athletico-pr": "athletico paranaense",
    "atletico mg": "atletico mineiro",
    "atletico mineiro": "atletico mineiro",
    "atlético mg": "atletico mineiro",
    "atlético mineiro": "atletico mineiro",
    "botafogo": "botafogo",
    "flamentego": "flamengo",
    "flamengo": "flamengo",
    "fluminense": "fluminense",
    "palmeiras": "palmeiras",
    "corinthians": "corinthians",
    "sao paulo": "sao paulo",
    "são paulo": "sao paulo",
    "internacional": "internacional",
    "inter porto alegre": "internacional",
    "gremio": "gremio",
    "grêmio": "gremio",
    "cruzeiro": "cruzeiro",
    "santos": "santos",
    "vasco": "vasco",
    "vasco da gama": "vasco",
    "bahia": "bahia",
    "ec bahia": "bahia",
    "vitoria": "vitoria",
    "vitória": "vitoria",
    "vitória ba": "vitoria",
    "rb bragantino": "rb bragantino",
    "red bull bragantino": "rb bragantino",
    "mirassol": "mirassol",
    "coritiba": "coritiba",
    "chapecoense": "chapecoense",
    "remo": "remo",
    "juventude": "juventude",
    "sport recife": "sport",
    "sport": "sport",
    "ceara": "ceara",
    "ceará": "ceara",
    "america mg": "america mineiro",
    "america mineiro": "america mineiro",
    "fortaleza": "fortaleza",
    "goias": "goias",
    "goiás": "goias",
    "cuiaba": "cuiaba",
    "cuiabá": "cuiaba",
    "avai": "avai",
    "avaí": "avai",
    # ── Copa Libertadores (com códigos de país removidos pela limpeza) ───────
    "dep. tachira": "deportivo tachira",
    "deportivo tachira": "deportivo tachira",
    "ind. medellin": "independiente medellin",
    "independiente medellin": "independiente medellin",
    "ind. medellín": "independiente medellin",
    "liverpool m.": "liverpool montevideo",
    "liverpool montev.": "liverpool montevideo",
    "o'higgins": "ohiggins",
    "ohiggins": "ohiggins",
    "nacional potosí": "nacional potosi",
    "nacional potosi": "nacional potosi",
    "universidad católica ecu": "universidad catolica ecu",
    "universidad católica": "universidad catolica",
    "universidad catolica": "universidad catolica",
    "sporting cristal": "sporting cristal",
    "argentinos juniors": "argentinos juniors",
    "alianza lima": "alianza lima",
    "the strongest": "the strongest",
    "guarani": "guarani py",
    "juventud": "juventud uru",
    "carabobo": "carabobo",
    "huachipato": "huachipato",
    "2 de mayo": "2 de mayo",
}


def _clean(name: str) -> str:
    """Limpeza genérica: remove sufixos, códigos de país e sufixos numéricos."""
    if not isinstance(name, str) or not name:
        return ""

    s = name

    # 1. Remove sufixos Flashscore contextuais (ex: "Botafogo (Bra)Avança na competição")
    s = _SUFFIX_PATTERNS.sub("", s)

    # 2. Remove código de país em parênteses: "(Bra)", "(Arg)", "(Per)", etc.
    s = re.sub(r"\s*\([A-Za-z]{2,4}\)\s*", " ", s)

    # 3. Remove sufixo numérico de equipes reserva: "Real Madrid2" → "Real Madrid"
    s = re.sub(r"\s*\d+\s*$", "", s)

    return s.strip()


def canonical(name: str) -> str:
    """
    Retorna a forma canônica do nome do time para comparação no JOIN.

    Exemplo:
        canonical("Bétis")            → "real betis"
        canonical("Real Betis")       → "real betis"
        canonical("Ath. Bilbao")      → "athletic club"
        canonical("Athletic Club")    → "athletic club"
        canonical("Brighton")         → "brighton"
        canonical("Brighton & Hove Albion") → "brighton"
        canonical("Botafogo (Bra)")   → "botafogo"
        canonical("Marseille2")       → "marseille"
    """
    s = _clean(name).lower()

    # Tenta alias direto
    if s in _ALIAS:
        return _ALIAS[s]

    # Remove prefixos descartáveis e tenta novamente
    s2 = _PREFIX_PATTERNS.sub("", s).strip()
    if s2 in _ALIAS:
        return _ALIAS[s2]

    # Retorna a forma limpa sem prefixos como fallback
    return s2 if s2 else s
