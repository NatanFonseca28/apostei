"""
discover_leagues.py
-------------------
Lista todos os torneios de futebol disponíveis no Sofascore por país/categoria.
Útil para descobrir o tournament_id correto antes de adicionar uma liga
ao LEAGUE_MAP em sofascore_collector.py.

Endpoint usado: GET api.sofascore.com/api/v1/category/{category_id}/unique-tournaments

Uso:
    python discover_leagues.py                         # lista todos os países
    python discover_leagues.py --country Brasil
    python discover_leagues.py --country Brasil Inglaterra Espanha
    python discover_leagues.py --search "serie a"      # filtra por nome
    python discover_leagues.py --search-api brasileirao  # busca via API
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")
from src.data.sofascore_collector import _API, _make_session  # noqa: E402

# category_id do Sofascore por país/região
# Descobertos via GET /api/v1/unique-tournament/{tid} → .uniqueTournament.category.id
CATEGORY_IDS: dict[str, int] = {
    "Brasil": 13,
    "Inglaterra": 1,
    "Espanha": 32,
    "Franca": 7,
    "Alemanha": 30,
    "Italia": 31,
    "Europa": 1465,  # UEFA / Champions / Europa
    "Portugal": 11,
    "Holanda": 35,
    "Belgica": 37,
    "Escocia": 130,
    "Argentina": 112,
    "Mexico": 162,
    "EUA": 211,
    "Colombia": 34,
    "Chile": 31506,
    "Turquia": 52,
    "Franca-D2": 7,  # mesmo cat da França
    "Grecia": 66,
}


def list_tournaments(
    categories: list[str],
    search: str | None = None,
) -> None:
    session = _make_session()
    found: list[dict] = []

    for cat_name in categories:
        cid = CATEGORY_IDS.get(cat_name)
        if cid is None:
            print(f"  Categoria '{cat_name}' não encontrada. Disponíveis: {list(CATEGORY_IDS)}")
            continue

        url = f"{_API}category/{cid}/unique-tournaments"
        resp = session.get(url, timeout=15)

        if resp.status_code != 200:
            print(f"  [{cat_name} cid={cid}] HTTP {resp.status_code}")
            time.sleep(0.8)
            continue

        data = resp.json()
        groups = data.get("groups", [])
        for group in groups:
            for t in group.get("uniqueTournaments", []):
                found.append(
                    {
                        "country": cat_name,
                        "id": t["id"],
                        "name": t["name"],
                        "slug": t.get("slug", ""),
                    }
                )

        time.sleep(0.8)

    # Filtra por nome
    if search:
        found = [t for t in found if search.lower() in t["name"].lower()]

    if not found:
        print("Nenhum torneio encontrado com os filtros aplicados.")
        return

    # Exibe tabela
    print(f"\n{'País':<15} {'ID':>6}  Nome")
    print("-" * 60)
    for t in sorted(found, key=lambda x: (x["country"], x["id"])):
        print(f"{t['country']:<15} {t['id']:>6}  {t['name']}")

    print(f"\nTotal: {len(found)} torneio(s).")
    print("Todos os torneios têm xG via GET /event/{game_id}/statistics.")


def search_by_name(query: str) -> None:
    """Busca torneio diretamente pelo nome via endpoint de search."""
    session = _make_session()
    resp = session.get(f"{_API}search/all?q={query.replace(' ', '+')}", timeout=15)
    if resp.status_code != 200:
        print(f"Erro na busca: HTTP {resp.status_code}")
        return

    results = resp.json().get("results", [])
    tournaments = [r for r in results if r.get("type") == "uniqueTournament"]
    if not tournaments:
        print("Nenhum torneio encontrado na busca.")
        return

    print(f"\nResultados para '{query}':")
    print(f"{'ID':>6}  {'Tipo':^25}  Nome")
    print("-" * 60)
    for r in tournaments[:20]:
        e = r.get("entity", {})
        cat = e.get("category", {}).get("name", "?")
        print(f"{e.get('id', '?'):>6}  {cat:<25}  {e.get('name', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descobre torneios disponíveis no Sofascore.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--country",
        nargs="+",
        default=list(CATEGORY_IDS.keys()),
        metavar="PAÍS",
        help=f"País(es) a listar. Padrão: todos.\nDisponíveis: {', '.join(CATEGORY_IDS)}",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        metavar="NOME",
        help="Filtra resultado por nome (case-insensitive).\nEx: --search 'serie a'",
    )
    parser.add_argument(
        "--search-api",
        type=str,
        default=None,
        metavar="QUERY",
        help="Busca diretamente pelo endpoint de search do Sofascore.\nEx: --search-api brasileirao",
    )
    args = parser.parse_args()

    if args.search_api:
        search_by_name(args.search_api)
    else:
        list_tournaments(
            categories=args.country,
            search=args.search,
        )


if __name__ == "__main__":
    main()
