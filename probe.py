import asyncio
import json

import aiohttp


async def probe():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://understat.com/league/EPL/2023",
        "Accept": "application/json, */*",
        "X-Requested-With": "XMLHttpRequest",
    }
    url = "https://understat.com/getLeagueData/EPL/2023"
    async with aiohttp.ClientSession(headers=headers) as s:
        r = await s.get(url)
        data = await r.json(content_type=None)
        print("Top-level keys:", list(data.keys()))
        dates = data.get("dates", [])
        print("Dates type:", type(dates))
        if isinstance(dates, list) and dates:
            print("First match sample:", json.dumps(dates[0], indent=2)[:600])
            print("Total matches:", len(dates))
        elif isinstance(dates, dict):
            first_key = list(dates.keys())[0]
            print("First key:", first_key, "sample:", json.dumps(dates[first_key], indent=2)[:400])


if __name__ == "__main__":
    asyncio.run(probe())
