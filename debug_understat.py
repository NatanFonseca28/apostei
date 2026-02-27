import typing

# Compatibility patch for Python 3.6.0
if not hasattr(typing, "Deque"):
    from typing import MutableSequence

    typing.Deque = MutableSequence
if not hasattr(typing, "Awaitable"):
    typing.Awaitable = typing.Generator
if not hasattr(typing, "AsyncGenerator"):
    typing.AsyncGenerator = typing.Generator

import asyncio

import aiohttp


async def test():
    url = "https://understat.com/league/EPL/2023"
    print(f"Fetching {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as response:
                print(f"Status: {response.status}")
                text = await response.text()
                print(f"Content length: {len(text)}")

                # Check for critical data variables
                print(f"Contains 'datesData': {'datesData' in text}")
                print(f"Contains 'teamsData': {'teamsData' in text}")
                print(f"Contains 'playersData': {'playersData' in text}")

                # Save for manual inspection if needed
                with open("debug_page.html", "w", encoding="utf-8") as f:
                    f.write(text)
                print("Saved HTML to debug_page.html")

    except Exception as e:
        print(f"Error: {e}")


loop = asyncio.get_event_loop()
loop.run_until_complete(test())
