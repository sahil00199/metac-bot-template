from __future__ import annotations

import asyncio
import os
import time

from asknews_sdk import AsyncAskNewsSDK, AskNewsSDK


if __name__ == "__main__":
    async def main() -> None:
        query = "Will humans go extinct before 2100?"

        client_id = os.environ.get("ASKNEWS_CLIENT_ID")
        client_secret = os.environ.get("ASKNEWS_SECRET")

        if not client_id or not client_secret:
            raise RuntimeError(
                "Environment variables ASKNEWS_CLIENT_ID and ASKNEWS_SECRET must be set"
            )

        async with AsyncAskNewsSDK(
            client_id=client_id,
            client_secret=client_secret,
            # scopes={"news"},
        ) as ask:

            # get the latest news related to the query (within the past 48 hours)
            hot_response = await ask.news.search_news(
                query=query,  # your natural language query
                n_articles=6,  # control the number of articles to include in the context
                return_type="both",
                strategy="news knowledge",  # enforces looking at the latest news only
            )
            print("Past response received")

            time.sleep(15)
            historical_response = await ask.news.search_news(
                query=query,
                n_articles=10,
                return_type="both",
                strategy="latest news",  # looks for relevant news within the past 60 days
            )
            print("hot response received")

        

    asyncio.run(main())