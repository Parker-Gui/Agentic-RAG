from tavily import TavilyClient

# Step 1. Instantiating your TavilyClient
tavily_client = TavilyClient(api_key="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")

# Step 2. Defining the starting URL
start_url = "https://www.nutra-max.cn/news/0.html"

# Step 3. Executing the crawl request with instructions to surface only pages about citrus fruits
response = tavily_client.crawl(
    url=start_url,
    max_depth=3,
    limit=50,
    instructions="把5.1号到8.25号的有关甜味剂的文章检索出来，并返回文章的标题、链接和内容摘要，按时间顺序排列。",
)

# Step 4. Printing pages matching the query with more detailed information
for result in response.get("results", []):
    title = result.get("title") or "无标题"
    url = result.get("url") or "无链接"
    content = result.get("raw_content") or result.get("content") or ""
    snippet = content[:1200].strip()

    print(f"标题: {title}")
    print(f"URL: {url}")
    print(f"内容摘要: {snippet[:1200]}{'...' if len(snippet) > 1200 else ''}")
    print("-" * 80)
