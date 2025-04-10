"""
Crawler modules for fetching and processing information from different sources.

Available classes:
- WebCrawler: General web page crawling
- ArxivSearcher: ArXiv paper lookup and parsing 
- DuckDuckGoSearcher: DuckDuckGo search API integration
- GitHubCrawler: GitHub repository cloning and extraction
"""

from .beautiful_soup import BSWebCrawler
from .arxiv_fetcher import ArxivFetcher  
from .ddg_search import DuckDuckGoSearcher
from .gh_crawler import GitHubCrawler
from .parquet_storage import ParquetStorage

__all__ = [
    "BSWebCrawler",
    "ArxivFetcher",
    "DuckDuckGoSearcher",  
    "GitHubCrawler",
    "ParquetStorage"
]