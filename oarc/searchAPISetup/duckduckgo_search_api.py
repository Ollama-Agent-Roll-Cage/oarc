"""
DuckDuckGo Search API with Extended Capabilities

This module provides an enhanced interface for performing DuckDuckGo text, image, and news searches,
with integrated support for storing query results. It facilitates asynchronous search operations,
formats the retrieved data, and persists results in a MongoDB database.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

import aiohttp
import pymongo
from fastapi import APIRouter, HTTPException

from oarc.utils.paths import Paths

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

DDG_URL = "https://duckduckgo.com/?q="

class DuckDuckGoSearch:


    def __init__(self, storage_type: str = "json", database_path: str = "searchResults.db"):
        """
        Initialize the DuckDuckGo Search API wrapper.
        
        Args:
            storage_type: Type of storage ("json" or "sqlite")
            database_path: Path to the database file
        """
        self.storage_type = storage_type
        self.database_path = database_path

    async def text_search(self, search_query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform an async text search using DuckDuckGo.
        
        Args:
            search_query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of formatted search results
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{DDG_URL}{search_query}&format=json&pretty=1") as response:
                    results = await response.json()
                    formatted_results = self.format_results(results)
                    await self.store_results(search_query, "text", formatted_results)
                    return formatted_results
        except Exception as e:
            log.error(f"Text search error: {e}")
            raise

    async def image_search(self, search_query: str, max_results: int = 10) -> List[Dict]:
        """Perform an async image search using DuckDuckGo."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{DDG_URL}{search_query}&format=json&pretty=1&iax=images&ia=images") as response:
                    results = await response.json()
                    formatted_results = self.format_image_results(results)
                    await self.store_results(search_query, "image", formatted_results)
                    return formatted_results
        except Exception as e:
            log.error(f"Image search error: {e}")
            raise

    async def news_search(self, search_query: str, max_results: int = 20) -> List[Dict]:
        """Perform an async news search using DuckDuckGo."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{DDG_URL}{search_query}&format=json&pretty=1&ia=news") as response:
                    results = await response.json()
                    formatted_results = self.format_news_results(results)
                    await self.store_results(search_query, "news", formatted_results)
                    return formatted_results
        except Exception as e:
            log.error(f"News search error: {e}")
            raise

    def format_results(self, results: List[Dict]) -> List[Dict]:
        """Format text search results."""
        return [{
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("body"),
            "source": result.get("source"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    def format_image_results(self, results: List[Dict]) -> List[Dict]:
        """Format image search results."""
        return [{
            "title": result.get("title"),
            "image_url": result.get("image"),
            "thumbnail": result.get("thumbnail"),
            "source_url": result.get("url"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    def format_news_results(self, results: List[Dict]) -> List[Dict]:
        """Format news search results."""
        return [{
            "title": result.get("title"),
            "link": result.get("link"),
            "snippet": result.get("excerpt"),
            "published": result.get("date"),
            "source": result.get("source"),
            "timestamp": datetime.now().isoformat()
        } for result in results]

    async def store_results(self, search_query: str, search_type: str, results: List[Dict]) -> None:
        """Store search results in the database."""
        client = pymongo.MongoClient(self.database_path)
        database = client.searchResults
        collection = database[search_type]
        for result in results:
            result["search_query"] = search_query
        collection.insert_many(results)
        client.close()
        
class DuckDuckGoSearchAPI:
    def __init__(self):
        """Initialize the DuckDuckGo Search API with routes and search engine."""
        self.router = APIRouter(prefix="/search", tags=["search"])
        self.search_engine = DuckDuckGoSearch()
        self.paths = Paths()  # Initialize paths utility
        self.setup_routes()
        log.info("DuckDuckGo Search API initialized")
        
    def setup_routes(self):
        """Set up API routes for search functionality"""
        log.info("Setting up DuckDuckGoSearchAPI routes")
        
        @self.router.get("/text")
        async def search_text(query: str, max_results: int = 10):
            """Search for text results via DuckDuckGo
            
            Args:
                query: Search query string
                max_results: Maximum number of results to return
                
            Returns:
                dict: Search results
            """
            try:
                log.info(f"Text search request: '{query}' (max: {max_results})")
                results = await self.search_engine.text_search(query, max_results)
                log.info(f"Found {len(results)} text results for: '{query}'")
                return {"results": results}
            except Exception as e:
                log.error(f"Text search failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
                
        @self.router.get("/images")
        async def search_images(query: str, max_results: int = 10):
            """Search for images via DuckDuckGo
            
            Args:
                query: Search query string
                max_results: Maximum number of results to return
                
            Returns:
                dict: Image search results
            """
            try:
                log.info(f"Image search request: '{query}' (max: {max_results})")
                results = await self.search_engine.image_search(query, max_results)
                log.info(f"Found {len(results)} image results for: '{query}'")
                return {"results": results}
            except Exception as e:
                log.error(f"Image search failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")
                
        @self.router.get("/news")
        async def search_news(query: str, max_results: int = 20):
            """Search for news via DuckDuckGo
            
            Args:
                query: Search query string
                max_results: Maximum number of results to return
                
            Returns:
                dict: News search results
            """
            try:
                log.info(f"News search request: '{query}' (max: {max_results})")
                results = await self.search_engine.news_search(query, max_results)
                log.info(f"Found {len(results)} news results for: '{query}'")
                return {"results": results}
            except Exception as e:
                log.error(f"News search failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"News search failed: {str(e)}")
                
        @self.router.get("/history")
        async def get_search_history(search_type: Optional[str] = None, limit: int = 50):
            """Get search history from the database
            
            Args:
                search_type: Type of search (text, image, news)
                limit: Maximum number of results to return
                
            Returns:
                dict: Search history
            """
            try:
                log.info(f"Search history request: type={search_type}, limit={limit}")
                # TODO: Implement history retrieval from database
                return {"history": []}
            except Exception as e:
                log.error(f"History retrieval failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")
                
        log.info("DuckDuckGoSearchAPI routes setup complete")
            
        @self.router.post("/synthesize") 
        async def synthesize_speech(self, text: str):
            pass