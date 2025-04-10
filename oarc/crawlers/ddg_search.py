"""
DuckDuckGo Search API with Extended Capabilities

This module provides an enhanced interface for performing DuckDuckGo text, image, and news searches.
This module also includes functionality to save search results in Parquet format for later analysis.

Author: @BorcherdingL
Date: 4/9/2025
"""

import re
import logging
import urllib.parse
import xml.etree.ElementTree as ET
import json

from datetime import datetime, UTC
from pathlib import Path
import aiohttp

from parquet_storage import ParquetStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuckDuckGoSearcher:
    """Class for performing searches using DuckDuckGo API."""
    
    def __init__(self, data_dir=None):
        """Initialize the DuckDuckGo Searcher.
        
        Args:
            data_dir (str, optional): Directory to store data.
        """
        self.data_dir = data_dir if data_dir else Path("./data")
        self.searches_dir = Path(f"{self.data_dir}/searches")
        self.searches_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def text_search(self, search_query, max_results=5):
        """Perform an async text search using DuckDuckGo.
        
        Args:
            search_query (str): Query to search for
            max_results (int): Maximum number of results to return
            
        Returns:
            str: Formatted search results in markdown
        """
        try:
            encoded_query = urllib.parse.quote(search_query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=1"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result_text = await response.text()
                        try:
                            results = json.loads(result_text)
                            
                            # Save search results to Parquet
                            search_data = {
                                'query': search_query,
                                'timestamp': datetime.now(UTC).isoformat(),
                                'raw_results': result_text
                            }
                            
                            # Generate a filename from the query
                            filename = re.sub(r'[^\w]', '_', search_query)[:50]
                            file_path = f"{self.data_dir}/searches/{filename}_{int(datetime.now().timestamp())}.parquet"
                            ParquetStorage.save_to_parquet(search_data, file_path)
                            
                            # Format the response nicely for Discord
                            formatted_results = "# DuckDuckGo Search Results\n\n"
                            
                            if 'AbstractText' in results and results['AbstractText']:
                                formatted_results += f"## Summary\n{results['AbstractText']}\n\n"
                                
                            if 'RelatedTopics' in results:
                                formatted_results += "## Related Topics\n\n"
                                count = 0
                                for topic in results['RelatedTopics']:
                                    if count >= max_results:
                                        break
                                    if 'Text' in topic and 'FirstURL' in topic:
                                        formatted_results += f"- [{topic['Text']}]({topic['FirstURL']})\n"
                                        count += 1
                            
                            return formatted_results
                        except json.JSONDecodeError:
                            return "Error: Could not parse the search results."
                    else:
                        return f"Error: Received status code {response.status} from DuckDuckGo API."
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            return f"An error occurred during the search: {str(e)}"

    async def image_search(self, search_query, max_results=10):
        """Perform an async image search using DuckDuckGo.

        Args:
            search_query (str): Query to search for images
            max_results (int): Maximum number of results to return
            
        Returns:
            str: Formatted image search results in markdown
        """
        try:
            encoded_query = urllib.parse.quote(search_query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=1&iax=images&ia=images"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result_text = await response.text()
                        try:
                            results = json.loads(result_text)
                            
                            # Save search results to Parquet
                            search_data = {
                                'query': search_query,
                                'type': 'image',
                                'timestamp': datetime.now(UTC).isoformat(),
                                'raw_results': result_text
                            }
                            
                            # Generate a filename from the query
                            filename = re.sub(r'[^\w]', '_', search_query)[:50]
                            file_path = f"{self.data_dir}/searches/img_{filename}_{int(datetime.now().timestamp())}.parquet"
                            ParquetStorage.save_to_parquet(search_data, file_path)
                            
                            # Format the response nicely
                            formatted_results = "# DuckDuckGo Image Search Results\n\n"
                            
                            # Extract image results
                            if 'Images' in results:
                                count = 0
                                for image in results['Images'][:max_results]:
                                    if count >= max_results:
                                        break
                                    if 'Image' in image and 'Source' in image:
                                        formatted_results += f"![{image.get('Title', 'Image')}]({image['Image']})\n"
                                        formatted_results += f"[Source: {image.get('Source', 'Unknown')}]({image.get('URL', '#')})\n\n"
                                        count += 1
                            elif 'RelatedTopics' in results:
                                formatted_results += "## Related Image Topics\n\n"
                                count = 0
                                for topic in results['RelatedTopics']:
                                    if count >= max_results:
                                        break
                                    if 'Icon' in topic and 'URL' in topic.get('Icon', {}):
                                        image_url = topic['Icon']['URL']
                                        if image_url:
                                            formatted_results += f"![{topic.get('Text', 'Image')}]({image_url})\n"
                                            if 'FirstURL' in topic:
                                                formatted_results += f"[Source]({topic['FirstURL']})\n\n"
                                            count += 1
                            
                            if not "![" in formatted_results:
                                formatted_results += "No image results found.\n"
                                
                            return formatted_results
                        except json.JSONDecodeError:
                            return "Error: Could not parse the image search results."
                    else:
                        return f"Error: Received status code {response.status} from DuckDuckGo API."
        except Exception as e:
            self.logger.error(f"DuckDuckGo image search error: {e}")
            return f"An error occurred during the image search: {str(e)}"

    async def news_search(self, search_query, max_results=20):
        """Perform an async news search using DuckDuckGo.
        
        Args:
            search_query (str): Query to search for news
            max_results (int): Maximum number of results to return
            
        Returns:
            str: Formatted news search results in markdown
        """
        try:
            encoded_query = urllib.parse.quote(search_query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=1&ia=news"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result_text = await response.text()
                        try:
                            results = json.loads(result_text)
                            
                            # Save search results to Parquet
                            search_data = {
                                'query': search_query,
                                'type': 'news',
                                'timestamp': datetime.now(UTC).isoformat(),
                                'raw_results': result_text
                            }
                            
                            # Generate a filename from the query
                            filename = re.sub(r'[^\w]', '_', search_query)[:50]
                            file_path = f"{self.data_dir}/searches/news_{filename}_{int(datetime.now().timestamp())}.parquet"
                            ParquetStorage.save_to_parquet(search_data, file_path)
                            
                            # Format the response nicely
                            formatted_results = "# DuckDuckGo News Search Results\n\n"
                            
                            # Try to extract news results
                            if 'News' in results:
                                count = 0
                                for news in results['News'][:max_results]:
                                    if count >= max_results:
                                        break
                                    if 'Title' in news and 'URL' in news:
                                        formatted_results += f"## {news['Title']}\n"
                                        if 'Source' in news:
                                            formatted_results += f"**Source:** {news['Source']}\n"
                                        if 'Date' in news:
                                            formatted_results += f"**Date:** {news['Date']}\n"
                                        if 'Excerpt' in news:
                                            formatted_results += f"\n{news['Excerpt']}\n"
                                        formatted_results += f"\n[Read more]({news['URL']})\n\n"
                                        count += 1
                            # Fallback to related topics if no news section
                            elif 'RelatedTopics' in results:
                                formatted_results += "## Related News Topics\n\n"
                                count = 0
                                for topic in results['RelatedTopics']:
                                    if count >= max_results:
                                        break
                                    if 'Text' in topic and 'FirstURL' in topic:
                                        formatted_results += f"- [{topic['Text']}]({topic['FirstURL']})\n"
                                        count += 1
                            
                            if formatted_results == "# DuckDuckGo News Search Results\n\n":
                                formatted_results += "No news results found.\n"
                                
                            return formatted_results
                        except json.JSONDecodeError:
                            return "Error: Could not parse the news search results."
                    else:
                        return f"Error: Received status code {response.status} from DuckDuckGo API."
        except Exception as e:
            self.logger.error(f"DuckDuckGo news search error: {e}")
            return f"An error occurred during the news search: {str(e)}"