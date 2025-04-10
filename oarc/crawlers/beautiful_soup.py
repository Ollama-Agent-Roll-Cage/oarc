"""Web Crawler Module

This module provides functions for crawling and extracting content from web pages
using BeautifulSoup. Includes specialized extractors for documentation sites, PyPI,
and general web content.

Author: @BorcherdingL
Date: 4/10/2025
"""

import re
import logging
from datetime import datetime, UTC
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

from .parquet_storage import ParquetStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BSWebCrawler:
    """Class for crawling web pages and extracting content."""
    
    def __init__(self, data_dir=None):
        """Initialize the Web Crawler.
        
        Args:
            data_dir (str, optional): Directory to store data. Defaults to DATA_DIR.
        """
        self.data_dir = data_dir
        self.crawl_data_dir = Path(f"{self.data_dir}/crawls")
        self.crawl_data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def fetch_url_content(url):
        """Fetch content from a URL.
        
        Args:
            url (str): The URL to fetch content from
            
        Returns:
            str: HTML content of the page or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Save crawled content
                        crawl_data = {
                            'url': url,
                            'timestamp': datetime.now(UTC).isoformat(),
                            'content': html[:100000]  # Limit content size
                        }
                        
                        # Generate a filename from the URL
                        filename = re.sub(r'[^\w]', '_', url.split('//')[-1])[:50]
                        file_path = f"{self.data_dir}/crawls/{filename}_{int(datetime.now().timestamp())}.parquet"
                        
                        # Ensure directory exists
                        Path(f"{self.data_dir}/crawls").mkdir(parents=True, exist_ok=True)
                        
                        # Save the data
                        ParquetStorage.save_to_parquet(crawl_data, file_path)
                        
                        return html
                    else:
                        logger.warning(f"Failed to fetch URL {url}: HTTP Status {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None

    @staticmethod
    async def extract_text_from_html(html):
        """Extract main text content from HTML using BeautifulSoup.
        
        Args:
            html (str): HTML content
            
        Returns:
            str: Extracted text content
        """
        if html:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                    
                # Get text
                text = soup.get_text(separator=' ', strip=True)
                
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Limit to first ~15,000 characters
                return text[:15000] + ("..." if len(text) > 15000 else "")
            except Exception as e:
                logger.error(f"Error parsing HTML: {e}")
                # Fall back to regex method if BeautifulSoup fails
                clean_html = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.DOTALL)
                clean_html = re.sub(r'<style.*?>.*?</style>', '', clean_html, flags=re.DOTALL)
                text = re.sub(r'<.*?>', ' ', clean_html)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:10000] + ("..." if len(text) > 10000 else "")
        return "Failed to extract text from the webpage."

    @staticmethod
    async def extract_pypi_content(html, package_name):
        """Specifically extract PyPI package documentation from HTML.
        
        Args:
            html (str): HTML content from PyPI page
            package_name (str): Name of the package
            
        Returns:
            dict: Structured package data or None if failed
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract package metadata from the sidebar
            metadata = {}
            sidebar = soup.find('div', {'class': 'sidebar'})
            if (sidebar):
                for section in sidebar.find_all('div', {'class': 'sidebar-section'}):
                    title_elem = section.find(['h3', 'h4'])
                    if title_elem:
                        section_title = title_elem.get_text().strip()
                        content_list = []
                        for p in section.find_all('p'):
                            content_list.append(p.get_text().strip())
                        metadata[section_title] = content_list
            
            # Find the project description section which contains the actual documentation
            description_div = soup.find('div', {'class': 'project-description'})
            
            if (description_div):
                # Extract text while preserving structure
                content = ""
                for element in description_div.children:
                    if hasattr(element, 'name'):  # Check if it's a tag
                        if element.name in ['h1', 'h2', 'h3', 'h4']:
                            heading_level = int(element.name[1])
                            heading_text = element.get_text().strip()
                            content += f"{'#' * heading_level} {heading_text}\n\n"
                        elif element.name == 'p':
                            content += f"{element.get_text().strip()}\n\n"
                        elif element.name == 'pre':
                            code = element.get_text().strip()
                            # Detect if there's a code element inside
                            code_element = element.find('code')
                            language = "python" if code_element and 'python' in str(code_element.get('class', [])).lower() else ""
                            content += f"```{language}\n{code}\n```\n\n"
                        elif element.name == 'ul':
                            for li in element.find_all('li', recursive=False):
                                content += f"- {li.get_text().strip()}\n"
                            content += "\n"
                
                # Construct a structured representation
                package_info = {
                    'name': package_name,
                    'metadata': metadata,
                    'documentation': content
                }
                
                return package_info
            else:
                logger.warning(f"No project description found for PyPI package {package_name}")
                return None
        except Exception as e:
            logger.error(f"Error extracting PyPI content for {package_name}: {e}")
            return None
    
    @staticmethod
    async def extract_documentation_content(html, url):
        """Extract content from documentation websites like ReadTheDocs, LlamaIndex, etc.
        
        Args:
            html (str): HTML content from the documentation site
            url (str): URL of the documentation page
            
        Returns:
            dict: Structured documentation data
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            doc_data = {
                'url': url,
                'title': '',
                'content': '',
                'toc': [],
                'metadata': {},
                'code_snippets': []
            }
            
            # Extract title - different websites have different structures
            title_candidates = [
                # ReadTheDocs/Sphinx style
                soup.find('div', {'class': 'document'}),
                # MkDocs style
                soup.find('div', {'class': 'md-content'}),
                # Generic
                soup.find('main'),
                soup.find('article')
            ]
            
            # Try to find the title
            for candidate in title_candidates:
                if candidate:
                    title_elem = candidate.find(['h1', 'h2'])
                    if title_elem:
                        doc_data['title'] = title_elem.get_text().strip()
                        break
            
            # If no title found using above methods, use the page title
            if not doc_data['title'] and soup.title:
                doc_data['title'] = soup.title.get_text().strip()
            
            # Identify main content container based on common documentation site structures
            content_candidates = [
                # ReadTheDocs/Sphinx
                soup.find('div', {'class': 'section'}),
                soup.find('div', {'class': 'body', 'role': 'main'}),
                soup.find('div', {'class': 'document'}),
                # MkDocs
                soup.find('div', {'class': 'md-content'}),
                soup.find('article', {'class': 'md-content__inner'}),
                # LlamaIndex style
                soup.find('div', {'class': 'prose'}),
                soup.find('div', {'class': 'content'}),
                # Generic fallbacks
                soup.find('main'),
                soup.find('article'),
                soup.find('div', {'id': 'content'})
            ]
            
            # Find the main content container
            main_content = None
            for candidate in content_candidates:
                if candidate:
                    main_content = candidate
                    break
            
            # If we still can't find a content container, use the body
            if main_content is None:
                main_content = soup.body
            
            if main_content:
                # Extract text preserving structure
                content = ""
                code_snippets = []
                
                # Extract table of contents if available
                toc_candidates = [
                    soup.find('div', {'class': 'toc'}),
                    soup.find('div', {'class': 'toctree'}),
                    soup.find('nav', {'class': 'md-nav'}),
                    soup.find('ul', {'class': 'toc'}),
                    soup.find('div', {'class': 'sidebar'})
                ]
                
                for toc in toc_candidates:
                    if toc:
                        # Extract TOC items
                        toc_items = []
                        for a in toc.find_all('a'):
                            if a.get_text().strip():
                                toc_items.append({
                                    'title': a.get_text().strip(),
                                    'url': a.get('href', '')
                                })
                        doc_data['toc'] = toc_items
                        break
                
                # Process the content by element type
                for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol', 'pre', 'code', 'div']):
                    if element.name in ['h1', 'h2', 'h3', 'h4']:
                        heading_text = element.get_text().strip()
                        if heading_text:
                            heading_level = int(element.name[1])
                            content += f"{'#' * heading_level} {heading_text}\n\n"
                    
                    elif element.name == 'p':
                        para_text = element.get_text().strip()
                        if para_text:
                            content += f"{para_text}\n\n"
                    
                    elif element.name in ['ul', 'ol']:
                        for li in element.find_all('li', recursive=False):
                            content += f"- {li.get_text().strip()}\n"
                        content += "\n"
                    
                    elif element.name == 'pre' or (element.name == 'div' and 'highlight' in element.get('class', [])):
                        code = element.get_text().strip()
                        if code:
                            # Try to detect language
                            lang = ""
                            if 'python' in str(element.get('class', [])).lower():
                                lang = "python"
                            elif 'javascript' in str(element.get('class', [])).lower():
                                lang = "javascript"
                            
                            # Add code to content
                            content += f"```{lang}\n{code}\n```\n\n"
                            
                            # Also save separately for easier access
                            code_snippets.append({
                                'language': lang,
                                'code': code
                            })
                
                doc_data['content'] = content
                doc_data['code_snippets'] = code_snippets
                
                # Extract metadata if available (e.g., version, last updated)
                meta_tags = soup.find_all('meta')
                for tag in meta_tags:
                    name = tag.get('name', '')
                    if name in ['description', 'keywords', 'author', 'version']:
                        doc_data['metadata'][name] = tag.get('content', '')
                
                # Get last updated time if available
                last_updated = None
                update_candidates = [
                    soup.find('time'),
                    soup.find(lambda tag: tag.name == 'p' and ('updated' in tag.text.lower() or 'modified' in tag.text.lower())),
                    soup.find('div', {'class': 'last-updated'})
                ]
                
                for update_elem in update_candidates:
                    if update_elem:
                        last_updated = update_elem.get_text().strip()
                        doc_data['metadata']['last_updated'] = last_updated
                        break
            
            return doc_data
            
        except Exception as e:
            logger.error(f"Error extracting documentation content from {url}: {e}")
            return {
                'url': url,
                'title': 'Error extracting content',
                'content': f"Failed to parse documentation: {str(e)}",
                'error': str(e)
            }

    @staticmethod
    async def format_pypi_info(package_data):
        """Format PyPI package data into a readable markdown format.
        
        Args:
            package_data (dict): Package data from PyPI API
            
        Returns:
            str: Formatted markdown text
        """
        if not package_data:
            return "Could not retrieve package information."
        
        info = package_data.get('info', {})
        
        # Basic package information
        name = info.get('name', 'Unknown')
        version = info.get('version', 'Unknown')
        summary = info.get('summary', 'No summary available')
        description = info.get('description', 'No description available')
        author = info.get('author', 'Unknown')
        author_email = info.get('author_email', 'No email available')
        home_page = info.get('home_page', '')
        project_urls = info.get('project_urls', {})
        requires_dist = info.get('requires_dist', [])
        
        # Format the markdown response
        md = f"""# {name} v{version}

## Summary
{summary}

## Basic Information
- **Author**: {author} ({author_email})
- **License**: {info.get('license', 'Not specified')}
- **Homepage**: {home_page}

## Project URLs
"""
        
        for name, url in project_urls.items():
            md += f"- **{name}**: {url}\n"
        
        md += "\n## Dependencies\n"
        
        if requires_dist:
            for dep in requires_dist:
                md += f"- {dep}\n"
        else:
            md += "No dependencies listed.\n"
        
        md += "\n## Quick Install\n```\npip install " + name + "\n```\n"
        
        # Truncate the description if it's too long
        if len(description) > 1000:
            short_desc = description[:1000] + "...\n\n(Description truncated for brevity)"
            md += f"\n## Description Preview\n{short_desc}"
        else:
            md += f"\n## Description\n{description}"
        
        return md

    @staticmethod
    async def format_documentation(doc_data):
        """Format extracted documentation content into readable markdown.
        
        Args:
            doc_data (dict): Documentation data extracted from the website
            
        Returns:
            str: Formatted markdown text
        """
        if not doc_data or 'error' in doc_data:
            return f"Error retrieving documentation: {doc_data.get('error', 'Unknown error')}"
        
        # Format the markdown response
        md = f"# {doc_data['title']}\n\n"
        
        # Add metadata if available
        if doc_data.get('metadata'):
            md += "## Page Information\n"
            for key, value in doc_data['metadata'].items():
                if value:
                    md += f"- **{key.title()}**: {value}\n"
            md += "\n"
        
        # Add table of contents if available
        if doc_data.get('toc') and len(doc_data['toc']) > 0:
            md += "## Table of Contents\n"
            for item in doc_data['toc'][:10]:  # Limit to 10 items
                md += f"- [{item['title']}]({item['url']})\n"
            if len(doc_data['toc']) > 10:
                md += f"- ... ({len(doc_data['toc']) - 10} more items)\n"
            md += "\n"
        
        # Add content
        if doc_data.get('content'):
            md += "## Content\n\n"
            # Limit content length for readability
            content = doc_data['content']
            if len(content) > 4000:
                md += content[:4000] + "\n\n... (content truncated for readability)\n"
            else:
                md += content + "\n"
        
        # Add code snippets section if available
        if doc_data.get('code_snippets') and len(doc_data['code_snippets']) > 0:
            md += "\n## Code Examples\n\n"
            for i, snippet in enumerate(doc_data['code_snippets'][:3]):  # Limit to 3 snippets
                lang = snippet['language'] or ""
                md += f"### Example {i+1}\n\n```{lang}\n{snippet['code']}\n```\n\n"
            
            if len(doc_data['code_snippets']) > 3:
                md += f"(+ {len(doc_data['code_snippets']) - 3} more code examples)\n"
        
        # Add source link
        md += f"\n[View original documentation]({doc_data['url']})\n"
        
        return md

    async def crawl_documentation_site(self, url: str) -> str:
        """Crawl a documentation website and extract formatted content.
        
        Args:
            url (str): URL of the documentation website
            
        Returns:
            str: Formatted documentation content as markdown
        """
        try:
            # Fetch the HTML content
            html = await self.fetch_url_content(url)
            
            if not html:
                return f"Failed to fetch content from {url}."
            
            # Extract documentation content
            doc_data = await self.extract_documentation_content(html, url)
            
            # Format the documentation data
            formatted_doc = await self.format_documentation(doc_data)
            
            # Save to Parquet
            doc_data['formatted_content'] = formatted_doc
            
            # Generate a filename from the URL
            filename = re.sub(r'[^\w]', '_', url.split('//')[-1])[:50]
            file_path = f"{self.data_dir}/crawls/doc_{filename}_{int(datetime.now().timestamp())}.parquet"
            
            ParquetStorage.save_to_parquet(doc_data, file_path)
            
            return formatted_doc
            
        except Exception as e:
            error_msg = f"Error crawling documentation site {url}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg