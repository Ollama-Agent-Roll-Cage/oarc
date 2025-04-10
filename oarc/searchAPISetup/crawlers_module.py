"""crawlers-module.py
This module contains classes for crawling and extracting information from different sources:
- WebCrawler: General web page crawling
- ArxivSearcher: ArXiv paper lookup and parsing
- DuckDuckGoSearcher: DuckDuckGo search API integration
- GitHubCrawler: GitHub repository cloning and extraction

Written By: @Borcherdingl
Date: 4/4/2025
"""

import os
import re
import asyncio
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import json
import shutil
import tempfile
import git
import glob
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timezone, UTC
from pathlib import Path
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup

# Configuration
DATA_DIR = os.getenv('DATA_DIR', 'data')

# Import our storage utilities
try:
    from utils import ParquetStorage
except ImportError:
    # Fallback implementation if utils is not importable
    class ParquetStorage:
        @staticmethod
        def save_to_parquet(data, file_path):
            """Save data to a Parquet file."""
            try:
                import pandas as pd
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                # Convert to DataFrame if it's a dictionary
                if isinstance(data, dict):
                    df = pd.DataFrame([data])
                elif isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = data
                    
                # Save to Parquet
                pq.write_table(pa.Table.from_pandas(df), file_path)
                logging.info(f"Data saved to {file_path}")
                return True
            except Exception as e:
                logging.error(f"Error saving to Parquet: {e}")
                return False
                
        @staticmethod
        def load_from_parquet(file_path):
            """Load data from a Parquet file."""
            try:
                import pandas as pd
                if not os.path.exists(file_path):
                    return None
                    
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                df = table.to_pandas()
                return df
            except Exception as e:
                logging.error(f"Error loading from Parquet: {e}")
                return None
                
        @staticmethod
        def append_to_parquet(data, file_path):
            """Append data to an existing Parquet file or create a new one."""
            try:
                import pandas as pd
                # Load existing data if available
                if os.path.exists(file_path):
                    existing_df = ParquetStorage.load_from_parquet(file_path)
                    if existing_df is not None:
                        # Convert new data to DataFrame
                        if isinstance(data, dict):
                            new_df = pd.DataFrame([data])
                        elif isinstance(data, list):
                            new_df = pd.DataFrame(data)
                        else:
                            new_df = data
                            
                        # Combine and save
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        return ParquetStorage.save_to_parquet(combined_df, file_path)
                
                # If file doesn't exist or loading failed, create new file
                return ParquetStorage.save_to_parquet(data, file_path)
            except Exception as e:
                logging.error(f"Error appending to Parquet: {e}")
                return False

# Initialize logging
logger = logging.getLogger(__name__)

class WebCrawler:
    """Class for crawling web pages and extracting content."""
    
    @staticmethod
    async def fetch_url_content(url):
        """Fetch content from a URL.
        
        Args:
            url (str): The URL to fetch content from
            
        Returns:
            str: HTML content of the page or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
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
                        file_path = f"{DATA_DIR}/crawls/{filename}_{int(datetime.now().timestamp())}.parquet"
                        
                        # Ensure directory exists
                        Path(f"{DATA_DIR}/crawls").mkdir(parents=True, exist_ok=True)
                        
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


class ArxivSearcher:
    """Class for searching and retrieving ArXiv papers."""
    
    @staticmethod
    def extract_arxiv_id(url_or_id):
        """Extract arXiv ID from a URL or direct ID string.
        
        Args:
            url_or_id (str): ArXiv URL or direct ID
            
        Returns:
            str: Extracted ArXiv ID
            
        Raises:
            ValueError: If ID cannot be extracted
        """
        patterns = [
            r'arxiv.org/abs/([\w.-]+)',
            r'arxiv.org/pdf/([\w.-]+)',
            r'^([\w.-]+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        raise ValueError("Could not extract arXiv ID from the provided input")

    @staticmethod
    async def fetch_paper_info(arxiv_id):
        """Fetch paper metadata from arXiv API.
        
        Args:
            arxiv_id (str): ArXiv paper ID
            
        Returns:
            dict: Paper metadata
            
        Raises:
            ValueError: If paper cannot be found
            ConnectionError: If connection to ArXiv fails
        """
        base_url = 'http://export.arxiv.org/api/query'
        query_params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(query_params)}"
        
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read().decode('utf-8')
            
            root = ET.fromstring(xml_data)
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entry = root.find('atom:entry', namespaces)
            if entry is None:
                raise ValueError("No paper found with the provided ID")
            
            paper_info = {
                'arxiv_id': arxiv_id,
                'title': entry.find('atom:title', namespaces).text.strip(),
                'authors': [author.find('atom:name', namespaces).text 
                           for author in entry.findall('atom:author', namespaces)],
                'abstract': entry.find('atom:summary', namespaces).text.strip(),
                'published': entry.find('atom:published', namespaces).text,
                'pdf_link': next(
                    link.get('href') for link in entry.findall('atom:link', namespaces)
                    if link.get('type') == 'application/pdf'
                ),
                'arxiv_url': next(
                    link.get('href') for link in entry.findall('atom:link', namespaces)
                    if link.get('rel') == 'alternate'
                ),
                'categories': [cat.get('term') for cat in entry.findall('atom:category', namespaces)],
                'timestamp': datetime.now(UTC).isoformat()
            }
            
            # Add optional fields if present
            optional_fields = ['comment', 'journal_ref', 'doi']
            for field in optional_fields:
                elem = entry.find(f'arxiv:{field}', namespaces)
                if elem is not None:
                    paper_info[field] = elem.text
                    
            # Ensure directory exists
            papers_dir = Path(f"{DATA_DIR}/papers")
            papers_dir.mkdir(parents=True, exist_ok=True)
                    
            # Save paper info to Parquet
            file_path = f"{DATA_DIR}/papers/{arxiv_id}.parquet"
            ParquetStorage.save_to_parquet(paper_info, file_path)
            
            # Also append to all papers list
            all_papers_path = f"{DATA_DIR}/papers/all_papers.parquet"
            ParquetStorage.append_to_parquet(paper_info, all_papers_path)
            
            return paper_info
            
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to arXiv API: {e}")
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse API response: {e}")

    @staticmethod
    async def format_paper_for_learning(paper_info):
        """Format paper information for learning.
        
        Args:
            paper_info (dict): Paper metadata
            
        Returns:
            str: Formatted markdown text
        """
        formatted_text = f"""# {paper_info['title']}

**Authors:** {', '.join(paper_info['authors'])}

**Published:** {paper_info['published'][:10]}

**Categories:** {', '.join(paper_info['categories'])}

## Abstract
{paper_info['abstract']}

**Links:**
- [ArXiv Page]({paper_info['arxiv_url']})
- [PDF Download]({paper_info['pdf_link']})
"""
        if 'comment' in paper_info and paper_info['comment']:
            formatted_text += f"\n**Comments:** {paper_info['comment']}\n"
            
        if 'journal_ref' in paper_info and paper_info['journal_ref']:
            formatted_text += f"\n**Journal Reference:** {paper_info['journal_ref']}\n"
            
        if 'doi' in paper_info and paper_info['doi']:
            formatted_text += f"\n**DOI:** {paper_info['doi']}\n"
            
        return formatted_text


class DuckDuckGoSearcher:
    """Class for performing searches using DuckDuckGo API."""
    
    @staticmethod
    async def text_search(search_query, max_results=5):
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
                            
                            # Ensure directory exists
                            searches_dir = Path(f"{DATA_DIR}/searches")
                            searches_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save search results to Parquet
                            search_data = {
                                'query': search_query,
                                'timestamp': datetime.now(UTC).isoformat(),
                                'raw_results': result_text
                            }
                            
                            # Generate a filename from the query
                            filename = re.sub(r'[^\w]', '_', search_query)[:50]
                            file_path = f"{DATA_DIR}/searches/{filename}_{int(datetime.now().timestamp())}.parquet"
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
            logger.error(f"DuckDuckGo search error: {e}")
            return f"An error occurred during the search: {str(e)}"


class GitHubCrawler:
    """Class for crawling and extracting content from GitHub repositories."""
    
    def __init__(self, data_dir=None):
        """Initialize the GitHub Crawler.
        
        Args:
            data_dir (str, optional): Directory to store data. Defaults to DATA_DIR.
        """
        self.data_dir = data_dir or DATA_DIR
        self.github_data_dir = Path(f"{self.data_dir}/github_repos")
        self.github_data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def extract_repo_info_from_url(url: str) -> Tuple[str, str, str]:
        """Extract repository owner and name from GitHub URL.
        
        Args:
            url (str): GitHub repository URL
            
        Returns:
            Tuple[str, str, str]: Repository owner, name, and branch (if available)
            
        Raises:
            ValueError: If URL is not a valid GitHub repository URL
        """
        # Handle different GitHub URL formats
        github_patterns = [
            r'github\.com[:/]([^/]+)/([^/]+)(?:/tree/([^/]+))?',  # Standard GitHub URL or git URL
            r'github\.com/([^/]+)/([^/\.]+)(?:\.git)?'  # GitHub URL with or without .git
        ]
        
        for pattern in github_patterns:
            match = re.search(pattern, url)
            if match:
                owner = match.group(1)
                repo_name = match.group(2)
                # Remove .git if it exists in the repo name
                repo_name = repo_name.replace('.git', '')
                
                # Extract branch if it exists (group 3)
                branch = match.group(3) if len(match.groups()) > 2 and match.group(3) else "main"
                return owner, repo_name, branch
                
        raise ValueError(f"Invalid GitHub repository URL: {url}")

    def get_repo_dir_path(self, owner: str, repo_name: str) -> Path:
        """Get the directory path for storing repository data.
        
        Args:
            owner (str): Repository owner
            repo_name (str): Repository name
            
        Returns:
            Path: Directory path
        """
        return self.github_data_dir / f"{owner}_{repo_name}"

    async def clone_repo(self, repo_url: str, temp_dir: Optional[str] = None) -> Path:
        """Clone a GitHub repository to a temporary directory.
        
        Args:
            repo_url (str): GitHub repository URL
            temp_dir (str, optional): Temporary directory path. If None, creates one.
            
        Returns:
            Path: Path to the cloned repository
            
        Raises:
            Exception: If cloning fails
        """
        try:
            owner, repo_name, branch = self.extract_repo_info_from_url(repo_url)
            
            # Create a temporary directory if not provided
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp(prefix=f"github_repo_{owner}_{repo_name}_")
            else:
                temp_dir = Path(temp_dir)
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Clone the repository
            self.logger.info(f"Cloning repository {repo_url} to {temp_dir}")
            repo = git.Repo.clone_from(repo_url, temp_dir)
            
            # Checkout the specified branch if not the default
            if branch != "main" and branch != "master":
                try:
                    repo.git.checkout(branch)
                except git.exc.GitCommandError:
                    self.logger.warning(f"Branch {branch} not found, staying on default branch")
            
            return Path(temp_dir)
            
        except Exception as e:
            self.logger.error(f"Error cloning repository {repo_url}: {e}")
            raise

    def is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if file is binary, False otherwise
        """
        # File extensions to exclude
        binary_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.exe', '.dll', '.so', '.dylib',
            '.pyc', '.pyd', '.pyo',
            '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.pdf',
            '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv',
            '.ttf', '.otf', '.woff', '.woff2'
        }
        
        _, ext = os.path.splitext(file_path.lower())
        if ext in binary_extensions:
            return True
            
        # Check file contents
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\0' in chunk  # Binary files typically contain null bytes
        except Exception:
            return True  # If we can't read it, treat as binary

    async def process_repo_to_dataframe(self, repo_path: Path, max_file_size_kb: int = 500) -> pd.DataFrame:
        """Process repository files and convert to DataFrame.
        
        Args:
            repo_path (Path): Path to cloned repository
            max_file_size_kb (int): Maximum file size in KB to process
            
        Returns:
            pd.DataFrame: DataFrame containing file information
        """
        data = []
        max_file_size = max_file_size_kb * 1024  # Convert to bytes
        
        # Get git repository object for metadata
        try:
            repo = git.Repo(repo_path)
        except git.exc.InvalidGitRepositoryError:
            # If it's not a valid git repo (shouldn't happen with clone but just in case)
            repo = None
        
        # Process each file
        for file_path in glob.glob(str(repo_path / '**' / '*'), recursive=True):
            file_path = Path(file_path)
            
            # Skip directories
            if file_path.is_dir():
                continue
                
            # Skip binary files and check file size
            if self.is_binary_file(str(file_path)) or file_path.stat().st_size > max_file_size:
                continue
            
            try:
                # Get relative path
                rel_path = str(file_path.relative_to(repo_path))
                
                # Get file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Get file metadata
                file_ext = file_path.suffix
                
                # Skip .git files
                if '.git' in str(file_path):
                    continue
                
                # Get language from extension
                language = self.get_language_from_extension(file_ext)
                
                # Get file metadata using git if available
                last_modified = None
                author = None
                
                if repo:
                    try:
                        # Try to get git blame information
                        for commit, lines in repo.git.blame('--incremental', str(rel_path)).items():
                            author = lines.split('author ')[1].split('\n')[0]
                            last_modified = lines.split('author-time ')[1].split('\n')[0]
                            break  # Just get the first author
                    except git.exc.GitCommandError:
                        # If blame fails, use file modification time
                        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                else:
                    last_modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                
                # Add to data
                data.append({
                    'file_path': rel_path,
                    'content': content,
                    'language': language,
                    'extension': file_ext,
                    'size_bytes': file_path.stat().st_size,
                    'last_modified': last_modified,
                    'author': author,
                    'line_count': len(content.splitlines()),
                    'timestamp': datetime.now(UTC).isoformat()
                })
                
            except Exception as e:
                self.logger.warning(f"Error processing file {file_path}: {e}")
                continue
        
        return pd.DataFrame(data)

    @staticmethod
    def get_language_from_extension(extension: str) -> str:
        """Get programming language name from file extension.
        
        Args:
            extension (str): File extension with leading dot
            
        Returns:
            str: Language name or 'Unknown'
        """
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React TypeScript',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.cs': 'C#',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.rs': 'Rust',
            '.sh': 'Shell',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.xml': 'XML',
            '.sql': 'SQL',
            '.r': 'R',
            '.m': 'Objective-C',
            '.dart': 'Dart',
            '.lua': 'Lua',
            '.pl': 'Perl',
            '.toml': 'TOML',
            '.ipynb': 'Jupyter Notebook'
        }
        
        return ext_to_lang.get(extension.lower(), 'Unknown')

    async def clone_and_store_repo(self, repo_url: str) -> str:
        """Clone a GitHub repository and store its data in Parquet format.
        
        Args:
            repo_url (str): GitHub repository URL
            
        Returns:
            str: Path to the Parquet file containing repository data
            
        Raises:
            Exception: If cloning or processing fails
        """
        try:
            # Extract repo information
            owner, repo_name, _ = self.extract_repo_info_from_url(repo_url)
            repo_dir = self.get_repo_dir_path(owner, repo_name)
            
            # Create a temporary directory for cloning
            temp_dir = tempfile.mkdtemp(prefix=f"github_repo_{owner}_{repo_name}_")
            
            try:
                # Clone the repository
                cloned_path = await self.clone_repo(repo_url, temp_dir)
                
                # Process repository to DataFrame
                df = await self.process_repo_to_dataframe(cloned_path)
                
                # Save to Parquet
                parquet_path = f"{self.github_data_dir}/{owner}_{repo_name}.parquet"
                ParquetStorage.save_to_parquet(df, parquet_path)
                
                self.logger.info(f"Successfully stored repository {repo_url} to {parquet_path}")
                return parquet_path
                
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            self.logger.error(f"Error cloning and storing repository {repo_url}: {e}")
            raise

    async def query_repo_content(self, repo_url: str, query: str) -> str:
        """Query repository content using natural language.
        
        Args:
            repo_url (str): GitHub repository URL
            query (str): Natural language query about the repository
            
        Returns:
            str: Query result formatted as markdown
            
        Raises:
            Exception: If querying fails
        """
        try:
            # Extract repo information
            owner, repo_name, _ = self.extract_repo_info_from_url(repo_url)
            
            # Check if repository data exists
            parquet_path = f"{self.github_data_dir}/{owner}_{repo_name}.parquet"
            
            if not os.path.exists(parquet_path):
                # Clone and store repository if not already done
                parquet_path = await self.clone_and_store_repo(repo_url)
            
            # Load repository data
            df = ParquetStorage.load_from_parquet(parquet_path)
            
            try:
                # Try to import PandasQueryEngine
                from utils import PandasQueryEngine
                
                # Execute query
                result = await PandasQueryEngine.execute_query(df, query)
                
                if result["success"]:
                    response = f"""# GitHub Repository Query Results
Repository: {owner}/{repo_name}
Query: `{query}`

{result["result"]}

"""
                    # Add summary if we have a count
                    if "count" in result:
                        response += f"Found {result['count']} matching results."
                else:
                    response = f"""# Query Error
Sorry, I couldn't process that query: {result["error"]}

Try queries like:
- "find all Python files"
- "count files by language"
- "find functions related to authentication"
- "show the largest files in the repository"
"""
                
                return response
                
            except ImportError:
                # Fallback if PandasQueryEngine is not available
                self.logger.warning("PandasQueryEngine not available, using basic filtering")
                
                # Basic filtering
                if "python" in query.lower():
                    filtered_df = df[df['language'] == 'Python']
                elif "javascript" in query.lower():
                    filtered_df = df[df['language'] == 'JavaScript']
                else:
                    # Text search in content
                    search_terms = query.lower().split()
                    filtered_df = df[df['content'].str.lower().apply(
                        lambda x: any(term in x.lower() for term in search_terms)
                    )]
                
                # Format results
                if len(filtered_df) > 0:
                    summary = f"Found {len(filtered_df)} files related to '{query}':\n\n"
                    for idx, row in filtered_df.head(10).iterrows():
                        summary += f"- {row['file_path']} ({row['language']}, {row['line_count']} lines)\n"
                    
                    if len(filtered_df) > 10:
                        summary += f"\n...and {len(filtered_df) - 10} more files."
                        
                    return summary
                else:
                    return f"No files found related to '{query}' in the repository."
                
        except Exception as e:
            self.logger.error(f"Error querying repository {repo_url}: {e}")
            raise

    async def get_repo_summary(self, repo_url: str) -> str:
        """Get a summary of the repository.
        
        Args:
            repo_url (str): GitHub repository URL
            
        Returns:
            str: Repository summary formatted as markdown
        """
        try:
            # Extract repo information
            owner, repo_name, _ = self.extract_repo_info_from_url(repo_url)
            
            # Check if repository data exists
            parquet_path = f"{self.github_data_dir}/{owner}_{repo_name}.parquet"
            
            if not os.path.exists(parquet_path):
                # Clone and store repository if not already done
                parquet_path = await self.clone_and_store_repo(repo_url)
            
            # Load repository data
            df = ParquetStorage.load_from_parquet(parquet_path)
            
            # Generate summary statistics
            total_files = len(df)
            total_lines = df['line_count'].sum()
            
            # Language distribution
            lang_counts = df['language'].value_counts().to_dict()
            
            # Format repository summary
            summary = f"""# GitHub Repository Summary: {owner}/{repo_name}

## Statistics
- **Total Files:** {total_files}
- **Total Lines of Code:** {total_lines:,}
- **Repository URL:** {repo_url}

## Language Distribution
"""
            
            for lang, count in lang_counts.items():
                percentage = (count / total_files) * 100
                summary += f"- **{lang}:** {count} files ({percentage:.1f}%)\n"
            
            # List main directories
            main_dirs = set()
            for path in df['file_path']:
                parts = path.split('/')
                if len(parts) > 1:
                    main_dirs.add(parts[0])
                    
            summary += "\n## Main Directories\n"
            for directory in sorted(main_dirs):
                summary += f"- {directory}/\n"
            
            # Include README if available
            readme_row = df[df['file_path'].str.lower().str.contains('readme.md')].head(1)
            if not readme_row.empty:
                readme_content = readme_row.iloc[0]['content']
                summary += "\n## README Preview\n"
                
                # Limit README preview to first 500 characters
                if len(readme_content) > 500:
                    summary += readme_content[:500] + "...\n"
                else:
                    summary += readme_content + "\n"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating repository summary for {repo_url}: {e}")
            return f"Error generating repository summary: {str(e)}"

    async def find_similar_code(self, repo_url: str, code_snippet: str) -> str:
        """Find similar code in the repository.
        
        Args:
            repo_url (str): GitHub repository URL
            code_snippet (str): Code snippet to find similar code for
            
        Returns:
            str: Similar code findings formatted as markdown
        """
        try:
            # Extract repo information
            owner, repo_name, _ = self.extract_repo_info_from_url(repo_url)
            
            # Check if repository data exists
            parquet_path = f"{self.github_data_dir}/{owner}_{repo_name}.parquet"
            
            if not os.path.exists(parquet_path):
                # Clone and store repository if not already done
                parquet_path = await self.clone_and_store_repo(repo_url)
            
            # Load repository data
            df = ParquetStorage.load_from_parquet(parquet_path)
            
            # Detect language from code snippet (basic detection)
            lang = "Unknown"
            if "def " in code_snippet and ":" in code_snippet:
                lang = "Python"
            elif "function" in code_snippet and "{" in code_snippet:
                lang = "JavaScript"
            elif "class" in code_snippet and "{" in code_snippet:
                lang = "Java"
            
            # Filter by language if detected
            if lang != "Unknown":
                df = df[df['language'] == lang]
            
            # Simple similarity function
            def simple_similarity(content):
                # Count how many non-trivial lines from code_snippet appear in content
                snippet_lines = set(line.strip() for line in code_snippet.splitlines() if len(line.strip()) > 10)
                if not snippet_lines:
                    return 0
                    
                content_lines = content.splitlines()
                matches = sum(1 for line in snippet_lines if any(line in c_line for c_line in content_lines))
                return matches / len(snippet_lines) if snippet_lines else 0
            
            # Calculate similarity
            df['similarity'] = df['content'].apply(simple_similarity)
            
            # Filter files with at least some similarity
            similar_files = df[df['similarity'] > 0.1].sort_values('similarity', ascending=False)
            
            if len(similar_files) == 0:
                return "No similar code found in the repository."
                
            # Format results
            results = f"""# Similar Code Findings

Found {len(similar_files)} files with potentially similar code:

"""
            for idx, row in similar_files.head(5).iterrows():
                similarity_percent = row['similarity'] * 100
                results += f"## {row['file_path']} ({similarity_percent:.1f}% similarity)\n\n"
                
                # Extract a relevant portion of the content
                content_lines = row['content'].splitlines()
                best_section = ""
                max_matches = 0
                
                for i in range(0, len(content_lines), 10):
                    section = '\n'.join(content_lines[i:i+20])
                    snippet_lines = set(line.strip() for line in code_snippet.splitlines() if len(line.strip()) > 10)
                    matches = sum(1 for line in snippet_lines if any(line in c_line for c_line in section.splitlines()))
                    
                    if matches > max_matches:
                        max_matches = matches
                        best_section = section
                
                # Display the best matching section
                if best_section:
                    results += f"```{row['language'].lower()}\n{best_section}\n```\n\n"
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding similar code in {repo_url}: {e}")
            return f"Error finding similar code: {str(e)}"
