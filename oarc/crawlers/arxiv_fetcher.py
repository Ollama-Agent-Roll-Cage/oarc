"""arxiv_fetcher.py

ArXiv Advanced Fetcher Module
This module provides functionality to fetch paper metadata and sources from arXiv,
and extract LaTeX content.

Classes:
    ArxivFetcher:
        Manages fetching paper metadata, source downloads, LaTeX extraction, and database storage.
Functions:
    main:
        Entry point for the interactive command line interface.
    A class to fetch and process papers from arXiv.

Author: @BorcherdingL
Date: 4/10/2025
"""

import re
import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from datetime import datetime, UTC
from pathlib import Path

from parquet_storage import ParquetStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivFetcher:
    """Class for searching and retrieving ArXiv papers."""
    
    def __init__(self, data_dir=None):
        """Initialize the ArXiv Fetcher.
        
        Args:
            data_dir (str, optional): Directory to store data.
        """
        self.data_dir = data_dir if data_dir else Path("./data")
        self.papers_dir = Path(f"{self.data_dir}/papers")
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def extract_arxiv_id(url_or_id):
        """Extract arXiv ID from a URL or direct ID string."""
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

    async def fetch_paper_info(self, arxiv_id):
        """Fetch paper metadata from arXiv API."""
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
                    
            # Save paper info to Parquet
            file_path = f"{self.papers_dir}/{arxiv_id}.parquet"
            ParquetStorage.save_to_parquet(paper_info, file_path)
            
            # Also append to all papers list
            all_papers_path = f"{self.papers_dir}/all_papers.parquet"
            ParquetStorage.append_to_parquet(paper_info, all_papers_path)
            
            return paper_info
            
        except urllib.error.URLError as e:
            self.logger.error(f"Failed to connect to arXiv API: {e}")
            raise ConnectionError(f"Failed to connect to arXiv API: {e}")
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse API response: {e}")
            raise ValueError(f"Failed to parse API response: {e}")

    @staticmethod
    async def format_paper_for_learning(paper_info):
        """Format paper information for learning."""
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