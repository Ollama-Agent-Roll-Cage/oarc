"""
ArXiv Advanced Fetcher Module
This module provides functionality to fetch paper metadata and sources from arXiv,
extract LaTeX content, and optionally store the results in a MongoDB database.
Classes:
    ArxivFetcher:
        Manages fetching paper metadata, source downloads, LaTeX extraction, and database storage.
Functions:
    main:
        Entry point for the interactive command line interface.
    A class to fetch and process papers from arXiv.
    This class handles:
      - Extracting the arXiv ID from a provided URL or direct identifier.
      - Retrieving paper metadata via the arXiv API.
      - Downloading and extracting the source files.
      - Extracting and concatenating LaTeX content from the downloaded sources.
      - Saving metadata and LaTeX content to a MongoDB database if configured.
    Attributes:
        download_path (Path): Directory where downloaded papers and extracted files are stored.
        mongo_client (MongoClient, optional): MongoDB client instance if provided.
        db (Database, optional): MongoDB database instance if provided.
    def __init__(self, download_path: str, mongo_uri: Optional[str] = None, db_name: Optional[str] = None):
            download_path (str): Path where papers will be downloaded.
            mongo_uri (Optional[str]): MongoDB connection URI.
            db_name (Optional[str]): MongoDB database name.
        Extract the arXiv ID from a URL or a direct ID string.
            url_or_id (str): An arXiv URL or a direct arXiv identifier.
        Returns:
            str: The extracted arXiv identifier.
        Raises:
            ValueError: If no valid arXiv ID can be extracted from the input.
        Retrieve paper metadata from the arXiv API.
            arxiv_id (str): The arXiv identifier for the paper.
        Returns:
            Dict[str, Any]: A dictionary containing paper metadata including title, authors, abstract,
                            publication date, links to the PDF and abstract, categories, and a download timestamp.
        Raises:
            ConnectionError: If there is an issue connecting to the arXiv API.
            ValueError: If the API response cannot be parsed or no paper is found.
        Download and extract source files for a given paper from arXiv.
            arxiv_id (str): The arXiv identifier for the paper.
        Returns:
            Optional[Path]: Path to the directory containing the extracted source files,
                            or None if the source could not be downloaded or extracted.
        Notes:
            The method first attempts to extract the downloaded file as a tar.gz archive.
            If that fails, it tries extraction as a gzip file.
        Extract and concatenate LaTeX content from .tex files in the provided directory.
            paper_dir (Path): The directory containing the paper's source files.
        Returns:
            Optional[str]: The combined LaTeX content, if available; otherwise, None.
        Notes:
            All .tex files found recursively within the directory are read and concatenated.
        Save paper metadata and LaTeX content to MongoDB.
            paper_info (Dict[str, Any]): The metadata of the paper.
            latex_content (Optional[str]): The extracted LaTeX content.
        Notes:
            If MongoDB is not configured, this function will skip saving.
            Paper metadata is stored in the 'papers' collection using the arXiv ID as the document ID.
            LaTeX content is stored in the 'latex' collection.
    Main entry point for the ArXiv Paper Fetcher interactive application.
    Functionality:
        - Parses command line arguments for download path and optional MongoDB configurations.
        - Initializes an instance of the ArxivFetcher.
        - Continuously prompts the user for an arXiv URL or ID.
        - For each input, it extracts the arXiv ID, fetches paper metadata,
          downloads and extracts the source files, and extracts LaTeX content.
        - Saves fetched data into MongoDB if configured.
        - Terminates upon receiving a 'quit' command.
"""
import argparse
import gzip
import logging
import re
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from pymongo import MongoClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArxivFetcher:
    def __init__(self,
                 download_path: str,
                 mongo_uri: Optional[str] = None,
                 db_name: Optional[str] = None):
        """
        Initialize the ArXiv fetcher with optional MongoDB connection.
        
        Args:
            download_path: Path where papers will be downloaded
            mongo_uri: MongoDB connection URI (optional)
            db_name: MongoDB database name (optional)
        """
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)

        # Initialize MongoDB connection if credentials provided
        self.mongo_client = None
        self.db = None
        if mongo_uri and db_name:
            try:
                self.mongo_client = MongoClient(mongo_uri)
                self.db = self.mongo_client[db_name]
                logger.info("Successfully connected to MongoDB")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise

    def extract_arxiv_id(self, url_or_id: str) -> str:
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

    def fetch_paper_info(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch paper metadata from arXiv API."""
        base_url = 'http://export.arxiv.org/api/query'
        query_params = {
            'id_list': arxiv_id,
            'max_results': 1
        }

        url = f"{base_url}?{urllib.parse.urlencode(query_params)}"

        try:
            time.sleep(3)  # Be nice to the API
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
                'download_timestamp': datetime.utcnow()
            }

            # Add optional fields if present
            optional_fields = ['comment', 'journal_ref', 'doi']
            for field in optional_fields:
                elem = entry.find(f'arxiv:{field}', namespaces)
                if elem is not None:
                    paper_info[field] = elem.text

            return paper_info

        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to arXiv API: {e}")
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse API response: {e}")

    def download_source(self, arxiv_id: str) -> Optional[Path]:
        """
        Download and extract source files for a paper.
        Returns path to extracted files or None if source not available.
        """
        # Construct source URL
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        paper_dir = self.download_path / arxiv_id
        paper_dir.mkdir(exist_ok=True)

        try:
            # Download source file
            logger.info(f"Downloading source for {arxiv_id}")
            temp_file = paper_dir / "temp_source"
            with urllib.request.urlopen(source_url) as response:
                with open(temp_file, 'wb') as f:
                    f.write(response.read())

            # Try to extract as tar.gz
            try:
                with tarfile.open(temp_file, 'r:gz') as tar:
                    tar.extractall(path=paper_dir)
                    logger.info(f"Extracted tar.gz source for {arxiv_id}")
            except tarfile.ReadError:
                # If not tar.gz, try as gzip
                try:
                    with gzip.open(temp_file, 'rb') as gz:
                        with open(paper_dir / 'main.tex', 'wb') as f:
                            f.write(gz.read())
                    logger.info(f"Extracted gzip source for {arxiv_id}")
                except Exception as e:
                    logger.error(f"Failed to extract source as gzip: {e}")
                    return None

            # Clean up temp file
            temp_file.unlink()
            return paper_dir

        except Exception as e:
            logger.error(f"Failed to download source for {arxiv_id}: {e}")
            return None

    def extract_latex(self, paper_dir: Path) -> Optional[str]:
        """
        Extract LaTeX content from source files.
        Returns concatenated LaTeX content or None if not found.
        """
        latex_content = []

        # Find all .tex files
        tex_files = list(paper_dir.glob('**/*.tex'))
        if not tex_files:
            logger.warning(f"No .tex files found in {paper_dir}")
            return None

        # Read and concatenate all .tex files
        for tex_file in tex_files:
            try:
                with open(tex_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    latex_content.append(f"% From file: {tex_file.name}\n{content}\n")
            except Exception as e:
                logger.error(f"Failed to read {tex_file}: {e}")
                continue

        return '\n'.join(latex_content) if latex_content else None

    def save_to_mongodb(self, paper_info: Dict[str, Any], latex_content: Optional[str] = None):
        """Save paper information and LaTeX content to MongoDB."""
        if not self.db:
            logger.warning("MongoDB not configured, skipping database save")
            return

        try:
            # Save paper metadata
            papers_collection = self.db.papers
            paper_info['_id'] = paper_info['arxiv_id']  # Use arxiv_id as document ID
            papers_collection.update_one(
                {'_id': paper_info['_id']},
                {'$set': paper_info},
                upsert=True
            )

            # Save LaTeX content if available
            if latex_content:
                latex_collection = self.db.latex
                latex_doc = {
                    '_id': paper_info['arxiv_id'],
                    'latex_content': latex_content,
                    'timestamp': datetime.utcnow()
                }
                latex_collection.update_one(
                    {'_id': latex_doc['_id']},
                    {'$set': latex_doc},
                    upsert=True
                )

            logger.info(f"Saved paper {paper_info['arxiv_id']} to MongoDB")

        except Exception as e:
            logger.error(f"Failed to save to MongoDB: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='ArXiv Paper Fetcher')
    parser.add_argument('--download-path', required=True, help='Path to download papers')
    parser.add_argument('--mongo-uri', help='MongoDB connection URI')
    parser.add_argument('--db-name', help='MongoDB database name')

    args = parser.parse_args()

    try:
        fetcher = ArxivFetcher(
            download_path=args.download_path,
            mongo_uri=args.mongo_uri,
            db_name=args.db_name
        )

        while True:
            url_or_id = input("\nEnter arXiv URL or ID (or 'quit' to exit): ").strip()

            if url_or_id.lower() == 'quit':
                break

            try:
                # Process paper
                arxiv_id = fetcher.extract_arxiv_id(url_or_id)
                paper_info = fetcher.fetch_paper_info(arxiv_id)

                # Download and extract source
                paper_dir = fetcher.download_source(arxiv_id)
                latex_content = None
                if paper_dir:
                    latex_content = fetcher.extract_latex(paper_dir)

                # Save to MongoDB if configured
                if args.mongo_uri and args.db_name:
                    fetcher.save_to_mongodb(paper_info, latex_content)

                # Display information
                print("\nPaper Information:")
                print(f"Title: {paper_info['title']}")
                print(f"Authors: {', '.join(paper_info['authors'])}")
                print(f"Categories: {', '.join(paper_info['categories'])}")
                print(f"Downloaded to: {paper_dir}")
                if latex_content:
                    print("LaTeX content extracted and saved to MongoDB")

            except Exception as e:
                logger.error(f"Error processing {url_or_id}: {e}")

    finally:
        if fetcher.mongo_client:
            fetcher.mongo_client.close()


if __name__ == "__main__":
    main()
