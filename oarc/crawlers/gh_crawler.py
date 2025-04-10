"""
This module enables the cloning of GitHub repositories and their storage in a structured format.
"""

import os
import re
import logging

import shutil
import tempfile
import git
import glob
from typing import Optional, Tuple
from datetime import datetime, UTC
from pathlib import Path

from .parquet_storage import ParquetStorage

import pandas as pd

class GitHubCrawler:
    """Class for crawling and extracting content from GitHub repositories."""
    
    def __init__(self, data_dir=None):
        """Initialize the GitHub Crawler.
        
        Args:
            data_dir (str, optional): Directory to store data. Defaults to DATA_DIR.
        """
        self.data_dir = data_dir
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


