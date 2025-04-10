"""YouTube Downloader Module

This module provides comprehensive functionality for downloading YouTube videos,
extracting information, and saving metadata using the pytube library.
Supports various formats, resolutions, and can extract captions/transcripts.

Author: @Borcherdingl
Date: 4/10/2025
"""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, UTC
from pathlib import Path

from pytube import YouTube, Playlist, Search, Caption
import pytube.exceptions

from .parquet_storage import ParquetStorage

class YouTubeDownloader:
    """Class for downloading and processing YouTube videos."""
    
    def __init__(self, data_dir=None):
        """Initialize the YouTube Downloader.
        
        Args:
            data_dir (str, optional): Directory to store data. Defaults to DATA_DIR.
        """
        self.data_dir = data_dir
        self.youtube_data_dir = Path(f"{self.data_dir}/youtube_data")
        self.youtube_data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    async def download_video(self, url: str, format: str = "mp4", 
                           resolution: str = "highest", output_path: Optional[str] = None,
                           filename: Optional[str] = None, extract_audio: bool = False) -> Dict:
        """Download a YouTube video with specified parameters.
        
        Args:
            url (str): YouTube video URL
            format (str): Video format (mp4, webm, etc.)
            resolution (str): Video resolution ("highest", "lowest", or specific like "720p")
            output_path (str, optional): Directory to save the video
            filename (str, optional): Custom filename for the downloaded video
            extract_audio (bool): Whether to extract audio only
            
        Returns:
            dict: Information about the downloaded video
        """
        try:
            # Create default output path if not specified
            if output_path is None:
                output_path = str(self.youtube_data_dir / "videos")
                os.makedirs(output_path, exist_ok=True)
            
            # Create YouTube object
            youtube = YouTube(url)
            video_info = self._extract_video_info(youtube)
            
            # Get appropriate stream based on parameters
            if extract_audio:
                stream = youtube.streams.filter(only_audio=True).first()
                file_path = stream.download(output_path=output_path, filename=filename)
                
                # Convert to mp3 if requested
                if format.lower() == "mp3":
                    from moviepy.editor import AudioFileClip
                    mp3_path = os.path.splitext(file_path)[0] + ".mp3"
                    audio_clip = AudioFileClip(file_path)
                    audio_clip.write_audiofile(mp3_path)
                    audio_clip.close()
                    os.remove(file_path)  # Remove the original file
                    file_path = mp3_path
            else:
                if resolution == "highest":
                    if format.lower() == "mp4":
                        stream = youtube.streams.filter(
                            progressive=True, file_extension=format).order_by('resolution').desc().first()
                    else:
                        stream = youtube.streams.filter(
                            file_extension=format).order_by('resolution').desc().first()
                elif resolution == "lowest":
                    stream = youtube.streams.filter(
                        file_extension=format).order_by('resolution').asc().first()
                else:
                    # Try to get the specific resolution
                    stream = youtube.streams.filter(
                        res=resolution, file_extension=format).first()
                    
                    # Fall back to highest if specified resolution not available
                    if not stream:
                        self.logger.warning(f"Resolution {resolution} not available, using highest available")
                        stream = youtube.streams.filter(
                            file_extension=format).order_by('resolution').desc().first()
                
                file_path = stream.download(output_path=output_path, filename=filename)
            
            # Update video info with downloaded file info
            video_info.update({
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'format': os.path.splitext(file_path)[1][1:],
                'download_time': datetime.now(UTC).isoformat()
            })
            
            # Save metadata to Parquet
            metadata_path = self._generate_metadata_path(youtube.video_id)
            ParquetStorage.save_to_parquet(video_info, metadata_path)
            
            return video_info
            
        except pytube.exceptions.PytubeError as e:
            error_msg = f"YouTube download error: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'url': url}
        except Exception as e:
            error_msg = f"Unexpected error downloading YouTube video: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'url': url}

    def _extract_video_info(self, youtube: YouTube) -> Dict:
        """Extract metadata information from a YouTube object.
        
        Args:
            youtube (YouTube): pytube YouTube object
            
        Returns:
            dict: Video metadata
        """
        try:
            video_info = {
                'title': youtube.title,
                'video_id': youtube.video_id,
                'url': f"https://www.youtube.com/watch?v={youtube.video_id}",
                'author': youtube.author,
                'channel_url': youtube.channel_url,
                'description': youtube.description,
                'length': youtube.length,
                'publish_date': youtube.publish_date.isoformat() if youtube.publish_date else None,
                'views': youtube.views,
                'rating': youtube.rating,
                'thumbnail_url': youtube.thumbnail_url,
                'keywords': youtube.keywords,
                'timestamp': datetime.now(UTC).isoformat()
            }
            return video_info
        except Exception as e:
            self.logger.error(f"Error extracting video info: {e}")
            return {
                'video_id': youtube.video_id,
                'url': f"https://www.youtube.com/watch?v={youtube.video_id}",
                'error': f"Error extracting metadata: {str(e)}"
            }

    def _generate_metadata_path(self, video_id: str) -> str:
        """Generate a file path for storing video metadata.
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            str: Path to metadata file
        """
        metadata_dir = self.youtube_data_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        return str(metadata_dir / f"{video_id}.parquet")

    async def download_playlist(self, playlist_url: str, format: str = "mp4", 
                              max_videos: int = 10, output_path: Optional[str] = None) -> Dict:
        """Download videos from a YouTube playlist.
        
        Args:
            playlist_url (str): YouTube playlist URL
            format (str): Video format (mp4, webm, etc.)
            max_videos (int): Maximum number of videos to download
            output_path (str, optional): Directory to save the videos
            
        Returns:
            dict: Information about the downloaded playlist
        """
        try:
            if output_path is None:
                output_path = str(self.youtube_data_dir / "playlists")
                os.makedirs(output_path, exist_ok=True)
            
            playlist = Playlist(playlist_url)
            
            # Extract playlist information
            playlist_info = {
                'title': playlist.title,
                'playlist_id': playlist.playlist_id,
                'url': playlist.playlist_url,
                'owner': playlist.owner,
                'total_videos': len(playlist.video_urls),
                'videos_to_download': min(max_videos, len(playlist.video_urls)),
                'videos': []
            }
            
            # Create subfolder for this playlist
            safe_title = re.sub(r'[^\w\s-]', '', playlist.title).strip().replace(' ', '_')
            playlist_dir = os.path.join(output_path, f"{safe_title}_{playlist.playlist_id}")
            os.makedirs(playlist_dir, exist_ok=True)
            
            # Download each video
            for i, video_url in enumerate(playlist.video_urls):
                if i >= max_videos:
                    break
                    
                self.logger.info(f"Downloading video {i+1}/{min(max_videos, len(playlist.video_urls))}")
                video_info = await self.download_video(video_url, format=format, output_path=playlist_dir)
                playlist_info['videos'].append(video_info)
            
            # Save playlist metadata
            metadata_path = os.path.join(playlist_dir, "playlist_metadata.parquet")
            ParquetStorage.save_to_parquet(playlist_info, metadata_path)
            
            return playlist_info
            
        except Exception as e:
            error_msg = f"Error downloading playlist: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'url': playlist_url}

    async def extract_captions(self, url: str, languages: List[str] = ['en']) -> Dict:
        """Extract captions/subtitles from a YouTube video.
        
        Args:
            url (str): YouTube video URL
            languages (list): List of language codes to extract 
                             (e.g., ['en', 'es', 'fr'])
            
        Returns:
            dict: Captions data
        """
        try:
            youtube = YouTube(url)
            video_info = self._extract_video_info(youtube)
            
            captions_data = {
                'video_id': youtube.video_id,
                'title': youtube.title,
                'url': url,
                'captions': {}
            }
            
            # Get available caption tracks
            caption_tracks = youtube.captions
            
            for lang in languages:
                # Find captions for requested language
                found = False
                for caption in caption_tracks.all():
                    if lang in caption.code:
                        try:
                            # Extract caption data
                            caption_content = caption.generate_srt_captions()
                            captions_data['captions'][caption.code] = caption_content
                            found = True
                            break
                        except Exception as e:
                            self.logger.warning(f"Error extracting {lang} captions: {e}")
                
                if not found and lang == 'en' and caption_tracks.all():
                    # If English not found but other captions exist, use the first available
                    try:
                        caption = caption_tracks.all()[0]
                        captions_data['captions'][caption.code] = caption.generate_srt_captions()
                        self.logger.info(f"Used {caption.code} captions as fallback for English")
                    except Exception as e:
                        self.logger.warning(f"Error extracting fallback captions: {e}")
            
            # Save captions to file
            captions_dir = self.youtube_data_dir / "captions"
            captions_dir.mkdir(exist_ok=True)
            
            for lang_code, content in captions_data['captions'].items():
                caption_file = captions_dir / f"{youtube.video_id}_{lang_code}.srt"
                with open(caption_file, "w", encoding="utf-8") as f:
                    f.write(content)
                captions_data['captions'][lang_code] = str(caption_file)
            
            # Save metadata
            metadata_path = captions_dir / f"{youtube.video_id}_caption_metadata.parquet"
            ParquetStorage.save_to_parquet(captions_data, str(metadata_path))
            
            return captions_data
            
        except Exception as e:
            error_msg = f"Error extracting captions: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'url': url}

    async def search_videos(self, query: str, limit: int = 10) -> Dict:
        """Search for YouTube videos using a query.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            dict: Search results
        """
        try:
            search_results = Search(query)
            videos = []
            
            for i, video in enumerate(search_results.results):
                if i >= limit:
                    break
                
                try:
                    video_info = {
                        'title': video.title,
                        'video_id': video.video_id,
                        'url': f"https://www.youtube.com/watch?v={video.video_id}",
                        'thumbnail_url': video.thumbnail_url,
                        'author': video.author,
                        'publish_date': video.publish_date.isoformat() if video.publish_date else None,
                        'description': video.description,
                        'length': video.length,
                        'views': video.views
                    }
                    videos.append(video_info)
                except Exception as e:
                    self.logger.warning(f"Error extracting info for video result: {e}")
            
            # Save search results
            search_data = {
                'query': query,
                'timestamp': datetime.now(UTC).isoformat(),
                'result_count': len(videos),
                'results': videos
            }
            
            # Save to Parquet
            search_dir = self.youtube_data_dir / "searches"
            search_dir.mkdir(exist_ok=True)
            safe_query = re.sub(r'[^\w\s-]', '', query).strip().replace(' ', '_')
            metadata_path = search_dir / f"{safe_query}_{int(datetime.now().timestamp())}.parquet"
            ParquetStorage.save_to_parquet(search_data, str(metadata_path))
            
            return search_data
            
        except Exception as e:
            error_msg = f"Error searching videos: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg, 'query': query}

    async def download_youtube_video(url, format="mp4", save_dir=None, filename=None):
        """Legacy function for backward compatibility.
        
        Args:
            url (str): YouTube video URL
            format (str): Video format
            save_dir (str): Directory to save the video
            filename (str): Custom filename
            
        Returns:
            str: Path to the downloaded file
        """
        downloader = YouTubeDownloader()
        result = await downloader.download_video(url, format=format, output_path=save_dir, filename=filename)
        return result.get('file_path')