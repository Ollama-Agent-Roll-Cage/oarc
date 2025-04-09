"""youtubeScript.py
This script downloads a YouTube video in the specified 
format and saves it to a given directory. Uses the pytube library.
Find more information at https://pytube.io/en/latest/

AUTHOR: @Borcherdingl
DATE: 4/4/2024
"""
from pytube import YouTube


def download_youtube_video(url, format, save_dir, filename):
    youtube = YouTube(url)
    
    streams = youtube.streams.filter(progressive=True,
                                     file_extension=format)
    save_path = streams.get_highest_resolution().download(output_path=save_dir,
                                              filename=filename)
    return save_path