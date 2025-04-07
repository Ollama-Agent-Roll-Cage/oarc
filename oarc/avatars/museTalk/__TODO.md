# MuseTalk TODO

Integrate MuseTalk from documentation for a simple VTuber avatar and other video processing effects.

## Relevant Repositories

1. [MuseTalk](https://github.com/TMElyralab/MuseTalk)
2. [MusePose](https://github.com/TMElyralab/MusePose)
3. [MuseV](https://github.com/TMElyralab/MuseV)
4. [lyraDiff](https://github.com/TMElyralab/lyraDiff)
5. [MMCM](https://github.com/TMElyralab/MMCM)

## Additional Notes

- Explore other useful resources in the [MMCM repository](https://github.com/TMElyralab/MMCM), such as:
    - [Music Map Script](https://github.com/TMElyralab/MMCM/blob/main/mmcm/music/music_map/music_map.py)

- Consider integrating this repository as well:
    - [Ditto Talking Head](https://github.com/antgroup/ditto-talkinghead)

## Example Script

Below is an example script for downloading YouTube videos:

```python
import os
from pytube import YouTube

def download_youtube(url, format, save_dir, filename):
        youtube = YouTube(url)
        streams = youtube.streams.filter(progressive=True, file_extension=format)
        save_path = streams.get_highest_resolution().download(output_path=save_dir, filename=filename)
        return save_path
```
