"""Video source helpers: webcam index, local file, direct URL, or YouTube link."""
import cv2


def open_source(source):
    """Return an opened cv2.VideoCapture for `source`.

    source may be:
      - an int (webcam index, e.g. 0)
      - a local file path
      - a direct video URL (http .mp4 / stream)
      - a YouTube watch/share link (resolved to a stream URL via yt-dlp)
    """
    if isinstance(source, str):
        s = source.strip()
        if s.isdigit():
            source = int(s)
        elif _is_youtube(s):
            source = _resolve_youtube(s)
        else:
            source = s
    return cv2.VideoCapture(source)


def _is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def _resolve_youtube(url: str) -> str:
    from yt_dlp import YoutubeDL
    opts = {
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if "url" in info:
            return info["url"]
        # some videos only expose formats list
        return info["formats"][-1]["url"]
