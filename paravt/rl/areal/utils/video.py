import base64
from pathlib import Path


def video2base64(videos: list[str] | str) -> list[str]:
    if isinstance(videos, str):
        videos = [videos]

    byte_videos = []
    for video_path in videos:
        path = Path(video_path)
        with path.open("rb") as f:
            byte_video = base64.b64encode(f.read()).decode("utf-8")
            byte_videos.append(byte_video)

    return byte_videos
