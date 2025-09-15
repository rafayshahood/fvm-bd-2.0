import subprocess
from pathlib import Path

def convert_mov_to_mp4(input_path, output_path=None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".mp4")

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vcodec", "libx264",
        "-acodec", "aac",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Converted to: {output_path}")

# Example usage
convert_mov_to_mp4("IMG_8835.MOV")