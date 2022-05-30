# Just a command line wrapper with some presets.
import os


def make_video(input_format: str, output_path: str, fps: int):
    os.system(f"ffmpeg -r {fps} -i {input_format} -crf 25 -pix_fmt yuv420p {output_path}")
