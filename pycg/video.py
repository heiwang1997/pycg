# Just a command line wrapper with some presets.
import os
from pathlib import Path


def make_video(input_format: str, output_path: str, fps: int):
    os.system(f"ffmpeg -r {fps} -i {input_format} -crf 25 -pix_fmt yuv420p {output_path}")


def make_video_xw264(input_format: str, output_path: str, crf: float = 23.5):
    """
    Call the x264 program with a best set of parameters found by Maruto-toolbox.
        Note that this does not support audio yet!
    :param input_format: could be path to mp4 file, png sequence wildcard, or its folder name
    :param output_path: output mp4 path
    :param crf: (float) the lower, the larger the file size and the better the video quality.
    """
    if isinstance(input_format, str):
        input_format = Path(input_format)

    if input_format.is_dir():
        example_file = input_format.glob("*.png").__iter__().__next__()
        stem_wildcard = len(example_file.stem)
        input_format = input_format / (f"%0{stem_wildcard}d" + example_file.suffix)

    os.system(f"x264 --crf {crf:.1f} --preset 8 -I 250 -r 4 -b 3 --me umh "
              "-i 1 --scenecut 60 -f 1:1 --qcomp 0.5 --psy-rd 0.3:0 --aq-mode 2 --aq-strength 0.8 "
              f"-o {output_path} "
              f"{input_format}")
