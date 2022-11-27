"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import glob
import os
from pathlib import Path
import tempfile
import shutil
import tqdm


def make_video(input_format: str, output_path: str, fps: int):
    os.system(f"ffmpeg -r {fps} -i {input_format} -crf 25 -pix_fmt yuv420p {output_path}")


def make_video_xw264(input_format: str, output_path: str, fps: int, crf: float = 23.5):
    """
    Call the x264 program with a best set of parameters found by Maruto-toolbox.
        Note that this does not support audio yet!
    :param input_format: could be path to mp4 file, png sequence wildcard, or its folder name
    :param output_path: output mp4 path
    :param crf: (float) the lower, the larger the file size and the better the video quality.
    :param fps: (int) frame-per-second
    """
    if isinstance(input_format, str):
        input_format = Path(input_format)

    # if input_format.is_dir():
    #     example_file = input_format.glob("*.png").__iter__().__next__()
    #     stem_wildcard = len(example_file.stem)
    #     input_format = input_format / (f"%0{stem_wildcard}d" + example_file.suffix)

    if input_format.is_dir():
        input_format = input_format / "*.png"

    input_files = glob.glob(str(input_format))
    input_named_files = []
    for i_files in input_files:
        i_files = Path(i_files)
        i_fname = i_files.stem
        i_idx = int(''.join(c for c in i_fname if c.isdigit()))
        input_named_files.append([i_files, i_idx])

    # Sort file according to extracted indices
    input_named_files = sorted(input_named_files, key=lambda t: t[1])
    input_indices = [t[1] for t in input_named_files]

    full_indices = list(range(input_indices[0], input_indices[-1] + 1))
    missing_indices = [t for t in full_indices if t not in input_indices]
    if len(missing_indices) > 0:
        print(f"Missing {len(missing_indices)} of {len(full_indices)} files.")

    # Copy to temp location.
    with tempfile.TemporaryDirectory() as video_tmp_dir:
        for idx, (fname, _) in enumerate(tqdm.tqdm(input_named_files)):
            shutil.copy(fname, Path(video_tmp_dir) / f"{idx:06d}.png")
        os.system(f"x264 --crf {crf:.1f} --fps {fps} --preset 8 -I 250 -r 4 -b 3 --me umh "
                  "-i 1 --scenecut 60 -f 1:1 --qcomp 0.5 --psy-rd 0.3:0 --aq-mode 2 --aq-strength 0.8 "
                  f"-o {output_path} {video_tmp_dir}/%06d.png")
