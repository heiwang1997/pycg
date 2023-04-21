"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import subprocess
from pathlib import Path


def compress_pdf(in_pdf: Path, out_pdf: Path, preset: str = None, dpi: int = 0, verbose: bool = False):
    """
    Borrowed from: https://tbrink.science/blog/2018/05/13/lossy-compression-for-pdfs/
    For lossless compression try: https://github.com/pts/pdfsizeopt

    :param in_pdf:
    :param out_pdf:
    :param preset:
    :param dpi:
    :return:
    """

    assert preset in ['screen', 'ebook', 'printer', 'prepress', 'default', None]

    GS_COMMAND = [
        'gs',
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.6',
        '-dNOPAUSE', '-dBATCH', '-dSAFER'
        '-dPrinted=false',
        f'-sOutputFile="{out_pdf}"',
        '-dEmbedAllFonts=true',
        '-dSubsetFonts=true'
    ]

    if not verbose:
        GS_COMMAND.append('-dQUIET')

    if preset is not None:
        GS_COMMAND.append(f'-dPDFSETTINGS=/{preset}')

    if dpi > 0:
        GS_COMMAND += [
            '-dColorImageDownsampleType=/Bicubic',
            f'-dColorImageResolution={dpi}',
            '-dGrayImageDownsampleType=/Bicubic',
            f'-dGrayImageResolution={dpi}',
            '-dMonoImageDownsampleType=/Bicubic',
            f'-dMonoImageResolution={dpi}'
        ]

    GS_COMMAND.append(str(in_pdf))

    subprocess.run(GS_COMMAND)
