# pycg.pdf

The `pdf` module provides a function for compressing PDF files using Ghostscript (gs). It allows you to reduce the file size of a PDF while maintaining a reasonable quality level.

## Requirements

- Ghostscript: The script relies on the `gs` command-line tool provided by Ghostscript.

## Function

### `compress_pdf(in_pdf: Path, out_pdf: Path, preset: str = None, dpi: int = 0, verbose: bool = False)`

This function compresses a PDF file specified by the `in_pdf` path and saves the compressed PDF to the `out_pdf` path. The compression options can be adjusted using the `preset` and `dpi` parameters.

- `in_pdf`: The input PDF file to be compressed (as a `Path` object).
- `out_pdf`: The output path for the compressed PDF file (as a `Path` object).
- `preset`: The compression preset to be used. Available options are: 'screen', 'ebook', 'printer', 'prepress', 'default', or `None` (default). Refer to the Ghostscript documentation for more details on each preset.
- `dpi`: The resolution (dots per inch) for downsampling images in the PDF. A value of 0 (default) disables downsampling.
- `verbose`: If set to `True`, enables verbose output from Ghostscript. If set to `False` (default), suppresses verbose output.

## Usage Example

```python
from pathlib import Path
from pycg import pdf

# Specify the input and output paths
input_path = Path("input.pdf")
output_path = Path("output.pdf")

# Compress the PDF with default options
pdf.compress_pdf(input_path, output_path)

# Compress the PDF with a specific preset and DPI
pdf.compress_pdf(input_path, output_path, preset="screen", dpi=150)
```

## Acknowledgements

This function is based on the script provided by [tbrink.science](https://tbrink.science/blog/2018/05/13/lossy-compression-for-pdfs/) for lossy compression of PDFs using Ghostscript.
