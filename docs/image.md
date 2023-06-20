# pycg.image

The `image` module provides various image operations implemented using the NumPy and Pillow libraries. These operations include cropping, alpha compositing, gamma transformation, auto cropping, placing images, creating chessboard patterns, creating solid color images, and assembling slides.

## Classes

### `ImageOperation`
- An abstract base class representing a generic image operation.

### `CropOperation`
- A class representing the crop operation, which crops an image to a specified region.

## Functions

### `ensure_float_image(img: np.ndarray) -> np.ndarray`
- Converts an image to a floating-point representation with values ranging from 0.0 to 1.0.

### `alpha_compositing(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray`
- Performs alpha compositing to blend two images using their alpha channels.

### `gamma_transform(img: np.ndarray, gamma: float = 1.0, alpha_only: bool = False) -> np.ndarray`
- Applies a gamma transform to an image.

### `auto_crop(img: np.ndarray, bound_color=None, tol: float = 2 / 255., enlarge: int = 0, return_op: bool = False) -> Union[np.ndarray, CropOperation]`
- Automatically crops an image by detecting the boundary color.

### `place_image(child_img: np.ndarray, parent_img: np.ndarray, pos_x: int, pos_y: int) -> np.ndarray`
- Places a child image over a parent image at the specified position.

### `chessboard(img_w: int, img_h: int, square_len: int = 20, seam_len: int = 0, color_a=None, color_b=None, color_seam=None) -> np.ndarray`
- Generates a chessboard pattern image.

### `solid(img_w: int, img_h: int, color=None) -> np.ndarray`
- Generates a solid color image.

### `vlayout_images(image_list: List[np.ndarray], slot_heights: List[int] = None, width: int = -1, gap: int = 0, halignment: str = 'center', background: List[Union[float, int]] = None, margin: int = 0) -> np.ndarray`
- Vertically layouts a list of images.

### `hlayout_images(image_list: List[np.ndarray], slot_widths: List[int] = None, height: int = -1, gap: int = 0, valignment: str = 'center', background: List[Union[float, int]] = None, margin: int = 0) -> np.ndarray`
- Horizontally layouts a list of images.

### `text(text, font='DejaVuSansMono.ttf', font_size=16, max_width=None) -> np.ndarray`
- Renders a text image using a specified font and font size.

### `assembled_slides(pic_matrix: List[List[Union[Path, str, np.ndarray]]], direction: str = "vertical", x_block_cm: float = 4.0, x_gap_cm: float = 0.5, y_block_cm: float = 4.0, y_gap_cm: float = 0.5) -> pptx.Presentation`
- Creates a PowerPoint presentation with a matrix of images.

### `from_mplot(fig, close: bool = False) -> np.ndarray`
- Converts a Matplotlib figure to an image.

### `color_palette(cmap_name: str) -> np.ndarray`
- Generates a color palette image based on a specified colormap name.

### `show(*imgs, reverse_rgb: bool = False, n_rows: int = 1, subfig_size: int = 3, shrink_batch_dim: bool = False)`
- Displays one or more images using Matplotlib.

### `add_alpha_color(img: np.ndarray, alpha_color, tol: float = 2 / 255.) -> np.ndarray`
- Adds alpha transparency to an image based on a specified color.

### `composite_solid(img: np.ndarray) -> np.ndarray`
- Composites an image over a solid color background.

### `read(path) -> np.ndarray`
- Reads an image from a file.

### `write(img: np.ndarray, path_or_f)`
- Writes an image to a file.

## Usage Examples

```python
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image_operations import (
    ensure_float_image, alpha_compositing, gamma_transform, auto_crop, place_image,
    chessboard, solid, vlayout_images, hlayout_images, text, assembled_slides,
    from_mplot, color_palette, show, add_alpha_color, composite_solid, read, write
)

# Example usage of ensure_float_image
img_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
img_float = ensure_float_image(img_uint8)

# Example usage of alpha_compositing
image_a = np.random.rand(100, 100, 4)
image_b = np.random.rand(100, 100, 4)
composite_image = alpha_compositing(image_a, image_b)

# Example usage of gamma_transform
img = np.random.rand(100, 100, 3)
gamma_img = gamma_transform(img, gamma=1.5)

# Example usage of auto_crop
img = np.random.rand(100, 100, 4)
cropped_img = auto_crop(img)

# Example usage of place_image
parent_img = np.random.rand(200, 200, 4)
child_img = np.random.rand(50, 50, 4)
placed_img = place_image(child_img, parent_img, pos_x=100, pos_y=100)

# Example usage of chessboard
chessboard_img = chessboard(400, 400, square_len=50, seam_len=5)

# Example usage of solid
solid_img = solid(200, 200, color=(0, 1, 0, 1))

# Example usage of vlayout_images
image_list = [np.random.rand(100, 100, 3), np.random.rand(150, 150, 3)]
layout_img = vlayout_images(image_list, slot_heights=[100, 150])

# Example usage of hlayout_images
image_list = [np.random.rand(100, 100, 3), np.random.rand(100, 200, 3)]
layout_img = hlayout_images(image_list, slot_widths=[100, 200])

# Example usage of text
text_img = text("Hello, world!")

# Example usage of assembled_slides
pic_matrix = [
    [np.random.rand(100, 100, 3), np.random.rand(150, 150, 3)],
    [np.random.rand(120, 120, 3), np.random.rand(200, 200, 3)],
]
presentation = assembled_slides(pic_matrix, direction="vertical")

# Example usage of from_mplot
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4])
image = from_mplot(fig, close=True)

# Example usage of color_palette
palette_img

 = color_palette("viridis")

# Example usage of show
img1 = np.random.rand(100, 100, 3)
img2 = np.random.rand(200, 200, 3)
show(img1, img2)

# Example usage of add_alpha_color
img = np.random.rand(100, 100, 4)
alpha_colored_img = add_alpha_color(img, alpha_color=[1, 0, 0])

# Example usage of composite_solid
img = np.random.rand(100, 100, 4)
solid_composite_img = composite_solid(img)

# Example usage of read and write
image = read("path/to/image.png")
write(image, "path/to/output.png")
```

Note: Some functions depend on additional libraries such as `pptx` and `imageio`. Ensure that these libraries are installed in your environment when using those functions.