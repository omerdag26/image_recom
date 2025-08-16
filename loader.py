# loader.py
from pathlib import Path
from PIL import Image, ImageFile

# tolerate truncated/corrupt JPEGs a bit
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Common image extensions
DEFAULT_EXTS = {
    ".jpg", ".jpeg",
    ".png", ".webp",
    ".bmp",
    ".gif",
    ".tif", ".tiff",
    ".jp2", ".j2k",
    ".jxl",
    ".avif",
    ".heif", ".heic",
    ".ppm", ".pgm", ".pbm"
}

# Generator to yield image file paths and their dimensions
def image_generator(image_root, exts=None):
    # Allowed extensions, defaulting to common image formats
    allowed = set(e.lower() for e in (exts or DEFAULT_EXTS))
    root = Path(image_root)
    # Iterate over all files in the root
    for p in root.rglob("*"):
        # Check if the file is an image based on its extension
        if p.is_file() and p.suffix.lower() in allowed:
            # Try to open the image to get its dimensions
            # If it fails, skip the file
            try:
                # Use PIL to open the image and get its dimensions
                with Image.open(p) as im:
                    w, h = im.size
                yield str(p), w, h
            except Exception:
                continue