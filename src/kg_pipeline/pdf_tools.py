from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image

from .schemas import BoundingBox


def render_pdf_to_images(pdf_path: str | Path, out_dir: str | Path, dpi: int = 350) -> list[Path]:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images: list[Path] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            path = out_dir / f"page_{page_idx:03d}.png"
            pix.save(path)
            images.append(path)
    return images


def crop_image_with_bbox(
    image_path: str | Path,
    bbox: BoundingBox,
    out_path: str | Path,
) -> Path:
    image_path = Path(image_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as img:
        width, height = img.size
        left = max(0, min(width, int(round(bbox.left * width))))
        top = max(0, min(height, int(round(bbox.top * height))))
        right = max(0, min(width, int(round(bbox.right * width))))
        bottom = max(0, min(height, int(round(bbox.bottom * height))))
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid bbox produced empty crop: {bbox.model_dump()}")
        cropped = img.crop((left, top, right, bottom))
        cropped.save(out_path)
    return out_path


def image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    with open(image_path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{suffix};base64,{payload}"


def pil_image_to_data_url(image: Image.Image, format_name: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=format_name)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{format_name.lower()};base64,{payload}"
