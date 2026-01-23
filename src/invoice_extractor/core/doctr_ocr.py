"""
OCR using DocTR library for German invoice text extraction.
"""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from doctr.models import ocr_predictor
from PIL import Image


@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    line_idx: int = 0


class DocTROCR:
    """
    Text extraction using doctr with parseq recognition.
    """

    def __init__(
        self, det_arch="fast_base", rec_arch="parseq", use_gpu=False, cache_dir=None
    ):
        self.det_arch = det_arch
        self.rec_arch = rec_arch
        self.use_gpu = use_gpu
        self._predictor = None
        self._cache = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = self._load_model()
        return self._predictor

    def _load_model(self):
        # use default pretrained models

        predictor = ocr_predictor(
            det_arch=self.det_arch,
            reco_arch=self.rec_arch,
            pretrained=True,
            assume_straight_pages=True,
            preserve_aspect_ratio=True,
        )
        return predictor

    def _hash_image(self, image):
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()
        elif isinstance(image, bytes):
            data = image
        else:
            with open(image, "rb") as f:
                data = f.read()
        return hashlib.md5(data).hexdigest()

    def _get_cached(self, key):
        if key in self._cache:
            return self._cache[key]
        if self._cache_dir:
            path = self._cache_dir / f"{key}.json"
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                blocks = [TextBlock(**b) for b in data]
                self._cache[key] = blocks
                return blocks
        return None

    def _set_cached(self, key, blocks):
        self._cache[key] = blocks
        if self._cache_dir:
            path = self._cache_dir / f"{key}.json"
            data = [
                {
                    "text": b.text,
                    "confidence": b.confidence,
                    "bbox": b.bbox,
                    "line_idx": b.line_idx,
                }
                for b in blocks
            ]
            with open(path, "w") as f:
                json.dump(data, f)

    def extract(self, image, use_cache=True) -> List[TextBlock]:
        """extract text blocks from image"""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        if use_cache:
            key = self._hash_image(image)
            cached = self._get_cached(key)
            if cached:
                return cached

        if image.mode != "RGB":
            image = image.convert("RGB")

        result = self.predictor([np.array(image)])
        page = result.pages[0]

        blocks = []
        line_num = 0
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geo = word.geometry
                    blocks.append(
                        TextBlock(
                            text=word.value,
                            confidence=word.confidence,
                            bbox=(geo[0][0], geo[0][1], geo[1][0], geo[1][1]),
                            line_idx=line_num,
                        )
                    )
                line_num += 1

        if use_cache:
            self._set_cached(key, blocks)

        return blocks

    def extract_text(self, image, use_cache=True) -> Tuple[str, float]:
        """get full text and average confidence"""
        blocks = self.extract(image, use_cache)
        if not blocks:
            return "", 0.0

        # group words by line
        lines = {}
        for b in blocks:
            lines.setdefault(b.line_idx, []).append(b)

        text_lines = []
        for idx in sorted(lines.keys()):
            words = sorted(lines[idx], key=lambda x: x.bbox[0])
            text_lines.append(" ".join(w.text for w in words))

        text = "\n".join(text_lines)
        conf = sum(b.confidence for b in blocks) / len(blocks)
        return text, conf
