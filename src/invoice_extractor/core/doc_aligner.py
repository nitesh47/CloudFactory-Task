"""
Document alignment using DocAligner ONNX model.
Detects document corners and applies correction.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class DocAligner:
    """
    aligns documents by detecting corners and transform
    uses fastvit onnx model for corner heatmap prediction
    """

    def __init__(self, model_path: str, input_size: int = 256):
        self._session = None
        self._model_path = model_path
        self._input_size = input_size

    @property
    def session(self):
        if self._session is None:
            self._session = ort.InferenceSession(
                self._model_path, providers=["CPUExecutionProvider"]
            )
        return self._session

    def align(
        self, image: Image.Image, confidence_threshold: float = 0.3
    ) -> Tuple[Image.Image, float]:
        """
        detect corners and align document
        """
        orig_w, orig_h = image.size

        # prep input
        img_resized = image.resize((self._input_size, self._input_size))
        img_array = np.array(img_resized).astype(np.float32) / 255.0

        # model expects CHW format
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, 0)

        # inference
        outputs = self.session.run(None, {"img": img_array})
        heatmap = outputs[0][0]  # shape: (4, 128, 128)

        # extract corners from heatmap
        corners, conf = self._extract_corners(heatmap)

        if conf < confidence_threshold:
            return image, conf

        # scale corners to original image size
        scale_x = orig_w / 128.0
        scale_y = orig_h / 128.0
        corners_scaled = corners * np.array([scale_x, scale_y])

        # apply perspective transform
        aligned = self._warp_perspective(image, corners_scaled)

        return aligned, conf

    def _extract_corners(self, heatmap: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        extract corner coordinates from heatmap
        """
        corners = []
        confidences = []

        for i in range(4):
            h = heatmap[i]
            # find max location
            max_val = h.max()
            confidences.append(max_val)

            y, x = np.unravel_index(h.argmax(), h.shape)
            corners.append([x, y])

        corners = np.array(corners, dtype=np.float32)
        avg_conf = np.mean(confidences)

        return corners, avg_conf

    def _warp_perspective(self, image: Image.Image, corners: np.ndarray) -> Image.Image:
        """apply perspective transform to rectify document"""
        img_cv = np.array(image)

        src_pts = corners.astype(np.float32)

        # compute target rectangle
        width = int(
            max(
                np.linalg.norm(src_pts[0] - src_pts[1]),
                np.linalg.norm(src_pts[3] - src_pts[2]),
            )
        )
        height = int(
            max(
                np.linalg.norm(src_pts[0] - src_pts[3]),
                np.linalg.norm(src_pts[1] - src_pts[2]),
            )
        )

        # keep reasonable bounds
        width = min(max(width, 100), 3000)
        height = min(max(height, 100), 4000)

        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        # compute and apply transform
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img_cv, matrix, (width, height))

        return Image.fromarray(warped)

    def check_rotation(self, image: Image.Image) -> int:
        """
        simple rotation check based on aspect ratio
        returns suggested rotation in degrees (0, 90, 180, 270)
        """
        w, h = image.size
        if h > w * 1.5:
            return 0
        elif w > h * 1.5:
            return 90
        return 0
