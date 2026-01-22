"""
Data loading
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Iterator, Any
from PIL import Image
import io
import pandas as pd

@dataclass
class InvoiceSample:
    """single invoice sample from the dataset"""
    index: int
    image: Image.Image
    ground_truth: Dict[str, str]
    image_bytes: bytes

    @property
    def fields(self) -> Dict[str, str]:
        """get the ground truth fields"""
        return self.ground_truth.get("gt_parse", {})


class DataLoader:
    """
    loads invoice data from parquet format
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self._df = None

    @property
    def df(self):
        """lazy load dataframe"""
        if self._df is None:
            self._df = pd.read_parquet(self.data_path)
        return self._df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> InvoiceSample:
        row = self.df.iloc[idx]
        return self._row_to_sample(idx, row)

    def _row_to_sample(self, idx: int, row: Any) -> InvoiceSample:
        """convert dataframe row to InvoiceSample"""
        # image data
        image_data = row["image"]
        if isinstance(image_data, dict):
            image_bytes = image_data["bytes"]
        else:
            image_bytes = image_data

        image = Image.open(io.BytesIO(image_bytes))

        # ground truth
        gt_str = row["ground_truth"]
        ground_truth = json.loads(gt_str) if isinstance(gt_str, str) else gt_str

        return InvoiceSample(
            index=idx,
            image=image,
            ground_truth=ground_truth,
            image_bytes=image_bytes
        )

    def iter_samples(self, limit: int = None) -> Iterator[InvoiceSample]:
        """iterate over samples"""
        n = len(self) if limit is None else min(limit, len(self))
        for i in range(n):
            yield self[i]

    def get_all_field_names(self) -> set:
        """get all unique field names in the dataset"""
        fields = set()
        for i in range(len(self)):
            sample = self[i]
            fields.update(sample.fields.keys())
        return fields

    def get_samples_with_field(self, field_name: str) -> List[InvoiceSample]:
        """get samples that have a specific field labeled"""
        samples = []
        for sample in self.iter_samples():
            if field_name in sample.fields:
                samples.append(sample)
        return samples

    def split(self, train_ratio: float = 0.8) -> tuple:
        """split into train/test sets"""
        n = len(self)
        train_n = int(n * train_ratio)

        train_indices = list(range(train_n))
        test_indices = list(range(train_n, n))

        return train_indices, test_indices


def find_parquet_files(directory: str) -> List[Path]:
    """find all parquet files in directory"""
    return list(Path(directory).glob("**/*.parquet"))