"""
Export sample invoice images from the dataset as PNG files.
"""

import argparse
import os
import sys
from pathlib import Path
from src.invoice_extractor.utils.data_loader import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def export_samples(
    data_path: str, output_dir: str, start_index: int = 0, count: int = 1
):
    """Export sample images from dataset"""
    loader = DataLoader(data_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Dataset has {len(loader)} samples")
    print(f"Exporting {count} image(s) starting from index {start_index}...")

    for i in range(start_index, min(start_index + count, len(loader))):
        sample = loader[i]
        filename = output_path / f"invoice_{i:04d}.png"
        sample.image.save(filename)
        print(f"  Saved: {filename}")

    print(f"\nDone! Images saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Export invoice images from dataset")
    parser.add_argument(
        "--index", "-i", type=int, default=0, help="Start index (default: 0)"
    )
    parser.add_argument(
        "--count", "-c", type=int, default=1, help="Number of images (default: 1)"
    )
    parser.add_argument(
        "--output", "-o", default="sample_invoices", help="Output directory"
    )
    parser.add_argument(
        "--data",
        "-d",
        default="data/train-00000-of-00001-07d07b95f758bb43.parquet",
        help="Path to data file",
    )

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        return

    export_samples(args.data, args.output, args.index, args.count)


if __name__ == "__main__":
    main()
