"""
Command line interface for invoice extraction.
"""

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .core.pipeline import ExtractionPipeline


def main():
    parser = argparse.ArgumentParser(description="extract fields from invoice images")
    parser.add_argument("image", help="path to invoice image")
    parser.add_argument("--gpu", action="store_true", help="use gpu for ocr")
    parser.add_argument("--no-cache", action="store_true", help="disable ocr caching")
    parser.add_argument("--raw-text", action="store_true", help="include raw ocr text")
    parser.add_argument(
        "--simple", action="store_true", help="output only field values"
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="hitl threshold")
    parser.add_argument("--no-llm", action="store_true", help="disable llm fallback")
    parser.add_argument("--aligner", action="store_true", help="enable doc aligner")

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # config
    if args.aligner:
        config = Config.with_aligner(use_gpu=args.gpu)
    else:
        config = Config()
        config.ocr.use_gpu = args.gpu

    config.enable_ocr_cache = not args.no_cache
    config.hitl.field_confidence_threshold = args.threshold

    # run extraction
    pipeline = ExtractionPipeline(config)
    result = pipeline.process(
        args.image, use_llm_fallback=not args.no_llm, include_raw_text=args.raw_text
    )

    # output
    if args.simple:
        output = result.to_simple_dict()
    else:
        output = result.to_dict()

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
