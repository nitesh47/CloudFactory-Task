"""
Configuration for invoice extraction system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _load_dotenv():
    """Load .env file if present."""
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in possible_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
            break


_load_dotenv()


@dataclass
class OCRConfig:
    """OCR model settings - uses DocTR pretrained models by default."""

    det_arch: str = "fast_base"
    rec_arch: str = "parseq"
    use_gpu: bool = False

    def __post_init__(self):
        if os.environ.get("USE_GPU", "").lower() == "true":
            self.use_gpu = True


@dataclass
class LLMConfig:
    """llm fallback settings"""

    api_key: str = None
    model: str = "gpt-4o-mini"
    use_local: bool = False
    local_url: str = "http://localhost:11434/v1"
    local_model: str = "llama3.2"
    fallback_threshold: float = 0.70

    def __post_init__(self):
        if os.environ.get("OPENAI_API_KEY"):
            self.api_key = os.environ["OPENAI_API_KEY"]


@dataclass
class HITLConfig:
    """human review routing settings"""

    confidence_threshold: float = 0.60
    critical_fields: list = field(
        default_factory=lambda: [
            "Summe",
            "IBAN",
            "Rechnungsnummer",
            "Falligkeitsdatum",
            "Rechnungsdatum",
        ]
    )


@dataclass
class Config:
    """Main configuration."""

    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    hitl: HITLConfig = field(default_factory=HITLConfig)

    aligner_path: Optional[str] = None
    cache_dir: Optional[str] = ".cache/ocr"
    enable_cache: bool = True

    def __post_init__(self):
        # Auto-detect aligner path from environment
        if self.aligner_path is None:
            self.aligner_path = os.environ.get("ALIGNER_PATH")

    @classmethod
    def default(cls) -> "Config":
        return cls()
