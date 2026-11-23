"""Keep curated fixtures separate from code and ignored benchmark downloads."""

from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
