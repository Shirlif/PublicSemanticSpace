"""Detection tiers: oracle boxes now, frozen detectors in W4."""

from .oracle import detect_oracle, write_detections

__all__ = ["detect_oracle", "write_detections"]
