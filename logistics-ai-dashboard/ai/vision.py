"""
ai/vision.py
Vision service abstraction — pallet/damage/label inspection, warehouse
photo understanding. Declared now so agents can request the VISION
capability; returns a clear "not configured" result until a vision NIM
model and a provider image method are wired.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from ai.router import AI

_log = config.get_logger(__name__)


@dataclass
class VisionResult:
    description: str
    ok: bool
    engine: str


def describe(image_bytes: bytes, prompt: str = "Describe this logistics image.") -> VisionResult:
    if not AI.status().get("vision", False):
        return VisionResult(
            description="Vision is not configured. Wire a vision NIM model to the "
                        "VISION capability (set NVIDIA_VISION_API_KEY and a model) "
                        "to enable image understanding.",
            ok=False, engine="unconfigured")
    # A vision-capable provider would add an image method; declared for extension.
    _log.info("Vision capability configured; image method pending provider support.")
    return VisionResult(description="Vision path pending provider image support.",
                        ok=False, engine="pending")
