"""Structured synthetic observations for the L0 sandbox."""

from .synthetic import (
    DEFAULT_CATEGORIES,
    DEFAULT_MODALITIES,
    SyntheticEntity,
    SyntheticScene,
    make_episode,
    make_large_scene,
    make_scene,
)

__all__ = [
    "DEFAULT_CATEGORIES",
    "DEFAULT_MODALITIES",
    "SyntheticEntity",
    "SyntheticScene",
    "make_scene",
    "make_episode",
    "make_large_scene",
]
