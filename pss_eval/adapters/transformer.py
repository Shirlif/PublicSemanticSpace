"""A simple transformer-state baseline for ``pss_eval``.

This is an external-style system: its only persistent state is a causal
self-attention KV cache and a running hidden vector.  It has no typed entities,
no module registry, no rollback, no audit log, and no budget controller.  It is
included to show that a plausible learned state model can run the same property
interface as the E5 flat runtime, and to pre-register which contract
properties it cannot pass.

The model is deliberately small and untrained.  No property in this package
scores accuracy; this adapter exists for protocol-level smoke tests and
contract-gap registration, not as a capability baseline.
"""

from __future__ import annotations

import numpy as np

from ..protocol import Observation, Prediction, StateSnapshot, state_digest

__all__ = ["TransformerStateAdapter"]


class TransformerStateAdapter:
    """Single-layer causal self-attention over a bounded context window."""

    name = "transformer-state"

    def __init__(
        self,
        dim: int = 32,
        model_dim: int = 16,
        n_heads: int = 2,
        context: int = 16,
        seed: int = 0,
    ) -> None:
        if model_dim <= 0 or n_heads <= 0 or model_dim % n_heads != 0:
            raise ValueError("model_dim must be positive and divisible by n_heads")
        if context <= 0:
            raise ValueError("context must be positive")
        self.dim = int(dim)
        self.model_dim = int(model_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.model_dim // self.n_heads
        self.context = int(context)
        self.reset(seed=seed)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed + 7919)
        scale = 1.0 / np.sqrt(self.dim)
        self.W_q = rng.normal(size=(self.dim, self.model_dim)) * scale
        self.W_k = rng.normal(size=(self.dim, self.model_dim)) * scale
        self.W_v = rng.normal(size=(self.dim, self.model_dim)) * scale
        self.W_o = rng.normal(size=(self.model_dim, self.model_dim)) / np.sqrt(
            self.model_dim
        )
        self._hidden = np.zeros(self.model_dim, dtype=np.float64)
        self._keys: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self._last: Observation | None = None
        self._n_ingested = 0

    # -- core protocol -----------------------------------------------------

    def ingest(self, observation: Observation) -> None:
        if observation.features.shape[0] != self.dim:
            raise ValueError(
                f"{self.name} expects dim {self.dim}, "
                f"got {observation.features.shape[0]}"
            )
        x = observation.features
        query = (x @ self.W_q).reshape(self.n_heads, self.head_dim)
        key = (x @ self.W_k).reshape(self.n_heads, self.head_dim)
        value = (x @ self.W_v).reshape(self.n_heads, self.head_dim)

        self._keys.append(key)
        self._values.append(value)
        if len(self._keys) > self.context:
            self._keys.pop(0)
            self._values.pop(0)

        keys = np.stack(self._keys)  # (T, H, Dh)
        values = np.stack(self._values)
        scores = np.einsum("hd,thd->ht", query, keys) / np.sqrt(self.head_dim)
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True)
        attended = np.einsum("ht,thd->hd", weights, values).reshape(self.model_dim)

        self._hidden = np.tanh(attended @ self.W_o + 0.5 * self._hidden)
        self._last = observation
        self._n_ingested += 1

    def step(self) -> Prediction:
        if self._last is None:
            raise RuntimeError("step() called before ingest()")
        return Prediction(
            observation_id=self._last.observation_id,
            embedding=self._hidden.copy(),
            label=None,
        )

    def state(self) -> StateSnapshot:
        empty = np.zeros((0, self.n_heads, self.head_dim), dtype=np.float64)
        return {
            "system": self.name,
            "seed": self.seed,
            "dim": self.dim,
            "model_dim": self.model_dim,
            "n_heads": self.n_heads,
            "context": self.context,
            "n_ingested": self._n_ingested,
            "context_len": len(self._keys),
            "hidden": self._hidden.copy(),
            "keys": np.stack(self._keys) if self._keys else empty,
            "values": np.stack(self._values) if self._values else empty,
            "param_digest": state_digest(
                {
                    "W_q": self.W_q,
                    "W_k": self.W_k,
                    "W_v": self.W_v,
                    "W_o": self.W_o,
                }
            ),
        }
