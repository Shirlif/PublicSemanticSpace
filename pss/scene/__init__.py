"""3D scene modelling layer for PSS.

The scene model is the latent 3D state; observations are projections.  This
package is intentionally separate from ``pss.reasoning.relations`` so the 1D
predicate fallback and the 3D model can coexist.
"""

from .model import Camera, ObjectState, Observation, SceneModel

__all__ = ["Camera", "ObjectState", "Observation", "SceneModel"]
