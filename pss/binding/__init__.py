from .entity_ops import (
    build_entity,
    match_modality,
    rebuild_for_slot,
    unbind_modality,
)
from .frames import (
    FrameOp,
    Pose,
    coords_in_frame,
    relation_in_frame,
    relation_in_frame_bool,
)
from .hrr import hrr_bind, hrr_unbind, make_roles
from .operators import (
    CompositeOperator,
    LowRankOperator,
    RelationOperatorBank,
    SignedPermutationOperator,
)
from .ops import (
    SLOT_EPS,
    bind,
    bind_sum,
    cosine,
    hadamard,
    make_slots,
    slot_orthogonality,
    unbind,
)
from .protocol import Operator, OperatorChain
from .rebind import DriftMonitor, RebindPolicy, relative_error
from .relations import ROLE_NAMES, EncodedRelation, RelationEncoder
from .slots import SlotPool

__all__ = [
    "SLOT_EPS",
    "ROLE_NAMES",
    "CompositeOperator",
    "DriftMonitor",
    "EncodedRelation",
    "FrameOp",
    "LowRankOperator",
    "Operator",
    "OperatorChain",
    "Pose",
    "RebindPolicy",
    "RelationEncoder",
    "RelationOperatorBank",
    "SignedPermutationOperator",
    "SlotPool",
    "bind",
    "bind_sum",
    "build_entity",
    "coords_in_frame",
    "cosine",
    "hadamard",
    "hrr_bind",
    "hrr_unbind",
    "make_roles",
    "make_slots",
    "match_modality",
    "rebuild_for_slot",
    "relation_in_frame",
    "relation_in_frame_bool",
    "relative_error",
    "slot_orthogonality",
    "unbind",
    "unbind_modality",
]
