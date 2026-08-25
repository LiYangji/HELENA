"""Experimental autodiff strategies kept separate from physics kernels."""

from .checkpoint import checkpointed_fixed_unroll
from .implicit import fixed_point_residual, implicit_adjoint_value_and_grad

__all__ = [
    "checkpointed_fixed_unroll",
    "fixed_point_residual",
    "implicit_adjoint_value_and_grad",
]
