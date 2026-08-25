"""Early process-wide JAX runtime configuration."""

from types import MappingProxyType

import jax

__all__ = ["configure"]


def configure(*, device="auto"):
    """Select the JAX device before backend initialization and require x64.

    ``device`` is ``"auto"``, ``"cpu"``, or ``"gpu"``. Explicit requests
    never fall back to another platform. The returned read-only mapping records
    the request, resolved JAX platform/device, and x64 state. This process-level
    operation changes execution only; it has no physical inputs or outputs.
    """
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")

    jax.config.update("jax_enable_x64", True)
    try:
        if device != "auto":
            jax.config.update("jax_platforms", "cpu" if device == "cpu" else "cuda")
        resolved_platform = jax.default_backend()
        resolved_devices = jax.devices()
    except RuntimeError as error:
        raise RuntimeError(
            f"HELENA could not initialize requested device {device!r}; "
            "no fallback or backend reset was attempted. Call configure before "
            "JAX backend initialization and verify the requested device is available."
        ) from error

    if device != "auto" and resolved_platform != device:
        raise RuntimeError(
            f"HELENA requested device {device!r}, but JAX had already resolved "
            f"platform {resolved_platform!r}. Call configure before any JAX backend "
            "initialization; no fallback or backend reset was attempted."
        )
    if not jax.config.jax_enable_x64:
        raise RuntimeError("HELENA scientific production requires JAX x64")

    return MappingProxyType(
        {
            "requested_device": device,
            "resolved_platform": resolved_platform,
            "resolved_device": str(resolved_devices[0]),
            "x64": True,
        }
    )
