"""Focused persistent Bethe--Heitler response-cache validation."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import helena.core.jax_backend.kernels.bethe_heitler as bh_module
import helena.runtime_cache as cache_module
from helena.core.jax_backend.kernels.bethe_heitler import (
    bethe_heitler_pair_injection_jax,
    prepare_bethe_heitler_pair_response_jax,
)
from helena.runtime_cache import (
    BetheHeitlerResponseCacheInvalid,
    BetheHeitlerResponseCacheMiss,
    bethe_heitler_response_fingerprint,
    load_or_prepare_bethe_heitler_pair_response_jax,
)

jax.config.update("jax_enable_x64", True)


def _grids():
    return (
        np.geomspace(1.1, 1.0e7, 5),
        np.geomspace(1.1, 1.0e8, 4),
        np.geomspace(1.0e-10, 1.0e-5, 3),
    )


def _paths(cache_dir, grids):
    fingerprint = bethe_heitler_response_fingerprint(*grids)
    stem = f"bh_pair_response_{fingerprint}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def test_persistent_cache_miss_build_save_and_hit(tmp_path, monkeypatch):
    grids = _grids()
    with pytest.raises(BetheHeitlerResponseCacheMiss, match="cache miss"):
        load_or_prepare_bethe_heitler_pair_response_jax(
            *grids, cache_dir=tmp_path, allow_build=False
        )

    built = load_or_prepare_bethe_heitler_pair_response_jax(
        *grids, cache_dir=tmp_path, allow_build=True
    )
    tensor_path, metadata_path = _paths(tmp_path, grids)
    assert tensor_path.is_file() and metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["compatibility"]["cache_schema_version"] == 1
    assert metadata["compatibility"]["physics"]["response_version"] == (
        bh_module.BH_PAIR_RESPONSE_VERSION
    )
    np.testing.assert_array_equal(np.load(tensor_path), np.asarray(built))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("persistent cache hit rebuilt the BH response")

    monkeypatch.setattr(bh_module, "prepare_bethe_heitler_pair_response_jax", forbidden)
    monkeypatch.setenv("HELENA_CACHE_DIR", str(tmp_path))
    loaded = load_or_prepare_bethe_heitler_pair_response_jax(*grids, allow_build=True)
    np.testing.assert_array_equal(loaded, built)


@pytest.mark.parametrize("grid_index", range(3), ids=("daughter", "proton", "target"))
def test_persistent_cache_grid_identity_is_exact(tmp_path, grid_index):
    grids = _grids()
    load_or_prepare_bethe_heitler_pair_response_jax(
        *grids, cache_dir=tmp_path, allow_build=True
    )
    changed = list(grids)
    changed[grid_index] = changed[grid_index].copy()
    changed[grid_index][1] = np.nextafter(changed[grid_index][1], np.inf)
    with pytest.raises(BetheHeitlerResponseCacheMiss):
        load_or_prepare_bethe_heitler_pair_response_jax(
            *changed, cache_dir=tmp_path, allow_build=False
        )
    with pytest.raises(ValueError, match="float64"):
        load_or_prepare_bethe_heitler_pair_response_jax(
            grids[0].astype(np.float32),
            grids[1],
            grids[2],
            cache_dir=tmp_path,
            allow_build=False,
        )


@pytest.mark.parametrize(
    "damage,exception",
    (
        ("version", BetheHeitlerResponseCacheInvalid),
        ("schema", BetheHeitlerResponseCacheInvalid),
        ("quadrature", BetheHeitlerResponseCacheInvalid),
        ("metadata", BetheHeitlerResponseCacheInvalid),
        ("tensor", BetheHeitlerResponseCacheInvalid),
        ("dtype", BetheHeitlerResponseCacheInvalid),
        ("shape", BetheHeitlerResponseCacheInvalid),
        ("incomplete", BetheHeitlerResponseCacheMiss),
    ),
)
def test_persistent_cache_rejects_invalid_or_incomplete_entries(
    tmp_path, damage, exception
):
    grids = _grids()
    response = load_or_prepare_bethe_heitler_pair_response_jax(
        *grids, cache_dir=tmp_path, allow_build=True
    )
    tensor_path, metadata_path = _paths(tmp_path, grids)
    if damage == "version":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["compatibility"]["physics"]["response_version"] = "stale"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "schema":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["compatibility"]["cache_schema_version"] = 0
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "quadrature":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["compatibility"]["physics"]["quadrature"]["omega_order"] = 16
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "metadata":
        metadata_path.write_text("{", encoding="utf-8")
    elif damage == "tensor":
        tensor_path.write_bytes(b"corrupt")
    elif damage == "dtype":
        np.save(tensor_path, np.asarray(response, dtype=np.float32))
    elif damage == "shape":
        np.save(tensor_path, np.asarray(response)[:-1])
    else:
        metadata_path.unlink()
    with pytest.raises(exception):
        load_or_prepare_bethe_heitler_pair_response_jax(
            *grids, cache_dir=tmp_path, allow_build=False
        )


def test_loaded_response_preserves_pair_injection_and_autodiff(tmp_path, monkeypatch):
    grids = _grids()
    fresh = prepare_bethe_heitler_pair_response_jax(*grids)
    loaded = load_or_prepare_bethe_heitler_pair_response_jax(
        *grids, cache_dir=tmp_path, allow_build=True
    )
    np.testing.assert_array_equal(loaded, fresh)
    proton_density = jnp.geomspace(1.0e-12, 4.0e-9, grids[1].size)
    photon_density = jnp.geomspace(2.0e12, 7.0e15, grids[2].size)

    def injection(response, scales):
        return bethe_heitler_pair_injection_jax(
            grids[0],
            grids[1],
            jnp.exp(scales[0]) * proton_density,
            grids[2],
            jnp.exp(scales[1]) * photon_density,
            pair_response=response,
        )

    parameters = jnp.zeros(2, dtype=jnp.float64)
    fresh_pair = injection(fresh, parameters)
    loaded_pair = injection(loaded, parameters)
    for actual, expected in zip(loaded_pair, fresh_pair, strict=True):
        np.testing.assert_array_equal(actual, expected)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("filesystem, builder, or direct BH fallback entered trace")

    monkeypatch.setattr(bh_module, "_pair_injection_kernel", forbidden)
    monkeypatch.setattr(bh_module, "prepare_bethe_heitler_pair_response_jax", forbidden)
    monkeypatch.setattr(cache_module.np, "load", forbidden)

    def objective(response, scales):
        return sum(jnp.sum(value) for value in injection(response, scales))

    fresh_gradient = jax.grad(lambda value: objective(fresh, value))(parameters)
    loaded_gradient = jax.jit(jax.grad(lambda value: objective(loaded, value)))(
        parameters
    )
    direction = jnp.asarray([0.4, -0.3], dtype=jnp.float64)
    _, loaded_jvp = jax.jvp(
        lambda value: objective(loaded, value),
        (parameters,),
        (direction,),
    )
    expected_jvp = jnp.vdot(loaded_gradient, direction)
    assert np.all(np.isfinite(np.asarray(loaded_gradient)))
    assert bool(jnp.isfinite(loaded_jvp))
    np.testing.assert_allclose(loaded_gradient, fresh_gradient, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(loaded_jvp, expected_jvp, rtol=2.0e-15, atol=0.0)
