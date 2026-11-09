"""Strict host-side persistent BH and gamma-gamma fixed-grid responses."""

import json
import os
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import jax
import jax.numpy as jnp
import numpy as np

import helena.core.jax_backend.kernels.bethe_heitler as bh_module
from helena._cache_manifest_v0_1_1 import OFFICIAL_RESPONSE_CACHE_RESOURCE

__all__ = [
    "BetheHeitlerResponseCacheInvalid",
    "BetheHeitlerResponseCacheMiss",
    "GammaGammaResponseCacheInvalid",
    "GammaGammaResponseCacheMiss",
    "bethe_heitler_response_fingerprint",
    "fetch_current_c513_response_cache",
    "gamma_gamma_response_fingerprint",
    "load_or_prepare_bethe_heitler_pair_response_jax",
    "load_or_prepare_gamma_gamma_response_jax",
    "prepare_current_c513_bethe_heitler_cache",
    "prepare_current_c513_response_cache",
]

_CACHE_SCHEMA_VERSION = 1
_CACHE_ENVIRONMENT_VARIABLE = "HELENA_CACHE_DIR"

_FINGERPRINT_64 = "807a424a15042b1e20906cd3f915b0b3a568d9a7926b3796e89cbd64a9d0bc19"
_FINGERPRINT_86 = "66d59b31d7ebfe7b904a4522070d0ff9999cddd074d342969fbc985703b364ad"
_FINGERPRINT_96 = "1c0eff6880cfaf20ea38774574d1fae02661375a10591df28902505f86420503"


def _resource_filename(fingerprint, suffix):
    return f"bh_pair_response_{fingerprint}.{suffix}"


# Immutable v0.1 resource binding, retained only for explicit legacy access.
# Current C513 grids have different fingerprints; never relabel this archive.
OFFICIAL_BH_CACHE_RESOURCE = {
    "resource_name": "helena_bethe_heitler_c513_responses",
    "cache_format_version": _CACHE_SCHEMA_VERSION,
    "physics_response_version": "ka2008_born_omega_lt_600_v1",
    "grid_identifier": "c513_daughter513_proton441_targets64-86-96_v1",
    "dtype": "<f8",
    "fingerprints": (_FINGERPRINT_64, _FINGERPRINT_86, _FINGERPRINT_96),
    "archive": {
        "format": "zip",
        "url": (
            "https://github.com/LiYangji/HELENA-resources/releases/download/"
            "bh-c513-v1/helena_bethe_heitler_c513_responses_"
            "ka2008_born_omega_lt_600_v1.zip"
        ),
        "sha256": "a3abbcf77452bcc5015e848130a463e691d717954150611b47f8f19569bd88d4",
    },
    "files": {
        _resource_filename(_FINGERPRINT_64, "json"): {
            "kind": "metadata",
            "fingerprint": _FINGERPRINT_64,
            "size": 1752,
            "sha256": (
                "2f4fb76f8b8aa5cb7e365d8fee715a95b5ced95116ffaa0caca39b2d2b871c4c"
            ),
        },
        _resource_filename(_FINGERPRINT_64, "npy"): {
            "kind": "response",
            "fingerprint": _FINGERPRINT_64,
            "size": 115831424,
            "sha256": (
                "50c52b6c6d1cf792b90b5342b43252bc3976af58314047cce13b4495fa2e54b2"
            ),
            "array_sha256": (
                "322f52b4d8c17fe49ba5e79a54524aacf4280e492f475f2bc54d64fc4f005ea6"
            ),
            "shape": (441, 64, 513),
            "dtype": "<f8",
        },
        _resource_filename(_FINGERPRINT_86, "json"): {
            "kind": "metadata",
            "fingerprint": _FINGERPRINT_86,
            "size": 1752,
            "sha256": (
                "22210fbeb7649f60649fc11c641f256a76bb2b404bad86fc3de4f7807eb9b31f"
            ),
        },
        _resource_filename(_FINGERPRINT_86, "npy"): {
            "kind": "response",
            "fingerprint": _FINGERPRINT_86,
            "size": 155648432,
            "sha256": (
                "e3f5515d8e7242440144d3d6c8d62c85eac1243d9d8caab64d488af90c91696e"
            ),
            "array_sha256": (
                "a729f3a45e2203d63b016cac3f93bdc14be09d4d5af715c6739d30f37352bc14"
            ),
            "shape": (441, 86, 513),
            "dtype": "<f8",
        },
        _resource_filename(_FINGERPRINT_96, "json"): {
            "kind": "metadata",
            "fingerprint": _FINGERPRINT_96,
            "size": 1752,
            "sha256": (
                "7b7dc4233913f6b8fa15a93c1955611afaf4b45d7b2c0c011073b8a27e34a029"
            ),
        },
        _resource_filename(_FINGERPRINT_96, "npy"): {
            "kind": "response",
            "fingerprint": _FINGERPRINT_96,
            "size": 173747072,
            "sha256": (
                "fc5440c390db9b711b4be2a2ab874a09470b7b1380d7194cda397fbef9630a1d"
            ),
            "array_sha256": (
                "ab0f9c7ec74d721afb5aafb0cf218497884a0ebc018d9905f0db5166dda73200"
            ),
            "shape": (441, 96, 513),
            "dtype": "<f8",
        },
    },
}


class BetheHeitlerResponseCacheMiss(FileNotFoundError):
    """Raised when an exact persistent response is unavailable."""


class BetheHeitlerResponseCacheInvalid(ValueError):
    """Raised when a persistent response fails strict validation."""


class BetheHeitlerCacheResourceUnavailable(RuntimeError):
    """Raised when the official release asset has not been bound."""


def _array_identity(name, values):
    if any(
        isinstance(value, jax.core.Tracer)
        for value in jax.tree_util.tree_leaves(values)
    ):
        raise ValueError(f"{name} must be concrete outside JAX tracing")
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be a one-dimensional float64 array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array = np.ascontiguousarray(array)
    return array, {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": sha256(array.tobytes()).hexdigest(),
    }


def _physics_identity():
    nodes = np.ascontiguousarray(np.asarray(bh_module._GAUSS_NODE, dtype=np.float64))
    weights = np.ascontiguousarray(
        np.asarray(bh_module._GAUSS_WEIGHT, dtype=np.float64)
    )
    return {
        "provider": "kelner_aharonian_2008_blumenthal_1970_born",
        "domain": "born_omega_lt_600",
        "omega_maximum_mec2": bh_module._BETHE_HEITLER_PAIR_OMEGA_MAX_MEC2,
        "pair_threshold": (
            "proton_gamma*target_epsilon>1; omega>2; strict daughter bounds"
        ),
        "quadrature": {
            "rule": "gauss_legendre",
            "omega_order": int(nodes.size),
            "electron_energy_order": int(nodes.size),
            "nodes_sha256": sha256(nodes.tobytes()).hexdigest(),
            "weights_sha256": sha256(weights.tobytes()).hexdigest(),
        },
        "response_layout": "proton,target_photon,daughter",
        "response_normalization": "cm3_erg_s_per_daughter_gamma_with_log_grid_weights",
        "response_version": bh_module.BH_PAIR_RESPONSE_VERSION,
        "float_precision": "float64",
    }


def _compatibility(daughter_grid, proton_grid, target_grid):
    daughter, daughter_identity = _array_identity("daughter_grid", daughter_grid)
    proton, proton_identity = _array_identity("proton_grid", proton_grid)
    target, target_identity = _array_identity("target_grid", target_grid)
    if np.any(daughter <= 1.0):
        raise ValueError("daughter_grid must contain only values > 1")
    if proton.size < 2 or np.any(proton <= 1.0) or np.any(np.diff(proton) <= 0.0):
        raise ValueError("proton_grid must be strictly increasing with values > 1")
    if target.size < 2 or np.any(target <= 0.0) or np.any(np.diff(target) <= 0.0):
        raise ValueError("target_grid must be strictly increasing and positive")
    compatibility = {
        "cache_schema_version": _CACHE_SCHEMA_VERSION,
        "grids": {
            "daughter": daughter_identity,
            "proton": proton_identity,
            "target_photon_energy_erg": target_identity,
        },
        "physics": _physics_identity(),
    }
    encoded = json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return (daughter, proton, target), compatibility, sha256(encoded).hexdigest()


def bethe_heitler_response_fingerprint(daughter_grid, proton_grid, target_grid):
    """Return the strict SHA-256 compatibility fingerprint for three grids."""
    return _compatibility(daughter_grid, proton_grid, target_grid)[2]


def _path(value, name):
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{name} must not be empty")
    try:
        return Path(value).expanduser()
    except TypeError as error:
        raise TypeError(f"{name} must be path-like") from error


def _platform_cache_root():
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "helena"
    if sys.platform.startswith("win"):
        local = os.environ.get("LOCALAPPDATA")
        base = (
            _path(local, "LOCALAPPDATA") if local else Path.home() / "AppData" / "Local"
        )
        return base / "helena" / "Cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = _path(xdg, "XDG_CACHE_HOME") if xdg else Path.home() / ".cache"
    if not base.is_absolute():
        base = Path.home() / ".cache"
    return base / "helena"


def bethe_heitler_cache_directory(cache_dir=None):
    """Resolve explicit, environment, then platform-default BH cache storage."""
    if cache_dir is not None:
        return _path(cache_dir, "cache_dir")
    configured = os.environ.get(_CACHE_ENVIRONMENT_VARIABLE)
    if configured:
        return _path(configured, _CACHE_ENVIRONMENT_VARIABLE)
    return _platform_cache_root() / "bethe_heitler"


def _directory(cache_dir):
    return bethe_heitler_cache_directory(cache_dir)


def _file_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_official_bethe_heitler_cache(cache_dir=None):
    """Validate the immutable legacy v0.1 resource, not current C513 grids."""
    directory = _directory(cache_dir)
    resource = OFFICIAL_BH_CACHE_RESOURCE
    if not directory.is_dir():
        raise BetheHeitlerResponseCacheMiss(
            f"official Bethe-Heitler cache is missing at {directory}"
        )

    for filename in resource["files"]:
        if not (directory / filename).is_file():
            raise BetheHeitlerResponseCacheMiss(
                f"official Bethe-Heitler cache is incomplete: missing {filename}"
            )

    for filename, expected in resource["files"].items():
        path = directory / filename
        if expected["kind"] == "metadata":
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise BetheHeitlerResponseCacheInvalid(
                    f"official cache metadata is unreadable: {filename}"
                ) from error
            compatibility = metadata.get("compatibility", {})
            physics = compatibility.get("physics", {})
            if metadata.get("fingerprint") != expected["fingerprint"]:
                raise BetheHeitlerResponseCacheInvalid(
                    f"official cache fingerprint mismatch: {filename}"
                )
            if (
                compatibility.get("cache_schema_version")
                != resource["cache_format_version"]
            ):
                raise BetheHeitlerResponseCacheInvalid(
                    f"official cache format version mismatch: {filename}"
                )
            if physics.get("response_version") != resource["physics_response_version"]:
                raise BetheHeitlerResponseCacheInvalid(
                    f"official cache physics-response version mismatch: {filename}"
                )
            response_expected = resource["files"][
                _resource_filename(expected["fingerprint"], "npy")
            ]
            response = metadata.get("response", {})
            if (
                response.get("dtype") != response_expected["dtype"]
                or tuple(response.get("shape", ())) != response_expected["shape"]
                or response.get("sha256") != response_expected["array_sha256"]
            ):
                raise BetheHeitlerResponseCacheInvalid(
                    f"official cache response identity mismatch: {filename}"
                )
        elif expected["kind"] == "response":
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise BetheHeitlerResponseCacheInvalid(
                    f"official cache response is unreadable: {filename}"
                ) from error
            if array.dtype.str != expected["dtype"] or array.shape != expected["shape"]:
                raise BetheHeitlerResponseCacheInvalid(
                    f"official cache response shape or dtype mismatch: {filename}"
                )

    for filename, expected in resource["files"].items():
        path = directory / filename
        if path.stat().st_size != expected["size"]:
            raise BetheHeitlerResponseCacheInvalid(
                f"official cache file size mismatch: {filename}"
            )
        if _file_sha256(path) != expected["sha256"]:
            raise BetheHeitlerResponseCacheInvalid(
                f"official cache SHA-256 mismatch: {filename}"
            )
    return directory


def _download_official_archive(url, expected_sha256, destination):
    digest = sha256()
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as source,
            destination.open("xb") as target,
        ):
            while block := source.read(1024 * 1024):
                target.write(block)
                digest.update(block)
            target.flush()
            os.fsync(target.fileno())
    except Exception as error:
        raise RuntimeError(
            f"official Bethe-Heitler cache download failed: {error}"
        ) from error
    if digest.hexdigest() != expected_sha256:
        raise BetheHeitlerResponseCacheInvalid(
            "official Bethe-Heitler cache archive SHA-256 mismatch"
        )


def fetch_official_bethe_heitler_cache(cache_dir=None):
    """Fetch the immutable legacy v0.1 BH resource with strict SHA checks."""
    directory = _directory(cache_dir)
    if directory.exists():
        return validate_official_bethe_heitler_cache(directory)

    resource = OFFICIAL_BH_CACHE_RESOURCE
    archive = resource["archive"]
    if not archive.get("url") or not archive.get("sha256"):
        raise BetheHeitlerCacheResourceUnavailable(
            "the official Bethe-Heitler cache release-asset URL and SHA-256 "
            "are not yet bound"
        )

    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        dir=directory.parent, prefix=f".{directory.name}-fetch-"
    ) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "resource.zip"
        _download_official_archive(archive["url"], archive["sha256"], archive_path)
        payload = temporary / "payload"
        payload.mkdir()
        try:
            with ZipFile(archive_path) as bundle:
                names = [info.filename for info in bundle.infolist()]
                if len(names) != len(resource["files"]) or set(names) != set(
                    resource["files"]
                ):
                    raise BetheHeitlerResponseCacheInvalid(
                        "official cache archive file manifest mismatch"
                    )
                for filename in resource["files"]:
                    with (
                        bundle.open(filename) as source,
                        (payload / filename).open("xb") as target,
                    ):
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                        target.flush()
                        os.fsync(target.fileno())
        except BetheHeitlerResponseCacheInvalid:
            raise
        except (BadZipFile, KeyError, OSError) as error:
            raise BetheHeitlerResponseCacheInvalid(
                f"official cache archive is invalid: {error}"
            ) from error
        validate_official_bethe_heitler_cache(payload)
        try:
            payload.rename(directory)
        except FileExistsError:
            return validate_official_bethe_heitler_cache(directory)
    return directory


def _paths(directory, fingerprint):
    stem = f"bh_pair_response_{fingerprint}"
    return directory / f"{stem}.npy", directory / f"{stem}.json"


def _response_identity(response):
    array = np.ascontiguousarray(np.asarray(response))
    return array, {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": sha256(array.tobytes()).hexdigest(),
    }


def _load(directory, fingerprint, compatibility, expected_shape):
    tensor_path, metadata_path = _paths(directory, fingerprint)
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise BetheHeitlerResponseCacheMiss(
            f"Bethe-Heitler response cache miss for {fingerprint}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            raise BetheHeitlerResponseCacheInvalid("cache fingerprint mismatch")
        if metadata.get("compatibility") != compatibility:
            raise BetheHeitlerResponseCacheInvalid("cache compatibility mismatch")
        response = np.load(tensor_path, allow_pickle=False)
        response, identity = _response_identity(response)
        if metadata.get("response") != identity:
            raise BetheHeitlerResponseCacheInvalid("cached response identity mismatch")
        if response.shape != expected_shape or response.dtype != np.dtype(np.float64):
            raise BetheHeitlerResponseCacheInvalid(
                "cached response shape or dtype mismatch"
            )
        if not np.all(np.isfinite(response)) or np.any(response < 0.0):
            raise BetheHeitlerResponseCacheInvalid(
                "cached response must be finite and non-negative"
            )
    except BetheHeitlerResponseCacheInvalid:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise BetheHeitlerResponseCacheInvalid(
            f"cached response is unreadable: {error}"
        ) from error
    return jnp.asarray(response)


def _atomic_save(directory, fingerprint, compatibility, response):
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = _paths(directory, fingerprint)
    response, identity = _response_identity(response)
    temporary_paths = []
    try:
        with NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as handle:
            temporary_tensor = Path(handle.name)
            temporary_paths.append(temporary_tensor)
            np.save(handle, response, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_tensor, tensor_path)
        temporary_paths.remove(temporary_tensor)
        metadata = {
            "fingerprint": fingerprint,
            "compatibility": compatibility,
            "response": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        with NamedTemporaryFile(
            mode="w", dir=directory, suffix=".json", encoding="utf-8", delete=False
        ) as handle:
            temporary_metadata = Path(handle.name)
            temporary_paths.append(temporary_metadata)
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_metadata, metadata_path)
        temporary_paths.remove(temporary_metadata)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def load_or_prepare_bethe_heitler_pair_response_jax(
    daughter_grid,
    proton_grid,
    target_grid,
    *,
    cache_dir=None,
    allow_build=False,
):
    """Load one exact fixed BH response, or explicitly build and persist it.

    Grids are concrete float64 blob-frame daughter/proton Lorentz factors and
    target photon energies in erg. The returned JAX float64 tensor has the
    production ``(proton, target, daughter)`` layout and units inherited from
    :func:`prepare_bethe_heitler_pair_response_jax`. Persistent I/O and
    construction are host-side only. Cache resolution uses explicit
    ``cache_dir``, then ``HELENA_CACHE_DIR``, then the platform-standard user
    cache. A miss or invalid entry raises unless ``allow_build=True``. No
    interpolation, reshaping, dtype coercion, stale reuse, automatic download,
    or implicit rebuild is performed.
    """
    if not isinstance(allow_build, (bool, np.bool_)):
        raise ValueError("allow_build must be boolean")
    grids, compatibility, fingerprint = _compatibility(
        daughter_grid, proton_grid, target_grid
    )
    directory = _directory(cache_dir)
    expected_shape = (grids[1].size, grids[2].size, grids[0].size)
    try:
        return _load(directory, fingerprint, compatibility, expected_shape)
    except (BetheHeitlerResponseCacheMiss, BetheHeitlerResponseCacheInvalid):
        if not allow_build:
            raise
    response = bh_module.prepare_bethe_heitler_pair_response_jax(*grids)
    response_array = np.asarray(response)
    if response_array.shape != expected_shape or response_array.dtype != np.float64:
        raise BetheHeitlerResponseCacheInvalid(
            "prepared response shape or dtype violates the cache contract"
        )
    _atomic_save(directory, fingerprint, compatibility, response_array)
    return _load(directory, fingerprint, compatibility, expected_shape)


def _current_c513_bh_grids():
    # Lazy import avoids a runtime_cache <-> c513 import cycle.
    from helena.core.jax_backend.chains.steady_state_hadronic_cascade import (
        _target_components,
    )
    from helena.models.c513 import build_c513_configuration

    setup = build_c513_configuration()
    configuration = setup["production_configuration"]
    components = _target_components(
        setup["initial_internal_photon_state_blob"],
        setup["fixed_external_photon_state_blob"],
    )
    return {
        name: (
            np.asarray(configuration["electron_gamma"]),
            np.asarray(configuration["proton_gamma"]),
            np.asarray(component["energy_erg"]),
        )
        for name, component in components.items()
    }


def prepare_current_c513_bethe_heitler_cache(cache_dir=None, *, allow_build=False):
    """Validate or explicitly build BH responses for the current C513 axes.

    Uses the same blob-frame float64 daughter/proton Lorentz factors and
    target energies (erg) as normal model preparation, including the external
    target. It performs no model iteration, gamma-gamma response construction,
    or network access. Exact grid, physics, dtype and SHA checks are inherited
    from the persistent loader. An explicit build can be expensive and writes
    only the selected cache directory. Returns that resolved directory.
    """
    directory = _directory(cache_dir)
    for grids in _current_c513_bh_grids().values():
        response = load_or_prepare_bethe_heitler_pair_response_jax(
            *grids, cache_dir=directory, allow_build=allow_build
        )
        response.block_until_ready()
    return directory


class GammaGammaResponseCacheMiss(BetheHeitlerResponseCacheMiss):
    """An exact persistent gamma-gamma response is unavailable."""


class GammaGammaResponseCacheInvalid(BetheHeitlerResponseCacheInvalid):
    """A gamma-gamma cache failed strict grid, method or payload validation."""


def _gamma_gamma_compatibility(photon_energy_erg, lepton_gamma, target_energy_erg=None):
    from helena.core.jax_backend.chains import gamma_gamma_pair_source as total
    from helena.core.jax_backend.kernels import gamma_gamma_absorption as absorption
    from helena.core.jax_backend.kernels import gamma_gamma_pair_injection as pair

    inputs = {"photon_energy_erg": photon_energy_erg, "lepton_gamma": lepton_gamma}
    if target_energy_erg is not None:
        inputs["target_photon_energy_erg"] = target_energy_erg
    arrays, identities = {}, {}
    for name, values in inputs.items():
        a, identity = _array_identity(name, values)
        minimum = 1.0 if name == "lepton_gamma" else 0.0
        if a.size < 2 or np.any(a <= minimum) or np.any(np.diff(a) <= 0):
            raise ValueError(f"{name} must be strictly increasing above {minimum}")
        arrays[name], identities[name] = a, identity
    physics = {
        "method": "Bottcher_Schlickeiser_1997_Breit_Wheeler",
        "configuration": json.loads(json.dumps(pair._PAIR_RESPONSE_CONFIGURATION)),
        "module_sha256": {
            module.__name__: _file_sha256(Path(module.__file__))
            for module in (pair, absorption, total)
        },
        "float_precision": "float64",
        "layout": "daughter,photon,target",
        "normalization": "c_and_both_dE_weights_no_same_population_half",
        "population": (
            "internal_x_external"
            if target_energy_erg is not None
            else "internal_x_internal"
        ),
    }
    compatibility = {"cache_schema_version": 1, "grids": identities, "physics": physics}
    encoded = json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode()
    return arrays, compatibility, sha256(encoded).hexdigest()


def gamma_gamma_response_fingerprint(
    photon_energy_erg, lepton_gamma, target_energy_erg=None
):
    """Exact blob-frame erg / daughter-gamma / method identity, outside tracing."""
    return _gamma_gamma_compatibility(
        photon_energy_erg, lepton_gamma, target_energy_erg
    )[2]


def _gamma_gamma_paths(directory, fingerprint):
    stem = f"gg_pair_response_{fingerprint}"
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _validate_gamma_gamma_payload(response, axes):
    external = "target_photon_energy_erg" in axes
    expected = set(axes) | {"pair_response"}
    if not external:
        expected |= {
            "photon_integration_weights_erg",
            "lepton_integration_weights",
            "absorption_kernel_cm2",
            "lower_support_mask",
            "upper_support_mask",
        }
    if set(response) != expected:
        raise GammaGammaResponseCacheInvalid("gamma-gamma payload fields mismatch")
    for name, axis in axes.items():
        if not np.array_equal(response[name], axis):
            raise GammaGammaResponseCacheInvalid(f"gamma-gamma {name} values mismatch")
    p, d = len(axes["photon_energy_erg"]), len(axes["lepton_gamma"])
    t = len(axes["target_photon_energy_erg"]) if external else p
    shapes = {name: a.shape for name, a in axes.items()}
    shapes["pair_response"] = (d, p, t)
    if not external:
        shapes.update(
            photon_integration_weights_erg=(p,),
            lepton_integration_weights=(d,),
            absorption_kernel_cm2=(p, p),
            lower_support_mask=(p, p),
            upper_support_mask=(p, p),
        )
    for name, value in response.items():
        dtype = np.dtype(bool if name.endswith("support_mask") else np.float64)
        if value.shape != shapes[name] or value.dtype != dtype:
            raise GammaGammaResponseCacheInvalid(
                f"gamma-gamma {name} shape/dtype mismatch"
            )
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise GammaGammaResponseCacheInvalid(
                f"gamma-gamma {name} must be finite nonnegative"
            )


def _load_gamma_gamma(directory, fingerprint, compatibility, axes):
    archive, metadata_path = _gamma_gamma_paths(directory, fingerprint)
    if not archive.is_file() or not metadata_path.is_file():
        raise GammaGammaResponseCacheMiss(
            f"gamma-gamma response cache miss for {fingerprint}; run 'helena cache fetch' "
            "or explicitly 'helena cache build' with the same cache directory"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("fingerprint") != fingerprint
            or metadata.get("compatibility") != compatibility
        ):
            raise GammaGammaResponseCacheInvalid(
                "gamma-gamma cache compatibility mismatch"
            )
        with np.load(archive, allow_pickle=False) as bundle:
            response = {name: bundle[name] for name in bundle.files}
        _validate_gamma_gamma_payload(response, axes)
        identities = {name: _response_identity(a)[1] for name, a in response.items()}
        if metadata.get("arrays") != identities:
            raise GammaGammaResponseCacheInvalid(
                "gamma-gamma cached array identity mismatch"
            )
    except GammaGammaResponseCacheInvalid:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise GammaGammaResponseCacheInvalid(
            f"unreadable gamma-gamma cache: {error}"
        ) from error
    return {name: jnp.asarray(a) for name, a in response.items()}


def _atomic_save_gamma_gamma(directory, fingerprint, compatibility, response):
    directory.mkdir(parents=True, exist_ok=True)
    archive, metadata_path = _gamma_gamma_paths(directory, fingerprint)
    arrays = {name: np.asarray(a) for name, a in response.items()}
    metadata = {
        "fingerprint": fingerprint,
        "compatibility": compatibility,
        "arrays": {name: _response_identity(a)[1] for name, a in arrays.items()},
    }
    with TemporaryDirectory(dir=directory, prefix=".gg-save-") as temporary:
        staged = Path(temporary)
        np.savez_compressed(staged / archive.name, **arrays)
        (staged / metadata_path.name).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staged / archive.name, archive)
        os.replace(staged / metadata_path.name, metadata_path)


def load_or_prepare_gamma_gamma_response_jax(
    photon_energy_erg,
    lepton_gamma,
    target_energy_erg=None,
    *,
    cache_dir=None,
    allow_build=False,
):
    """Load exact fixed gamma-gamma arrays, or explicitly prepare and persist them.

    Photon/target energies are blob-frame erg; daughters are total-energy Lorentz
    factors. Internal responses retain both density factors and the external
    response is labelled internal x reservoir (no event half). Only fixed grids
    and kernels are stored, never densities or fitted parameters. Normal loads
    never download, rebuild, interpolate or coerce precision. ``target_energy_erg``
    selects the external response; the same existing physical builders are used.
    """
    if not isinstance(allow_build, (bool, np.bool_)):
        raise TypeError("allow_build must be boolean")
    axes, compatibility, fingerprint = _gamma_gamma_compatibility(
        photon_energy_erg, lepton_gamma, target_energy_erg
    )
    directory = _directory(cache_dir)
    try:
        return _load_gamma_gamma(directory, fingerprint, compatibility, axes)
    except (GammaGammaResponseCacheMiss, GammaGammaResponseCacheInvalid):
        if not allow_build:
            raise
    from helena.core.jax_backend.chains.gamma_gamma_pair_source import (
        prepare_gamma_gamma_total_field_response_jax,
    )
    from helena.core.jax_backend.kernels.gamma_gamma_pair_injection import (
        prepare_gamma_gamma_pair_response_jax,
    )

    if target_energy_erg is None:
        response = prepare_gamma_gamma_total_field_response_jax(
            photon_energy_erg, lepton_gamma
        )
    else:
        response = {
            **axes,
            "pair_response": prepare_gamma_gamma_pair_response_jax(
                lepton_gamma, photon_energy_erg, target_energy_erg
            ),
        }
    arrays = {name: np.asarray(a) for name, a in response.items()}
    _validate_gamma_gamma_payload(arrays, axes)
    _atomic_save_gamma_gamma(directory, fingerprint, compatibility, arrays)
    return _load_gamma_gamma(directory, fingerprint, compatibility, axes)


def _current_c513_gamma_gamma_grids():
    from helena.models.c513 import build_c513_configuration

    setup = build_c513_configuration()
    config = setup["production_configuration"]
    common = (
        np.asarray(config["gamma_ray_energy_erg"]),
        np.asarray(config["pair_daughter_gamma"]),
    )
    return {
        "internal": (*common, None),
        "external": (
            *common,
            np.asarray(setup["fixed_external_photon_state_blob"]["energy_erg"]),
        ),
    }


def prepare_current_c513_response_cache(cache_dir=None, *, allow_build=False):
    """Strict current BH and gamma-gamma internal/external preparation, no network."""
    directory = prepare_current_c513_bethe_heitler_cache(
        cache_dir, allow_build=allow_build
    )
    for grids in _current_c513_gamma_gamma_grids().values():
        response = load_or_prepare_gamma_gamma_response_jax(
            *grids, cache_dir=directory, allow_build=allow_build
        )
        jax.block_until_ready(response)
    return directory


def fetch_current_c513_response_cache(cache_dir=None):
    """Fetch individually versioned v0.1.1 assets, SHA-check before installation.

    Existing files are never overwritten: a conflicting file is an explicit
    error. Missing files are downloaded into staging first. The old v0.1.0
    binding and explicit legacy operations remain unchanged.
    """
    directory = _directory(cache_dir)
    resource = OFFICIAL_RESPONSE_CACHE_RESOURCE
    files = resource.get("files", {})
    if not files:
        raise BetheHeitlerCacheResourceUnavailable(
            "v0.1.1 cache resources are not yet bound; use 'helena cache build'"
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".helena-fetch-") as temporary:
        staged = Path(temporary)
        missing = []
        for filename, expected in files.items():
            if Path(filename).name != filename or filename in (".", ".."):
                raise BetheHeitlerResponseCacheInvalid("unsafe cache resource filename")
            path = directory / filename
            if path.exists():
                if (
                    path.stat().st_size != expected["size"]
                    or _file_sha256(path) != expected["sha256"]
                ):
                    raise BetheHeitlerResponseCacheInvalid(
                        f"existing cache file conflicts with release: {filename}; use an empty --cache-dir"
                    )
                continue
            _download_official_archive(
                resource["base_url"] + "/" + filename,
                expected["sha256"],
                staged / filename,
            )
            if (staged / filename).stat().st_size != expected["size"]:
                raise BetheHeitlerResponseCacheInvalid(
                    f"cache resource size mismatch: {filename}"
                )
            missing.append(filename)
        directory.mkdir(exist_ok=True)
        for filename in missing:
            destination = directory / filename
            if destination.exists():
                if _file_sha256(destination) != files[filename]["sha256"]:
                    raise BetheHeitlerResponseCacheInvalid(
                        f"cache file appeared during fetch: {filename}"
                    )
            else:
                os.replace(staged / filename, destination)
    return prepare_current_c513_response_cache(directory)
