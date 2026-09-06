"""Portable Bethe--Heitler release-resource contract checks."""

from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import jax
import numpy as np
import pytest

from helena import ModelConfig, OneZoneModel
import helena.cli as cli
import helena.runtime_cache as cache_module
from helena.runtime_cache import (
    BetheHeitlerCacheResourceUnavailable,
    BetheHeitlerResponseCacheInvalid,
    bethe_heitler_cache_directory,
    fetch_official_bethe_heitler_cache,
    validate_official_bethe_heitler_cache,
)

jax.config.update("jax_enable_x64", True)


def _mock_resource():
    fingerprint = "0" * 64
    array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    array_buffer = BytesIO()
    np.save(array_buffer, array, allow_pickle=False)
    array_bytes = array_buffer.getvalue()
    array_sha256 = sha256(array.tobytes()).hexdigest()
    metadata = {
        "fingerprint": fingerprint,
        "compatibility": {
            "cache_schema_version": 1,
            "physics": {"response_version": "ka2008_born_omega_lt_600_v1"},
        },
        "response": {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": array_sha256,
        },
    }
    metadata_bytes = (json.dumps(metadata, sort_keys=True) + "\n").encode()
    stem = f"bh_pair_response_{fingerprint}"
    payload = {
        f"{stem}.json": metadata_bytes,
        f"{stem}.npy": array_bytes,
    }
    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", compression=ZIP_STORED) as archive:
        for filename, content in payload.items():
            archive.writestr(filename, content)
    archive_bytes = archive_buffer.getvalue()
    resource = {
        "resource_name": "test_bethe_heitler_response",
        "cache_format_version": 1,
        "physics_response_version": "ka2008_born_omega_lt_600_v1",
        "grid_identifier": "test-grid-v1",
        "dtype": "<f8",
        "fingerprints": (fingerprint,),
        "archive": {
            "format": "zip",
            "url": "https://example.invalid/bh-cache.zip",
            "sha256": sha256(archive_bytes).hexdigest(),
        },
        "files": {
            f"{stem}.json": {
                "kind": "metadata",
                "fingerprint": fingerprint,
                "size": len(metadata_bytes),
                "sha256": sha256(metadata_bytes).hexdigest(),
            },
            f"{stem}.npy": {
                "kind": "response",
                "fingerprint": fingerprint,
                "size": len(array_bytes),
                "sha256": sha256(array_bytes).hexdigest(),
                "array_sha256": array_sha256,
                "shape": array.shape,
                "dtype": array.dtype.str,
            },
        },
    }
    return resource, archive_bytes, payload


def _config():
    return ModelConfig(
        magnetic_field_G=0.1,
        radius_cm=1.0e16,
        redshift=0.1,
        doppler_factor=30.0,
        luminosity_distance_cm=1.0e27,
        electron_kinetic_injection_luminosity_erg_s=3.7e40,
        electron_injection_index=2.0,
        electron_gamma_max=3.0e5,
        proton_kinetic_injection_luminosity_erg_s=2.8e46,
        proton_injection_index=2.0,
        proton_gamma_max=1.0e7,
    )


def test_cache_path_precedence_and_platform_defaults(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"
    home = tmp_path / "home"
    monkeypatch.setenv("HELENA_CACHE_DIR", str(configured))
    assert bethe_heitler_cache_directory(explicit) == explicit
    assert bethe_heitler_cache_directory() == configured

    monkeypatch.delenv("HELENA_CACHE_DIR")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cache_module.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert bethe_heitler_cache_directory() == (
        tmp_path / "xdg" / "helena" / "bethe_heitler"
    )
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-is-invalid")
    assert bethe_heitler_cache_directory() == (
        home / ".cache" / "helena" / "bethe_heitler"
    )

    monkeypatch.setattr(cache_module.sys, "platform", "darwin")
    assert bethe_heitler_cache_directory() == (
        home / "Library" / "Caches" / "helena" / "bethe_heitler"
    )

    monkeypatch.setattr(cache_module.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert bethe_heitler_cache_directory() == (
        tmp_path / "local" / "helena" / "Cache" / "bethe_heitler"
    )


def test_manifest_identity_and_mocked_cli_fetch(tmp_path, monkeypatch, capsys):
    resource, archive_bytes, payload = _mock_resource()
    monkeypatch.setattr(cache_module, "OFFICIAL_BH_CACHE_RESOURCE", resource)
    monkeypatch.setattr(
        cache_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(archive_bytes),
    )
    required = {
        "resource_name",
        "cache_format_version",
        "physics_response_version",
        "grid_identifier",
        "dtype",
        "files",
        "archive",
    }
    assert required <= resource.keys()

    destination = tmp_path / "custom-cache"
    assert cli.main(["cache", "path", "--cache-dir", str(destination)]) == 0
    assert Path(capsys.readouterr().out.strip()) == destination
    assert cli.main(["cache", "status", "--cache-dir", str(destination)]) == 1
    assert "missing:" in capsys.readouterr().out
    assert cli.main(["cache", "fetch", "--cache-dir", str(destination)]) == 0
    capsys.readouterr()
    assert validate_official_bethe_heitler_cache(destination) == destination
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == payload
    assert cli.main(["cache", "status", "--cache-dir", str(destination)]) == 0
    assert "compatible:" in capsys.readouterr().out

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a compatible cache was downloaded again")

    monkeypatch.setattr(cache_module.urllib.request, "urlopen", forbidden)
    assert fetch_official_bethe_heitler_cache(destination) == destination


def test_unbound_official_resource_never_accesses_network(tmp_path, monkeypatch):
    monkeypatch.setitem(cache_module.OFFICIAL_BH_CACHE_RESOURCE["archive"], "url", None)
    monkeypatch.setitem(cache_module.OFFICIAL_BH_CACHE_RESOURCE["archive"], "sha256", None)
    assert cache_module.OFFICIAL_BH_CACHE_RESOURCE["archive"]["url"] is None
    assert cache_module.OFFICIAL_BH_CACHE_RESOURCE["archive"]["sha256"] is None

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an unbound resource attempted a download")

    monkeypatch.setattr(cache_module.urllib.request, "urlopen", forbidden)
    destination = tmp_path / "unbound"
    with pytest.raises(BetheHeitlerCacheResourceUnavailable, match="not yet bound"):
        fetch_official_bethe_heitler_cache(destination)
    assert not destination.exists()


def test_fetch_rejects_bad_archive_and_interruption_atomically(tmp_path, monkeypatch):
    resource, archive_bytes, _ = _mock_resource()
    resource["archive"] = {**resource["archive"], "sha256": "f" * 64}
    monkeypatch.setattr(cache_module, "OFFICIAL_BH_CACHE_RESOURCE", resource)
    monkeypatch.setattr(
        cache_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(archive_bytes),
    )
    bad_destination = tmp_path / "bad-checksum"
    with pytest.raises(BetheHeitlerResponseCacheInvalid, match="archive SHA-256"):
        fetch_official_bethe_heitler_cache(bad_destination)
    assert not bad_destination.exists()

    resource["archive"]["sha256"] = sha256(archive_bytes).hexdigest()

    class InterruptedDownload(BytesIO):
        def read(self, *_args, **_kwargs):
            if self.tell():
                raise OSError("interrupted")
            return super().read(64)

    monkeypatch.setattr(
        cache_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: InterruptedDownload(archive_bytes),
    )
    interrupted_destination = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="interrupted"):
        fetch_official_bethe_heitler_cache(interrupted_destination)
    assert not interrupted_destination.exists()
    assert not list(tmp_path.glob(".*-fetch-*"))


@pytest.mark.parametrize(
    "extra", ["duplicate", "directory", "traversal", "unexpected", "missing"]
)
def test_fetch_requires_exact_archive_members(tmp_path, monkeypatch, extra):
    resource, _, payload = _mock_resource()
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as bundle:
        for name, content in payload.items():
            if extra != "missing" or name.endswith(".json"):
                bundle.writestr(name, content)
        if extra == "duplicate":
            name = next(iter(payload))
            with pytest.warns(UserWarning, match="Duplicate name"):
                bundle.writestr(name, payload[name])
        elif extra == "directory":
            bundle.writestr("unexpected/", b"")
        elif extra == "traversal":
            bundle.writestr("../escape.txt", b"outside")
        elif extra == "unexpected":
            bundle.writestr("README.txt", b"extra")
    archive_bytes = buffer.getvalue()
    resource["archive"]["sha256"] = sha256(archive_bytes).hexdigest()
    monkeypatch.setattr(cache_module, "OFFICIAL_BH_CACHE_RESOURCE", resource)
    monkeypatch.setattr(
        cache_module.urllib.request, "urlopen", lambda *_a, **_k: BytesIO(archive_bytes)
    )
    destination = tmp_path / "cache"
    with pytest.raises(BetheHeitlerResponseCacheInvalid, match="file manifest mismatch"):
        fetch_official_bethe_heitler_cache(destination)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_installed_resource_rejects_wrong_version_and_corruption(
    tmp_path, monkeypatch
):
    resource, archive_bytes, _ = _mock_resource()
    monkeypatch.setattr(cache_module, "OFFICIAL_BH_CACHE_RESOURCE", resource)
    monkeypatch.setattr(
        cache_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(archive_bytes),
    )
    destination = tmp_path / "cache"
    fetch_official_bethe_heitler_cache(destination)
    metadata_path = next(destination.glob("*.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["compatibility"]["physics"]["response_version"] = "stale"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(BetheHeitlerResponseCacheInvalid, match="version mismatch"):
        validate_official_bethe_heitler_cache(destination)

    destination = tmp_path / "corrupt-cache"
    fetch_official_bethe_heitler_cache(destination)
    response_path = next(destination.glob("*.npy"))
    content = response_path.read_bytes()
    response_path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(BetheHeitlerResponseCacheInvalid, match="SHA-256 mismatch"):
        validate_official_bethe_heitler_cache(destination)


def test_reference_cache_parity_and_normal_model_load_never_downloads(monkeypatch):
    cache_dir = os.environ.get("HELENA_REFERENCE_BH_CACHE_DIR")
    if not cache_dir:
        pytest.skip("HELENA_REFERENCE_BH_CACHE_DIR is not set")
    directory = validate_official_bethe_heitler_cache(cache_dir)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal model resource discovery accessed the network")

    monkeypatch.setattr(cache_module.urllib.request, "urlopen", forbidden)
    prepared = OneZoneModel(_config(), cache_dir=directory)._get_prepared_responses()
    fingerprints = prepared["bethe_heitler_response_fingerprints"]
    for name, response in prepared["bethe_heitler_pair_responses"].items():
        fingerprint = fingerprints[name]
        filename = f"bh_pair_response_{fingerprint}.npy"
        expected = cache_module.OFFICIAL_BH_CACHE_RESOURCE["files"][filename]
        actual = np.ascontiguousarray(np.asarray(response))
        assert actual.shape == expected["shape"]
        assert actual.dtype.str == expected["dtype"]
        assert sha256(actual.tobytes()).hexdigest() == expected["array_sha256"]
