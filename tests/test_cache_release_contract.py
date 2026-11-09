"""Portable Bethe--Heitler release-resource contract checks."""

import json
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import jax
import numpy as np
import pytest

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
    assert (
        cli.main(["cache", "status", "--legacy", "--cache-dir", str(destination)]) == 1
    )
    assert "missing:" in capsys.readouterr().out
    assert (
        cli.main(["cache", "fetch", "--legacy", "--cache-dir", str(destination)]) == 0
    )
    capsys.readouterr()
    assert validate_official_bethe_heitler_cache(destination) == destination
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == payload
    assert (
        cli.main(["cache", "status", "--legacy", "--cache-dir", str(destination)]) == 0
    )
    assert "compatible:" in capsys.readouterr().out

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a compatible cache was downloaded again")

    monkeypatch.setattr(cache_module.urllib.request, "urlopen", forbidden)
    assert fetch_official_bethe_heitler_cache(destination) == destination


def test_unbound_official_resource_never_accesses_network(tmp_path, monkeypatch):
    monkeypatch.setitem(cache_module.OFFICIAL_BH_CACHE_RESOURCE["archive"], "url", None)
    monkeypatch.setitem(
        cache_module.OFFICIAL_BH_CACHE_RESOURCE["archive"], "sha256", None
    )
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
    with pytest.raises(
        BetheHeitlerResponseCacheInvalid, match="file manifest mismatch"
    ):
        fetch_official_bethe_heitler_cache(destination)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_installed_resource_rejects_wrong_version_and_corruption(tmp_path, monkeypatch):
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


def test_reference_legacy_resource_identity():
    cache_dir = os.environ.get("HELENA_REFERENCE_BH_CACHE_DIR")
    if not cache_dir:
        pytest.skip("HELENA_REFERENCE_BH_CACHE_DIR is not set")
    directory = validate_official_bethe_heitler_cache(cache_dir)
    for filename, expected in cache_module.OFFICIAL_BH_CACHE_RESOURCE["files"].items():
        if expected["kind"] == "response":
            actual = np.load(directory / filename, allow_pickle=False)
            assert actual.shape == expected["shape"]
            assert sha256(actual.tobytes()).hexdigest() == expected["array_sha256"]


def test_current_model_cache_identity_is_not_legacy_resource():
    current = cache_module._current_c513_bh_grids()
    assert {name: len(grids[2]) for name, grids in current.items()} == {
        "external": 64,
        "internal": 249,
    }
    for grids in current.values():
        assert len(grids[0]) == 513  # BH uses transport daughters, not gamma-gamma537.
        fingerprint = cache_module.bethe_heitler_response_fingerprint(*grids)
        assert (
            fingerprint not in cache_module.OFFICIAL_BH_CACHE_RESOURCE["fingerprints"]
        )


def test_unbound_current_fetch_does_not_install_legacy_grid(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cache_module, "OFFICIAL_RESPONSE_CACHE_RESOURCE", {"files": {}})

    def forbidden(*_args, **_kwargs):
        raise AssertionError("current fetch attempted an incompatible download")

    monkeypatch.setattr(cache_module.urllib.request, "urlopen", forbidden)
    destination = tmp_path / "missing-current"
    assert cli.main(["cache", "fetch", "--cache-dir", str(destination)]) == 1
    assert "helena cache build" in capsys.readouterr().err
    assert not destination.exists()


def test_current_cli_real_small_build_and_strict_reload(tmp_path, monkeypatch, capsys):
    daughter = np.geomspace(1.1, 1e7, 5)
    proton = np.geomspace(1.1, 1e8, 4)
    target = np.geomspace(1e-10, 1e-5, 3)
    grids = {
        "external": (daughter, proton, target),
        "internal": (daughter, proton, target * 2),
    }
    monkeypatch.setattr(cache_module, "_current_c513_bh_grids", lambda: grids)
    monkeypatch.setattr(
        cache_module,
        "_current_c513_gamma_gamma_grids",
        lambda: {
            "internal": (target * 1e3, daughter, None),
            "external": (target * 1e3, daughter, target),
        },
    )
    destination = tmp_path / "fresh-current"
    assert cli.main(["cache", "status", "--cache-dir", str(destination)]) == 1
    assert "missing:" in capsys.readouterr().out
    assert not destination.exists()
    assert cli.main(["cache", "build", "--cache-dir", str(destination)]) == 0
    capsys.readouterr()
    assert len(list(destination.glob("*.npy"))) == 2
    assert len(list(destination.glob("*.npz"))) == 2
    assert any(np.any(np.load(p) > 0) for p in destination.glob("*.npy"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("strict current cache hit rebuilt or downloaded")

    monkeypatch.setattr(
        cache_module.bh_module, "prepare_bethe_heitler_pair_response_jax", forbidden
    )
    monkeypatch.setattr(cache_module.urllib.request, "urlopen", forbidden)
    assert cli.main(["cache", "status", "--cache-dir", str(destination)]) == 0
    assert "current C513 grids" in capsys.readouterr().out
    assert cli.main(["cache", "build", "--cache-dir", str(destination)]) == 0
    assert (
        cli.main(["cache", "status", "--legacy", "--cache-dir", str(destination)]) == 1
    )


def test_current_cache_rejects_legacy_only_directory(tmp_path, monkeypatch, capsys):
    resource, archive_bytes, _ = _mock_resource()
    monkeypatch.setattr(cache_module, "OFFICIAL_BH_CACHE_RESOURCE", resource)
    monkeypatch.setattr(
        cache_module.urllib.request, "urlopen", lambda *_a, **_k: BytesIO(archive_bytes)
    )
    destination = tmp_path / "legacy-only"
    fetch_official_bethe_heitler_cache(destination)
    assert cli.main(["cache", "status", "--cache-dir", str(destination)]) == 1
    assert "missing:" in capsys.readouterr().out


def test_gamma_gamma_strict_cache_roundtrip_and_identity(tmp_path, monkeypatch):
    from helena.core.jax_backend.kernels import gamma_gamma_pair_injection as pair

    photon = np.geomspace(2e-7, 3e-4, 5)
    daughter = np.geomspace(1.000001, 1000, 9)
    target = np.geomspace(1e-8, 5e-5, 4)
    load = cache_module.load_or_prepare_gamma_gamma_response_jax
    for external in (None, target):
        with pytest.raises(cache_module.GammaGammaResponseCacheMiss):
            load(photon, daughter, external, cache_dir=tmp_path)
        built = load(photon, daughter, external, cache_dir=tmp_path, allow_build=True)
        assert np.any(np.asarray(built["pair_response"]) > 0)
        loaded = load(photon, daughter, external, cache_dir=tmp_path)
        for key in built:
            np.testing.assert_array_equal(built[key], loaded[key])
    assert cache_module.gamma_gamma_response_fingerprint(
        photon, daughter
    ) != cache_module.gamma_gamma_response_fingerprint(photon, daughter, photon)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("normal cache access must not build or download")

    monkeypatch.setattr(pair, "prepare_gamma_gamma_pair_response_jax", forbidden)
    monkeypatch.setattr(cache_module.urllib.request, "urlopen", forbidden)
    load(photon, daughter, cache_dir=tmp_path)
    with pytest.raises(cache_module.GammaGammaResponseCacheMiss):
        load(photon * 1.01, daughter, cache_dir=tmp_path)
    fingerprint = cache_module.gamma_gamma_response_fingerprint(photon, daughter)
    archive, metadata_path = cache_module._gamma_gamma_paths(tmp_path, fingerprint)
    with np.load(archive) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["pair_response"] *= 2
    np.savez_compressed(archive, **arrays)
    with pytest.raises(
        cache_module.GammaGammaResponseCacheInvalid, match="identity mismatch"
    ):
        load(photon, daughter, cache_dir=tmp_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["compatibility"]["physics"]["float_precision"] = "float32"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(
        cache_module.GammaGammaResponseCacheInvalid, match="compatibility mismatch"
    ):
        load(photon, daughter, cache_dir=tmp_path)


def test_current_fetch_all_assets_sha_and_reload(tmp_path, monkeypatch, capsys):
    daughter = np.geomspace(1.000001, 1000, 7)
    proton = np.geomspace(1.1, 1e6, 4)
    photon = np.geomspace(2e-7, 3e-4, 4)
    target = np.geomspace(1e-8, 5e-5, 3)
    monkeypatch.setattr(
        cache_module,
        "_current_c513_bh_grids",
        lambda: {
            "internal": (daughter, proton, photon),
            "external": (daughter, proton, target),
        },
    )
    monkeypatch.setattr(
        cache_module,
        "_current_c513_gamma_gamma_grids",
        lambda: {
            "internal": (photon, daughter, None),
            "external": (photon, daughter, target),
        },
    )
    source = tmp_path / "built"
    cache_module.prepare_current_c513_response_cache(source, allow_build=True)
    payload = {p.name: p.read_bytes() for p in source.iterdir()}
    resource = {
        "version": "0.1.1",
        "base_url": "https://example.invalid/v0.1.1",
        "files": {
            n: {"size": len(v), "sha256": sha256(v).hexdigest()}
            for n, v in payload.items()
        },
    }
    monkeypatch.setattr(cache_module, "OFFICIAL_RESPONSE_CACHE_RESOURCE", resource)
    calls = []

    def download(url, **_kwargs):
        calls.append(url)
        return BytesIO(payload[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(cache_module.urllib.request, "urlopen", download)
    destination = tmp_path / "downloaded"
    assert cli.main(["cache", "fetch", "--cache-dir", str(destination)]) == 0
    assert len(calls) == 8
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == payload
    assert cli.main(["cache", "status", "--cache-dir", str(destination)]) == 0
    assert cli.main(["cache", "fetch", "--cache-dir", str(destination)]) == 0
    assert len(calls) == 8
    first = next(iter(resource["files"]))
    resource["files"][first]["sha256"] = "f" * 64
    missing = tmp_path / "bad-download"
    assert cli.main(["cache", "fetch", "--cache-dir", str(missing)]) == 1
    assert not missing.exists()
    assert "SHA-256" in capsys.readouterr().err


def test_c513_gamma_gamma_preparation_uses_persistent_loader(tmp_path, monkeypatch):
    from helena.models import c513

    setup = c513.build_c513_configuration()
    setup["production_configuration"]["include_external_gamma_gamma_target"] = True
    calls = []
    monkeypatch.setattr(
        c513,
        "load_or_prepare_bethe_heitler_pair_response_jax",
        lambda *a, **k: np.zeros(1),
    )

    def load(photon, daughter, target=None, **kwargs):
        calls.append((np.asarray(photon), np.asarray(daughter), target, kwargs))
        return (
            {"target_photon_energy_erg": target, "pair_response": np.zeros(1)}
            if target is not None
            else {"pair_response": np.zeros(1)}
        )

    monkeypatch.setattr(c513, "load_or_prepare_gamma_gamma_response_jax", load)
    result = c513.prepare_c513_responses(setup, cache_dir=tmp_path)
    assert len(calls) == 2
    assert all(c[3] == {"cache_dir": tmp_path, "allow_build": False} for c in calls)
    assert len(calls[0][0]) == 249 and len(calls[0][1]) == 537
    assert len(calls[1][2]) == 64
    assert "external_pair_response" in result["gamma_gamma_pair_response"]
