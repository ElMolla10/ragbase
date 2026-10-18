"""Immutable originals must survive reopening and reject path/integrity attacks."""

from dataclasses import replace
from uuid import uuid4

import pytest

from backend.storage.local import ArtifactError, LocalArtifactStore


def test_durable_roundtrip_and_immutable_version(tmp_path):
    workspace, version = uuid4(), uuid4()
    store = LocalArtifactStore(tmp_path, max_bytes=1024)
    artifact = store.put(workspace, version, "بيانات الاختبار".encode())
    assert (
        LocalArtifactStore(tmp_path, max_bytes=1024).read(workspace, artifact)
        == "بيانات الاختبار".encode()
    )
    with pytest.raises(ArtifactError, match="immutable"):
        store.put(workspace, version, b"replacement")
    assert store.read(workspace, artifact) == "بيانات الاختبار".encode()
    assert not list((tmp_path / ".staging").iterdir())


def test_wrong_workspace_and_tampered_file_are_rejected(tmp_path):
    workspace = uuid4()
    store = LocalArtifactStore(tmp_path, max_bytes=1024)
    artifact = store.put(workspace, uuid4(), b"original")
    with pytest.raises(ArtifactError):
        store.read(uuid4(), artifact)
    (tmp_path / artifact.key).write_bytes(b"tampered")
    with pytest.raises(ArtifactError, match="integrity"):
        store.read(workspace, artifact)


@pytest.mark.parametrize(
    "key",
    [
        "../secret",
        "/etc/passwd",
        "C:\\secrets.txt",
        "x/../../secret",
        "-" * 36 + "/" + "-" * 36 + "/" + "a" * 64 + ".bin",
    ],
)
def test_untrusted_paths_cannot_escape(tmp_path, key):
    store = LocalArtifactStore(tmp_path, max_bytes=1024)
    workspace = uuid4()
    artifact = store.put(workspace, uuid4(), b"test")
    with pytest.raises(ArtifactError):
        store.read(workspace, replace(artifact, key=key))


def test_symlinked_workspace_cannot_redirect_to_another_workspace(tmp_path):
    store = LocalArtifactStore(tmp_path, max_bytes=1024)
    first, second = uuid4(), uuid4()
    (tmp_path / str(second)).mkdir()
    (tmp_path / str(first)).symlink_to(tmp_path / str(second), target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlinks"):
        store.put(first, uuid4(), b"blocked")


@pytest.mark.parametrize("data", [b"", b"oversized"])
def test_empty_or_oversized_writes_are_rejected(tmp_path, data):
    with pytest.raises(ArtifactError, match="byte limit"):
        LocalArtifactStore(tmp_path, max_bytes=3).put(uuid4(), uuid4(), data)
