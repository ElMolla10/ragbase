"""Reconciliation reports failed commits, missing files and corruption without deletion."""

from uuid import uuid4

from backend.storage.local import LocalArtifactStore
from backend.storage.reconcile import reconcile


def test_orphan_and_corrupt_references_are_reported_without_deleting(tmp_path):
    store = LocalArtifactStore(tmp_path, max_bytes=100)
    workspace = uuid4()
    orphan = store.put(workspace, uuid4(), b"uncommitted")
    referenced = store.put(workspace, uuid4(), b"committed")
    (tmp_path / referenced.key).write_bytes(b"corruption")
    report = reconcile(store, [(workspace, referenced)])
    assert report["orphan_keys"] == [orphan.key]
    assert report["invalid_references"] == [referenced.key]
    assert (tmp_path / orphan.key).is_file()
    assert (tmp_path / referenced.key).is_file()
