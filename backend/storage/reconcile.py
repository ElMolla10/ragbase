"""Read-only artifact reconciliation: report inconsistencies, never delete files."""

from pathlib import Path
from uuid import UUID

from backend.storage.local import KEY, ArtifactError, LocalArtifactStore, StoredArtifact


def reconcile(
    store: LocalArtifactStore, references: list[tuple[UUID, StoredArtifact]]
) -> dict:
    referenced = {artifact.key for _, artifact in references}
    inventory = set()
    invalid_paths = []
    for path in store.root.glob("*/*/*.bin"):
        key = path.relative_to(store.root).as_posix()
        match = KEY.fullmatch(key)
        try:
            if not match:
                raise ArtifactError("Invalid key")
            store._path(UUID(match[1]), key)
            if not path.is_file():
                raise ArtifactError("Not a file")
            inventory.add(key)
        except (ValueError, OSError):
            invalid_paths.append(key)
    invalid_references = []
    for workspace, artifact in references:
        try:
            store.read(workspace, artifact)
        except (ArtifactError, OSError):
            invalid_references.append(artifact.key)
    return {
        "mode": "report_only_no_deletions",
        "referenced_count": len(references),
        "orphan_keys": sorted(inventory - referenced),
        "invalid_references": sorted(invalid_references),
        "invalid_paths": sorted(invalid_paths),
        "staging_file_count": sum(1 for path in Path(store.staging).iterdir()),
    }
