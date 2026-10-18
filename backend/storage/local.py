"""Immutable local artifacts with safe keys, atomic publication and integrity checks."""

import hashlib
import hmac
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

KEY = re.compile(r"([0-9a-f-]{36})/([0-9a-f-]{36})/([0-9a-f]{64})\.bin\Z")


class ArtifactError(ValueError):
    """Artifact path, size, immutability or integrity validation failed."""


@dataclass(frozen=True)
class StoredArtifact:
    key: str
    sha256: str
    size: int


class LocalArtifactStore:
    def __init__(self, root: Path, *, max_bytes: int):
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ArtifactError("Storage root must not be a symbolic link")
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self.staging = self.root / ".staging"
        self.staging.mkdir(exist_ok=True)
        if self.staging.is_symlink():
            raise ArtifactError("Staging directory must not be a symbolic link")

    def _path(self, workspace: UUID, key: str) -> Path:
        match = KEY.fullmatch(key)
        try:
            valid = (
                match
                and str(UUID(match[1])) == str(workspace)
                and str(UUID(match[2])) == match[2]
            )
        except ValueError:
            valid = False
        if not valid:
            raise ArtifactError("Invalid artifact key or workspace")
        path = self.root / key
        for part in (path.parent.parent, path.parent, path):
            if part.is_symlink():
                raise ArtifactError("Artifact symlinks are forbidden")
        if not path.resolve().is_relative_to(self.root):
            raise ArtifactError("Artifact path escaped storage root")
        return path

    def put(self, workspace: UUID, version: UUID, data: bytes) -> StoredArtifact:
        if not data or len(data) > self.max_bytes:
            raise ArtifactError("Artifact is empty or exceeds byte limit")
        digest = hashlib.sha256(data).hexdigest()
        key = f"{workspace}/{version}/{digest}.bin"
        path = self._path(workspace, key)
        path.parent.parent.mkdir(exist_ok=True)
        try:
            path.parent.mkdir()
        except FileExistsError as exc:
            raise ArtifactError("Artifact version is immutable") from exc
        self._path(workspace, key)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.staging, delete=False) as file:
                temporary = Path(file.name)
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            # Atomic, same filesystem, and unlike rename never overwrites a target.
            os.link(temporary, path)
            if os.name == "posix":
                for directory in (path.parent, path.parent.parent, self.root):
                    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
        except FileExistsError as exc:
            raise ArtifactError("Artifact version is immutable") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return StoredArtifact(key, digest, len(data))

    def read(self, workspace: UUID, artifact: StoredArtifact) -> bytes:
        path = self._path(workspace, artifact.key)
        if artifact.size < 1 or artifact.size > self.max_bytes:
            raise ArtifactError("Artifact size is invalid")
        try:
            fd = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            with os.fdopen(fd, "rb") as file:
                metadata = os.fstat(file.fileno())
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_size != artifact.size
                ):
                    raise ArtifactError("Artifact integrity check failed")
                data = file.read(self.max_bytes + 1)
        except OSError as exc:
            raise ArtifactError("Artifact unavailable") from exc
        if len(data) != artifact.size or not hmac.compare_digest(
            hashlib.sha256(data).hexdigest(), artifact.sha256
        ):
            raise ArtifactError("Artifact integrity check failed")
        return data
