"""Enterprise configuration is separate from the legacy demo's environment."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True)
class EnterpriseSettings:
    profile: str = "dev-lite"
    database_url: str | None = field(default=None, repr=False)
    storage_root: Path | None = None
    dev_tokens: dict[str, UUID] = field(default_factory=dict, repr=False)
    max_upload_bytes: int = 5_000_000

    def __post_init__(self):
        if self.profile != "dev-lite":
            raise ValueError("Only the development enterprise profile is implemented")
        if self.database_url and not self.database_url.startswith(
            "postgresql+asyncpg://"
        ):
            raise ValueError("Enterprise database requires the async PostgreSQL driver")
        if self.database_url and not self.storage_root:
            raise ValueError("Enterprise storage root is required with a database")
        if not 1 <= self.max_upload_bytes <= 50_000_000:
            raise ValueError("Upload byte limit must be between 1 and 50000000")
        if any(
            len(token) < 32 or not isinstance(actor, UUID)
            for token, actor in self.dev_tokens.items()
        ):
            raise ValueError(
                "Development tokens require random secrets and UUID identities"
            )

    @classmethod
    def from_env(cls):
        try:
            tokens = json.loads(os.environ.get("ENTERPRISE_DEV_TOKENS", "{}"))
            if not isinstance(tokens, dict):
                raise ValueError
            return cls(
                profile=os.environ.get("RAGBASE_PROFILE", "dev-lite"),
                database_url=os.environ.get("ENTERPRISE_DATABASE_URL") or None,
                storage_root=Path(os.environ["ENTERPRISE_STORAGE_ROOT"])
                if os.environ.get("ENTERPRISE_STORAGE_ROOT")
                else None,
                dev_tokens={token: UUID(actor) for token, actor in tokens.items()},
                max_upload_bytes=int(
                    os.environ.get("ENTERPRISE_MAX_UPLOAD_BYTES", "5000000")
                ),
            )
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid enterprise development configuration") from exc
