"""Enterprise configuration is separate from the legacy demo's environment."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID


@dataclass(frozen=True)
class EnterpriseSettings:
    profile: str = "dev-lite"
    database_url: str | None = field(default=None, repr=False)
    storage_root: Path | None = None
    dev_tokens: dict[str, UUID] = field(default_factory=dict, repr=False)
    max_upload_bytes: int = 5_000_000
    generation_mode: str = "disabled"
    model_endpoint: str = "http://model-relay:8001"
    model_name: str = ""
    model_digest: str = ""

    def __post_init__(self):
        endpoint = urlsplit(self.model_endpoint)
        if self.generation_mode not in {"disabled", "fixture", "ollama"}:
            raise ValueError("Unknown development generation mode")
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"model-relay", "localhost", "127.0.0.1"}
            or endpoint.username
            or endpoint.password
            or endpoint.query
            or endpoint.fragment
            or endpoint.path not in {"", "/"}
        ):
            raise ValueError(
                "Generation endpoint must be an approved local development service"
            )
        if self.generation_mode == "ollama" and (
            not self.model_name
            or len(self.model_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.model_digest)
        ):
            raise ValueError("Ollama mode requires a pinned installed checkpoint")
        if self.profile != "dev-lite":
            raise ValueError("Only the development enterprise profile is implemented")
        if self.database_url and not self.database_url.startswith(
            "postgresql+asyncpg://"
        ):
            raise ValueError("Enterprise database requires the async PostgreSQL driver")
        if self.database_url and not self.storage_root:
            raise ValueError("Enterprise storage root is required with a database")
        if not 1 <= self.max_upload_bytes <= 5_000_000:
            raise ValueError(
                "Upload byte limit must be between 1 and 5000000 for the bounded parser"
            )
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
                generation_mode=os.environ.get(
                    "ENTERPRISE_GENERATION_MODE", "disabled"
                ),
                model_endpoint=os.environ.get(
                    "ENTERPRISE_MODEL_ENDPOINT", "http://model-relay:8001"
                ),
                model_name=os.environ.get("ENTERPRISE_MODEL_NAME", ""),
                model_digest=os.environ.get("ENTERPRISE_MODEL_DIGEST", ""),
            )
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid enterprise development configuration") from exc
